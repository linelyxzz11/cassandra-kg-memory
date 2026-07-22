import argparse
import csv
import hashlib
import json
import random
import statistics
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cassandra.cluster import Cluster

OUT_DIR = Path("D:/memorytable/cassandra-kg-memory/results/system")
RAW_DIR = OUT_DIR / "raw_latency_logs"

RELATIONS = ["likes", "suitable_for", "related_to", "suggests", "visited", "talked_to", "helped", "works_at", "bought", "reviewed", "attended", "planned", "remembered"]
RELATION_PATH_B2B = ["likes", "suitable_for", "related_to", "suggests"]
NOISE_RELATIONS = ["visited", "talked_to", "helped", "works_at", "bought", "reviewed", "attended", "planned", "remembered"]
B2_DEFAULT_SELECTIVITIES = [0.01, 0.10, 0.50]
GRAPH_ID = "synth_1M"
CACHE_CAP = 200
DEGREE_THRESHOLD = 100
REPEATS = 5
KEYSPACE = "ai_memory"
CASSANDRA_HOSTS = ["127.0.0.1"]
CASSANDRA_PORT = 9042
SEED = 42

INSERT_SRC = """
INSERT INTO kg_edges_by_src (graph_id, src_id, relation, dst_id, edge_id, src_type, dst_type, confidence, source, created_at)
VALUES (?, ?, ?, ?, now(), ?, ?, ?, ?, toTimestamp(now()))
"""
INSERT_SRC_RELATION = """
INSERT INTO kg_edges_by_src_relation (graph_id, src_id, relation, dst_id, edge_id, src_type, dst_type, confidence, source, created_at)
VALUES (?, ?, ?, ?, now(), ?, ?, ?, ?, toTimestamp(now()))
"""
INSERT_DST = """
INSERT INTO kg_edges_by_dst (graph_id, dst_id, relation, src_id, edge_id, src_type, dst_type, confidence, source, created_at)
VALUES (?, ?, ?, ?, now(), ?, ?, ?, ?, toTimestamp(now()))
"""
INSERT_RELATION_BUCKET = """
INSERT INTO kg_edges_by_relation_bucket (graph_id, relation, bucket, src_id, dst_id, edge_id, src_type, dst_type, confidence, source, created_at)
VALUES (?, ?, ?, ?, ?, now(), ?, ?, ?, ?, toTimestamp(now()))
"""
TRUNCATE_TABLES = [
    "TRUNCATE kg_edges_by_src",
    "TRUNCATE kg_edges_by_dst",
    "TRUNCATE kg_edges_by_relation_bucket",
    "TRUNCATE kg_edges_by_src_relation",
]

B1_SUMMARY = OUT_DIR / "layerB_B1_parallel_summary.csv"
B2_SUMMARY = OUT_DIR / "layerB_B2_relation_index_summary.csv"
B3_SUMMARY = OUT_DIR / "layerB_B3_cache_summary.csv"
B1_DETAIL = OUT_DIR / "layerB_parallel_worker_sweep.csv"
B2_DETAIL = OUT_DIR / "layerB_relation_index_characterization.csv"
B3_DETAIL = OUT_DIR / "layerB_cache_effective_latency.csv"


def stable_bucket(value, bucket_count=64):
    """Deterministic bucket assignment.

    Do not use Python's built-in hash() here because it is salted per process,
    which makes bucket assignment non-reproducible across runs.
    """
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % bucket_count


def percentile(data, p):
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (k - f) * (s[c] - s[f]) if c > f else s[f]


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def append_csv(path, row, fieldnames):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if p.exists() else "w"
    with p.open(mode, encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        w.writerow(row)


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def make_edge(graph_id, src_id, relation, dst_id, rng):
    sid = rng.randint(1, 100)
    ev = f"D{sid}:{rng.randint(1, 50)}"
    return {
        "graph_id": graph_id,
        "src_id": src_id,
        "relation": relation,
        "dst_id": dst_id,
        "src_type": "ENTITY",
        "dst_type": "ENTITY",
        "confidence": round(rng.uniform(0.5, 1.0), 2),
        "source": f"synthetic|{ev}",
    }


class SyntheticGraph:
    def __init__(self, n_entities=10000, n_edges=1000000, seed=SEED,
                 high_degree_frac=0.02, high_degree_mult=20, relation_types=None):
        self.n_entities = n_entities
        self.n_edges = n_edges
        self.seed = seed
        self.high_degree_frac = high_degree_frac
        self.high_degree_mult = high_degree_mult
        self.relation_types = relation_types or RELATIONS
        self.rng = random.Random(seed)
        self.entity_ids = [f"entity_{i}" for i in range(n_entities)]
        self.edges = []
        self.entity_outdegree = defaultdict(int)
        self.entity_edges = defaultdict(list)
        self.high_degree_entities = set()
        self.n_actual = 0
        self.b2_sources = {}
        self.b2b_starts = []
        self.b2_config = {}
        self.b3_hubs = []
        self.b3_normals = []
        self.b3_query_stream = []
        self.b3_config = {}

    def generate(self):
        n_high = max(1, int(self.n_entities * self.high_degree_frac))
        self.high_degree_entities = set(self.rng.sample(self.entity_ids, n_high))
        edges_per_entity = self.n_edges // self.n_entities
        generated = 0
        for entity in self.entity_ids:
            if entity in self.high_degree_entities:
                n_out = edges_per_entity * self.high_degree_mult
            else:
                n_out = edges_per_entity
            for _ in range(n_out):
                if generated >= self.n_edges:
                    break
                rel = self.rng.choice(self.relation_types)
                dst = self.rng.choice(self.entity_ids)
                sid = self.rng.randint(1, 100)
                ev = f"D{sid}:{self.rng.randint(1, 50)}"
                edge = {
                    "graph_id": GRAPH_ID,
                    "src_id": entity,
                    "relation": rel,
                    "dst_id": dst,
                    "src_type": "ENTITY",
                    "dst_type": "ENTITY",
                    "confidence": round(self.rng.uniform(0.5, 1.0), 2),
                    "source": f"synthetic|{ev}",
                }
                self.edges.append(edge)
                self.entity_outdegree[entity] += 1
                self.entity_edges[entity].append(edge)
                generated += 1
                if generated >= self.n_edges:
                    break
        self.n_actual = len(self.edges)
        return self

    def generate_b2_selectivity_graph(self, n_sources=10, outdegree=5000,
                                       selectivities=None, target_relation="likes",
                                       b2b_path=None, b2b_path_fanout=1):
        """Generate a B2-specific graph for relation-index characterization.

        B2a: high-outdegree one-hop nodes with controlled per-source relation
        selectivity. For each source, src_scan reads `outdegree` rows, while
        src_relation_index reads only target-relation rows.

        B2b: fixed linear 4-hop relation path with high noise on every source
        node. The linear chain avoids accidental path breaks and keeps the
        expected indexed raw scan easy to verify.
        """
        if selectivities is None:
            selectivities = B2_DEFAULT_SELECTIVITIES
        if b2b_path is None:
            b2b_path = RELATION_PATH_B2B
        noise_relations = [r for r in self.relation_types if r != target_relation]
        if not noise_relations:
            noise_relations = [r for r in NOISE_RELATIONS if r != target_relation]
        if outdegree < 2:
            raise ValueError("b2 outdegree must be >= 2")

        self.edges = []
        self.entity_outdegree.clear()
        self.entity_edges.clear()
        self.high_degree_entities.clear()
        self.b2_sources = {}
        self.b2b_starts = []
        self.b2_config = {}
        self.b3_hubs = []
        self.b3_normals = []
        self.b3_query_stream = []
        self.b3_config = {}

        def add_edge(src, rel, dst):
            edge = make_edge(GRAPH_ID, src, rel, dst, self.rng)
            self.edges.append(edge)
            self.entity_outdegree[src] += 1
            self.entity_edges[src].append(edge)

        # B2a: per-source selectivity nodes.
        for sel in selectivities:
            sel_label = f"sel{int(round(sel * 1000)):03d}"
            srcs = [f"b2_{sel_label}_src_{i}" for i in range(n_sources)]
            self.b2_sources[sel] = srcs
            target_count = max(1, int(round(outdegree * sel)))
            target_count = min(target_count, outdegree)
            noise_count = outdegree - target_count
            self.b2_config[sel] = {
                "n_sources": n_sources,
                "outdegree": outdegree,
                "target_edges": target_count,
                "noise_edges": noise_count,
                "target_relation": target_relation,
            }
            for src_idx, src in enumerate(srcs):
                for j in range(target_count):
                    add_edge(src, target_relation, f"b2_{sel_label}_src{src_idx}_target_{j}")
                for j in range(noise_count):
                    rel = self.rng.choice(noise_relations)
                    add_edge(src, rel, f"b2_{sel_label}_src{src_idx}_noise_{j}")

        # B2b: fixed linear path. No branching by default.
        # start_i -> pref_i -> need_i -> state_i -> strategy_i
        self.b2b_starts = [f"b2_path_start_{i}" for i in range(n_sources)]
        self.b2_config["B2b"] = {
            "n_sources": n_sources,
            "outdegree": outdegree,
            "path": list(b2b_path),
            "path_fanout": b2b_path_fanout,
            "target_relation": target_relation,
            "note": "linear_chain_with_noise",
        }
        node_levels = ["start", "pref", "need", "state", "strategy"]
        for i in range(n_sources):
            nodes = {
                "start": f"b2_path_start_{i}",
                "pref": f"b2_path_pref_{i}",
                "need": f"b2_path_need_{i}",
                "state": f"b2_path_state_{i}",
                "strategy": f"b2_path_strategy_{i}",
            }
            src_dst_pairs = [
                (nodes["start"], b2b_path[0], nodes["pref"]),
                (nodes["pref"], b2b_path[1], nodes["need"]),
                (nodes["need"], b2b_path[2], nodes["state"]),
                (nodes["state"], b2b_path[3], nodes["strategy"]),
            ]
            for src, rel, dst in src_dst_pairs:
                # Add the target relation edge(s). Default fanout=1 for a clean path.
                add_edge(src, rel, dst)
                for extra in range(1, max(1, b2b_path_fanout)):
                    add_edge(src, rel, f"{dst}_extra_{extra}")
                # Fill the rest of this source's outdegree with non-current-relation noise.
                current_noise_relations = [r for r in self.relation_types if r != rel]
                if not current_noise_relations:
                    current_noise_relations = [r for r in NOISE_RELATIONS if r != rel]
                target_edges_here = max(1, b2b_path_fanout)
                for j in range(max(0, outdegree - target_edges_here)):
                    noise_rel = self.rng.choice(current_noise_relations)
                    add_edge(src, noise_rel, f"{src}_noise_{j}")

        self.entity_ids = sorted(self.entity_outdegree.keys())
        self.high_degree_entities = {e for e, d in self.entity_outdegree.items() if d >= outdegree}
        self.n_actual = len(self.edges)
        return self

    def generate_b3_cache_graph(self, n_hubs=20, hub_outdegree=5000,
                                 n_normal=500, normal_outdegree=20,
                                 hub_repeat=30, hub_query_ratio=0.7):
        """Generate a B3-specific graph for cache-effectiveness experiments.

        The workload intentionally repeats expensive high-degree hub reads so
        that cache behavior can be observed cleanly. This graph is for cache
        benchmarking, not for relation-index benchmarking.

        Query stream:
        - hub queries = n_hubs * hub_repeat
        - normal queries are added so hub queries are approximately
          `hub_query_ratio` of the stream, but at least `n_normal` normal
          queries are included.
        """
        self.edges = []
        self.entity_outdegree.clear()
        self.entity_edges.clear()
        self.high_degree_entities.clear()
        self.b3_hubs = []
        self.b3_normals = []
        self.b3_query_stream = []
        self.b3_config = {
            "n_hubs": n_hubs,
            "hub_outdegree": hub_outdegree,
            "n_normal": n_normal,
            "normal_outdegree": normal_outdegree,
            "hub_repeat": hub_repeat,
            "hub_query_ratio": hub_query_ratio,
        }

        def add_edge(src, rel, dst):
            edge = make_edge(GRAPH_ID, src, rel, dst, self.rng)
            self.edges.append(edge)
            self.entity_outdegree[src] += 1
            self.entity_edges[src].append(edge)

        hubs = [f"b3_hub_{i}" for i in range(n_hubs)]
        for hub in hubs:
            for j in range(hub_outdegree):
                rel = self.rng.choice(self.relation_types)
                # Some overlap among dst ids is intentional; the cache key is
                # the source node, so the expensive object is the hub partition.
                dst = f"b3_hub_{hub}_dst_{j}"
                add_edge(hub, rel, dst)
            self.high_degree_entities.add(hub)
        self.b3_hubs = hubs

        normals = [f"b3_normal_{i}" for i in range(n_normal)]
        for normal in normals:
            for j in range(normal_outdegree):
                rel = self.rng.choice(self.relation_types)
                dst = f"b3_normal_{normal}_dst_{j}"
                add_edge(normal, rel, dst)
        self.b3_normals = normals
        self.entity_ids = hubs + normals

        n_hub_queries = n_hubs * hub_repeat
        n_normal_queries = max(
            1,
            int(n_hub_queries * (1.0 - hub_query_ratio) / max(hub_query_ratio, 1e-6)),
        )

        hub_queries = []
        for hub in hubs:
            hub_queries.extend([hub] * hub_repeat)
        self.rng.shuffle(hub_queries)

        normal_queries = [self.rng.choice(normals) for _ in range(n_normal_queries)]

        # Interleave hubs and normals instead of putting all repeated hubs next
        # to each other. This is still cache-friendly but less artificial than a
        # pure block sequence.
        stream = []
        hub_idx = 0
        normal_idx = 0
        hub_block = max(1, n_hub_queries // max(n_normal_queries, 1))
        while hub_idx < len(hub_queries) or normal_idx < len(normal_queries):
            for _ in range(hub_block):
                if hub_idx >= len(hub_queries):
                    break
                stream.append(hub_queries[hub_idx])
                hub_idx += 1
            if normal_idx < len(normal_queries):
                stream.append(normal_queries[normal_idx])
                normal_idx += 1

        self.b3_query_stream = stream
        self.b3_config["n_hub_queries"] = n_hub_queries
        self.b3_config["n_normal_queries"] = n_normal_queries
        self.b3_config["total_queries"] = len(stream)
        self.n_actual = len(self.edges)
        return self

    def get_seed_entities(self, n=200, high_deg_bias=True):
        """Return seed entities that actually have outgoing edges.

        The synthetic generator intentionally creates a skewed graph. For small
        test runs, some tail entities may receive zero outgoing edges. Sampling
        from all entity_ids can therefore pick a valid but uninformative seed
        whose one-hop result is empty. For traversal benchmarks, seeds should
        come from the positive-outdegree population.
        """
        positive_entities = [e for e in self.entity_ids if self.entity_outdegree.get(e, 0) > 0]
        if not positive_entities:
            positive_entities = list(self.entity_ids)

        if high_deg_bias:
            sorted_entities = sorted(
                [(e, self.entity_outdegree.get(e, 0)) for e in positive_entities],
                key=lambda x: -x[1],
            )
            # Bias toward costly frontier nodes, but still avoid zero-outdegree seeds.
            top_n = max(1, len(sorted_entities) // 2)
            top = [e for e, _ in sorted_entities[:top_n]]
            seeds = self.rng.sample(top, min(n, len(top))) if top else []
        else:
            seeds = self.rng.sample(positive_entities, min(n, len(positive_entities)))
        return seeds

    def get_relation_selective_seeds_per_source(self, target_relation, selectivity):
        seeds = []
        for entity, edges in self.entity_edges.items():
            if len(edges) < 10:
                continue
            total = len(edges)
            matching = sum(1 for e in edges if e["relation"] == target_relation)
            actual_sel = matching / total
            if selectivity == 0.01:
                target_range = (0.005, 0.02)
            elif selectivity == 0.10:
                target_range = (0.05, 0.15)
            elif selectivity == 0.50:
                target_range = (0.35, 0.65)
            else:
                target_range = (selectivity - 0.02, selectivity + 0.02)
            if target_range[0] <= actual_sel <= target_range[1]:
                seeds.append(entity)
        return seeds

    def get_skewed_seed_sequence(self, n_query_cycles=3, seeds_per_cycle=50):
        high_deg_seeds = list(self.high_degree_entities)
        normal_seeds = [e for e in self.entity_ids if e not in self.high_degree_entities]
        sequence = []
        for cycle in range(n_query_cycles):
            hd = self.rng.sample(high_deg_seeds, min(seeds_per_cycle // 2, len(high_deg_seeds)))
            nd = self.rng.sample(normal_seeds, min(seeds_per_cycle // 2, len(normal_seeds)))
            cycle_seeds = hd + nd
            self.rng.shuffle(cycle_seeds)
            sequence.extend(cycle_seeds)
        return sequence

    def config_dict(self):
        d = {
            "n_entities": self.n_entities,
            "n_edges": self.n_edges,
            "n_actual": self.n_actual,
            "seed": self.seed,
            "high_degree_frac": self.high_degree_frac,
            "high_degree_mult": self.high_degree_mult,
            "n_high_degree_entities": len(self.high_degree_entities),
            "n_relation_types": len(self.relation_types),
            "relation_types": self.relation_types,
            "graph_id": GRAPH_ID,
        }
        if self.b2_config:
            d["b2_config"] = self.b2_config
            d["b2b_starts"] = self.b2b_starts
            d["b2_sources"] = {str(k): v for k, v in self.b2_sources.items()}
        if self.b3_config:
            d["b3_config"] = self.b3_config
        return d


class HighDegreeCache:
    def __init__(self, capacity=200, degree_threshold=100):
        self.capacity = capacity
        self.degree_threshold = degree_threshold
        self.store = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.degree_map = {}

    def set_degree(self, entity, degree):
        self.degree_map[entity] = degree

    def get(self, key):
        with self.lock:
            entity = key[1] if isinstance(key, tuple) and len(key) >= 2 else key
            degree = self.degree_map.get(entity, 0)
            if degree < self.degree_threshold:
                self.misses += 1
                return None, False
            if key not in self.store:
                self.misses += 1
                return None, False
            edges = self.store.pop(key)
            self.store[key] = edges
            self.hits += 1
            return edges.copy(), True

    def set(self, key, edges):
        entity = key[1] if isinstance(key, tuple) and len(key) >= 2 else key
        degree = self.degree_map.get(entity, 0)
        if degree < self.degree_threshold:
            return
        with self.lock:
            if key in self.store:
                self.store.pop(key)
            elif len(self.store) >= self.capacity:
                self.store.popitem(last=False)
            self.store[key] = list(edges)

    def clear(self):
        with self.lock:
            self.store.clear()
            self.hits = 0
            self.misses = 0

    def reset_stats(self):
        with self.lock:
            self.hits = 0
            self.misses = 0

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache:
    def __init__(self, capacity=200):
        self.capacity = capacity
        self.store = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self.lock:
            if key not in self.store:
                self.misses += 1
                return None, False
            edges = self.store.pop(key)
            self.store[key] = edges
            self.hits += 1
            return edges.copy(), True

    def set(self, key, edges):
        with self.lock:
            if key in self.store:
                self.store.pop(key)
            elif len(self.store) >= self.capacity:
                self.store.popitem(last=False)
            self.store[key] = list(edges)

    def clear(self):
        with self.lock:
            self.store.clear()
            self.hits = 0
            self.misses = 0

    def reset_stats(self):
        with self.lock:
            self.hits = 0
            self.misses = 0

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def get_session():
    cluster = Cluster(CASSANDRA_HOSTS, port=CASSANDRA_PORT)
    return cluster.connect(KEYSPACE), cluster


def fetch_all_edges(session, src_id, graph_id):
    """Fetch all outgoing edges for src_id.

    Returns:
        edges: list of returned edge tuples
        raw_count: number of edge rows read from Cassandra
        returned_count: number of edges returned to the traversal
    """
    rows = session.execute(
        "SELECT src_id, relation, dst_id, source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
        (graph_id, src_id),
    )
    edges = [(r.src_id, r.relation, r.dst_id, r.source) for r in rows]
    return edges, len(edges), len(edges)


def fetch_filtered_edges(session, src_id, relation, graph_id):
    """Fetch all outgoing edges, then filter relation in Python.

    This is the src-scan baseline for relation-selective queries. raw_count
    counts all rows read before filtering; returned_count counts only rows after
    relation filtering.
    """
    rows = session.execute(
        "SELECT src_id, relation, dst_id, source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
        (graph_id, src_id),
    )
    raw_edges = [(r.src_id, r.relation, r.dst_id, r.source) for r in rows]
    filtered = [e for e in raw_edges if e[1] == relation]
    return filtered, len(raw_edges), len(filtered)


def fetch_indexed_edges(session, src_id, relation, graph_id):
    """Fetch relation-filtered outgoing edges using kg_edges_by_src_relation."""
    rows = session.execute(
        "SELECT src_id, relation, dst_id, source FROM kg_edges_by_src_relation WHERE graph_id=%s AND src_id=%s AND relation=%s",
        (graph_id, src_id, relation),
    )
    edges = [(r.src_id, r.relation, r.dst_id, r.source) for r in rows]
    return edges, len(edges), len(edges)


def frontier_traverse(session, seed, graph_id, workers=1, relation_path=None, max_depth=2,
                      cache=None, use_index=False, raise_on_error=True,
                      max_frontier=None, fanout_per_node=None):
    """Run one KG frontier traversal query.

    Important metric definitions:
    - partition_reads: number of Cassandra point/partition queries issued.
    - raw_edges_from_db: number of edge rows actually read from Cassandra before
      Python-side filtering. For cache hits this is 0.
    - returned_edges: number of edges returned into the traversal after any
      relation filtering.

    Parallelism here is frontier-level parallelism inside a single query, not
    query-level concurrency.
    """
    q_start = time.perf_counter()
    frontier = {seed}
    total_partition_reads = 0
    total_raw_edges_from_db = 0
    total_returned = 0
    cache_hits = 0
    errors = 0
    first_error_reported = False
    max_frontier_seen = 0
    frontier_truncated = 0
    stats_lock = threading.Lock()

    def fetch_one(src, depth):
        nonlocal cache_hits, errors, first_error_reported

        current_relation = None
        if relation_path and depth < len(relation_path):
            current_relation = relation_path[depth]

        # Cache keys must include relation when the cached value is relation-filtered.
        # This avoids returning edges filtered by a previous relation at another hop.
        if current_relation is not None:
            cache_key = (graph_id, src, current_relation)
        else:
            cache_key = (graph_id, src)

        if cache:
            cached, hit = cache.get(cache_key)
            if hit:
                with stats_lock:
                    cache_hits += 1
                return cached, 0, 0, len(cached)

        try:
            if use_index and current_relation is not None:
                edges, raw_count, returned_count = fetch_indexed_edges(session, src, current_relation, graph_id)
            elif current_relation is not None:
                edges, raw_count, returned_count = fetch_filtered_edges(session, src, current_relation, graph_id)
            else:
                edges, raw_count, returned_count = fetch_all_edges(session, src, graph_id)
        except Exception as ex:
            with stats_lock:
                errors += 1
                should_print = not first_error_reported
                if should_print:
                    first_error_reported = True
            if should_print:
                print(
                    f"  [FETCH ERROR] src={src} graph={graph_id} depth={depth} "
                    f"relation={current_relation} use_index={use_index}: {repr(ex)}"
                )
            if raise_on_error:
                raise
            return [], 0, 0, 0

        if cache:
            cache.set(cache_key, edges)

        # One Cassandra query was issued on every miss/successful DB read.
        return edges, 1, raw_count, returned_count

    for depth in range(max_depth):
        next_frontier = set()
        src_list = sorted(frontier)
        max_frontier_seen = max(max_frontier_seen, len(src_list))
        if max_frontier is not None and len(src_list) > max_frontier:
            frontier_truncated += len(src_list) - max_frontier
            src_list = src_list[:max_frontier]
        if not src_list:
            break

        if workers == 1:
            for src in src_list:
                edges, partition_reads, raw_count, n_ret = fetch_one(src, depth)
                total_partition_reads += partition_reads
                total_raw_edges_from_db += raw_count
                total_returned += n_ret
                expand_edges = edges[:fanout_per_node] if fanout_per_node is not None else edges
                for _, _, dst, _ in expand_edges:
                    next_frontier.add(dst)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(fetch_one, src, depth): src for src in src_list}
                for f in as_completed(futures):
                    edges, partition_reads, raw_count, n_ret = f.result()
                    total_partition_reads += partition_reads
                    total_raw_edges_from_db += raw_count
                    total_returned += n_ret
                    expand_edges = edges[:fanout_per_node] if fanout_per_node is not None else edges
                    for _, _, dst, _ in expand_edges:
                        next_frontier.add(dst)
        frontier = next_frontier

    elapsed = (time.perf_counter() - q_start) * 1000
    return {
        "latency_ms": elapsed,
        "partition_reads": total_partition_reads,
        "raw_edges_from_db": total_raw_edges_from_db,
        # Backward-compatible alias used by existing summary/log code.
        "raw_reads": total_raw_edges_from_db,
        "returned_edges": total_returned,
        "cache_hits": cache_hits,
        "errors": errors,
        "max_frontier_seen": max_frontier_seen,
        "frontier_truncated": frontier_truncated,
    }


def bench_onehop_src_scan(session, src, relation, graph_id):
    t0 = time.perf_counter()
    edges, raw_count, returned_count = fetch_filtered_edges(session, src, relation, graph_id)
    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "latency_ms": elapsed,
        "raw_edges_from_db": raw_count,
        "returned_edges": returned_count,
        "edges": edges,
    }


def bench_onehop_indexed(session, src, relation, graph_id):
    t0 = time.perf_counter()
    edges, raw_count, returned_count = fetch_indexed_edges(session, src, relation, graph_id)
    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "latency_ms": elapsed,
        "raw_edges_from_db": raw_count,
        "returned_edges": returned_count,
        "edges": edges,
    }

def log_raw_result(experiment, mode, query_id, seed_entity, hop, relation_path,
                   latency_ms, raw_reads, returned_edges, cache_hit, error):
    row = {
        "experiment": experiment,
        "mode": mode,
        "query_id": query_id,
        "seed_entity": seed_entity,
        "hop": hop,
        "relation_path": ";".join(relation_path) if relation_path else "",
        "latency_ms": round(latency_ms, 6),
        "raw_edges_from_db": raw_reads,
        "returned_edges": returned_edges,
        "cache_hit": cache_hit,
        "error": error,
    }
    raw_fields = list(row.keys())
    append_csv(RAW_DIR / f"{experiment}_raw.csv", row, raw_fields)


def collect_latency_stats(latencies):
    if not latencies:
        return {"mean": 0, "p50": 0, "p95": 0, "p99": 0}
    return {
        "mean": round(statistics.mean(latencies), 3),
        "p50": round(percentile(latencies, 50), 3),
        "p95": round(percentile(latencies, 95), 3),
        "p99": round(percentile(latencies, 99), 3),
    }


def import_to_cassandra(session, edges, truncate=False):
    if truncate:
        print("Truncating tables...")
        for stmt in TRUNCATE_TABLES:
            print(f"  {stmt} ...", flush=True)
            session.execute(stmt)
        print("  Truncate done.", flush=True)
    else:
        print(f"Skipping TRUNCATE. Importing into isolated graph_id={GRAPH_ID}", flush=True)

    insert_src = session.prepare(INSERT_SRC)
    insert_src_rel = session.prepare(INSERT_SRC_RELATION)
    insert_dst = session.prepare(INSERT_DST)
    insert_rel_bucket = session.prepare(INSERT_RELATION_BUCKET)

    physical = 0
    for i, e in enumerate(edges):
        bucket = stable_bucket(e["dst_id"])
        try:
            session.execute(insert_src, (GRAPH_ID, e["src_id"], e["relation"], e["dst_id"],
                            e["src_type"], e["dst_type"], e["confidence"], e["source"]))
            session.execute(insert_src_rel, (GRAPH_ID, e["src_id"], e["relation"], e["dst_id"],
                            e["src_type"], e["dst_type"], e["confidence"], e["source"]))
            session.execute(insert_dst, (GRAPH_ID, e["dst_id"], e["relation"], e["src_id"],
                            e["src_type"], e["dst_type"], e["confidence"], e["source"]))
            session.execute(insert_rel_bucket, (GRAPH_ID, e["relation"], bucket, e["src_id"], e["dst_id"],
                            e["src_type"], e["dst_type"], e["confidence"], e["source"]))
            physical += 4
        except Exception as ex:
            print(f"  WARN: insert failed at logical={i}: {ex}")
            continue
        if physical % 20000 == 0:
            print(f"  {physical} physical writes ({i+1}/{len(edges)} logical)...")
    print(f"  Import done: {len(edges)} logical, {physical} physical writes")



def import_b2_to_cassandra(session, edges):
    """Import only the tables needed by B2.

    B2 compares kg_edges_by_src against kg_edges_by_src_relation, so writing
    dst/relation_bucket would only add unnecessary import time.
    """
    print(f"B2-specific import (src + src_relation only). graph_id={GRAPH_ID}", flush=True)
    insert_src = session.prepare(INSERT_SRC)
    insert_src_rel = session.prepare(INSERT_SRC_RELATION)

    physical = 0
    for i, e in enumerate(edges):
        try:
            session.execute(insert_src, (GRAPH_ID, e["src_id"], e["relation"], e["dst_id"],
                            e["src_type"], e["dst_type"], e["confidence"], e["source"]))
            session.execute(insert_src_rel, (GRAPH_ID, e["src_id"], e["relation"], e["dst_id"],
                            e["src_type"], e["dst_type"], e["confidence"], e["source"]))
            physical += 2
        except Exception as ex:
            print(f"  WARN: insert failed at logical={i}: {ex}")
            continue
        if physical % 20000 == 0:
            print(f"  {physical} physical writes ({i+1}/{len(edges)} logical)...", flush=True)
    print(f"  B2 import done: {len(edges)} logical, {physical} physical writes", flush=True)




def import_b3_to_cassandra(session, edges):
    """Import only kg_edges_by_src for B3 cache experiments.

    B3 reads only outgoing edges by source. Writing dst/relation_bucket or
    src_relation tables would increase import time without changing the cache
    benchmark.
    """
    print(f"B3-specific import (src only). graph_id={GRAPH_ID}", flush=True)
    insert_src = session.prepare(INSERT_SRC)
    physical = 0
    for i, e in enumerate(edges):
        try:
            session.execute(insert_src, (GRAPH_ID, e["src_id"], e["relation"], e["dst_id"],
                            e["src_type"], e["dst_type"], e["confidence"], e["source"]))
            physical += 1
        except Exception as ex:
            print(f"  WARN: insert failed at logical={i}: {ex}")
            continue
        if physical % 20000 == 0:
            print(f"  {physical} physical writes ({i+1}/{len(edges)} logical)...", flush=True)
    print(f"  B3 import done: {len(edges)} logical, {physical} physical writes", flush=True)

def run_b1_parallel_worker_sweep(session, graph, args):
    print("\n=== B1: Parallel Worker Sweep ===")
    seeds = graph.get_seed_entities(n=args.n_queries, high_deg_bias=False)
    smoke_seed = seeds[0]
    smoke_degree = graph.entity_outdegree.get(smoke_seed, 0)
    print(f"  Smoke query: trying src={smoke_seed} (synthetic_outdegree={smoke_degree})...")
    try:
        r = frontier_traverse(session, smoke_seed, GRAPH_ID, workers=1, max_depth=1, cache=None, use_index=False)
        print(
            f"  Smoke: latency={r['latency_ms']:.2f}ms, "
            f"partition_reads={r['partition_reads']}, raw_edges={r['raw_edges_from_db']}, "
            f"returned={r['returned_edges']}, errors={r['errors']}"
        )
        if r["partition_reads"] == 0 or r["returned_edges"] == 0 or r["errors"] > 0:
            print(
                "  WARNING: Smoke query returned no edges. This now means either "
                "the chosen seed has no imported rows, or Cassandra read consistency/import visibility needs checking."
            )
    except Exception as ex:
        print(f"  Smoke query FAILED: {ex}")
    worker_levels = [1, 4, 8, 16, 32]
    hop_depths = [2]
    if args.b1_hop4:
        hop_depths.append(4)
    summary_rows = []

    for hop in hop_depths:
        print(f"  Hop={hop}... max_frontier={args.max_frontier}, fanout_per_node={args.fanout_per_node}", flush=True)
        print("    Naive workers=1...", flush=True)
        naive_latencies = []
        for rep in range(REPEATS):
            print(f"      naive rep {rep+1}/{REPEATS}", flush=True)
            for i, seed in enumerate(seeds):
                r = frontier_traverse(session, seed, GRAPH_ID, workers=1,
                                      relation_path=None, max_depth=hop, cache=None, use_index=False,
                                      max_frontier=args.max_frontier, fanout_per_node=args.fanout_per_node)
                naive_latencies.append(r["latency_ms"])
                log_raw_result("B1_parallel", "Cassandra_naive", f"hop{hop}_q{i}", seed, hop, None,
                               r["latency_ms"], r["raw_reads"], r["returned_edges"], 0, r["errors"])

        naive_stats = collect_latency_stats(naive_latencies)
        naive_mean = naive_stats["mean"]
        naive_qps = len(naive_latencies) / (sum(naive_latencies) / 1000.0) if sum(naive_latencies) > 0 else 0

        summary_rows.append({
            "experiment": "B1_parallel",
            "mode": "Cassandra_naive",
            "workload_type": "cold_path_frontier",
            "hop": hop,
            "workers": 1,
            "cache_policy": "none",
            "index_enabled": False,
            "cache_state": "cold",
            "n_queries": len(seeds) * REPEATS,
            "mean_latency_ms": naive_stats["mean"],
            "p50_latency_ms": naive_stats["p50"],
            "p95_latency_ms": naive_stats["p95"],
            "p99_latency_ms": naive_stats["p99"],
            "qps": round(naive_qps, 3),
            "raw_edges_from_db": r["raw_reads"],
            "returned_edges": r["returned_edges"],
            "cache_hit_rate": 0.0,
            "effective_latency_ms": naive_stats["mean"],
            "speedup_vs_naive": 1.0,
            "notes": f"B1 hop={hop} baseline; max_frontier={args.max_frontier}; fanout_per_node={args.fanout_per_node}",
        })

        for workers in worker_levels:
            if workers == 1:
                continue
            print(f"    Workers={workers}...", flush=True)
            all_latencies = []
            for rep in range(REPEATS):
                print(f"      workers={workers} rep {rep+1}/{REPEATS}", flush=True)
                for i, seed in enumerate(seeds):
                    r = frontier_traverse(session, seed, GRAPH_ID, workers=workers,
                                          relation_path=None, max_depth=hop, cache=None, use_index=False,
                                          max_frontier=args.max_frontier, fanout_per_node=args.fanout_per_node)
                    all_latencies.append(r["latency_ms"])
                    log_raw_result("B1_parallel", f"Cassandra_parallel_w{workers}", f"hop{hop}_q{i}", seed, hop, None,
                                   r["latency_ms"], r["raw_reads"], r["returned_edges"], 0, r["errors"])

            stats = collect_latency_stats(all_latencies)
            pqps = len(all_latencies) / (sum(all_latencies) / 1000.0) if sum(all_latencies) > 0 else 0
            summary_rows.append({
                "experiment": "B1_parallel",
                "mode": f"Cassandra_parallel_w{workers}",
                "workload_type": "cold_path_frontier",
                "hop": hop,
                "workers": workers,
                "cache_policy": "none",
                "index_enabled": False,
                "cache_state": "cold",
                "n_queries": len(seeds) * REPEATS,
                "mean_latency_ms": stats["mean"],
                "p50_latency_ms": stats["p50"],
                "p95_latency_ms": stats["p95"],
                "p99_latency_ms": stats["p99"],
                "qps": round(pqps, 3),
                "raw_edges_from_db": r["raw_reads"],
                "returned_edges": r["returned_edges"],
                "cache_hit_rate": 0.0,
                "effective_latency_ms": stats["mean"],
                "speedup_vs_naive": round(naive_mean / max(stats["mean"], 0.001), 2),
                "notes": f"B1 hop={hop}; max_frontier={args.max_frontier}; fanout_per_node={args.fanout_per_node}",
            })

    out = B1_DETAIL
    fields = list(summary_rows[0].keys()) if summary_rows else []
    write_csv(out, summary_rows, fields)
    print(f"  -> {out}")
    if summary_rows:
        write_csv(B1_SUMMARY, summary_rows, fields)
        print(f"  -> {B1_SUMMARY}")
    return summary_rows


def run_b2_relation_index(session, graph, args):
    print("\n=== B2: Relation-Index Characterization ===")
    summary_rows = []
    target_relation = args.b2_target_relation

    print("  B2a: One-hop src+relation microbenchmark...")
    for selectivity in args.b2_selectivities:
        sel_sources = graph.b2_sources.get(selectivity)
        if not sel_sources:
            # Backward-compatible fallback for random graphs.
            sel_sources = graph.get_relation_selective_seeds_per_source(target_relation, selectivity)
            sel_sources = sel_sources[:min(len(sel_sources), args.n_queries)]
        if not sel_sources:
            print(f"    selectivity={selectivity}: no sources found, skipping")
            continue
        print(f"    selectivity={selectivity}, n_sources={len(sel_sources)}")

        cfg = graph.b2_config.get(selectivity, {})
        target_edges = cfg.get("target_edges", "")
        noise_edges = cfg.get("noise_edges", "")
        outdegree = cfg.get("outdegree", "")

        scan_times, idx_times = [], []
        scan_raw_counts, scan_ret_counts = [], []
        idx_raw_counts, idx_ret_counts = [], []

        for rep in range(REPEATS):
            print(f"      B2a selectivity={selectivity} rep {rep+1}/{REPEATS}", flush=True)
            for i, src in enumerate(sel_sources):
                r_scan = bench_onehop_src_scan(session, src, target_relation, GRAPH_ID)
                scan_times.append(r_scan["latency_ms"])
                scan_raw_counts.append(r_scan["raw_edges_from_db"])
                scan_ret_counts.append(r_scan["returned_edges"])
                log_raw_result("B2a", "src_scan", f"sel{selectivity}_q{i}", src, 1, [target_relation],
                               r_scan["latency_ms"], r_scan["raw_edges_from_db"], r_scan["returned_edges"], 0, 0)

                r_idx = bench_onehop_indexed(session, src, target_relation, GRAPH_ID)
                idx_times.append(r_idx["latency_ms"])
                idx_raw_counts.append(r_idx["raw_edges_from_db"])
                idx_ret_counts.append(r_idx["returned_edges"])
                log_raw_result("B2a", "src_relation_index", f"sel{selectivity}_q{i}", src, 1, [target_relation],
                               r_idx["latency_ms"], r_idx["raw_edges_from_db"], r_idx["returned_edges"], 0, 0)

        scan_stats = collect_latency_stats(scan_times)
        idx_stats = collect_latency_stats(idx_times)
        speedup = round(scan_stats["mean"] / max(idx_stats["mean"], 0.001), 2)

        for mode, stats, use_idx, raw_counts, ret_counts in [
            ("src_scan", scan_stats, False, scan_raw_counts, scan_ret_counts),
            ("src_relation_index", idx_stats, True, idx_raw_counts, idx_ret_counts),
        ]:
            summary_rows.append({
                "experiment": "B2a_onehop",
                "mode": mode,
                "workload_type": f"relation_selective_{selectivity}",
                "selectivity": selectivity,
                "target_relation": target_relation,
                "n_sources": len(sel_sources),
                "outdegree": outdegree,
                "target_edges": target_edges,
                "noise_edges": noise_edges,
                "hop": 1,
                "workers": 1,
                "cache_policy": "none",
                "index_enabled": use_idx,
                "cache_state": "cold",
                "n_queries": len(sel_sources) * REPEATS,
                "mean_latency_ms": stats["mean"],
                "p50_latency_ms": stats["p50"],
                "p95_latency_ms": stats["p95"],
                "p99_latency_ms": stats["p99"],
                "qps": round(len(raw_counts) / (sum(scan_times if not use_idx else idx_times) / 1000.0), 3) if (sum(scan_times if not use_idx else idx_times) > 0) else 0,
                "raw_edges_from_db": round(statistics.mean(raw_counts), 3) if raw_counts else 0,
                "returned_edges": round(statistics.mean(ret_counts), 3) if ret_counts else 0,
                "cache_hit_rate": 0.0,
                "effective_latency_ms": stats["mean"],
                "speedup_vs_src_scan": speedup if use_idx else 1.0,
                "speedup_vs_naive": speedup if use_idx else 1.0,
                "notes": f"B2a target={target_relation}; selectivity={selectivity}; high-outdegree={bool(graph.b2_config)}",
            })

    print("  B2b: Multi-hop fixed relation path (linear chain + noise)...")
    path = RELATION_PATH_B2B
    seeds_b2b = list(graph.b2b_starts) if graph.b2b_starts else graph.get_seed_entities(n=min(args.n_queries, 50), high_deg_bias=False)
    if seeds_b2b:
        scan_times_b2b, idx_times_b2b = [], []
        scan_raw_b2b, scan_ret_b2b = [], []
        idx_raw_b2b, idx_ret_b2b = [], []
        for rep in range(REPEATS):
            print(f"      B2b rep {rep+1}/{REPEATS}", flush=True)
            for i, seed in enumerate(seeds_b2b):
                r_scan = frontier_traverse(session, seed, GRAPH_ID, workers=16,
                                           relation_path=path, max_depth=len(path), cache=None, use_index=False,
                                           max_frontier=args.max_frontier, fanout_per_node=args.fanout_per_node)
                scan_times_b2b.append(r_scan["latency_ms"])
                scan_raw_b2b.append(r_scan["raw_edges_from_db"])
                scan_ret_b2b.append(r_scan["returned_edges"])
                log_raw_result("B2b", "src_scan", f"q{i}", seed, len(path), path,
                               r_scan["latency_ms"], r_scan["raw_reads"], r_scan["returned_edges"], 0, r_scan["errors"])

                r_idx = frontier_traverse(session, seed, GRAPH_ID, workers=16,
                                          relation_path=path, max_depth=len(path), cache=None, use_index=True,
                                          max_frontier=args.max_frontier, fanout_per_node=args.fanout_per_node)
                idx_times_b2b.append(r_idx["latency_ms"])
                idx_raw_b2b.append(r_idx["raw_edges_from_db"])
                idx_ret_b2b.append(r_idx["returned_edges"])
                log_raw_result("B2b", "src_relation_index", f"q{i}", seed, len(path), path,
                               r_idx["latency_ms"], r_idx["raw_reads"], r_idx["returned_edges"], 0, r_idx["errors"])

        scan_stats_b = collect_latency_stats(scan_times_b2b)
        idx_stats_b = collect_latency_stats(idx_times_b2b)
        speedup_b = round(scan_stats_b["mean"] / max(idx_stats_b["mean"], 0.001), 2)

        for mode, stats, use_idx, raw_counts, ret_counts, times in [
            ("src_scan", scan_stats_b, False, scan_raw_b2b, scan_ret_b2b, scan_times_b2b),
            ("src_relation_index", idx_stats_b, True, idx_raw_b2b, idx_ret_b2b, idx_times_b2b),
        ]:
            summary_rows.append({
                "experiment": "B2b_multihop",
                "mode": mode,
                "workload_type": "fixed_relation_path_4hop_linear_noise",
                "selectivity": "path",
                "target_relation": ";".join(path),
                "n_sources": len(seeds_b2b),
                "outdegree": args.b2_outdegree if graph.b2_config else "",
                "target_edges": "4-hop-linear",
                "noise_edges": "per-source-outdegree-minus-target",
                "hop": len(path),
                "workers": 16,
                "cache_policy": "none",
                "index_enabled": use_idx,
                "cache_state": "cold",
                "n_queries": len(seeds_b2b) * REPEATS,
                "mean_latency_ms": stats["mean"],
                "p50_latency_ms": stats["p50"],
                "p95_latency_ms": stats["p95"],
                "p99_latency_ms": stats["p99"],
                "qps": round(len(times) / (sum(times) / 1000.0), 3) if sum(times) > 0 else 0,
                "raw_edges_from_db": round(statistics.mean(raw_counts), 3) if raw_counts else 0,
                "returned_edges": round(statistics.mean(ret_counts), 3) if ret_counts else 0,
                "cache_hit_rate": 0.0,
                "effective_latency_ms": stats["mean"],
                "speedup_vs_src_scan": speedup_b if use_idx else 1.0,
                "speedup_vs_naive": speedup_b if use_idx else 1.0,
                "notes": "B2b linear path start->pref->need->state->strategy with high noise per source",
            })

    out = B2_DETAIL
    fields = sorted({k for row in summary_rows for k in row.keys()}) if summary_rows else []
    write_csv(out, summary_rows, fields)
    print(f"  -> {out}")

    if summary_rows:
        write_csv(B2_SUMMARY, summary_rows, fields)
        print(f"  -> {B2_SUMMARY}")
    return summary_rows

def run_b3_cache_effective_latency(session, graph, args):
    print("\n=== B3: Cache Effective-Latency (High-Degree Repeated Workload) ===")
    print(
        f"  B3 traversal depth={args.b3_depth}, "
        f"max_frontier={args.max_frontier}, fanout_per_node={args.fanout_per_node}"
    )

    if graph.b3_query_stream:
        query_stream = list(graph.b3_query_stream)
        print(
            f"  Using B3-specific graph: {graph.b3_config.get('n_hubs', '?')} hubs x "
            f"{graph.b3_config.get('hub_outdegree', '?')} outdegree; "
            f"{graph.b3_config.get('n_normal', '?')} normal nodes x "
            f"{graph.b3_config.get('normal_outdegree', '?')} outdegree"
        )
        print(
            f"  Query stream: {len(query_stream)} queries "
            f"({graph.b3_config.get('n_hub_queries', '?')} hub, "
            f"{graph.b3_config.get('n_normal_queries', '?')} normal)"
        )
    else:
        print("  WARNING: no B3-specific query stream; falling back to skewed seed sequence.")
        query_stream = graph.get_skewed_seed_sequence(n_query_cycles=3, seeds_per_cycle=args.n_queries)
        print(f"  Fallback stream: {len(query_stream)} queries")

    summary_rows = []

    def measure_stream(cache, label):
        if cache:
            cache.clear()
        latencies = []
        raw_edges_list = []
        returned_list = []
        partition_reads_list = []
        cache_hits_list = []
        error_count = 0

        for i, seed in enumerate(query_stream):
            r = frontier_traverse(
                session,
                seed,
                GRAPH_ID,
                workers=16,
                relation_path=None,
                max_depth=args.b3_depth,
                cache=cache,
                use_index=False,
                max_frontier=args.max_frontier,
                fanout_per_node=args.fanout_per_node,
            )
            hit_flag = 1 if r["cache_hits"] > 0 else 0
            latencies.append(r["latency_ms"])
            raw_edges_list.append(r["raw_edges_from_db"])
            returned_list.append(r["returned_edges"])
            partition_reads_list.append(r["partition_reads"])
            cache_hits_list.append(r["cache_hits"])
            error_count += r["errors"]
            log_raw_result(
                "B3_cache",
                label,
                f"q{i}",
                seed,
                args.b3_depth,
                None,
                r["latency_ms"],
                r["raw_reads"],
                r["returned_edges"],
                hit_flag,
                r["errors"],
            )

        stats = collect_latency_stats(latencies)
        hit_rate = cache.hit_rate() if cache else 0.0
        cache_hits = cache.hits if cache else 0
        cache_misses = cache.misses if cache else 0
        return {
            "stats": stats,
            "latencies": latencies,
            "avg_raw": round(statistics.mean(raw_edges_list), 3) if raw_edges_list else 0,
            "avg_returned": round(statistics.mean(returned_list), 3) if returned_list else 0,
            "avg_partition_reads": round(statistics.mean(partition_reads_list), 3) if partition_reads_list else 0,
            "avg_cache_hits_per_query": round(statistics.mean(cache_hits_list), 3) if cache_hits_list else 0,
            "cache_hit_rate": hit_rate,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "error_count": error_count,
        }

    print("  B3a: No cache stream baseline...")
    no_cache = measure_stream(None, "no_cache")

    print("  B3b: LRU cache stream...")
    lru = LRUCache(capacity=CACHE_CAP)
    lru_result = measure_stream(lru, "LRU")

    print(f"  B3c: HighDegree cache stream (capacity={CACHE_CAP}, threshold={DEGREE_THRESHOLD})...")
    hd = HighDegreeCache(capacity=CACHE_CAP, degree_threshold=DEGREE_THRESHOLD)
    for eid, deg in graph.entity_outdegree.items():
        hd.set_degree(eid, deg)
    hd_result = measure_stream(hd, "HighDegree")

    base_mean = no_cache["stats"]["mean"]
    rows = [
        ("no_cache", "none", no_cache),
        ("LRU", "LRU", lru_result),
        ("HighDegree", "high_degree", hd_result),
    ]

    for label, cache_policy, result in rows:
        stats = result["stats"]
        total_time = sum(result["latencies"])
        n_queries = len(query_stream)
        summary_rows.append({
            "experiment": "B3_cache",
            "mode": f"Cassandra+parallel+{label}",
            "workload_type": "high_degree_repeated_stream",
            "hop": args.b3_depth,
            "workers": 16,
            "cache_policy": cache_policy,
            "index_enabled": False,
            "cache_state": "stream",
            "n_queries": n_queries,
            "mean_latency_ms": stats["mean"],
            "p50_latency_ms": stats["p50"],
            "p95_latency_ms": stats["p95"],
            "p99_latency_ms": stats["p99"],
            "qps": round(n_queries / (total_time / 1000.0), 3) if total_time > 0 else 0,
            "raw_edges_from_db": result["avg_raw"],
            "returned_edges": result["avg_returned"],
            "partition_reads": result["avg_partition_reads"],
            "cache_hit_rate": round(result["cache_hit_rate"], 4),
            "cache_hits": result["cache_hits"],
            "cache_misses": result["cache_misses"],
            "cache_hits_per_query": result["avg_cache_hits_per_query"],
            # This is the measured stream mean under the observed hit rate.
            # Do not recombine cold/warm means here.
            "effective_latency_ms": stats["mean"],
            "speedup_vs_naive": round(base_mean / max(stats["mean"], 0.001), 2),
            "error_count": result["error_count"],
            "notes": (
                f"B3 {label}; depth={args.b3_depth}; "
                f"hubs={graph.b3_config.get('n_hubs', '?')}x{graph.b3_config.get('hub_outdegree', '?')}; "
                f"normals={graph.b3_config.get('n_normal', '?')}x{graph.b3_config.get('normal_outdegree', '?')}; "
                f"stream={n_queries}q; max_frontier={args.max_frontier}; "
                f"fanout_per_node={args.fanout_per_node}"
            ),
        })

    out = B3_DETAIL
    fields = sorted({k for row in summary_rows for k in row.keys()}) if summary_rows else []
    write_csv(out, summary_rows, fields)
    print(f"  -> {out}")
    if summary_rows:
        write_csv(B3_SUMMARY, summary_rows, fields)
        print(f"  -> {B3_SUMMARY}")
    return summary_rows


def generate_summary(b1_rows=None, b2_rows=None, b3_rows=None):
    print("\n=== Layer B Total Summary ===")
    # Persisted summaries are the source of truth, so running only B2/B3 will not
    # erase previously completed B1 rows.
    all_rows = []
    for path in [B1_SUMMARY, B2_SUMMARY, B3_SUMMARY]:
        rows = read_csv_rows(path)
        if rows:
            print(f"  loaded {len(rows)} rows from {path.name}")
            all_rows.extend(rows)
    # Also include current in-memory rows if they have not already been written.
    for rows in [b1_rows or [], b2_rows or [], b3_rows or []]:
        if rows and not all_rows:
            all_rows.extend(rows)
    if not all_rows:
        print("  No results to summarize")
        return
    out = OUT_DIR / "layerB_cassandra_internal_ablation_summary.csv"
    # Union fields so B1/B2/B3-specific columns are not silently dropped.
    preferred = [
        "experiment", "mode", "workload_type", "selectivity", "target_relation",
        "hop", "workers", "cache_policy", "index_enabled", "cache_state",
        "n_sources", "outdegree", "target_edges", "noise_edges", "n_queries",
        "mean_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
        "qps", "raw_edges_from_db", "returned_edges", "cache_hit_rate",
        "cold_latency_ms", "warm_latency_ms", "effective_latency_ms",
        "speedup_vs_naive", "speedup_vs_src_scan", "notes",
    ]
    union = []
    for row in all_rows:
        for k in row.keys():
            if k not in union:
                union.append(k)
    fields = [k for k in preferred if k in union] + [k for k in union if k not in preferred]
    write_csv(out, all_rows, fields)
    print(f"  -> {out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-entities", type=int, default=10000)
    parser.add_argument("--n-edges", type=int, default=1000000)
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5, help="Repeat count per benchmark condition.")
    parser.add_argument("--max-frontier", type=int, default=None, help="Optional cap on expanded frontier nodes per depth; use for controlled hop=4 runs.")
    parser.add_argument("--fanout-per-node", type=int, default=None, help="Optional cap on outgoing edges expanded per node; raw DB edges are still counted.")
    parser.add_argument("--graph-id", type=str, default=None, help="Graph id to use. If omitted, a unique graph id is generated for this run.")
    parser.add_argument("--truncate", action="store_true", help="Explicitly TRUNCATE all KG tables before import. Default is false to avoid Cassandra TRUNCATE stalls.")
    parser.add_argument("--b1-hop4", action="store_true", help="Also run hop=4 in B1")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--skip-b1", action="store_true")
    parser.add_argument("--skip-b2", action="store_true")
    parser.add_argument("--skip-b3", action="store_true")
    parser.add_argument("--b2-selectivity-graph", action="store_true",
                        help="Generate a B2-specific high-outdegree relation-selective graph.")
    parser.add_argument("--b2-n-sources", type=int, default=10,
                        help="Number of B2 source nodes per selectivity and path starts.")
    parser.add_argument("--b2-outdegree", type=int, default=5000,
                        help="Outdegree per source in the B2-specific graph.")
    parser.add_argument("--b2-selectivities", type=str, default="0.01,0.10,0.50",
                        help="Comma-separated selectivities, e.g. 0.01,0.10,0.50")
    parser.add_argument("--b2-target-relation", type=str, default="likes")
    parser.add_argument("--b2b-path-fanout", type=int, default=1,
                        help="Target relation fanout per B2b path source. Default 1 keeps a linear path.")
    parser.add_argument("--b3-cache-graph", action="store_true",
                        help="Generate a B3-specific high-degree repeated-workload graph for cache evaluation.")
    parser.add_argument("--b3-depth", type=int, default=1,
                        help="Traversal depth for B3 cache benchmark. Default 1 isolates repeated expensive hub reads. Use 2 only with --max-frontier/--fanout-per-node controls.")
    parser.add_argument("--b3-n-hubs", type=int, default=20)
    parser.add_argument("--b3-hub-outdegree", type=int, default=5000)
    parser.add_argument("--b3-n-normal", type=int, default=500)
    parser.add_argument("--b3-normal-outdegree", type=int, default=20)
    parser.add_argument("--b3-hub-repeat", type=int, default=30)
    parser.add_argument("--b3-hub-query-ratio", type=float, default=0.7)
    args = parser.parse_args()
    args.b2_selectivities = [float(x.strip()) for x in args.b2_selectivities.split(",") if x.strip()]

    global REPEATS
    REPEATS = args.repeats

    global GRAPH_ID
    if args.graph_id:
        GRAPH_ID = args.graph_id
    elif not args.skip_import:
        if args.b3_cache_graph:
            prefix = "b3_cache"
            edge_label = args.b3_n_hubs * args.b3_hub_outdegree + args.b3_n_normal * args.b3_normal_outdegree
        elif args.b2_selectivity_graph:
            prefix = "b2_selectivity"
            edge_label = args.b2_n_sources * args.b2_outdegree
        else:
            prefix = "synth"
            edge_label = args.n_edges
        GRAPH_ID = f"{prefix}_{edge_label}_{int(time.time())}"
    elif not args.skip_generate:
        GRAPH_ID = f"synth_{args.n_edges}_{int(time.time())}"
    # When --skip-import is used, caller should pass --graph-id for existing data.

    print(f"Using GRAPH_ID={GRAPH_ID}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
        if args.b3_cache_graph:
            print("Generating B3 cache-specific graph...")
            graph = SyntheticGraph(n_entities=args.n_entities, n_edges=args.n_edges,
                                   high_degree_frac=0.02, high_degree_mult=20)
            graph.generate_b3_cache_graph(
                n_hubs=args.b3_n_hubs,
                hub_outdegree=args.b3_hub_outdegree,
                n_normal=args.b3_n_normal,
                normal_outdegree=args.b3_normal_outdegree,
                hub_repeat=args.b3_hub_repeat,
                hub_query_ratio=args.b3_hub_query_ratio,
            )
            print(f"  {graph.n_actual} B3 logical edges generated; query stream={len(graph.b3_query_stream)}")
        elif args.b2_selectivity_graph:
            print("Generating B2 relation-selective graph...")
            graph = SyntheticGraph(n_entities=args.n_entities, n_edges=args.n_edges,
                                   high_degree_frac=0.02, high_degree_mult=20)
            graph.generate_b2_selectivity_graph(
                n_sources=args.b2_n_sources,
                outdegree=args.b2_outdegree,
                selectivities=args.b2_selectivities,
                target_relation=args.b2_target_relation,
                b2b_path_fanout=args.b2b_path_fanout,
            )
            print(f"  {graph.n_actual} B2 logical edges generated")
        else:
            print(f"Generating synthetic graph: {args.n_entities} entities, {args.n_edges} edges...")
            graph = SyntheticGraph(n_entities=args.n_entities, n_edges=args.n_edges,
                                   high_degree_frac=0.02, high_degree_mult=20).generate()
            print(f"  {graph.n_actual} edges, {len(graph.high_degree_entities)} high-degree entities")
        config_path = OUT_DIR / "layerB_graph_config.json"
        with open(config_path, "w") as f:
            json.dump(graph.config_dict(), f, indent=2)
        print(f"  Graph config saved: {config_path}")
    else:
        print("Skipping graph generation (in-memory regeneration, no import)")
        graph = SyntheticGraph(n_entities=args.n_entities, n_edges=args.n_edges,
                               high_degree_frac=0.02, high_degree_mult=20)
        if args.b3_cache_graph:
            graph.generate_b3_cache_graph(
                n_hubs=args.b3_n_hubs,
                hub_outdegree=args.b3_hub_outdegree,
                n_normal=args.b3_n_normal,
                normal_outdegree=args.b3_normal_outdegree,
                hub_repeat=args.b3_hub_repeat,
                hub_query_ratio=args.b3_hub_query_ratio,
            )
            print(f"  B3 graph regenerated in-memory: {graph.n_actual} edges; query stream={len(graph.b3_query_stream)}")
        elif args.b2_selectivity_graph:
            graph.generate_b2_selectivity_graph(
                n_sources=args.b2_n_sources,
                outdegree=args.b2_outdegree,
                selectivities=args.b2_selectivities,
                target_relation=args.b2_target_relation,
                b2b_path_fanout=args.b2b_path_fanout,
            )
            print(f"  B2 graph regenerated in-memory: {graph.n_actual} edges")
        else:
            graph.generate()

    print("Connecting to Cassandra...")
    session, cluster = get_session()

    if not args.skip_import:
        if args.b2_selectivity_graph:
            import_b2_to_cassandra(session, graph.edges)
        elif args.b3_cache_graph:
            import_b3_to_cassandra(session, graph.edges)
        else:
            import_to_cassandra(session, graph.edges, truncate=args.truncate)

    b1_rows, b2_rows, b3_rows = [], [], []

    try:
        if not args.skip_b1:
            b1_rows = run_b1_parallel_worker_sweep(session, graph, args)
        if not args.skip_b2:
            b2_rows = run_b2_relation_index(session, graph, args)
        if not args.skip_b3:
            b3_rows = run_b3_cache_effective_latency(session, graph, args)
        if b1_rows or b2_rows or b3_rows:
            generate_summary(b1_rows, b2_rows, b3_rows)
    finally:
        cluster.shutdown()

    print("\nDone.")


if __name__ == "__main__":
    main()
