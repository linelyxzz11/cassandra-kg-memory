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

    def get_seed_entities(self, n=200, high_deg_bias=True):
        if high_deg_bias:
            sorted_entities = sorted(self.entity_outdegree.items(), key=lambda x: -x[1])
            half = len(sorted_entities) // 2
            top = [e for e, _ in sorted_entities[:half]]
            seeds = self.rng.sample(top, min(n, len(top))) if top else []
        else:
            seeds = self.rng.sample(self.entity_ids, min(n, len(self.entity_ids)))
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
        return {
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
                      cache=None, use_index=False, raise_on_error=True):
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
        if not src_list:
            break

        if workers == 1:
            for src in src_list:
                edges, partition_reads, raw_count, n_ret = fetch_one(src, depth)
                total_partition_reads += partition_reads
                total_raw_edges_from_db += raw_count
                total_returned += n_ret
                for _, _, dst, _ in edges:
                    next_frontier.add(dst)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(fetch_one, src, depth): src for src in src_list}
                for f in as_completed(futures):
                    edges, partition_reads, raw_count, n_ret = f.result()
                    total_partition_reads += partition_reads
                    total_raw_edges_from_db += raw_count
                    total_returned += n_ret
                    for _, _, dst, _ in edges:
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


def run_b1_parallel_worker_sweep(session, graph, args):
    print("\n=== B1: Parallel Worker Sweep ===")
    seeds = graph.get_seed_entities(n=args.n_queries, high_deg_bias=False)
    print(f"  Smoke query: trying src={seeds[0]}...")
    try:
        r = frontier_traverse(session, seeds[0], GRAPH_ID, workers=1, max_depth=1, cache=None, use_index=False)
        print(f"  Smoke: latency={r['latency_ms']:.2f}ms, reads={r['raw_reads']}, returned={r['returned_edges']}, errors={r['errors']}")
        if r["returned_edges"] == 0 or r["errors"] > 0:
            print("  WARNING: Smoke query returned 0 edges! Cassandra data may be empty or schema mismatch.")
    except Exception as ex:
        print(f"  Smoke query FAILED: {ex}")
    worker_levels = [1, 4, 8, 16, 32]
    hop_depths = [2]
    if args.b1_hop4:
        hop_depths.append(4)
    summary_rows = []

    for hop in hop_depths:
        print(f"  Hop={hop}...")
        naive_latencies = []
        for rep in range(REPEATS):
            for i, seed in enumerate(seeds):
                r = frontier_traverse(session, seed, GRAPH_ID, workers=1,
                                      relation_path=None, max_depth=hop, cache=None, use_index=False)
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
            "notes": f"B1 hop={hop} baseline",
        })

        for workers in worker_levels:
            if workers == 1:
                continue
            print(f"    Workers={workers}...")
            all_latencies = []
            for rep in range(REPEATS):
                for i, seed in enumerate(seeds):
                    r = frontier_traverse(session, seed, GRAPH_ID, workers=workers,
                                          relation_path=None, max_depth=hop, cache=None, use_index=False)
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
                "notes": f"B1 hop={hop}",
            })

    out = OUT_DIR / "layerB_parallel_worker_sweep.csv"
    fields = list(summary_rows[0].keys()) if summary_rows else []
    write_csv(out, summary_rows, fields)
    print(f"  -> {out}")
    return summary_rows


def run_b2_relation_index(session, graph, args):
    print("\n=== B2: Relation-Index Characterization ===")
    summary_rows = []

    print("  B2a: One-hop src+relation microbenchmark...")
    for selectivity in [0.01, 0.10, 0.50]:
        seeds = graph.get_relation_selective_seeds_per_source("likes", selectivity)
        if not seeds:
            print(f"    selectivity={selectivity}: no seeds found, skipping")
            continue
        seeds = seeds[:min(len(seeds), args.n_queries)]
        print(f"    selectivity={selectivity}, n_seeds={len(seeds)}")

        scan_times, idx_times = [], []
        scan_raw_counts, scan_ret_counts = [], []
        idx_raw_counts, idx_ret_counts = [], []
        for rep in range(REPEATS):
            for i, seed in enumerate(seeds):
                r_scan = frontier_traverse(session, seed, GRAPH_ID, workers=16,
                                           relation_path=["likes"], max_depth=1, cache=None, use_index=False)
                scan_times.append(r_scan["latency_ms"])
                scan_raw_counts.append(r_scan["raw_edges_from_db"])
                scan_ret_counts.append(r_scan["returned_edges"])
                log_raw_result("B2a", "src_scan", f"sel{selectivity}_q{i}", seed, 1, ["likes"],
                               r_scan["latency_ms"], r_scan["raw_reads"], r_scan["returned_edges"], 0, r_scan["errors"])

                r_idx = frontier_traverse(session, seed, GRAPH_ID, workers=16,
                                          relation_path=["likes"], max_depth=1, cache=None, use_index=True)
                idx_times.append(r_idx["latency_ms"])
                idx_raw_counts.append(r_idx["raw_edges_from_db"])
                idx_ret_counts.append(r_idx["returned_edges"])
                log_raw_result("B2a", "src_relation_index", f"sel{selectivity}_q{i}", seed, 1, ["likes"],
                               r_idx["latency_ms"], r_idx["raw_reads"], r_idx["returned_edges"], 0, r_idx["errors"])

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
                "hop": 1,
                "workers": 16,
                "cache_policy": "none",
                "index_enabled": use_idx,
                "cache_state": "cold",
                "n_queries": len(seeds) * REPEATS,
                "mean_latency_ms": stats["mean"],
                "p50_latency_ms": stats["p50"],
                "p95_latency_ms": stats["p95"],
                "p99_latency_ms": stats["p99"],
                "qps": 0,
                "raw_edges_from_db": round(statistics.mean(raw_counts), 3) if raw_counts else 0,
                "returned_edges": round(statistics.mean(ret_counts), 3) if ret_counts else 0,
                "cache_hit_rate": 0.0,
                "effective_latency_ms": stats["mean"],
                "speedup_vs_naive": speedup if use_idx else 1.0,
                "notes": f"B2a selectivity={selectivity}",
            })

    print("  B2b: Multi-hop fixed relation path (likes -> suitable_for -> related_to -> suggests)...")
    path = RELATION_PATH_B2B
    seeds_b2b = graph.get_seed_entities(n=min(args.n_queries, 50), high_deg_bias=False)
    if seeds_b2b:
        scan_times_b2b, idx_times_b2b = [], []
        scan_raw_b2b, scan_ret_b2b = [], []
        idx_raw_b2b, idx_ret_b2b = [], []
        for rep in range(REPEATS):
            for i, seed in enumerate(seeds_b2b):
                r_scan = frontier_traverse(session, seed, GRAPH_ID, workers=16,
                                           relation_path=path, max_depth=len(path), cache=None, use_index=False)
                scan_times_b2b.append(r_scan["latency_ms"])
                scan_raw_b2b.append(r_scan["raw_edges_from_db"])
                scan_ret_b2b.append(r_scan["returned_edges"])
                log_raw_result("B2b", "src_scan", f"q{i}", seed, len(path), path,
                               r_scan["latency_ms"], r_scan["raw_reads"], r_scan["returned_edges"], 0, r_scan["errors"])

                r_idx = frontier_traverse(session, seed, GRAPH_ID, workers=16,
                                          relation_path=path, max_depth=len(path), cache=None, use_index=True)
                idx_times_b2b.append(r_idx["latency_ms"])
                idx_raw_b2b.append(r_idx["raw_edges_from_db"])
                idx_ret_b2b.append(r_idx["returned_edges"])
                log_raw_result("B2b", "src_relation_index", f"q{i}", seed, len(path), path,
                               r_idx["latency_ms"], r_idx["raw_reads"], r_idx["returned_edges"], 0, r_idx["errors"])

        scan_stats_b = collect_latency_stats(scan_times_b2b)
        idx_stats_b = collect_latency_stats(idx_times_b2b)
        speedup_b = round(scan_stats_b["mean"] / max(idx_stats_b["mean"], 0.001), 2)

        for mode, stats, use_idx, raw_counts, ret_counts in [
            ("src_scan", scan_stats_b, False, scan_raw_b2b, scan_ret_b2b),
            ("src_relation_index", idx_stats_b, True, idx_raw_b2b, idx_ret_b2b),
        ]:
            summary_rows.append({
                "experiment": "B2b_multihop",
                "mode": mode,
                "workload_type": "fixed_relation_path_4hop",
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
                "qps": 0,
                "raw_edges_from_db": round(statistics.mean(raw_counts), 3) if raw_counts else 0,
                "returned_edges": round(statistics.mean(ret_counts), 3) if ret_counts else 0,
                "cache_hit_rate": 0.0,
                "effective_latency_ms": stats["mean"],
                "speedup_vs_naive": speedup_b if use_idx else 1.0,
                "notes": "B2b path=likes->suitable_for->related_to->suggests",
            })

    out = OUT_DIR / "layerB_relation_index_characterization.csv"
    fields = list(summary_rows[0].keys()) if summary_rows else []
    write_csv(out, summary_rows, fields)
    print(f"  -> {out}")
    return summary_rows


def run_b3_cache_effective_latency(session, graph, args):
    print("\n=== B3: Cache Effective-Latency Characterization ===")
    seeds = graph.get_seed_entities(n=args.n_queries, high_deg_bias=True)
    print(f"  n_seeds={len(seeds)} (high-degree biased)")
    summary_rows = []

    def evaluate_cache(cache, seeds, workers, label, clear_before=True, repeat=1):
        if clear_before and cache:
            cache.clear()
        latencies = []
        for _ in range(repeat):
            for i, seed in enumerate(seeds):
                r = frontier_traverse(session, seed, GRAPH_ID, workers=workers,
                                      relation_path=None, max_depth=2, cache=cache, use_index=False)
                hit_flag = 1 if r["cache_hits"] > 0 else 0
                latencies.append(r["latency_ms"])
                log_raw_result("B3_cache", label, f"q{i}", seed, 2, None,
                               r["latency_ms"], r["raw_reads"], r["returned_edges"], hit_flag, r["errors"])
        stats = collect_latency_stats(latencies)
        hr = cache.hit_rate() if cache else 0.0
        return stats, hr, latencies

    print("  B3a: No cache (cold baseline)...")
    nc_stats, _, nc_lat = evaluate_cache(None, seeds, 16, "no_cache_cold")

    print("  B3b: Warm-cache (LRU, repeated same queries)...")
    lru = LRUCache(capacity=CACHE_CAP)
    evaluate_cache(lru, seeds, 16, "LRU_warmup", clear_before=False, repeat=2)
    lru.reset_stats()
    lru_warm_stats, lru_hr, lru_warm_lat = evaluate_cache(lru, seeds, 16, "LRU_warm", clear_before=False, repeat=1)

    lru_cold_stats, _, lru_cold_lat = evaluate_cache(LRUCache(capacity=CACHE_CAP), seeds, 16, "LRU_cold", clear_before=True, repeat=1)
    lru_eff = lru_hr * lru_warm_stats["mean"] + (1.0 - lru_hr) * lru_cold_stats["mean"]

    print("  B3b: Warm-cache (HighDegree, repeated same queries)...")
    hd = HighDegreeCache(capacity=CACHE_CAP, degree_threshold=DEGREE_THRESHOLD)
    for eid, deg in graph.entity_outdegree.items():
        hd.set_degree(eid, deg)
    evaluate_cache(hd, seeds, 16, "HD_warmup", clear_before=False, repeat=2)
    hd.reset_stats()
    hd_warm_stats, hd_hr, hd_warm_lat = evaluate_cache(hd, seeds, 16, "HD_warm", clear_before=False, repeat=1)

    hd_cold = HighDegreeCache(capacity=CACHE_CAP, degree_threshold=DEGREE_THRESHOLD)
    for eid, deg in graph.entity_outdegree.items():
        hd_cold.set_degree(eid, deg)
    hd_cold_stats, _, hd_cold_lat = evaluate_cache(hd_cold, seeds, 16, "HD_cold", clear_before=True, repeat=1)
    hd_eff = hd_hr * hd_warm_stats["mean"] + (1.0 - hd_hr) * hd_cold_stats["mean"]

    print("  B3c: Realistic-cache (skewed continuous stream)...")
    realistic_seeds = graph.get_skewed_seed_sequence(n_query_cycles=3, seeds_per_cycle=args.n_queries)
    print(f"  realistic n_seeds={len(realistic_seeds)}")

    lru_real = LRUCache(capacity=CACHE_CAP)
    real_times_lru = []
    for i, seed in enumerate(realistic_seeds):
        r = frontier_traverse(session, seed, GRAPH_ID, workers=16,
                              relation_path=None, max_depth=2, cache=lru_real, use_index=False)
        hit_flag = 1 if r["cache_hits"] > 0 else 0
        real_times_lru.append(r["latency_ms"])
        log_raw_result("B3_cache", "LRU_realistic", f"q{i}", seed, 2, None,
                       r["latency_ms"], r["raw_reads"], r["returned_edges"], hit_flag, r["errors"])
    lru_real_stats = collect_latency_stats(real_times_lru)
    lru_real_hr = lru_real.hit_rate()
    lru_real_eff = lru_real_hr * lru_warm_stats["mean"] + (1.0 - lru_real_hr) * lru_cold_stats["mean"]

    hd_real = HighDegreeCache(capacity=CACHE_CAP, degree_threshold=DEGREE_THRESHOLD)
    for eid, deg in graph.entity_outdegree.items():
        hd_real.set_degree(eid, deg)
    real_times_hd = []
    for i, seed in enumerate(realistic_seeds):
        r = frontier_traverse(session, seed, GRAPH_ID, workers=16,
                              relation_path=None, max_depth=2, cache=hd_real, use_index=False)
        hit_flag = 1 if r["cache_hits"] > 0 else 0
        real_times_hd.append(r["latency_ms"])
        log_raw_result("B3_cache", "HD_realistic", f"q{i}", seed, 2, None,
                       r["latency_ms"], r["raw_reads"], r["returned_edges"], hit_flag, r["errors"])
    hd_real_stats = collect_latency_stats(real_times_hd)
    hd_real_hr = hd_real.hit_rate()
    hd_real_eff = hd_real_hr * hd_warm_stats["mean"] + (1.0 - hd_real_hr) * hd_cold_stats["mean"]

    cache_rows = [
        ("no_cache", "none", "cold", nc_stats, 0.0, nc_stats["mean"], nc_stats["mean"], nc_stats["mean"]),
        ("LRU", "LRU", "cold", lru_cold_stats, 0.0, lru_cold_stats["mean"], lru_cold_stats["mean"], lru_cold_stats["mean"]),
        ("LRU", "LRU", "warm", lru_warm_stats, lru_hr, lru_cold_stats["mean"], lru_warm_stats["mean"], lru_eff),
        ("LRU", "LRU", "realistic", lru_real_stats, lru_real_hr, lru_cold_stats["mean"], lru_warm_stats["mean"], lru_real_eff),
        ("HighDegree", "high_degree", "cold", hd_cold_stats, 0.0, hd_cold_stats["mean"], hd_cold_stats["mean"], hd_cold_stats["mean"]),
        ("HighDegree", "high_degree", "warm", hd_warm_stats, hd_hr, hd_cold_stats["mean"], hd_warm_stats["mean"], hd_eff),
        ("HighDegree", "high_degree", "realistic", hd_real_stats, hd_real_hr, hd_cold_stats["mean"], hd_warm_stats["mean"], hd_real_eff),
    ]

    effective_base = nc_stats["mean"]
    for mode_label, cache_policy, cache_state, stats, hr, cold_lat, warm_lat, eff_lat in cache_rows:
        speedup = round(effective_base / max(eff_lat, 0.001), 2)
        summary_rows.append({
            "experiment": "B3_cache",
            "mode": f"Cassandra+parallel+{mode_label}",
            "workload_type": f"repeated_skewed_{cache_state}",
            "hop": 2,
            "workers": 16,
            "cache_policy": cache_policy,
            "index_enabled": False,
            "cache_state": cache_state,
            "n_queries": len(seeds) if cache_state != "realistic" else len(realistic_seeds),
            "mean_latency_ms": stats["mean"],
            "p50_latency_ms": stats["p50"],
            "p95_latency_ms": stats["p95"],
            "p99_latency_ms": stats["p99"],
            "qps": 0,
            "raw_edges_from_db": 0,
            "cache_hit_rate": round(hr, 4),
            "cold_latency_ms": round(cold_lat, 3),
            "warm_latency_ms": round(warm_lat, 3),
            "effective_latency_ms": round(eff_lat, 3),
            "speedup_vs_naive": speedup,
            "notes": f"B3 {cache_state}; headline=realistic only",
        })

    out = OUT_DIR / "layerB_cache_effective_latency.csv"
    fields = [k for k in summary_rows[0].keys()] if summary_rows else []
    write_csv(out, summary_rows, fields)
    print(f"  -> {out}")
    return summary_rows


def generate_summary(b1, b2, b3):
    print("\n=== Layer B Total Summary ===")
    all_rows = b1 + b2 + b3
    if not all_rows:
        print("  No results to summarize")
        return
    out = OUT_DIR / "layerB_cassandra_internal_ablation_summary.csv"
    fields = list(all_rows[0].keys())
    write_csv(out, all_rows, fields)
    print(f"  -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-entities", type=int, default=10000)
    parser.add_argument("--n-edges", type=int, default=1000000)
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--graph-id", type=str, default=None, help="Graph id to use. If omitted, a unique graph id is generated for this run.")
    parser.add_argument("--truncate", action="store_true", help="Explicitly TRUNCATE all KG tables before import. Default is false to avoid Cassandra TRUNCATE stalls.")
    parser.add_argument("--b1-hop4", action="store_true", help="Also run hop=4 in B1")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--skip-b1", action="store_true")
    parser.add_argument("--skip-b2", action="store_true")
    parser.add_argument("--skip-b3", action="store_true")
    args = parser.parse_args()

    global GRAPH_ID
    if args.graph_id:
        GRAPH_ID = args.graph_id
    elif not args.skip_import:
        GRAPH_ID = f"synth_{args.n_edges}_{int(time.time())}"
    # When --skip-import is used, caller should pass --graph-id for existing data.

    print(f"Using GRAPH_ID={GRAPH_ID}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
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
                               high_degree_frac=0.02, high_degree_mult=20).generate()

    print("Connecting to Cassandra...")
    session, cluster = get_session()

    if not args.skip_import:
        import_to_cassandra(session, graph.edges)

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