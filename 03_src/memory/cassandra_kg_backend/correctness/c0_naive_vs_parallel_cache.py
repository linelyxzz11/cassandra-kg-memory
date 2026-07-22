import argparse
import json
import statistics
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cassandra.cluster import Cluster


def percentile(data, p):
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (k - f) * (s[c] - s[f]) if c > f else s[f]


class DegreeAwareCache:
    def __init__(self, capacity, degree_threshold):
        self.capacity = capacity
        self.degree_threshold = degree_threshold
        self._entries = OrderedDict()
        self._degrees = {}
        self.hits = 0
        self.misses = 0
        self.admitted = set()

    def key(self, graph_id, src_id):
        return (graph_id, src_id)

    def set_degree(self, graph_id, src_id, outdegree):
        self._degrees[(graph_id, src_id)] = outdegree

    def get(self, graph_id, src_id):
        k = self.key(graph_id, src_id)
        degree = self._degrees.get(k, 0)
        if degree < self.degree_threshold:
            self.misses += 1
            return None
        if k not in self._entries:
            self.misses += 1
            return None
        self._entries.move_to_end(k)
        self.hits += 1
        return list(self._entries[k])

    def put(self, graph_id, src_id, edges):
        k = self.key(graph_id, src_id)
        degree = self._degrees.get(k, 0)
        if degree < self.degree_threshold:
            return
        self.admitted.add(src_id)
        if k in self._entries:
            self._entries.pop(k)
        elif len(self._entries) >= self.capacity:
            self._entries.popitem(last=False)  # LRU eviction: pop the least-recently-used entry
        self._entries[k] = list(edges)

    def clear(self):
        self._entries.clear()
        self._degrees.clear()
        self.hits = 0
        self.misses = 0


def fetch_edges(session, graph_id, src_id):
    rows = session.execute(
        "SELECT src_id, relation, dst_id, source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
        (graph_id, src_id),
    )
    edges = [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]
    edges.sort(key=lambda e: (e[1], e[2], e[3]))
    return edges


def traverse(session, seed, graph_id, workers, cache, max_depth, fanout):
    t0 = time.perf_counter()
    frontier = {(seed, (seed,), ())}
    total_raw = 0
    queried = {}

    for depth in range(max_depth):
        sources = sorted({s[0] for s in frontier})
        src_edges = {}
        if workers == 1:
            for src in sources:
                cached = cache.get(graph_id, src) if cache else None
                if cached is not None:
                    src_edges[src] = cached
                else:
                    edges = fetch_edges(session, graph_id, src)
                    total_raw += len(edges)
                    queried[src] = len(edges)
                    if cache:
                        cache.set_degree(graph_id, src, len(edges))
                        cache.put(graph_id, src, edges)
                    src_edges[src] = edges
        else:
            pending = []
            for src in sources:
                if cache:
                    cached = cache.get(graph_id, src)
                    if cached is not None:
                        src_edges[src] = cached
                        continue
                pending.append(src)

            def _do(src):
                return src, fetch_edges(session, graph_id, src)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_do, src): src for src in pending}
                for f in as_completed(futures):
                    src, edges = f.result()
                    total_raw += len(edges)
                    queried[src] = len(edges)
                    if cache:
                        cache.set_degree(graph_id, src, len(edges))
                        cache.put(graph_id, src, edges)
                    src_edges[src] = edges

        next_frontier = set()
        for src, node_path, edge_path in frontier:
            edges = src_edges.get(src, [])
            candidates = [e for e in edges if e[2] not in node_path]
            for _, rel, dst, source in candidates[:fanout]:
                next_frontier.add((dst, node_path + (dst,), edge_path + (source,)))
        frontier = next_frontier
        if not frontier:
            break

    elapsed = (time.perf_counter() - t0) * 1000
    paths = tuple(sorted({s[2] for s in frontier}))
    return {
        "paths": paths,
        "latency_ms": round(elapsed, 3),
        "raw_edges": total_raw,
        "cache_hits": cache.hits if cache else 0,
        "cache_misses": cache.misses if cache else 0,
        "queried_nodes": {k: v for k, v in sorted(queried.items(), key=lambda x: -x[1])},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-id", required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--hop", type=int, default=4)
    p.add_argument("--fanout", type=int, default=20)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--cache-capacity", type=int, default=200)
    p.add_argument("--degree-threshold", type=int, default=100)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9042)
    p.add_argument("--keyspace", default="ai_memory")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Graph: {args.graph_id}  Seed: {args.seed}  Hop: {args.hop}  Fanout: {args.fanout}")
    print(f"Workers: {args.workers}  Cache: cap={args.cache_capacity}  threshold={args.degree_threshold}")
    print()

    cluster = Cluster([args.host], port=args.port)
    session = cluster.connect(args.keyspace)

    # Verify seed exists
    check = list(session.execute(
        "SELECT count(*) FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
        (args.graph_id, args.seed),
    ))
    outdeg = check[0].count if check else 0
    if outdeg == 0:
        print(f"ERROR: seed '{args.seed}' has no edges in graph {args.graph_id}")
        cluster.shutdown()
        return
    print(f"Seed outdegree: {outdeg}")

    # Naive run
    print("\n[1] Naive (workers=1, no cache)...")
    r_naive = traverse(session, args.seed, args.graph_id, 1, None, args.hop, args.fanout)
    print(f"  paths={len(r_naive['paths'])}  latency={r_naive['latency_ms']}ms  raw_edges={r_naive['raw_edges']}")

    # Optimized cold run
    print("\n[2] Optimized COLD (workers=16, high-degree cache, fresh)...")
    cache_cold = DegreeAwareCache(args.cache_capacity, args.degree_threshold)
    r_cold = traverse(session, args.seed, args.graph_id, args.workers, cache_cold, args.hop, args.fanout)
    print(f"  paths={len(r_cold['paths'])}  latency={r_cold['latency_ms']}ms  raw_edges={r_cold['raw_edges']}")
    print(f"  cache_hits={r_cold['cache_hits']}  cache_misses={r_cold['cache_misses']}")

    # Optimized warm run (reuse same cache, same seed)
    cold_hits_snapshot = cache_cold.hits
    cold_misses_snapshot = cache_cold.misses
    print("\n[3] Optimized WARM (workers=16, same cache, same seed)...")
    r_warm = traverse(session, args.seed, args.graph_id, args.workers, cache_cold, args.hop, args.fanout)
    warm_hits = cache_cold.hits - cold_hits_snapshot
    warm_misses = cache_cold.misses - cold_misses_snapshot
    print(f"  paths={len(r_warm['paths'])}  latency={r_warm['latency_ms']}ms  raw_edges={r_warm['raw_edges']}")
    print(f"  cache_hits={warm_hits}  cache_misses={warm_misses}")

    cluster.shutdown()

    # Diagnostics
    print(f"\n===== DIAGNOSTICS =====")
    print(f"Queried nodes (outdegree) during cold run:")
    for src, deg in sorted(r_cold["queried_nodes"].items(), key=lambda x: -x[1]):
        flag = " [ADMITTED]" if src in cache_cold.admitted else ""
        print(f"  {src}: {deg}{flag}")
    print(f"\ncache_admitted_nodes (outdegree >= {args.degree_threshold}): {sorted(cache_cold.admitted)}")
    max_deg = max(r_cold["queried_nodes"].values()) if r_cold["queried_nodes"] else 0
    print(f"max_degree_in_traversal: {max_deg}")

    # Comparisons
    def diff(a, b, label):
        a_set = set(a["paths"])
        b_set = set(b["paths"])
        missing = sorted(a_set - b_set)
        extra = sorted(b_set - a_set)
        match = len(missing) == 0 and len(extra) == 0
        print(f"\n{label}: {'MATCH' if match else 'MISMATCH'}")
        if not match:
            print(f"  missing={len(missing)}  extra={len(extra)}")
        return match, missing, extra

    match_cold, miss_cold, ext_cold = diff(r_naive, r_cold, "naive vs cold")
    match_warm, miss_warm, ext_warm = diff(r_naive, r_warm, "naive vs warm")

    # Mismatches
    if miss_cold or ext_cold:
        with (out_dir / "mismatches_cold.jsonl").open("w") as f:
            f.write(json.dumps({"missing": miss_cold[:50], "extra": ext_cold[:50]}) + "\n")
    if miss_warm or ext_warm:
        with (out_dir / "mismatches_warm.jsonl").open("w") as f:
            f.write(json.dumps({"missing": miss_warm[:50], "extra": ext_warm[:50]}) + "\n")

    summary = {
        "experiment": "C0-B Cassandra naive vs parallel+cache semantic gate",
        "graph_id": args.graph_id,
        "seed": args.seed,
        "hop": args.hop,
        "fanout": args.fanout,
        "naive_workers": 1,
        "parallel_workers": args.workers,
        "cache_policy": "high_degree",
        "cache_capacity": args.cache_capacity,
        "degree_threshold": args.degree_threshold,
        "relation_index": "disabled",
        "naive": {
            "paths": len(r_naive["paths"]),
            "latency_ms": r_naive["latency_ms"],
            "raw_edges": r_naive["raw_edges"],
        },
        "optimized_cold": {
            "paths": len(r_cold["paths"]),
            "latency_ms": r_cold["latency_ms"],
            "raw_edges": r_cold["raw_edges"],
            "cache_hits": r_cold["cache_hits"],
            "cache_misses": r_cold["cache_misses"],
        },
        "optimized_warm": {
            "paths": len(r_warm["paths"]),
            "latency_ms": r_warm["latency_ms"],
            "raw_edges": r_warm["raw_edges"],
            "cache_hits": r_warm["cache_hits"],
            "cache_misses": r_warm["cache_misses"],
        },
        "naive_vs_cold_disagreements": 0 if match_cold else 1,
        "naive_vs_warm_disagreements": 0 if match_warm else 1,
        "all_pass": match_cold and match_warm,
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n===== C0-B SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
