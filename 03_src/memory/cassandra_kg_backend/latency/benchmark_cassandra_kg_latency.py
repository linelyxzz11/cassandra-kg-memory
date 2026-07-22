import argparse
import csv
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from cassandra.cluster import Cluster

BASELINE_QUERY = """
SELECT src_id, relation, dst_id, source
FROM kg_edges_by_src
WHERE graph_id=%s AND src_id=%s
"""

INDEX_QUERY = """
SELECT src_id, relation, dst_id, source
FROM kg_edges_by_src_relation
WHERE graph_id=%s AND src_id=%s AND relation=%s
"""

BASELINE_RELATION_QUERY = """
SELECT src_id, relation, dst_id, source
FROM kg_edges_by_src
WHERE graph_id=%s AND src_id=%s AND relation=%s
"""


class BoundedOneHopCache:
    def __init__(self, capacity=200):
        self.capacity = capacity
        self.store = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        if self.capacity <= 0:
            return None, False
        with self.lock:
            if key not in self.store:
                return None, False
            edges = self.store.pop(key)
            self.store[key] = edges
        return edges, True

    def set(self, key, edges):
        if self.capacity <= 0:
            return
        with self.lock:
            if key in self.store:
                self.store.pop(key)
            elif len(self.store) >= self.capacity:
                self.store.popitem(last=False)
            self.store[key] = edges

    def size(self):
        with self.lock:
            return len(self.store)


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((p / 100) * (len(values) - 1)))
    return values[idx]


def load_test_entities(edges_csv, n=100):
    entities = []
    with open(edges_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            entities.append((row["src_id"].strip(), row["relation"].strip(), row["graph_id"].strip()))
    random.seed(42)
    random.shuffle(entities)
    return entities[:n]


def run_baseline(session, graph_id, src_id):
    start = time.perf_counter()
    rows = list(session.execute(BASELINE_QUERY, (graph_id, src_id)))
    elapsed = (time.perf_counter() - start) * 1000
    return {"latency_ms": elapsed, "edge_count": len(rows), "raw_count": len(rows)}


def run_baseline_with_relation(session, graph_id, src_id, relation):
    start = time.perf_counter()
    rows = list(session.execute(BASELINE_RELATION_QUERY, (graph_id, src_id, relation)))
    elapsed = (time.perf_counter() - start) * 1000
    return {"latency_ms": elapsed, "edge_count": len(rows), "raw_count": len(rows)}


def run_index(session, graph_id, src_id, relation):
    start = time.perf_counter()
    rows = list(session.execute(INDEX_QUERY, (graph_id, src_id, relation)))
    elapsed = (time.perf_counter() - start) * 1000
    return {"latency_ms": elapsed, "edge_count": len(rows), "raw_count": len(rows)}


def run_mode(session, entities, mode, workers=1, cache=None, warmup=3, repeat=5):
    run_fn = None
    if mode == "baseline":
        run_fn = lambda gid, src, rel: run_baseline(session, gid, src)
    elif mode == "index":
        run_fn = lambda gid, src, rel: run_index(session, gid, src, rel)
    elif mode == "parallel":
        run_fn = lambda gid, src, rel: run_baseline(session, gid, src)
    elif mode == "cache":
        run_fn = lambda gid, src, rel: run_baseline(session, gid, src)
    elif mode == "all":
        run_fn = lambda gid, src, rel: run_index(session, gid, src, rel)
    else:
        raise ValueError(mode)

    use_parallel = (mode in ("parallel", "all"))
    use_cache_flag = (mode in ("cache", "all"))
    use_index_flag = (mode in ("index", "all"))
    n_workers = workers if use_parallel else 1

    for _ in range(warmup):
        bench_entities(session, entities, use_index_flag, n_workers, cache if use_cache_flag else None)

    all_runs = []
    for _ in range(repeat):
        run_result = bench_entities(session, entities, use_index_flag, n_workers, cache if use_cache_flag else None)
        all_runs.append(run_result)

    latencies = []
    total_raws = []
    c_hits = []
    for rr in all_runs:
        latencies.append(rr["total_wall_ms"])
        total_raws.append(rr["total_raw_reads"])
        c_hits.append(rr["cache_hits"])

    return {
        "mode": mode,
        "n_entities": len(entities),
        "workers": n_workers,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "mean_ms": mean(latencies),
        "avg_raw_reads": mean(total_raws),
        "avg_cache_hits": mean(c_hits),
        "cache_hit_rate": mean([rr["cache_hit_rate"] for rr in all_runs]),
        "cache_size": cache.size() if cache else 0,
    }


def bench_entities(session, entities, use_index, workers, cache):
    total_start = time.perf_counter()
    total_raw = 0
    cache_hits = 0
    cache_misses = 0

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for gid, src, rel in entities:
                ckey = (gid, src)
                if cache is not None:
                    cached, hit = cache.get(ckey)
                    if hit:
                        cache_hits += 1
                        with threading.Lock():
                            total_raw += cached["raw_count"]
                        continue

                if use_index:
                    f = executor.submit(run_index, session, gid, src, rel)
                else:
                    f = executor.submit(run_baseline, session, gid, src)
                futures[f] = (gid, src, rel, ckey)

            for f in as_completed(futures):
                gid, src, rel, ckey = futures[f]
                r = f.result()
                total_raw += r["raw_count"]
                if cache is not None:
                    cache.set(ckey, r)
                    cache_misses += 1
    else:
        for gid, src, rel in entities:
            ckey = (gid, src)
            if cache is not None:
                cached, hit = cache.get(ckey)
                if hit:
                    cache_hits += 1
                    total_raw += cached["raw_count"]
                    continue

            if use_index:
                r = run_index(session, gid, src, rel)
            else:
                r = run_baseline(session, gid, src)

            total_raw += r["raw_count"]
            if cache is not None:
                cache.set(ckey, r)
                cache_misses += 1

    total_elapsed = (time.perf_counter() - total_start) * 1000
    total = cache_hits + cache_misses if cache else len(entities)
    return {
        "total_wall_ms": total_elapsed,
        "total_raw_reads": total_raw,
        "cache_hits": cache_hits,
        "cache_hit_rate": cache_hits / total if cache else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9042)
    parser.add_argument("--keyspace", default="ai_memory")
    parser.add_argument("--edges-csv", default=str(Path(__file__).resolve().parent.parent.parent / "results/locomo_kg_edges_spacy.csv"))
    parser.add_argument("--n-entities", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cache-capacity", type=int, default=200)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent.parent.parent / "results/final/final_cassandra_kg_serving_ablation.csv"))
    args = parser.parse_args()

    print("=" * 70)
    print("Cassandra-KG Serving Latency Ablation")
    print("=" * 70)

    print(f"\nLoading test entities from {args.edges_csv}...")
    entities = load_test_entities(args.edges_csv, args.n_entities)
    print(f"  {len(entities)} entities selected")
    relations = set(rel for _, rel, _ in entities)
    print(f"  {len(relations)} unique relations")

    cluster = Cluster([args.host], port=args.port)
    session = cluster.connect(args.keyspace)

    modes = [
        ("baseline", "No optimization (full src scan)"),
        ("index", "+ relation-aware source index"),
        ("parallel", "+ parallel frontier (workers={})".format(args.workers)),
        ("cache", "+ high-degree cache (cap={})".format(args.cache_capacity)),
        ("all", "All optimizations combined"),
    ]

    results = []
    baseline_mean = None

    for mode, desc in modes:
        print(f"\n--- {mode}: {desc} ---")
        cache = BoundedOneHopCache(capacity=args.cache_capacity) if mode in ("cache", "all") else None
        run = run_mode(session, entities, mode,
                       workers=args.workers, cache=cache,
                       warmup=args.warmup, repeat=args.repeat)

        if mode == "baseline":
            baseline_mean = run["mean_ms"]

        speedup = baseline_mean / run["mean_ms"] if baseline_mean else 1.0

        results.append({
            "mode": mode,
            "description": desc,
            "n_entities": run["n_entities"],
            "workers": run["workers"],
            "p50_ms": f"{run['p50_ms']:.3f}",
            "p95_ms": f"{run['p95_ms']:.3f}",
            "p99_ms": f"{run['p99_ms']:.3f}",
            "mean_ms": f"{run['mean_ms']:.3f}",
            "avg_raw_reads": f"{run['avg_raw_reads']:.1f}",
            "avg_cache_hits": f"{run['avg_cache_hits']:.1f}",
            "cache_hit_rate": f"{run['cache_hit_rate']:.4f}",
            "cache_size": run["cache_size"],
            "speedup": f"{speedup:.2f}x",
        })

        print(f"  mean={run['mean_ms']:.3f}ms  p95={run['p95_ms']:.3f}ms  p99={run['p99_ms']:.3f}ms  "
              f"reads={run['avg_raw_reads']:.1f}  ch_rate={run['cache_hit_rate']:.4f}  speedup={speedup:.2f}x")

    cluster.shutdown()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode", "description", "n_entities", "workers",
        "p50_ms", "p95_ms", "p99_ms", "mean_ms",
        "avg_raw_reads", "avg_cache_hits", "cache_hit_rate",
        "cache_size", "speedup",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print(f"\n{'=' * 70}")
    print(f"Results saved: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
