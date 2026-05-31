import argparse
import csv
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from cassandra.cluster import Cluster


DEFAULT_RELATION_WHITELIST = {
    "likes",
    "suitable_for",
    "related_to",
    "suggests",
    "leads_to",
    "studies",
    "includes",
    "used_for",
    "requires",
    "uses",
    "has_feature",
    "supports",
    "prevents",
    "solved_by",
    "improves",
    "mentions",
    "located_in",
}


class BoundedOneHopCache:
    def __init__(self, capacity, policy="lru", degree_threshold=100):
        self.capacity = capacity
        self.policy = policy
        self.degree_threshold = degree_threshold
        self.store = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        if self.capacity <= 0:
            return None, False

        with self.lock:
            if key not in self.store:
                return None, False

            item = self.store.pop(key)
            self.store[key] = item
            return item["edges"], True

    def set(self, key, edges):
        if self.capacity <= 0:
            return {"admitted": 0, "rejected": 1, "evicted": 0}

        degree = len(edges)
        item = {
            "edges": edges,
            "degree": degree,
            "is_high_degree": degree >= self.degree_threshold,
        }

        with self.lock:
            if key in self.store:
                self.store.pop(key)
                self.store[key] = item
                return {"admitted": 1, "rejected": 0, "evicted": 0}

            if len(self.store) < self.capacity:
                self.store[key] = item
                return {"admitted": 1, "rejected": 0, "evicted": 0}

            if self.policy == "lru":
                self.store.popitem(last=False)
                self.store[key] = item
                return {"admitted": 1, "rejected": 0, "evicted": 1}

            if self.policy == "high_degree":
                return self._set_high_degree_priority_locked(key, item)

            raise ValueError("Unsupported cache policy.")

    def _set_high_degree_priority_locked(self, key, item):
        if item["is_high_degree"]:
            victim_key = self._find_oldest_non_high_degree_key_locked()

            if victim_key is None:
                self.store.popitem(last=False)
            else:
                self.store.pop(victim_key)

            self.store[key] = item
            return {"admitted": 1, "rejected": 0, "evicted": 1}

        victim_key = self._find_oldest_non_high_degree_key_locked()

        if victim_key is None:
            return {"admitted": 0, "rejected": 1, "evicted": 0}

        self.store.pop(victim_key)
        self.store[key] = item
        return {"admitted": 1, "rejected": 0, "evicted": 1}

    def _find_oldest_non_high_degree_key_locked(self):
        for key, item in self.store.items():
            if not item["is_high_degree"]:
                return key

        return None

    def size(self):
        with self.lock:
            return len(self.store)

    def high_degree_size(self):
        with self.lock:
            return sum(1 for item in self.store.values() if item["is_high_degree"])


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)
    index = int(round((p / 100) * (len(values) - 1)))
    return values[index]


def dedupe_edges(edges):
    seen = set()
    result = []

    for edge in edges:
        key = (edge.src_id, edge.relation, edge.dst_id)

        if key in seen:
            continue

        seen.add(key)
        result.append(edge)

    return result


def fetch_edges(session, graph_id, src_id, relation_whitelist, max_fanout, cache=None):
    cache_key = (graph_id, src_id)

    if cache is not None:
        cached_edges, hit = cache.get(cache_key)

        if hit:
            raw_edges = cached_edges

            return {
                "src_id": src_id,
                "edges": prepare_edges(raw_edges, relation_whitelist, max_fanout),
                "raw_edge_count": len(raw_edges),
                "raw_edges_from_cassandra": 0,
                "cassandra_query_count": 0,
                "cache_hit_count": 1,
                "cache_miss_count": 0,
                "cache_admit_count": 0,
                "cache_reject_count": 0,
                "cache_eviction_count": 0,
            }

    query = """
    SELECT src_id, relation, dst_id, confidence
    FROM kg_edges_by_src
    WHERE graph_id=%s AND src_id=%s
    """

    rows = session.execute(query, (graph_id, src_id))
    raw_edges = list(rows)

    admission = {"admitted": 0, "rejected": 0, "evicted": 0}

    if cache is not None:
        admission = cache.set(cache_key, raw_edges)

    return {
        "src_id": src_id,
        "edges": prepare_edges(raw_edges, relation_whitelist, max_fanout),
        "raw_edge_count": len(raw_edges),
        "raw_edges_from_cassandra": len(raw_edges),
        "cassandra_query_count": 1,
        "cache_hit_count": 0,
        "cache_miss_count": 1 if cache is not None else 0,
        "cache_admit_count": admission["admitted"],
        "cache_reject_count": admission["rejected"],
        "cache_eviction_count": admission["evicted"],
    }


def prepare_edges(raw_edges, relation_whitelist, max_fanout):
    if relation_whitelist is not None:
        filtered_edges = [
            edge for edge in raw_edges
            if edge.relation in relation_whitelist
        ]
    else:
        filtered_edges = raw_edges

    deduped_edges = dedupe_edges(filtered_edges)

    return deduped_edges[:max_fanout]


def expand_frontier_parallel(session, graph_id, frontier, relation_whitelist, max_fanout, workers, cache):
    query_items = []

    # Each node in the frontier is queried concurrently.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_state = {}

        for node_id, path, visited in frontier:
            future = executor.submit(
                fetch_edges,
                session,
                graph_id,
                node_id,
                relation_whitelist,
                max_fanout,
                cache,
            )
            future_to_state[future] = (node_id, path, visited)

        for future in as_completed(future_to_state):
            node_id, path, visited = future_to_state[future]
            query_result = future.result()

            query_items.append({
                "node_id": node_id,
                "path": path,
                "visited": visited,
                "query_result": query_result,
            })

    return query_items


def profile_once(session, args, cache):
    frontier = [
        (args.start, [], {args.start})
    ]

    all_paths = []
    level_profiles = []

    total_start = time.perf_counter()

    for level in range(1, args.depth + 1):
        level_start = time.perf_counter()

        query_items = expand_frontier_parallel(
            session=session,
            graph_id=args.graph_id,
            frontier=frontier,
            relation_whitelist=DEFAULT_RELATION_WHITELIST,
            max_fanout=args.fanout,
            workers=args.workers,
            cache=cache,
        )

        cassandra_query_count = 0
        cache_hit_count = 0
        cache_miss_count = 0
        cache_admit_count = 0
        cache_reject_count = 0
        cache_eviction_count = 0
        raw_edge_count = 0
        raw_edges_from_cassandra = 0
        expanded_edge_count = 0
        new_path_count = 0
        next_frontier = []

        for item in query_items:
            path = item["path"]
            visited = item["visited"]
            result = item["query_result"]

            edges = result["edges"]

            cassandra_query_count += result["cassandra_query_count"]
            cache_hit_count += result["cache_hit_count"]
            cache_miss_count += result["cache_miss_count"]
            cache_admit_count += result["cache_admit_count"]
            cache_reject_count += result["cache_reject_count"]
            cache_eviction_count += result["cache_eviction_count"]
            raw_edge_count += result["raw_edge_count"]
            raw_edges_from_cassandra += result["raw_edges_from_cassandra"]
            expanded_edge_count += len(edges)

            for edge in edges:
                dst = edge.dst_id

                if dst in visited:
                    continue

                new_path = path + [edge]
                new_visited = set(visited)
                new_visited.add(dst)

                all_paths.append(new_path)
                next_frontier.append((dst, new_path, new_visited))
                new_path_count += 1

        level_end = time.perf_counter()

        level_profiles.append({
            "level": level,
            "frontier_size": len(frontier),
            "cassandra_query_count": cassandra_query_count,
            "cache_hit_count": cache_hit_count,
            "cache_miss_count": cache_miss_count,
            "cache_admit_count": cache_admit_count,
            "cache_reject_count": cache_reject_count,
            "cache_eviction_count": cache_eviction_count,
            "raw_edge_count": raw_edge_count,
            "raw_edges_from_cassandra": raw_edges_from_cassandra,
            "expanded_edge_count": expanded_edge_count,
            "new_path_count": new_path_count,
            "next_frontier_size": len(next_frontier),
            "total_paths_so_far": len(all_paths),
            "level_wall_latency_ms": (level_end - level_start) * 1000,
        })

        frontier = next_frontier

        if not frontier:
            break

    total_end = time.perf_counter()

    return {
        "profiles": level_profiles,
        "total_paths": len(all_paths),
        "total_wall_latency_ms": (total_end - total_start) * 1000,
    }


def aggregate_runs(runs):
    levels = sorted({
        row["level"]
        for run in runs
        for row in run["profiles"]
    })

    profiles = []

    for level in levels:
        rows = [
            row for run in runs for row in run["profiles"]
            if row["level"] == level
        ]

        wall_latencies = [row["level_wall_latency_ms"] for row in rows]

        profiles.append({
            "level": level,
            "frontier_size": mean([row["frontier_size"] for row in rows]),
            "cassandra_query_count": mean([row["cassandra_query_count"] for row in rows]),
            "cache_hit_count": mean([row["cache_hit_count"] for row in rows]),
            "cache_miss_count": mean([row["cache_miss_count"] for row in rows]),
            "cache_admit_count": mean([row["cache_admit_count"] for row in rows]),
            "cache_reject_count": mean([row["cache_reject_count"] for row in rows]),
            "cache_eviction_count": mean([row["cache_eviction_count"] for row in rows]),
            "raw_edge_count": mean([row["raw_edge_count"] for row in rows]),
            "raw_edges_from_cassandra": mean([row["raw_edges_from_cassandra"] for row in rows]),
            "expanded_edge_count": mean([row["expanded_edge_count"] for row in rows]),
            "new_path_count": mean([row["new_path_count"] for row in rows]),
            "next_frontier_size": mean([row["next_frontier_size"] for row in rows]),
            "avg_level_wall_latency_ms": mean(wall_latencies),
            "p95_level_wall_latency_ms": percentile(wall_latencies, 95),
        })

    total_latencies = [run["total_wall_latency_ms"] for run in runs]

    total_hits = mean([
        sum(row["cache_hit_count"] for row in run["profiles"])
        for run in runs
    ])

    total_misses = mean([
        sum(row["cache_miss_count"] for row in run["profiles"])
        for run in runs
    ])

    total_access = total_hits + total_misses
    hit_rate = total_hits / total_access if total_access > 0 else 0.0

    return {
        "profiles": profiles,
        "avg_total_wall_latency_ms": mean(total_latencies),
        "p95_total_wall_latency_ms": percentile(total_latencies, 95),
        "avg_total_paths": mean([run["total_paths"] for run in runs]),
        "avg_total_cassandra_queries": mean([
            sum(row["cassandra_query_count"] for row in run["profiles"])
            for run in runs
        ]),
        "avg_total_raw_edges_from_cassandra": mean([
            sum(row["raw_edges_from_cassandra"] for row in run["profiles"])
            for run in runs
        ]),
        "cache_hit_rate": hit_rate,
    }


def run_benchmark(session, args):
    cache = None

    if args.cache_policy != "none" and args.cache_capacity > 0:
        cache = BoundedOneHopCache(
            capacity=args.cache_capacity,
            policy=args.cache_policy,
            degree_threshold=args.degree_threshold,
        )

    # Warmup allows us to evaluate repeated-query cache behavior.
    for _ in range(args.warmup):
        profile_once(session, args, cache)

    runs = []

    for _ in range(args.repeat):
        runs.append(profile_once(session, args, cache))

    result = aggregate_runs(runs)
    result["cache_size"] = cache.size() if cache is not None else 0
    result["high_degree_cache_size"] = cache.high_degree_size() if cache is not None else 0

    return result


def print_result(args, result):
    print("Parallel + Cache Benchmark")
    print("--------------------------")
    print(f"graph_id         : {args.graph_id}")
    print(f"start            : {args.start}")
    print(f"depth            : {args.depth}")
    print(f"fanout           : {args.fanout}")
    print(f"workers          : {args.workers}")
    print(f"cache_policy     : {args.cache_policy}")
    print(f"cache_capacity   : {args.cache_capacity}")
    print(f"degree_threshold : {args.degree_threshold}")
    print()

    print(
        f"{'level':<7}"
        f"{'frontier':<10}"
        f"{'cass':<8}"
        f"{'hit':<8}"
        f"{'miss':<8}"
        f"{'raw_db':<9}"
        f"{'expanded':<11}"
        f"{'wall_avg':<12}"
        f"{'wall_p95':<12}"
    )

    print("-" * 88)

    for row in result["profiles"]:
        print(
            f"{row['level']:<7}"
            f"{row['frontier_size']:<10.1f}"
            f"{row['cassandra_query_count']:<8.1f}"
            f"{row['cache_hit_count']:<8.1f}"
            f"{row['cache_miss_count']:<8.1f}"
            f"{row['raw_edges_from_cassandra']:<9.1f}"
            f"{row['expanded_edge_count']:<11.1f}"
            f"{row['avg_level_wall_latency_ms']:<12.3f}"
            f"{row['p95_level_wall_latency_ms']:<12.3f}"
        )

    print()
    print(f"Average total paths: {result['avg_total_paths']:.1f}")
    print(f"Average total wall latency: {result['avg_total_wall_latency_ms']:.3f} ms")
    print(f"P95 total wall latency: {result['p95_total_wall_latency_ms']:.3f} ms")
    print(f"Average Cassandra queries: {result['avg_total_cassandra_queries']:.1f}")
    print(f"Average raw edges from Cassandra: {result['avg_total_raw_edges_from_cassandra']:.1f}")
    print(f"Cache hit rate: {result['cache_hit_rate']:.3f}")
    print(f"Final cache size: {result['cache_size']}")
    print(f"High-degree entries in cache: {result['high_degree_cache_size']}")


def write_csv(args, result):
    if not args.output:
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "graph_id",
        "start_node",
        "depth",
        "fanout",
        "workers",
        "cache_policy",
        "cache_capacity",
        "degree_threshold",
        "avg_total_paths",
        "avg_total_wall_latency_ms",
        "p95_total_wall_latency_ms",
        "avg_total_cassandra_queries",
        "avg_total_raw_edges_from_cassandra",
        "cache_hit_rate",
        "cache_size",
        "high_degree_cache_size",
    ]

    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "graph_id": args.graph_id,
            "start_node": args.start,
            "depth": args.depth,
            "fanout": args.fanout,
            "workers": args.workers,
            "cache_policy": args.cache_policy,
            "cache_capacity": args.cache_capacity,
            "degree_threshold": args.degree_threshold,
            "avg_total_paths": round(result["avg_total_paths"], 3),
            "avg_total_wall_latency_ms": round(result["avg_total_wall_latency_ms"], 3),
            "p95_total_wall_latency_ms": round(result["p95_total_wall_latency_ms"], 3),
            "avg_total_cassandra_queries": round(result["avg_total_cassandra_queries"], 3),
            "avg_total_raw_edges_from_cassandra": round(result["avg_total_raw_edges_from_cassandra"], 3),
            "cache_hit_rate": round(result["cache_hit_rate"], 5),
            "cache_size": result["cache_size"],
            "high_degree_cache_size": result["high_degree_cache_size"],
        })


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark parallel frontier traversal with bounded one-hop cache."
    )

    parser.add_argument("--graph-id", default="synthetic_high_degree_21k")
    parser.add_argument("--start", default="user_big")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--fanout", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cache-policy", choices=["none", "lru", "high_degree"], default="none")
    parser.add_argument("--cache-capacity", type=int, default=20)
    parser.add_argument("--degree-threshold", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--keyspace", default="ai_memory")
    parser.add_argument(
        "--output",
        default="results/benchmark_parallel_cache_results.csv",
    )

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    result = run_benchmark(session, args)

    print_result(args, result)
    write_csv(args, result)

    cluster.shutdown()


if __name__ == "__main__":
    main()