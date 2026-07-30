import argparse
import csv
import time
from collections import OrderedDict
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

    def get(self, key):
        if self.capacity <= 0:
            return None, False

        if key not in self.store:
            return None, False

        item = self.store.pop(key)
        self.store[key] = item
        return item["edges"], True

    def set(self, key, edges):
        if self.capacity <= 0:
            return {
                "admitted": 0,
                "rejected": 1,
                "eviction": 0,
            }

        degree = len(edges)
        is_high_degree = degree >= self.degree_threshold

        item = {
            "edges": edges,
            "degree": degree,
            "is_high_degree": is_high_degree,
        }

        if key in self.store:
            self.store.pop(key)
            self.store[key] = item
            return {
                "admitted": 1,
                "rejected": 0,
                "eviction": 0,
            }

        if len(self.store) < self.capacity:
            self.store[key] = item
            return {
                "admitted": 1,
                "rejected": 0,
                "eviction": 0,
            }

        if self.policy == "lru":
            self.store.popitem(last=False)
            self.store[key] = item
            return {
                "admitted": 1,
                "rejected": 0,
                "eviction": 1,
            }

        if self.policy == "high_degree":
            return self._set_high_degree_priority(key, item)

        raise ValueError("Unsupported cache policy.")

    def _set_high_degree_priority(self, key, item):
        new_is_high_degree = item["is_high_degree"]

        if new_is_high_degree:
            victim_key = self._find_oldest_non_high_degree_key()

            if victim_key is None:
                self.store.popitem(last=False)
            else:
                self.store.pop(victim_key)

            self.store[key] = item
            return {
                "admitted": 1,
                "rejected": 0,
                "eviction": 1,
            }

        victim_key = self._find_oldest_non_high_degree_key()

        if victim_key is None:
            return {
                "admitted": 0,
                "rejected": 1,
                "eviction": 0,
            }

        self.store.pop(victim_key)
        self.store[key] = item
        return {
            "admitted": 1,
            "rejected": 0,
            "eviction": 1,
        }

    def _find_oldest_non_high_degree_key(self):
        for key, item in self.store.items():
            if not item["is_high_degree"]:
                return key

        return None

    def size(self):
        return len(self.store)

    def high_degree_size(self):
        return sum(
            1 for item in self.store.values()
            if item["is_high_degree"]
        )


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


def fetch_raw_edges(session, graph_id, src_id, cache=None):
    cache_key = (graph_id, src_id)

    if cache is not None:
        cached_edges, is_hit = cache.get(cache_key)

        if is_hit:
            return {
                "raw_edges": cached_edges,
                "cache_hit": 1,
                "cache_miss": 0,
                "cache_admit": 0,
                "cache_reject": 0,
                "cache_eviction": 0,
                "cassandra_query": 0,
                "raw_edges_from_cassandra": 0,
            }

    query = """
    SELECT src_id, relation, dst_id, confidence
    FROM kg_edges_by_src
    WHERE graph_id=%s AND src_id=%s
    """

    rows = session.execute(query, (graph_id, src_id))
    raw_edges = list(rows)

    admission = {
        "admitted": 0,
        "rejected": 0,
        "eviction": 0,
    }

    if cache is not None:
        admission = cache.set(cache_key, raw_edges)

    return {
        "raw_edges": raw_edges,
        "cache_hit": 0,
        "cache_miss": 1 if cache is not None else 0,
        "cache_admit": admission["admitted"],
        "cache_reject": admission["rejected"],
        "cache_eviction": admission["eviction"],
        "cassandra_query": 1,
        "raw_edges_from_cassandra": len(raw_edges),
    }


def get_out_edges_with_cache(session, graph_id, src_id, relation_whitelist, max_fanout, cache=None):
    fetch_result = fetch_raw_edges(
        session=session,
        graph_id=graph_id,
        src_id=src_id,
        cache=cache,
    )

    raw_edges = fetch_result["raw_edges"]

    if relation_whitelist is not None:
        filtered_edges = [
            edge for edge in raw_edges
            if edge.relation in relation_whitelist
        ]
    else:
        filtered_edges = raw_edges

    deduped_edges = dedupe_edges(filtered_edges)
    expanded_edges = deduped_edges[:max_fanout]

    return {
        "edges": expanded_edges,
        "raw_edge_count": len(raw_edges),
        "expanded_edge_count": len(expanded_edges),
        "cache_hit_count": fetch_result["cache_hit"],
        "cache_miss_count": fetch_result["cache_miss"],
        "cache_admit_count": fetch_result["cache_admit"],
        "cache_reject_count": fetch_result["cache_reject"],
        "cache_eviction_count": fetch_result["cache_eviction"],
        "cassandra_query_count": fetch_result["cassandra_query"],
        "raw_edges_from_cassandra": fetch_result["raw_edges_from_cassandra"],
    }


def profile_once(session, graph_id, start_id, max_depth, max_fanout, relation_whitelist, cache=None):
    frontier = [
        (start_id, [], {start_id})
    ]

    all_paths = []
    level_profiles = []

    total_start = time.perf_counter()

    for level in range(1, max_depth + 1):
        level_start = time.perf_counter()

        current_frontier_size = len(frontier)
        logical_query_count = 0
        cassandra_query_count = 0
        cache_hit_count = 0
        cache_miss_count = 0
        cache_admit_count = 0
        cache_reject_count = 0
        cache_eviction_count = 0
        raw_edges_from_cassandra = 0
        expanded_edge_count = 0
        new_path_count = 0
        next_frontier = []

        for node_id, path, visited_in_path in frontier:
            logical_query_count += 1

            query_result = get_out_edges_with_cache(
                session=session,
                graph_id=graph_id,
                src_id=node_id,
                relation_whitelist=relation_whitelist,
                max_fanout=max_fanout,
                cache=cache,
            )

            edges = query_result["edges"]

            cassandra_query_count += query_result["cassandra_query_count"]
            cache_hit_count += query_result["cache_hit_count"]
            cache_miss_count += query_result["cache_miss_count"]
            cache_admit_count += query_result["cache_admit_count"]
            cache_reject_count += query_result["cache_reject_count"]
            cache_eviction_count += query_result["cache_eviction_count"]
            raw_edges_from_cassandra += query_result["raw_edges_from_cassandra"]
            expanded_edge_count += query_result["expanded_edge_count"]

            for edge in edges:
                dst = edge.dst_id

                if dst in visited_in_path:
                    continue

                new_path = path + [edge]
                new_visited = set(visited_in_path)
                new_visited.add(dst)

                all_paths.append(new_path)
                next_frontier.append((dst, new_path, new_visited))
                new_path_count += 1

        level_end = time.perf_counter()

        level_profiles.append({
            "level": level,
            "frontier_size": current_frontier_size,
            "logical_query_count": logical_query_count,
            "cassandra_query_count": cassandra_query_count,
            "cache_hit_count": cache_hit_count,
            "cache_miss_count": cache_miss_count,
            "cache_admit_count": cache_admit_count,
            "cache_reject_count": cache_reject_count,
            "cache_eviction_count": cache_eviction_count,
            "raw_edges_from_cassandra": raw_edges_from_cassandra,
            "expanded_edge_count": expanded_edge_count,
            "new_path_count": new_path_count,
            "next_frontier_size": len(next_frontier),
            "total_paths_so_far": len(all_paths),
            "level_latency_ms": (level_end - level_start) * 1000,
        })

        frontier = next_frontier

        if not frontier:
            break

    total_end = time.perf_counter()

    return {
        "profiles": level_profiles,
        "total_paths": len(all_paths),
        "total_latency_ms": (total_end - total_start) * 1000,
    }


def aggregate_runs(runs):
    levels = sorted({
        row["level"]
        for run in runs
        for row in run["profiles"]
    })

    aggregated_profiles = []

    for level in levels:
        rows = [
            row
            for run in runs
            for row in run["profiles"]
            if row["level"] == level
        ]

        latencies = [row["level_latency_ms"] for row in rows]

        aggregated_profiles.append({
            "level": level,
            "frontier_size": mean([row["frontier_size"] for row in rows]),
            "logical_query_count": mean([row["logical_query_count"] for row in rows]),
            "cassandra_query_count": mean([row["cassandra_query_count"] for row in rows]),
            "cache_hit_count": mean([row["cache_hit_count"] for row in rows]),
            "cache_miss_count": mean([row["cache_miss_count"] for row in rows]),
            "cache_admit_count": mean([row["cache_admit_count"] for row in rows]),
            "cache_reject_count": mean([row["cache_reject_count"] for row in rows]),
            "cache_eviction_count": mean([row["cache_eviction_count"] for row in rows]),
            "raw_edges_from_cassandra": mean([row["raw_edges_from_cassandra"] for row in rows]),
            "expanded_edge_count": mean([row["expanded_edge_count"] for row in rows]),
            "new_path_count": mean([row["new_path_count"] for row in rows]),
            "next_frontier_size": mean([row["next_frontier_size"] for row in rows]),
            "total_paths_so_far": mean([row["total_paths_so_far"] for row in rows]),
            "avg_level_latency_ms": mean(latencies),
            "p95_level_latency_ms": percentile(latencies, 95),
        })

    total_latencies = [run["total_latency_ms"] for run in runs]
    total_paths = [run["total_paths"] for run in runs]

    avg_total_hits = mean([
        sum(row["cache_hit_count"] for row in run["profiles"])
        for run in runs
    ])

    avg_total_misses = mean([
        sum(row["cache_miss_count"] for row in run["profiles"])
        for run in runs
    ])

    avg_total_admits = mean([
        sum(row["cache_admit_count"] for row in run["profiles"])
        for run in runs
    ])

    avg_total_rejects = mean([
        sum(row["cache_reject_count"] for row in run["profiles"])
        for run in runs
    ])

    avg_total_evictions = mean([
        sum(row["cache_eviction_count"] for row in run["profiles"])
        for run in runs
    ])

    avg_total_cassandra_queries = mean([
        sum(row["cassandra_query_count"] for row in run["profiles"])
        for run in runs
    ])

    avg_total_raw_db = mean([
        sum(row["raw_edges_from_cassandra"] for row in run["profiles"])
        for run in runs
    ])

    total_cache_access = avg_total_hits + avg_total_misses
    hit_rate = avg_total_hits / total_cache_access if total_cache_access > 0 else 0.0

    return {
        "profiles": aggregated_profiles,
        "avg_total_paths": mean(total_paths),
        "avg_total_latency_ms": mean(total_latencies),
        "p95_total_latency_ms": percentile(total_latencies, 95),
        "avg_total_cache_hits": avg_total_hits,
        "avg_total_cache_misses": avg_total_misses,
        "avg_total_cache_admits": avg_total_admits,
        "avg_total_cache_rejects": avg_total_rejects,
        "avg_total_cache_evictions": avg_total_evictions,
        "avg_total_cassandra_queries": avg_total_cassandra_queries,
        "avg_total_raw_edges_from_cassandra": avg_total_raw_db,
        "cache_hit_rate": hit_rate,
    }


def run_method(session, args, method_name, cache_capacity, cache_policy):
    cache = None

    if cache_capacity > 0:
        cache = BoundedOneHopCache(
            capacity=cache_capacity,
            policy=cache_policy,
            degree_threshold=args.degree_threshold,
        )

    # Warmup fills the cache before timed runs.
    for _ in range(args.warmup):
        profile_once(
            session=session,
            graph_id=args.graph_id,
            start_id=args.start,
            max_depth=args.depth,
            max_fanout=args.fanout,
            relation_whitelist=DEFAULT_RELATION_WHITELIST,
            cache=cache,
        )

    timed_runs = []

    for _ in range(args.repeat):
        result = profile_once(
            session=session,
            graph_id=args.graph_id,
            start_id=args.start,
            max_depth=args.depth,
            max_fanout=args.fanout,
            relation_whitelist=DEFAULT_RELATION_WHITELIST,
            cache=cache,
        )

        timed_runs.append(result)

    aggregated = aggregate_runs(timed_runs)
    aggregated["method"] = method_name
    aggregated["cache_capacity"] = cache_capacity
    aggregated["cache_policy"] = cache_policy
    aggregated["degree_threshold"] = args.degree_threshold
    aggregated["cache_size"] = cache.size() if cache is not None else 0
    aggregated["high_degree_cache_size"] = cache.high_degree_size() if cache is not None else 0

    return aggregated


def print_result(result):
    print()
    print(f"Method: {result['method']}")
    print(f"Cache policy: {result['cache_policy']}")
    print(f"Cache capacity: {result['cache_capacity']}")
    print(f"Degree threshold: {result['degree_threshold']}")
    print("-" * 50)

    print(
        f"{'level':<7}"
        f"{'frontier':<10}"
        f"{'logical':<10}"
        f"{'cass':<8}"
        f"{'hit':<8}"
        f"{'miss':<8}"
        f"{'admit':<8}"
        f"{'reject':<8}"
        f"{'evict':<8}"
        f"{'raw_db':<9}"
        f"{'avg_ms':<11}"
        f"{'p95_ms':<11}"
    )

    print("-" * 118)

    for row in result["profiles"]:
        print(
            f"{row['level']:<7}"
            f"{row['frontier_size']:<10.1f}"
            f"{row['logical_query_count']:<10.1f}"
            f"{row['cassandra_query_count']:<8.1f}"
            f"{row['cache_hit_count']:<8.1f}"
            f"{row['cache_miss_count']:<8.1f}"
            f"{row['cache_admit_count']:<8.1f}"
            f"{row['cache_reject_count']:<8.1f}"
            f"{row['cache_eviction_count']:<8.1f}"
            f"{row['raw_edges_from_cassandra']:<9.1f}"
            f"{row['avg_level_latency_ms']:<11.3f}"
            f"{row['p95_level_latency_ms']:<11.3f}"
        )

    print()
    print(f"Average total paths: {result['avg_total_paths']:.1f}")
    print(f"Average total latency: {result['avg_total_latency_ms']:.3f} ms")
    print(f"P95 total latency: {result['p95_total_latency_ms']:.3f} ms")
    print(f"Average Cassandra queries: {result['avg_total_cassandra_queries']:.1f}")
    print(f"Average raw edges from Cassandra: {result['avg_total_raw_edges_from_cassandra']:.1f}")
    print(f"Cache hit rate: {result['cache_hit_rate']:.3f}")
    print(f"Average admits: {result['avg_total_cache_admits']:.1f}")
    print(f"Average rejects: {result['avg_total_cache_rejects']:.1f}")
    print(f"Average evictions: {result['avg_total_cache_evictions']:.1f}")
    print(f"Final cache size: {result['cache_size']}")
    print(f"High-degree entries in cache: {result['high_degree_cache_size']}")


def write_csv(args, result):
    if not args.output:
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "cache_policy",
        "graph_id",
        "start_node",
        "max_depth",
        "fanout",
        "repeat",
        "warmup",
        "cache_capacity",
        "degree_threshold",
        "cache_size",
        "high_degree_cache_size",
        "level",
        "frontier_size",
        "logical_query_count",
        "cassandra_query_count",
        "cache_hit_count",
        "cache_miss_count",
        "cache_admit_count",
        "cache_reject_count",
        "cache_eviction_count",
        "raw_edges_from_cassandra",
        "expanded_edge_count",
        "new_path_count",
        "next_frontier_size",
        "total_paths_so_far",
        "avg_level_latency_ms",
        "p95_level_latency_ms",
        "avg_total_paths",
        "avg_total_latency_ms",
        "p95_total_latency_ms",
        "avg_total_cassandra_queries",
        "avg_total_raw_edges_from_cassandra",
        "cache_hit_rate",
        "avg_total_cache_admits",
        "avg_total_cache_rejects",
        "avg_total_cache_evictions",
    ]

    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in result["profiles"]:
            writer.writerow({
                "method": result["method"],
                "cache_policy": result["cache_policy"],
                "graph_id": args.graph_id,
                "start_node": args.start,
                "max_depth": args.depth,
                "fanout": args.fanout,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "cache_capacity": result["cache_capacity"],
                "degree_threshold": result["degree_threshold"],
                "cache_size": result["cache_size"],
                "high_degree_cache_size": result["high_degree_cache_size"],
                "level": row["level"],
                "frontier_size": round(row["frontier_size"], 3),
                "logical_query_count": round(row["logical_query_count"], 3),
                "cassandra_query_count": round(row["cassandra_query_count"], 3),
                "cache_hit_count": round(row["cache_hit_count"], 3),
                "cache_miss_count": round(row["cache_miss_count"], 3),
                "cache_admit_count": round(row["cache_admit_count"], 3),
                "cache_reject_count": round(row["cache_reject_count"], 3),
                "cache_eviction_count": round(row["cache_eviction_count"], 3),
                "raw_edges_from_cassandra": round(row["raw_edges_from_cassandra"], 3),
                "expanded_edge_count": round(row["expanded_edge_count"], 3),
                "new_path_count": round(row["new_path_count"], 3),
                "next_frontier_size": round(row["next_frontier_size"], 3),
                "total_paths_so_far": round(row["total_paths_so_far"], 3),
                "avg_level_latency_ms": round(row["avg_level_latency_ms"], 3),
                "p95_level_latency_ms": round(row["p95_level_latency_ms"], 3),
                "avg_total_paths": round(result["avg_total_paths"], 3),
                "avg_total_latency_ms": round(result["avg_total_latency_ms"], 3),
                "p95_total_latency_ms": round(result["p95_total_latency_ms"], 3),
                "avg_total_cassandra_queries": round(result["avg_total_cassandra_queries"], 3),
                "avg_total_raw_edges_from_cassandra": round(result["avg_total_raw_edges_from_cassandra"], 3),
                "cache_hit_rate": round(result["cache_hit_rate"], 5),
                "avg_total_cache_admits": round(result["avg_total_cache_admits"], 3),
                "avg_total_cache_rejects": round(result["avg_total_cache_rejects"], 3),
                "avg_total_cache_evictions": round(result["avg_total_cache_evictions"], 3),
            })


def main():
    parser = argparse.ArgumentParser(
        description="Compare LRU and high-degree-priority one-hop cache."
    )

    parser.add_argument("--graph-id", default="synthetic_10k")
    parser.add_argument("--start", default="user_000001")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--fanout", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cache-capacity", type=int, default=20)
    parser.add_argument(
        "--cache-policy",
        choices=["lru", "high_degree"],
        default="lru",
    )
    parser.add_argument("--degree-threshold", type=int, default=100)
    parser.add_argument("--keyspace", default="ai_memory")
    parser.add_argument(
        "--output",
        default="results/benchmark_cache_policy_results.csv",
    )

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    print("Cache Policy Benchmark")
    print("----------------------")
    print(f"graph_id         : {args.graph_id}")
    print(f"start            : {args.start}")
    print(f"depth            : {args.depth}")
    print(f"fanout           : {args.fanout}")
    print(f"repeat           : {args.repeat}")
    print(f"warmup           : {args.warmup}")
    print(f"cache_capacity   : {args.cache_capacity}")
    print(f"cache_policy     : {args.cache_policy}")
    print(f"degree_threshold : {args.degree_threshold}")

    no_cache_result = run_method(
        session=session,
        args=args,
        method_name="no_cache",
        cache_capacity=0,
        cache_policy=args.cache_policy,
    )

    cache_result = run_method(
        session=session,
        args=args,
        method_name=args.cache_policy,
        cache_capacity=args.cache_capacity,
        cache_policy=args.cache_policy,
    )

    print_result(no_cache_result)
    print_result(cache_result)

    write_csv(args, no_cache_result)
    write_csv(args, cache_result)

    cluster.shutdown()


if __name__ == "__main__":
    main()