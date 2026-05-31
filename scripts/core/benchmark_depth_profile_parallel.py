import argparse
import csv
import time
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


def get_out_edges_with_counts(session, graph_id, src_id, relation_whitelist, max_fanout):
    query = """
    SELECT src_id, relation, dst_id, confidence
    FROM kg_edges_by_src
    WHERE graph_id=%s AND src_id=%s
    """

    start = time.perf_counter()
    rows = session.execute(query, (graph_id, src_id))
    raw_edges = list(rows)
    end = time.perf_counter()

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
        "src_id": src_id,
        "edges": expanded_edges,
        "raw_edge_count": len(raw_edges),
        "expanded_edge_count": len(expanded_edges),
        "query_latency_ms": (end - start) * 1000,
    }


def expand_frontier_serial(session, graph_id, frontier, relation_whitelist, max_fanout):
    results = []

    # Serial mode: query frontier nodes one by one.
    for node_id, path, visited_in_path in frontier:
        query_result = get_out_edges_with_counts(
            session=session,
            graph_id=graph_id,
            src_id=node_id,
            relation_whitelist=relation_whitelist,
            max_fanout=max_fanout,
        )

        results.append({
            "node_id": node_id,
            "path": path,
            "visited": visited_in_path,
            "query_result": query_result,
        })

    return results


def expand_frontier_parallel(session, graph_id, frontier, relation_whitelist, max_fanout, workers):
    results = []

    # Parallel mode: query all nodes in the same frontier layer concurrently.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {}

        for node_id, path, visited_in_path in frontier:
            future = executor.submit(
                get_out_edges_with_counts,
                session,
                graph_id,
                node_id,
                relation_whitelist,
                max_fanout,
            )
            future_to_item[future] = (node_id, path, visited_in_path)

        for future in as_completed(future_to_item):
            node_id, path, visited_in_path = future_to_item[future]
            query_result = future.result()

            results.append({
                "node_id": node_id,
                "path": path,
                "visited": visited_in_path,
                "query_result": query_result,
            })

    return results


def profile_once(session, graph_id, start_id, max_depth, max_fanout, workers, mode):
    frontier = [
        (start_id, [], {start_id})
    ]

    all_paths = []
    level_profiles = []

    total_start = time.perf_counter()

    for level in range(1, max_depth + 1):
        level_start = time.perf_counter()

        current_frontier_size = len(frontier)

        if mode == "serial":
            query_items = expand_frontier_serial(
                session=session,
                graph_id=graph_id,
                frontier=frontier,
                relation_whitelist=DEFAULT_RELATION_WHITELIST,
                max_fanout=max_fanout,
            )
        elif mode == "parallel":
            query_items = expand_frontier_parallel(
                session=session,
                graph_id=graph_id,
                frontier=frontier,
                relation_whitelist=DEFAULT_RELATION_WHITELIST,
                max_fanout=max_fanout,
                workers=workers,
            )
        else:
            raise ValueError("mode must be serial or parallel")

        cassandra_query_count = len(query_items)
        raw_edge_count = 0
        expanded_edge_count = 0
        query_latency_sum_ms = 0.0
        new_path_count = 0
        next_frontier = []

        for item in query_items:
            path = item["path"]
            visited_in_path = item["visited"]
            query_result = item["query_result"]

            raw_edge_count += query_result["raw_edge_count"]
            expanded_edge_count += query_result["expanded_edge_count"]
            query_latency_sum_ms += query_result["query_latency_ms"]

            for edge in query_result["edges"]:
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
            "cassandra_query_count": cassandra_query_count,
            "raw_edge_count": raw_edge_count,
            "expanded_edge_count": expanded_edge_count,
            "new_path_count": new_path_count,
            "next_frontier_size": len(next_frontier),
            "total_paths_so_far": len(all_paths),
            "query_latency_sum_ms": query_latency_sum_ms,
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
            row
            for run in runs
            for row in run["profiles"]
            if row["level"] == level
        ]

        wall_latencies = [row["level_wall_latency_ms"] for row in rows]

        profiles.append({
            "level": level,
            "frontier_size": mean([row["frontier_size"] for row in rows]),
            "cassandra_query_count": mean([row["cassandra_query_count"] for row in rows]),
            "raw_edge_count": mean([row["raw_edge_count"] for row in rows]),
            "expanded_edge_count": mean([row["expanded_edge_count"] for row in rows]),
            "new_path_count": mean([row["new_path_count"] for row in rows]),
            "next_frontier_size": mean([row["next_frontier_size"] for row in rows]),
            "total_paths_so_far": mean([row["total_paths_so_far"] for row in rows]),
            "avg_query_latency_sum_ms": mean([row["query_latency_sum_ms"] for row in rows]),
            "avg_level_wall_latency_ms": mean(wall_latencies),
            "p95_level_wall_latency_ms": percentile(wall_latencies, 95),
        })

    total_latencies = [run["total_wall_latency_ms"] for run in runs]
    total_paths = [run["total_paths"] for run in runs]

    return {
        "profiles": profiles,
        "avg_total_paths": mean(total_paths),
        "avg_total_wall_latency_ms": mean(total_latencies),
        "p95_total_wall_latency_ms": percentile(total_latencies, 95),
    }


def run_benchmark(session, args, mode):
    for _ in range(args.warmup):
        profile_once(
            session=session,
            graph_id=args.graph_id,
            start_id=args.start,
            max_depth=args.depth,
            max_fanout=args.fanout,
            workers=args.workers,
            mode=mode,
        )

    runs = []

    for _ in range(args.repeat):
        result = profile_once(
            session=session,
            graph_id=args.graph_id,
            start_id=args.start,
            max_depth=args.depth,
            max_fanout=args.fanout,
            workers=args.workers,
            mode=mode,
        )
        runs.append(result)

    aggregated = aggregate_runs(runs)
    aggregated["mode"] = mode

    return aggregated


def print_result(args, result):
    print()
    print(f"Mode: {result['mode']}")
    print("-" * 30)

    print(
        f"{'level':<7}"
        f"{'frontier':<10}"
        f"{'queries':<9}"
        f"{'raw':<9}"
        f"{'expanded':<11}"
        f"{'new_paths':<11}"
        f"{'next_front':<12}"
        f"{'wall_avg':<12}"
        f"{'wall_p95':<12}"
    )

    print("-" * 93)

    for row in result["profiles"]:
        print(
            f"{row['level']:<7}"
            f"{row['frontier_size']:<10.1f}"
            f"{row['cassandra_query_count']:<9.1f}"
            f"{row['raw_edge_count']:<9.1f}"
            f"{row['expanded_edge_count']:<11.1f}"
            f"{row['new_path_count']:<11.1f}"
            f"{row['next_frontier_size']:<12.1f}"
            f"{row['avg_level_wall_latency_ms']:<12.3f}"
            f"{row['p95_level_wall_latency_ms']:<12.3f}"
        )

    print()
    print(f"Average total paths: {result['avg_total_paths']:.1f}")
    print(f"Average total wall latency: {result['avg_total_wall_latency_ms']:.3f} ms")
    print(f"P95 total wall latency: {result['p95_total_wall_latency_ms']:.3f} ms")


def write_csv(args, result):
    if not args.output:
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "mode",
        "graph_id",
        "start_node",
        "depth",
        "fanout",
        "workers",
        "repeat",
        "warmup",
        "level",
        "frontier_size",
        "cassandra_query_count",
        "raw_edge_count",
        "expanded_edge_count",
        "new_path_count",
        "next_frontier_size",
        "total_paths_so_far",
        "avg_query_latency_sum_ms",
        "avg_level_wall_latency_ms",
        "p95_level_wall_latency_ms",
        "avg_total_paths",
        "avg_total_wall_latency_ms",
        "p95_total_wall_latency_ms",
    ]

    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in result["profiles"]:
            writer.writerow({
                "mode": result["mode"],
                "graph_id": args.graph_id,
                "start_node": args.start,
                "depth": args.depth,
                "fanout": args.fanout,
                "workers": args.workers,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "level": row["level"],
                "frontier_size": round(row["frontier_size"], 3),
                "cassandra_query_count": round(row["cassandra_query_count"], 3),
                "raw_edge_count": round(row["raw_edge_count"], 3),
                "expanded_edge_count": round(row["expanded_edge_count"], 3),
                "new_path_count": round(row["new_path_count"], 3),
                "next_frontier_size": round(row["next_frontier_size"], 3),
                "total_paths_so_far": round(row["total_paths_so_far"], 3),
                "avg_query_latency_sum_ms": round(row["avg_query_latency_sum_ms"], 3),
                "avg_level_wall_latency_ms": round(row["avg_level_wall_latency_ms"], 3),
                "p95_level_wall_latency_ms": round(row["p95_level_wall_latency_ms"], 3),
                "avg_total_paths": round(result["avg_total_paths"], 3),
                "avg_total_wall_latency_ms": round(result["avg_total_wall_latency_ms"], 3),
                "p95_total_wall_latency_ms": round(result["p95_total_wall_latency_ms"], 3),
            })


def main():
    parser = argparse.ArgumentParser(
        description="Compare serial and parallel frontier expansion."
    )

    parser.add_argument("--graph-id", default="synthetic_high_degree_21k")
    parser.add_argument("--start", default="user_big")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--fanout", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--keyspace", default="ai_memory")
    parser.add_argument(
        "--output",
        default="results/benchmark_parallel_frontier_results.csv",
    )

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    print("Parallel Frontier Benchmark")
    print("---------------------------")
    print(f"graph_id : {args.graph_id}")
    print(f"start    : {args.start}")
    print(f"depth    : {args.depth}")
    print(f"fanout   : {args.fanout}")
    print(f"workers  : {args.workers}")
    print(f"repeat   : {args.repeat}")
    print(f"warmup   : {args.warmup}")

    serial_result = run_benchmark(session, args, mode="serial")
    parallel_result = run_benchmark(session, args, mode="parallel")

    print_result(args, serial_result)
    print_result(args, parallel_result)

    if parallel_result["avg_total_wall_latency_ms"] > 0:
        speedup = (
            serial_result["avg_total_wall_latency_ms"]
            / parallel_result["avg_total_wall_latency_ms"]
        )
        print()
        print(f"Total speedup: {speedup:.2f}x")

    write_csv(args, serial_result)
    write_csv(args, parallel_result)

    cluster.shutdown()


if __name__ == "__main__":
    main()