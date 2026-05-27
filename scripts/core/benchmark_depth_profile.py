import argparse
import csv
import time
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


def get_out_edges_with_counts(session, graph_id, src_id, relation_whitelist=None, max_fanout=50):
    query = """
    SELECT src_id, relation, dst_id, confidence
    FROM kg_edges_by_src
    WHERE graph_id=%s AND src_id=%s
    """

    rows = session.execute(query, (graph_id, src_id))
    raw_edges = list(rows)
    raw_edge_count = len(raw_edges)

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
        "raw_edge_count": raw_edge_count,
        "filtered_edge_count": len(filtered_edges),
        "deduped_edge_count": len(deduped_edges),
        "expanded_edge_count": len(expanded_edges),
    }


def profile_once(session, graph_id, start_id, max_depth, max_fanout, relation_whitelist):
    frontier = [
        (start_id, [], {start_id})
    ]

    all_paths = []
    level_profiles = []

    total_start = time.perf_counter()

    for level in range(1, max_depth + 1):
        level_start = time.perf_counter()

        current_frontier_size = len(frontier)
        cassandra_query_count = 0
        raw_edge_count = 0
        filtered_edge_count = 0
        deduped_edge_count = 0
        expanded_edge_count = 0
        new_path_count = 0
        next_frontier = []

        for node_id, path, visited_in_path in frontier:
            cassandra_query_count += 1

            query_result = get_out_edges_with_counts(
                session=session,
                graph_id=graph_id,
                src_id=node_id,
                relation_whitelist=relation_whitelist,
                max_fanout=max_fanout,
            )

            edges = query_result["edges"]

            raw_edge_count += query_result["raw_edge_count"]
            filtered_edge_count += query_result["filtered_edge_count"]
            deduped_edge_count += query_result["deduped_edge_count"]
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
            "cassandra_query_count": cassandra_query_count,
            "raw_edge_count": raw_edge_count,
            "filtered_edge_count": filtered_edge_count,
            "deduped_edge_count": deduped_edge_count,
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
            "cassandra_query_count": mean([row["cassandra_query_count"] for row in rows]),
            "raw_edge_count": mean([row["raw_edge_count"] for row in rows]),
            "filtered_edge_count": mean([row["filtered_edge_count"] for row in rows]),
            "deduped_edge_count": mean([row["deduped_edge_count"] for row in rows]),
            "expanded_edge_count": mean([row["expanded_edge_count"] for row in rows]),
            "new_path_count": mean([row["new_path_count"] for row in rows]),
            "next_frontier_size": mean([row["next_frontier_size"] for row in rows]),
            "total_paths_so_far": mean([row["total_paths_so_far"] for row in rows]),
            "avg_level_latency_ms": mean(latencies),
            "p95_level_latency_ms": percentile(latencies, 95),
            "min_level_latency_ms": min(latencies),
            "max_level_latency_ms": max(latencies),
        })

    total_latencies = [run["total_latency_ms"] for run in runs]
    total_paths = [run["total_paths"] for run in runs]

    return {
        "profiles": aggregated_profiles,
        "avg_total_paths": mean(total_paths),
        "avg_total_latency_ms": mean(total_latencies),
        "p95_total_latency_ms": percentile(total_latencies, 95),
        "min_total_latency_ms": min(total_latencies),
        "max_total_latency_ms": max(total_latencies),
    }


def run_profile(session, args):
    for _ in range(args.warmup):
        profile_once(
            session=session,
            graph_id=args.graph_id,
            start_id=args.start,
            max_depth=args.depth,
            max_fanout=args.fanout,
            relation_whitelist=DEFAULT_RELATION_WHITELIST,
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
        )

        timed_runs.append(result)

    return aggregate_runs(timed_runs)


def print_profile(args, result):
    print("Depth Profile Benchmark")
    print("-----------------------")
    print(f"graph_id      : {args.graph_id}")
    print(f"start node    : {args.start}")
    print(f"max_depth     : {args.depth}")
    print(f"fanout        : {args.fanout}")
    print(f"repeat        : {args.repeat}")
    print(f"warmup        : {args.warmup}")
    print(f"keyspace      : {args.keyspace}")
    print()

    print(
        f"{'level':<7}"
        f"{'frontier':<10}"
        f"{'queries':<9}"
        f"{'raw':<9}"
        f"{'expanded':<11}"
        f"{'new_paths':<11}"
        f"{'next_front':<12}"
        f"{'total_paths':<13}"
        f"{'avg_ms':<11}"
        f"{'p95_ms':<11}"
    )

    print("-" * 104)

    for row in result["profiles"]:
        print(
            f"{row['level']:<7}"
            f"{row['frontier_size']:<10.1f}"
            f"{row['cassandra_query_count']:<9.1f}"
            f"{row['raw_edge_count']:<9.1f}"
            f"{row['expanded_edge_count']:<11.1f}"
            f"{row['new_path_count']:<11.1f}"
            f"{row['next_frontier_size']:<12.1f}"
            f"{row['total_paths_so_far']:<13.1f}"
            f"{row['avg_level_latency_ms']:<11.3f}"
            f"{row['p95_level_latency_ms']:<11.3f}"
        )

    print()
    print(f"Average total paths: {result['avg_total_paths']:.1f}")
    print(f"Average total latency: {result['avg_total_latency_ms']:.3f} ms")
    print(f"P95 total latency: {result['p95_total_latency_ms']:.3f} ms")


def write_csv(args, result):
    if not args.output:
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "graph_id",
        "start_node",
        "max_depth",
        "fanout",
        "repeat",
        "warmup",
        "level",
        "frontier_size",
        "cassandra_query_count",
        "raw_edge_count",
        "filtered_edge_count",
        "deduped_edge_count",
        "expanded_edge_count",
        "new_path_count",
        "next_frontier_size",
        "total_paths_so_far",
        "avg_level_latency_ms",
        "p95_level_latency_ms",
        "min_level_latency_ms",
        "max_level_latency_ms",
        "avg_total_paths",
        "avg_total_latency_ms",
        "p95_total_latency_ms",
        "min_total_latency_ms",
        "max_total_latency_ms",
    ]

    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in result["profiles"]:
            writer.writerow({
                "graph_id": args.graph_id,
                "start_node": args.start,
                "max_depth": args.depth,
                "fanout": args.fanout,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "level": row["level"],
                "frontier_size": round(row["frontier_size"], 3),
                "cassandra_query_count": round(row["cassandra_query_count"], 3),
                "raw_edge_count": round(row["raw_edge_count"], 3),
                "filtered_edge_count": round(row["filtered_edge_count"], 3),
                "deduped_edge_count": round(row["deduped_edge_count"], 3),
                "expanded_edge_count": round(row["expanded_edge_count"], 3),
                "new_path_count": round(row["new_path_count"], 3),
                "next_frontier_size": round(row["next_frontier_size"], 3),
                "total_paths_so_far": round(row["total_paths_so_far"], 3),
                "avg_level_latency_ms": round(row["avg_level_latency_ms"], 3),
                "p95_level_latency_ms": round(row["p95_level_latency_ms"], 3),
                "min_level_latency_ms": round(row["min_level_latency_ms"], 3),
                "max_level_latency_ms": round(row["max_level_latency_ms"], 3),
                "avg_total_paths": round(result["avg_total_paths"], 3),
                "avg_total_latency_ms": round(result["avg_total_latency_ms"], 3),
                "p95_total_latency_ms": round(result["p95_total_latency_ms"], 3),
                "min_total_latency_ms": round(result["min_total_latency_ms"], 3),
                "max_total_latency_ms": round(result["max_total_latency_ms"], 3),
            })


def main():
    parser = argparse.ArgumentParser(
        description="Profile multi-hop traversal by depth level."
    )

    parser.add_argument("--graph-id", default="synthetic_10k")
    parser.add_argument("--start", default="user_000001")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--fanout", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--keyspace", default="ai_memory")
    parser.add_argument(
        "--output",
        default="results/benchmark_depth_profile_results.csv",
    )

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    result = run_profile(session, args)
    print_profile(args, result)
    write_csv(args, result)

    cluster.shutdown()


if __name__ == "__main__":
    main()