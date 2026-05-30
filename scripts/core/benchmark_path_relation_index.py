import argparse
import csv
import time
from pathlib import Path
from statistics import mean

from cassandra.cluster import Cluster


BASELINE_QUERY = """
SELECT src_id, relation, dst_id, confidence
FROM kg_edges_by_src
WHERE graph_id=%s AND src_id=%s
"""


INDEX_QUERY = """
SELECT src_id, relation, dst_id, confidence
FROM kg_edges_by_src_relation
WHERE graph_id=%s AND src_id=%s AND relation=%s
"""


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


def get_edges_baseline(session, graph_id, src_id, relation, fanout):
    # Baseline reads all outgoing edges, then filters relation in Python.
    rows = list(session.execute(BASELINE_QUERY, (graph_id, src_id)))
    raw_count = len(rows)

    matched_edges = [
        row for row in rows
        if row.relation == relation
    ]

    matched_edges = dedupe_edges(matched_edges)
    expanded_edges = matched_edges[:fanout]

    return {
        "edges": expanded_edges,
        "raw_count": raw_count,
        "matched_count": len(matched_edges),
        "expanded_count": len(expanded_edges),
    }


def get_edges_index(session, graph_id, src_id, relation, fanout):
    # Index query directly reads edges under graph_id + src_id + relation.
    rows = list(session.execute(INDEX_QUERY, (graph_id, src_id, relation)))
    raw_count = len(rows)

    matched_edges = dedupe_edges(rows)
    expanded_edges = matched_edges[:fanout]

    return {
        "edges": expanded_edges,
        "raw_count": raw_count,
        "matched_count": len(matched_edges),
        "expanded_count": len(expanded_edges),
    }


def profile_once(session, graph_id, start_id, relation_sequence, fanout, method):
    frontier = [
        (start_id, [], {start_id})
    ]

    all_paths = []
    level_profiles = []

    total_start = time.perf_counter()

    for level, relation in enumerate(relation_sequence, start=1):
        level_start = time.perf_counter()

        current_frontier_size = len(frontier)
        query_count = 0
        raw_count = 0
        matched_count = 0
        expanded_count = 0
        new_path_count = 0
        next_frontier = []

        for node_id, path, visited in frontier:
            query_count += 1

            if method == "baseline":
                result = get_edges_baseline(
                    session=session,
                    graph_id=graph_id,
                    src_id=node_id,
                    relation=relation,
                    fanout=fanout,
                )
            elif method == "index":
                result = get_edges_index(
                    session=session,
                    graph_id=graph_id,
                    src_id=node_id,
                    relation=relation,
                    fanout=fanout,
                )
            else:
                raise ValueError("method must be baseline or index")

            raw_count += result["raw_count"]
            matched_count += result["matched_count"]
            expanded_count += result["expanded_count"]

            for edge in result["edges"]:
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
            "relation": relation,
            "frontier_size": current_frontier_size,
            "query_count": query_count,
            "raw_count": raw_count,
            "matched_count": matched_count,
            "expanded_count": expanded_count,
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

    profiles = []

    for level in levels:
        rows = [
            row
            for run in runs
            for row in run["profiles"]
            if row["level"] == level
        ]

        latencies = [row["level_latency_ms"] for row in rows]

        profiles.append({
            "level": level,
            "relation": rows[0]["relation"],
            "frontier_size": mean([row["frontier_size"] for row in rows]),
            "query_count": mean([row["query_count"] for row in rows]),
            "raw_count": mean([row["raw_count"] for row in rows]),
            "matched_count": mean([row["matched_count"] for row in rows]),
            "expanded_count": mean([row["expanded_count"] for row in rows]),
            "new_path_count": mean([row["new_path_count"] for row in rows]),
            "next_frontier_size": mean([row["next_frontier_size"] for row in rows]),
            "total_paths_so_far": mean([row["total_paths_so_far"] for row in rows]),
            "avg_level_latency_ms": mean(latencies),
            "p95_level_latency_ms": percentile(latencies, 95),
        })

    total_latencies = [run["total_latency_ms"] for run in runs]

    return {
        "profiles": profiles,
        "avg_total_paths": mean([run["total_paths"] for run in runs]),
        "avg_total_latency_ms": mean(total_latencies),
        "p95_total_latency_ms": percentile(total_latencies, 95),
        "avg_total_raw_count": mean([
            sum(row["raw_count"] for row in run["profiles"])
            for run in runs
        ]),
        "avg_total_matched_count": mean([
            sum(row["matched_count"] for row in run["profiles"])
            for run in runs
        ]),
    }


def run_benchmark(session, args, method):
    relation_sequence = [
        item.strip()
        for item in args.relations.split(",")
        if item.strip()
    ]

    for _ in range(args.warmup):
        profile_once(
            session=session,
            graph_id=args.graph_id,
            start_id=args.start,
            relation_sequence=relation_sequence,
            fanout=args.fanout,
            method=method,
        )

    runs = []

    for _ in range(args.repeat):
        runs.append(
            profile_once(
                session=session,
                graph_id=args.graph_id,
                start_id=args.start,
                relation_sequence=relation_sequence,
                fanout=args.fanout,
                method=method,
            )
        )

    result = aggregate_runs(runs)
    result["method"] = method

    return result


def print_result(args, result):
    print()
    print(f"Method: {result['method']}")
    print("-" * 32)

    print(
        f"{'level':<7}"
        f"{'relation':<14}"
        f"{'frontier':<10}"
        f"{'queries':<9}"
        f"{'raw':<9}"
        f"{'matched':<10}"
        f"{'expanded':<11}"
        f"{'avg_ms':<11}"
        f"{'p95_ms':<11}"
    )

    print("-" * 92)

    for row in result["profiles"]:
        print(
            f"{row['level']:<7}"
            f"{row['relation']:<14}"
            f"{row['frontier_size']:<10.1f}"
            f"{row['query_count']:<9.1f}"
            f"{row['raw_count']:<9.1f}"
            f"{row['matched_count']:<10.1f}"
            f"{row['expanded_count']:<11.1f}"
            f"{row['avg_level_latency_ms']:<11.3f}"
            f"{row['p95_level_latency_ms']:<11.3f}"
        )

    print()
    print(f"Average total paths: {result['avg_total_paths']:.1f}")
    print(f"Average total latency: {result['avg_total_latency_ms']:.3f} ms")
    print(f"P95 total latency: {result['p95_total_latency_ms']:.3f} ms")
    print(f"Average total raw count: {result['avg_total_raw_count']:.1f}")
    print(f"Average total matched count: {result['avg_total_matched_count']:.1f}")


def write_csv(args, result):
    if not args.output:
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "graph_id",
        "start_node",
        "relations",
        "fanout",
        "repeat",
        "warmup",
        "level",
        "relation",
        "frontier_size",
        "query_count",
        "raw_count",
        "matched_count",
        "expanded_count",
        "new_path_count",
        "next_frontier_size",
        "total_paths_so_far",
        "avg_level_latency_ms",
        "p95_level_latency_ms",
        "avg_total_paths",
        "avg_total_latency_ms",
        "p95_total_latency_ms",
        "avg_total_raw_count",
        "avg_total_matched_count",
    ]

    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in result["profiles"]:
            writer.writerow({
                "method": result["method"],
                "graph_id": args.graph_id,
                "start_node": args.start,
                "relations": args.relations,
                "fanout": args.fanout,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "level": row["level"],
                "relation": row["relation"],
                "frontier_size": round(row["frontier_size"], 3),
                "query_count": round(row["query_count"], 3),
                "raw_count": round(row["raw_count"], 3),
                "matched_count": round(row["matched_count"], 3),
                "expanded_count": round(row["expanded_count"], 3),
                "new_path_count": round(row["new_path_count"], 3),
                "next_frontier_size": round(row["next_frontier_size"], 3),
                "total_paths_so_far": round(row["total_paths_so_far"], 3),
                "avg_level_latency_ms": round(row["avg_level_latency_ms"], 3),
                "p95_level_latency_ms": round(row["p95_level_latency_ms"], 3),
                "avg_total_paths": round(result["avg_total_paths"], 3),
                "avg_total_latency_ms": round(result["avg_total_latency_ms"], 3),
                "p95_total_latency_ms": round(result["p95_total_latency_ms"], 3),
                "avg_total_raw_count": round(result["avg_total_raw_count"], 3),
                "avg_total_matched_count": round(result["avg_total_matched_count"], 3),
            })


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline traversal and relation-aware indexed traversal."
    )

    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument(
        "--relations",
        required=True,
        help="Comma-separated relation sequence, e.g. likes,suitable_for,related_to",
    )
    parser.add_argument("--fanout", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--keyspace", default="ai_memory")
    parser.add_argument(
        "--output",
        default="results/benchmark_path_relation_index_results.csv",
    )

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    print("Path Relation Index Benchmark")
    print("-----------------------------")
    print(f"graph_id  : {args.graph_id}")
    print(f"start     : {args.start}")
    print(f"relations : {args.relations}")
    print(f"fanout    : {args.fanout}")
    print(f"repeat    : {args.repeat}")
    print(f"warmup    : {args.warmup}")

    baseline = run_benchmark(session, args, method="baseline")
    index = run_benchmark(session, args, method="index")

    print_result(args, baseline)
    print_result(args, index)

    if index["avg_total_latency_ms"] > 0:
        speedup = baseline["avg_total_latency_ms"] / index["avg_total_latency_ms"]
        print()
        print(f"Index traversal speedup: {speedup:.2f}x")

    raw_reduction = baseline["avg_total_raw_count"] - index["avg_total_raw_count"]
    print(f"Total raw edge reduction: {raw_reduction:.1f}")

    write_csv(args, baseline)
    write_csv(args, index)

    cluster.shutdown()


if __name__ == "__main__":
    main()