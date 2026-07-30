import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    values = sorted(values)
    if not values:
        return 0.0
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


def query_baseline(session, graph_id, src_id, relation, fanout):
    rows = list(session.execute(BASELINE_QUERY, (graph_id, src_id)))
    matched = [row for row in rows if row.relation == relation]
    matched = dedupe_edges(matched)

    return {
        "src_id": src_id,
        "edges": matched[:fanout],
        "raw_count": len(rows),
        "matched_count": len(matched),
    }


def query_index(session, graph_id, src_id, relation, fanout):
    rows = list(session.execute(INDEX_QUERY, (graph_id, src_id, relation)))
    matched = dedupe_edges(rows)

    return {
        "src_id": src_id,
        "edges": matched[:fanout],
        "raw_count": len(rows),
        "matched_count": len(matched),
    }


def expand_frontier_parallel(session, graph_id, frontier, relation, fanout, workers, method):
    results = []

    query_fn = query_baseline if method == "baseline" else query_index

    # Query all nodes in the same frontier layer concurrently.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_state = {}

        for node_id, path, visited in frontier:
            future = executor.submit(
                query_fn,
                session,
                graph_id,
                node_id,
                relation,
                fanout,
            )
            future_to_state[future] = (node_id, path, visited)

        for future in as_completed(future_to_state):
            node_id, path, visited = future_to_state[future]
            query_result = future.result()

            results.append({
                "node_id": node_id,
                "path": path,
                "visited": visited,
                "query_result": query_result,
            })

    return results


def profile_once(session, graph_id, start_id, relations, fanout, workers, method):
    frontier = [(start_id, [], {start_id})]
    all_paths = []
    level_profiles = []

    total_start = time.perf_counter()

    for level, relation in enumerate(relations, start=1):
        level_start = time.perf_counter()

        query_items = expand_frontier_parallel(
            session=session,
            graph_id=graph_id,
            frontier=frontier,
            relation=relation,
            fanout=fanout,
            workers=workers,
            method=method,
        )

        raw_count = 0
        matched_count = 0
        expanded_count = 0
        new_path_count = 0
        next_frontier = []

        for item in query_items:
            path = item["path"]
            visited = item["visited"]
            result = item["query_result"]

            raw_count += result["raw_count"]
            matched_count += result["matched_count"]
            expanded_count += len(result["edges"])

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
            "frontier_size": len(frontier),
            "query_count": len(query_items),
            "raw_count": raw_count,
            "matched_count": matched_count,
            "expanded_count": expanded_count,
            "new_path_count": new_path_count,
            "next_frontier_size": len(next_frontier),
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


def run_method(session, args, method):
    relations = [
        item.strip()
        for item in args.relations.split(",")
        if item.strip()
    ]

    for _ in range(args.warmup):
        profile_once(
            session=session,
            graph_id=args.graph_id,
            start_id=args.start,
            relations=relations,
            fanout=args.fanout,
            workers=args.workers,
            method=method,
        )

    runs = []

    for _ in range(args.repeat):
        runs.append(
            profile_once(
                session=session,
                graph_id=args.graph_id,
                start_id=args.start,
                relations=relations,
                fanout=args.fanout,
                workers=args.workers,
                method=method,
            )
        )

    result = aggregate_runs(runs)
    result["method"] = method

    return result


def print_result(result):
    print()
    print(f"Method: {result['method']}")
    print("-" * 32)

    print(
        f"{'level':<7}"
        f"{'relation':<14}"
        f"{'frontier':<10}"
        f"{'queries':<9}"
        f"{'raw':<10}"
        f"{'matched':<10}"
        f"{'expanded':<11}"
        f"{'avg_ms':<11}"
        f"{'p95_ms':<11}"
    )

    print("-" * 96)

    for row in result["profiles"]:
        print(
            f"{row['level']:<7}"
            f"{row['relation']:<14}"
            f"{row['frontier_size']:<10.1f}"
            f"{row['query_count']:<9.1f}"
            f"{row['raw_count']:<10.1f}"
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


def main():
    parser = argparse.ArgumentParser(
        description="Compare parallel baseline traversal and parallel relation-index traversal."
    )

    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--relations", required=True)
    parser.add_argument("--fanout", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--keyspace", default="ai_memory")

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    print("Parallel Path Relation Index Benchmark")
    print("--------------------------------------")
    print(f"graph_id  : {args.graph_id}")
    print(f"start     : {args.start}")
    print(f"relations : {args.relations}")
    print(f"fanout    : {args.fanout}")
    print(f"workers   : {args.workers}")
    print(f"repeat    : {args.repeat}")
    print(f"warmup    : {args.warmup}")

    baseline = run_method(session, args, method="baseline")
    index = run_method(session, args, method="index")

    print_result(baseline)
    print_result(index)

    if index["avg_total_latency_ms"] > 0:
        speedup = baseline["avg_total_latency_ms"] / index["avg_total_latency_ms"]
        print()
        print(f"Parallel index traversal speedup: {speedup:.2f}x")

    raw_reduction = baseline["avg_total_raw_count"] - index["avg_total_raw_count"]
    print(f"Total raw edge reduction: {raw_reduction:.1f}")

    cluster.shutdown()


if __name__ == "__main__":
    main()