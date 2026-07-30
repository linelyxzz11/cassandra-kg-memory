import argparse
import time
from statistics import mean

from cassandra.cluster import Cluster


RELATION_WHITELIST = {
    "mentions",
    "has_feature",
    "supports",
    "related_to",
    "suggests",
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


def get_out_edges(session, graph_id, src_id, relation_whitelist=None, max_fanout=None):
    query = """
    SELECT src_id, relation, dst_id, confidence
    FROM kg_edges_by_src
    WHERE graph_id=%s AND src_id=%s
    """

    rows = session.execute(query, (graph_id, src_id))
    edges = list(rows)

    if relation_whitelist is not None:
        edges = [
            edge for edge in edges
            if edge.relation in relation_whitelist
        ]

    edges = dedupe_edges(edges)

    if max_fanout is not None:
        edges = edges[:max_fanout]

    return edges


def query_full_out_edges(session, graph_id, src_id):
    return get_out_edges(
        session=session,
        graph_id=graph_id,
        src_id=src_id,
        relation_whitelist=None,
        max_fanout=None,
    )


def expand_k_hop(session, graph_id, start_id, max_depth, max_fanout):
    frontier = [
        (start_id, [], {start_id})
    ]

    results = []

    for _ in range(1, max_depth + 1):
        next_frontier = []

        for node_id, path, visited_in_path in frontier:
            edges = get_out_edges(
                session=session,
                graph_id=graph_id,
                src_id=node_id,
                relation_whitelist=RELATION_WHITELIST,
                max_fanout=max_fanout,
            )

            for edge in edges:
                dst = edge.dst_id

                if dst in visited_in_path:
                    continue

                new_path = path + [edge]
                new_visited = set(visited_in_path)
                new_visited.add(dst)

                results.append(new_path)
                next_frontier.append((dst, new_path, new_visited))

        frontier = next_frontier

    return results


def benchmark_one(name, func, repeat, warmup):
    for _ in range(warmup):
        func()

    times_ms = []
    last_result = None

    for _ in range(repeat):
        start = time.perf_counter()
        last_result = func()
        end = time.perf_counter()

        times_ms.append((end - start) * 1000)

    return {
        "name": name,
        "avg_ms": mean(times_ms),
        "p95_ms": percentile(times_ms, 95),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "count": len(last_result) if last_result is not None else 0,
    }


def print_result(result):
    print(
        f"{result['name']:<42} "
        f"avg={result['avg_ms']:.3f} ms | "
        f"p95={result['p95_ms']:.3f} ms | "
        f"min={result['min_ms']:.3f} ms | "
        f"max={result['max_ms']:.3f} ms | "
        f"count={result['count']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark high-degree node queries on Cassandra KG."
    )

    parser.add_argument(
        "--graph-id",
        default="synthetic_high_degree_21k",
        help="Graph id."
    )

    parser.add_argument(
        "--start",
        default="user_big",
        help="High-degree start node."
    )

    parser.add_argument(
        "--fanout",
        type=int,
        default=20,
        help="Max fanout for multi-hop query."
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Timed repetitions."
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup repetitions."
    )

    parser.add_argument(
        "--keyspace",
        default="ai_memory",
        help="Cassandra keyspace."
    )

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    print("High-Degree Node Benchmark")
    print("--------------------------")
    print(f"graph_id : {args.graph_id}")
    print(f"start    : {args.start}")
    print(f"fanout   : {args.fanout}")
    print(f"repeat   : {args.repeat}")
    print(f"warmup   : {args.warmup}")
    print()

    benchmarks = [
        (
            "Full 1-hop outgoing query",
            lambda: query_full_out_edges(
                session=session,
                graph_id=args.graph_id,
                src_id=args.start,
            ),
        ),
        (
            "Path query depth=1",
            lambda: expand_k_hop(
                session=session,
                graph_id=args.graph_id,
                start_id=args.start,
                max_depth=1,
                max_fanout=args.fanout,
            ),
        ),
        (
            "Path query depth=2",
            lambda: expand_k_hop(
                session=session,
                graph_id=args.graph_id,
                start_id=args.start,
                max_depth=2,
                max_fanout=args.fanout,
            ),
        ),
        (
            "Path query depth=3",
            lambda: expand_k_hop(
                session=session,
                graph_id=args.graph_id,
                start_id=args.start,
                max_depth=3,
                max_fanout=args.fanout,
            ),
        ),
        (
            "Path query depth=4",
            lambda: expand_k_hop(
                session=session,
                graph_id=args.graph_id,
                start_id=args.start,
                max_depth=4,
                max_fanout=args.fanout,
            ),
        ),
    ]

    for name, func in benchmarks:
        result = benchmark_one(
            name=name,
            func=func,
            repeat=args.repeat,
            warmup=args.warmup,
        )
        print_result(result)

    cluster.shutdown()


if __name__ == "__main__":
    main()
