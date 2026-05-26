import argparse
import time
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


def get_out_edges(session, graph_id, src_id, relation_whitelist=None, max_fanout=50):
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

    return edges[:max_fanout]


def expand_k_hop(session, graph_id, start_id, max_depth, relation_whitelist, max_fanout):
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
                relation_whitelist=relation_whitelist,
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


def query_forward(session, graph_id, start_id):
    query = """
    SELECT src_id, relation, dst_id
    FROM kg_edges_by_src
    WHERE graph_id=%s AND src_id=%s
    """

    rows = session.execute(query, (graph_id, start_id))
    return dedupe_edges(list(rows))


def query_reverse(session, graph_id, dst_id):
    query = """
    SELECT src_id, relation, dst_id
    FROM kg_edges_by_dst
    WHERE graph_id=%s AND dst_id=%s
    """

    rows = session.execute(query, (graph_id, dst_id))
    return dedupe_edges(list(rows))


def query_by_relation(session, graph_id, relation, bucket_count):
    query = """
    SELECT src_id, relation, dst_id
    FROM kg_edges_by_relation_bucket
    WHERE graph_id=%s AND relation=%s AND bucket=%s
    """

    all_edges = []

    for bucket in range(bucket_count):
        rows = session.execute(query, (graph_id, relation, bucket))
        all_edges.extend(list(rows))

    return dedupe_edges(all_edges)


def benchmark_one(name, func, repeat=20, warmup=3):
    for _ in range(warmup):
        func()

    times_ms = []
    last_result = None

    for _ in range(repeat):
        start = time.perf_counter()
        last_result = func()
        end = time.perf_counter()

        times_ms.append((end - start) * 1000)

    result_count = len(last_result) if last_result is not None else 0

    return {
        "name": name,
        "avg_ms": mean(times_ms),
        "p95_ms": percentile(times_ms, 95),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "result_count": result_count,
    }


def print_result(result):
    print(
        f"{result['name']:<45} "
        f"avg={result['avg_ms']:.3f} ms | "
        f"p95={result['p95_ms']:.3f} ms | "
        f"min={result['min_ms']:.3f} ms | "
        f"max={result['max_ms']:.3f} ms | "
        f"count={result['result_count']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark synthetic KG queries on Cassandra."
    )

    parser.add_argument(
        "--graph-id",
        default="synthetic_10k",
        help="Graph id to benchmark."
    )

    parser.add_argument(
        "--start",
        default="user_000001",
        help="Start node for forward and multi-hop queries."
    )

    parser.add_argument(
        "--reverse-dst",
        default="need_000001",
        help="Destination node for reverse query."
    )

    parser.add_argument(
        "--relation",
        default="suitable_for",
        help="Relation name for relation query."
    )

    parser.add_argument(
        "--bucket-count",
        type=int,
        default=64,
        help="Bucket count used during insertion."
    )

    parser.add_argument(
        "--fanout",
        type=int,
        default=20,
        help="Max fanout per node for multi-hop query."
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Number of timed repetitions."
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup repetitions."
    )

    parser.add_argument(
        "--keyspace",
        default="ai_memory",
        help="Cassandra keyspace."
    )

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    print("Synthetic KG Benchmark")
    print("----------------------")
    print(f"graph_id       : {args.graph_id}")
    print(f"start node     : {args.start}")
    print(f"reverse dst    : {args.reverse_dst}")
    print(f"relation       : {args.relation}")
    print(f"bucket_count   : {args.bucket_count}")
    print(f"fanout         : {args.fanout}")
    print(f"repeat         : {args.repeat}")
    print(f"warmup         : {args.warmup}")
    print()

    benchmarks = [
        (
            "Forward 1-hop query",
            lambda: query_forward(
                session=session,
                graph_id=args.graph_id,
                start_id=args.start,
            ),
        ),
        (
            "Reverse 1-hop query",
            lambda: query_reverse(
                session=session,
                graph_id=args.graph_id,
                dst_id=args.reverse_dst,
            ),
        ),
        (
            "Relation query across buckets",
            lambda: query_by_relation(
                session=session,
                graph_id=args.graph_id,
                relation=args.relation,
                bucket_count=args.bucket_count,
            ),
        ),
        (
            "Path query depth=2",
            lambda: expand_k_hop(
                session=session,
                graph_id=args.graph_id,
                start_id=args.start,
                max_depth=2,
                relation_whitelist=DEFAULT_RELATION_WHITELIST,
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
                relation_whitelist=DEFAULT_RELATION_WHITELIST,
                max_fanout=args.fanout,
            ),
        ),
        (
            "Path query depth=5",
            lambda: expand_k_hop(
                session=session,
                graph_id=args.graph_id,
                start_id=args.start,
                max_depth=5,
                relation_whitelist=DEFAULT_RELATION_WHITELIST,
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
