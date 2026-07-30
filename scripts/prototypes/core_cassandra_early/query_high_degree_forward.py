import argparse
import time

from cassandra.cluster import Cluster


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


def query_out_edges(session, graph_id, src_id):
    query = """
    SELECT src_id, relation, dst_id, confidence, source
    FROM kg_edges_by_src
    WHERE graph_id=%s AND src_id=%s
    """

    rows = session.execute(query, (graph_id, src_id))
    return dedupe_edges(list(rows))


def main():
    parser = argparse.ArgumentParser(
        description="Query outgoing edges for a high-degree node."
    )

    parser.add_argument(
        "--graph-id",
        default="synthetic_high_degree_21k",
        help="Graph id."
    )

    parser.add_argument(
        "--src",
        default="user_big",
        help="Source node id."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of edges to print."
    )

    parser.add_argument(
        "--keyspace",
        default="ai_memory",
        help="Cassandra keyspace."
    )

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    start = time.perf_counter()
    edges = query_out_edges(session, args.graph_id, args.src)
    end = time.perf_counter()

    print(f"Graph ID: {args.graph_id}")
    print(f"Source node: {args.src}")
    print(f"Total outgoing logical edges: {len(edges)}")
    print(f"Query time: {(end - start) * 1000:.3f} ms")
    print(f"Showing first {min(args.limit, len(edges))} edges")
    print("-" * 60)

    for edge in edges[:args.limit]:
        print(
            f"{edge.src_id} --{edge.relation}--> {edge.dst_id} "
            f"| confidence={edge.confidence} | source={edge.source}"
        )

    cluster.shutdown()


if __name__ == "__main__":
    main()