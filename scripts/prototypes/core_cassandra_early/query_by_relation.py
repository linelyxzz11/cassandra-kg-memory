import argparse

from cassandra.cluster import Cluster


def get_edges_by_relation(session, graph_id, relation, bucket_count):
    all_edges = []

    query = """
    SELECT src_id, relation, dst_id, confidence, source
    FROM kg_edges_by_relation_bucket
    WHERE graph_id=%s AND relation=%s AND bucket=%s
    """

    for bucket in range(bucket_count):
        rows = session.execute(query, (graph_id, relation, bucket))
        all_edges.extend(list(rows))

    return all_edges


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


def print_edges(title, edges, limit):
    print()
    print(title)
    print("-" * len(title))

    if not edges:
        print("No edges found.")
        return

    edges = dedupe_edges(edges)

    print(f"Total logical edges: {len(edges)}")
    print(f"Showing first {min(limit, len(edges))} edges")

    for edge in edges[:limit]:
        print(
            f"{edge.src_id} --{edge.relation}--> {edge.dst_id} "
            f"| confidence={edge.confidence} | source={edge.source}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Query KG edges by relation across multiple buckets."
    )

    parser.add_argument(
        "--graph-id",
        default="synthetic_10k",
        help="Graph id to query."
    )

    parser.add_argument(
        "--relation",
        default="suitable_for",
        help="Relation to query."
    )

    parser.add_argument(
        "--bucket-count",
        type=int,
        default=64,
        help="Bucket count used during insertion."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=30,
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

    edges = get_edges_by_relation(
        session=session,
        graph_id=args.graph_id,
        relation=args.relation,
        bucket_count=args.bucket_count,
    )

    print_edges(
        title=f"All {args.relation} edges in {args.graph_id}",
        edges=edges,
        limit=args.limit,
    )

    cluster.shutdown()


if __name__ == "__main__":
    main()