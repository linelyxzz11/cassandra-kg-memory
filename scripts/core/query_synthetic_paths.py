import argparse

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

    seen = set()
    deduped_edges = []

    for edge in edges:
        key = (edge.src_id, edge.relation, edge.dst_id)

        if key in seen:
            continue

        seen.add(key)
        deduped_edges.append(edge)

    return deduped_edges[:max_fanout]


def expand_k_hop(
    session,
    graph_id,
    start_id,
    max_depth=5,
    relation_whitelist=None,
    max_fanout=50,
):
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


def print_paths(paths, limit=30):
    if not paths:
        print("No paths found.")
        return

    print(f"Total paths found: {len(paths)}")
    print(f"Showing first {min(limit, len(paths))} paths")
    print("-" * 60)

    for index, path in enumerate(paths[:limit], start=1):
        parts = [path[0].src_id]
        score = 1.0

        for edge in path:
            parts.append(f"--{edge.relation}-->")
            parts.append(edge.dst_id)

            if edge.confidence is not None:
                score *= edge.confidence

        print(f"[Path {index}] " + " ".join(parts) + f" | score={score:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Query multi-hop paths from synthetic KG in Cassandra."
    )

    parser.add_argument(
        "--graph-id",
        default="synthetic_1k",
        help="Graph id to query."
    )

    parser.add_argument(
        "--start",
        default="user_000001",
        help="Start node id."
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Max hop depth."
    )

    parser.add_argument(
        "--fanout",
        type=int,
        default=20,
        help="Max fanout per node."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Number of paths to print."
    )

    parser.add_argument(
        "--keyspace",
        default="ai_memory",
        help="Cassandra keyspace."
    )

    parser.add_argument(
        "--longest-first",
        action="store_true",
        help="Print longer paths first."
    )

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    paths = expand_k_hop(
        session=session,
        graph_id=args.graph_id,
        start_id=args.start,
        max_depth=args.depth,
        relation_whitelist=DEFAULT_RELATION_WHITELIST,
        max_fanout=args.fanout,
    )

    if args.longest_first:
        paths = sorted(paths, key=lambda item: len(item), reverse=True)

    print(f"Graph ID: {args.graph_id}")
    print(f"Start node: {args.start}")
    print(f"Max depth: {args.depth}")
    print(f"Max fanout: {args.fanout}")
    print()

    print_paths(paths, limit=args.limit)

    cluster.shutdown()


if __name__ == "__main__":
    main()
