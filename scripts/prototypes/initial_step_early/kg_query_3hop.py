
from cassandra.cluster import Cluster


KEYSPACE = "ai_memory"
GRAPH_ID = "memory_graph"

RELATION_WHITELIST = {
    "likes",
    "suitable_for",
    "related_to",
    "studies",
    "includes",
    "used_for",
}

MAX_DEPTH = 4
MAX_FANOUT = 20


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


def get_out_edges(session, src_id):
    query = """
    SELECT src_id, relation, dst_id, confidence
    FROM kg_edges_by_src
    WHERE graph_id=%s AND src_id=%s
    """

    rows = session.execute(query, (GRAPH_ID, src_id))
    edges = list(rows)

    edges = [
        edge for edge in edges
        if edge.relation in RELATION_WHITELIST
    ]

    edges = dedupe_edges(edges)

    return edges[:MAX_FANOUT]


def expand_paths(session, start_id, max_depth):
    frontier = [
        (start_id, [], {start_id})
    ]

    results = []

    for _ in range(1, max_depth + 1):
        next_frontier = []

        for node_id, path, visited in frontier:
            edges = get_out_edges(session, node_id)

            for edge in edges:
                dst = edge.dst_id

                if dst in visited:
                    continue

                new_path = path + [edge]
                new_visited = set(visited)
                new_visited.add(dst)

                results.append(new_path)
                next_frontier.append((dst, new_path, new_visited))

        frontier = next_frontier

    return results


def print_paths(paths):
    for index, path in enumerate(paths, start=1):
        parts = [path[0].src_id]
        score = 1.0

        for edge in path:
            parts.append(f"--{edge.relation}-->")
            parts.append(edge.dst_id)

            if edge.confidence is not None:
                score *= edge.confidence

        print(f"[Path {index}] " + " ".join(parts) + f" | score={score:.4f}")


def main():
    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(KEYSPACE)

    paths = expand_paths(
        session=session,
        start_id="user_001",
        max_depth=MAX_DEPTH,
    )

    print_paths(paths)

    cluster.shutdown()


if __name__ == "__main__":
    main()

