from cassandra.cluster import Cluster


KEYSPACE = "ai_memory"
GRAPH_ID = "memory_graph"


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


def query_in_edges(session, dst_id, relation=None):
    if relation is None:
        query = """
        SELECT src_id, relation, dst_id, confidence, source
        FROM kg_edges_by_dst
        WHERE graph_id=%s AND dst_id=%s
        """
        rows = session.execute(query, (GRAPH_ID, dst_id))
    else:
        query = """
        SELECT src_id, relation, dst_id, confidence, source
        FROM kg_edges_by_dst
        WHERE graph_id=%s AND dst_id=%s AND relation=%s
        """
        rows = session.execute(query, (GRAPH_ID, dst_id, relation))

    return dedupe_edges(list(rows))


def print_edges(title, edges):
    print()
    print(title)
    print("-" * len(title))

    if not edges:
        print("No edges found.")
        return

    for edge in edges:
        print(
            f"{edge.src_id} --{edge.relation}--> {edge.dst_id} "
            f"| confidence={edge.confidence} | source={edge.source}"
        )


def main():
    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(KEYSPACE)

    print_edges(
        "Who is suitable_for rest?",
        query_in_edges(session, "rest", "suitable_for"),
    )

    print_edges(
        "Who is related_to good_sleep?",
        query_in_edges(session, "good_sleep", "related_to"),
    )

    print_edges(
        "Who includes cassandra?",
        query_in_edges(session, "cassandra", "includes"),
    )

    cluster.shutdown()


if __name__ == "__main__":
    main()
