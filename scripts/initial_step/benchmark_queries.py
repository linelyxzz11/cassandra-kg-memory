import time
from statistics import mean

from cassandra.cluster import Cluster


KEYSPACE = "ai_memory"
GRAPH_ID = "memory_graph"
REPEAT = 20


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


def query_forward(session):
    query = """
    SELECT src_id, relation, dst_id
    FROM kg_edges_by_src
    WHERE graph_id=%s AND src_id=%s
    """

    rows = session.execute(query, (GRAPH_ID, "user_001"))
    return dedupe_edges(list(rows))


def query_reverse(session):
    query = """
    SELECT src_id, relation, dst_id
    FROM kg_edges_by_dst
    WHERE graph_id=%s AND dst_id=%s
    """

    rows = session.execute(query, (GRAPH_ID, "rest"))
    return dedupe_edges(list(rows))


def query_relation(session):
    query = """
    SELECT src_id, relation, dst_id
    FROM kg_edges_by_relation_bucket
    WHERE graph_id=%s AND relation=%s AND bucket=%s
    """

    all_edges = []

    for bucket in range(8):
        rows = session.execute(query, (GRAPH_ID, "suitable_for", bucket))
        all_edges.extend(list(rows))

    return dedupe_edges(all_edges)


def query_two_hop(session):
    first_hop = query_forward(session)
    results = []

    for edge in first_hop:
        if edge.relation != "likes":
            continue

        query = """
        SELECT src_id, relation, dst_id
        FROM kg_edges_by_src
        WHERE graph_id=%s AND src_id=%s
        """

        rows = session.execute(query, (GRAPH_ID, edge.dst_id))
        results.extend(list(rows))

    return dedupe_edges(results)


def benchmark(name, func):
    times_ms = []
    last_result = None

    for _ in range(REPEAT):
        start = time.perf_counter()
        last_result = func()
        end = time.perf_counter()

        times_ms.append((end - start) * 1000)

    print(
        f"{name}: avg={mean(times_ms):.3f} ms "
        f"| result_count={len(last_result) if last_result is not None else 0}"
    )


def main():
    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(KEYSPACE)

    print("Benchmark result")
    print("----------------")

    benchmark(
        "Forward query: user_001 outgoing edges",
        lambda: query_forward(session),
    )

    benchmark(
        "Reverse query: incoming edges to rest",
        lambda: query_reverse(session),
    )

    benchmark(
        "Relation query: all suitable_for edges",
        lambda: query_relation(session),
    )

    benchmark(
        "Two-hop query: user_001 likes -> suitable_for",
        lambda: query_two_hop(session),
    )

    cluster.shutdown()


if __name__ == "__main__":
    main()