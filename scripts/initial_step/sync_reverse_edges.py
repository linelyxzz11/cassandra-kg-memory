import uuid
from datetime import datetime, timezone

from cassandra.cluster import Cluster


KEYSPACE = "ai_memory"
GRAPH_ID = "memory_graph"


def fetch_src_edges(session):
    query = """
    SELECT src_id, relation, dst_id, src_type, dst_type, confidence
    FROM kg_edges_by_src
    WHERE graph_id=%s
    ALLOW FILTERING
    """

    return list(session.execute(query, (GRAPH_ID,)))


def sync_edge_to_dst(session, edge):
    query = """
    INSERT INTO kg_edges_by_dst (
        graph_id, dst_id, relation, src_id, edge_id,
        src_type, dst_type, confidence, source, created_at
    )
    VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s
    )
    """

    session.execute(
        query,
        (
            GRAPH_ID,
            edge.dst_id,
            edge.relation,
            edge.src_id,
            uuid.uuid1(),
            edge.src_type,
            edge.dst_type,
            edge.confidence,
            "python_reverse_sync",
            datetime.now(timezone.utc),
        ),
    )


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


def main():
    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(KEYSPACE)

    edges = dedupe_edges(fetch_src_edges(session))

    for edge in edges:
        sync_edge_to_dst(session, edge)

    print(f"Synced {len(edges)} logical edges into kg_edges_by_dst.")

    cluster.shutdown()


if __name__ == "__main__":
    main()