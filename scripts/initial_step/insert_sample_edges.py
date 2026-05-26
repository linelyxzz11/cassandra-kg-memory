
import uuid
from datetime import datetime, timezone

from cassandra.cluster import Cluster


KEYSPACE = "ai_memory"
GRAPH_ID = "memory_graph"


SAMPLE_EDGES = [
    ("user_001", "user", "likes", "quiet_hotel", "preference", 0.92, "manual_seed"),
    ("quiet_hotel", "preference", "suitable_for", "rest", "need", 0.85, "manual_seed"),
    ("rest", "need", "related_to", "good_sleep", "state", 0.80, "manual_seed"),
    ("user_001", "user", "likes", "slow_music", "preference", 0.88, "manual_seed"),
    ("slow_music", "preference", "suitable_for", "relaxation", "need", 0.86, "manual_seed"),
    ("slow_music", "preference", "suitable_for", "rest", "need", 0.82, "manual_seed"),
    ("user_001", "user", "studies", "knowledge_graph", "topic", 0.95, "manual_seed"),
    ("knowledge_graph", "topic", "related_to", "graph_database", "topic", 0.90, "manual_seed"),
    ("graph_database", "topic", "includes", "cassandra", "technology", 0.87, "manual_seed"),
    ("cassandra", "technology", "used_for", "kg_storage", "task", 0.89, "manual_seed"),
]


def insert_edge(session, edge):
    src_id, src_type, relation, dst_id, dst_type, confidence, source = edge

    query = """
    INSERT INTO kg_edges_by_src (
        graph_id, src_id, relation, dst_id, edge_id,
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
            src_id,
            relation,
            dst_id,
            uuid.uuid1(),
            src_type,
            dst_type,
            confidence,
            source,
            datetime.now(timezone.utc),
        ),
    )


def main():
    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(KEYSPACE)

    for edge in SAMPLE_EDGES:
        insert_edge(session, edge)

    print(f"Inserted {len(SAMPLE_EDGES)} sample edges into kg_edges_by_src.")

    cluster.shutdown()


if __name__ == "__main__":
    main()