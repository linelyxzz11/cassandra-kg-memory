import argparse
import csv
from pathlib import Path

from cassandra.cluster import Cluster
from cassandra.query import BatchStatement, BatchType


INSERT_SRC = """
INSERT INTO kg_edges_by_src (
    graph_id, src_id, relation, dst_id, edge_id,
    src_type, dst_type, confidence, source, created_at
)
VALUES (?, ?, ?, ?, now(), ?, ?, ?, ?, toTimestamp(now()))
"""

INSERT_DST = """
INSERT INTO kg_edges_by_dst (
    graph_id, dst_id, relation, src_id, edge_id,
    src_type, dst_type, confidence, source, created_at
)
VALUES (?, ?, ?, ?, now(), ?, ?, ?, ?, toTimestamp(now()))
"""

INSERT_RELATION_BUCKET = """
INSERT INTO kg_edges_by_relation_bucket (
    graph_id, relation, bucket, src_id, dst_id, edge_id,
    src_type, dst_type, confidence, source, created_at
)
VALUES (?, ?, ?, ?, ?, now(), ?, ?, ?, ?, toTimestamp(now()))
"""

INSERT_SRC_RELATION = """
INSERT INTO kg_edges_by_src_relation (
    graph_id, src_id, relation, dst_id, edge_id,
    src_type, dst_type, confidence, source, created_at
)
VALUES (?, ?, ?, ?, now(), ?, ?, ?, ?, toTimestamp(now()))
"""


def stable_bucket(value, bucket_count):
    return abs(hash(value)) % bucket_count


def load_edges(file_path):
    with Path(file_path).open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            yield {
                "src_id": row["src_id"],
                "src_type": row["src_type"],
                "relation": row["relation"],
                "dst_id": row["dst_id"],
                "dst_type": row["dst_type"],
                "confidence": float(row["confidence"]),
                "source": row["source"],
            }


def insert_edges(session, args):
    insert_src = session.prepare(INSERT_SRC)
    insert_dst = session.prepare(INSERT_DST)
    insert_relation_bucket = session.prepare(INSERT_RELATION_BUCKET)
    insert_src_relation = session.prepare(INSERT_SRC_RELATION)

    logical_edges = 0
    physical_writes = 0
    batch = BatchStatement(batch_type=BatchType.UNLOGGED)

    for edge in load_edges(args.file):
        bucket = stable_bucket(edge["dst_id"], args.bucket_count)

        # Forward adjacency table.
        batch.add(
            insert_src,
            (
                args.graph_id,
                edge["src_id"],
                edge["relation"],
                edge["dst_id"],
                edge["src_type"],
                edge["dst_type"],
                edge["confidence"],
                edge["source"],
            ),
        )

        # Reverse adjacency table.
        batch.add(
            insert_dst,
            (
                args.graph_id,
                edge["dst_id"],
                edge["relation"],
                edge["src_id"],
                edge["src_type"],
                edge["dst_type"],
                edge["confidence"],
                edge["source"],
            ),
        )

        # Relation-level bucket table.
        batch.add(
            insert_relation_bucket,
            (
                args.graph_id,
                edge["relation"],
                bucket,
                edge["src_id"],
                edge["dst_id"],
                edge["src_type"],
                edge["dst_type"],
                edge["confidence"],
                edge["source"],
            ),
        )

        # Relation-aware source index.
        batch.add(
            insert_src_relation,
            (
                args.graph_id,
                edge["src_id"],
                edge["relation"],
                edge["dst_id"],
                edge["src_type"],
                edge["dst_type"],
                edge["confidence"],
                edge["source"],
            ),
        )

        logical_edges += 1
        physical_writes += 4

        if logical_edges % args.batch_size == 0:
            session.execute(batch)
            batch.clear()

            if logical_edges % 1000 == 0:
                print(f"Inserted {logical_edges} logical edges...")

    if len(batch) > 0:
        session.execute(batch)

    print()
    print("Bulk insert v3 completed.")
    print("-------------------------")
    print(f"Input file          : {args.file}")
    print(f"Graph id            : {args.graph_id}")
    print(f"Inserted edges      : {logical_edges}")
    print(f"Physical writes     : {physical_writes}")
    print(f"Write amplification : 4x")
    print(f"Bucket count        : {args.bucket_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk insert KG edges into four Cassandra access tables."
    )

    parser.add_argument("--file", required=True)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--bucket-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--keyspace", default="ai_memory")

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    insert_edges(session, args)

    cluster.shutdown()


if __name__ == "__main__":
    main()