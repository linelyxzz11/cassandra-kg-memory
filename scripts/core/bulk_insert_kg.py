import argparse
import csv
import uuid
import zlib
from datetime import datetime, timezone

from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement


def stable_bucket(src_id, bucket_count):
    if bucket_count <= 1:
        return 0

    return zlib.crc32(src_id.encode("utf-8")) % bucket_count


def read_edges(csv_file):
    with open(csv_file, "r", encoding="utf-8") as file:
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


def insert_edge(session, graph_id, edge, bucket_count):
    edge_id = uuid.uuid1()
    created_at = datetime.now(timezone.utc)
    bucket = stable_bucket(edge["src_id"], bucket_count)

    insert_src = SimpleStatement("""
        INSERT INTO kg_edges_by_src (
            graph_id, src_id, relation, dst_id, edge_id,
            src_type, dst_type, confidence, source, created_at
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
    """)

    insert_dst = SimpleStatement("""
        INSERT INTO kg_edges_by_dst (
            graph_id, dst_id, relation, src_id, edge_id,
            src_type, dst_type, confidence, source, created_at
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
    """)

    insert_relation = SimpleStatement("""
        INSERT INTO kg_edges_by_relation_bucket (
            graph_id, relation, bucket,
            src_id, dst_id, edge_id,
            src_type, dst_type, confidence, source, created_at
        )
        VALUES (
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s, %s
        )
    """)

    session.execute(
        insert_src,
        (
            graph_id,
            edge["src_id"],
            edge["relation"],
            edge["dst_id"],
            edge_id,
            edge["src_type"],
            edge["dst_type"],
            edge["confidence"],
            edge["source"],
            created_at,
        )
    )

    session.execute(
        insert_dst,
        (
            graph_id,
            edge["dst_id"],
            edge["relation"],
            edge["src_id"],
            edge_id,
            edge["src_type"],
            edge["dst_type"],
            edge["confidence"],
            edge["source"],
            created_at,
        )
    )

    session.execute(
        insert_relation,
        (
            graph_id,
            edge["relation"],
            bucket,
            edge["src_id"],
            edge["dst_id"],
            edge_id,
            edge["src_type"],
            edge["dst_type"],
            edge["confidence"],
            edge["source"],
            created_at,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Bulk insert KG CSV into Cassandra three-table schema."
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Input CSV file."
    )

    parser.add_argument(
        "--graph-id",
        default="synthetic_1k",
        help="Graph id written into Cassandra."
    )

    parser.add_argument(
        "--keyspace",
        default="ai_memory",
        help="Cassandra keyspace."
    )

    parser.add_argument(
        "--bucket-count",
        type=int,
        default=32,
        help="Bucket count for relation table."
    )

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    count = 0

    for edge in read_edges(args.file):
        insert_edge(
            session=session,
            graph_id=args.graph_id,
            edge=edge,
            bucket_count=args.bucket_count,
        )

        count += 1

        if count % 1000 == 0:
            print(f"Inserted {count} edges...")

    print("Bulk insert completed.")
    print(f"Input file: {args.file}")
    print(f"Graph id: {args.graph_id}")
    print(f"Inserted logical edges: {count}")
    print(f"Physical writes: {count * 3}")
    print(f"Bucket count: {args.bucket_count}")

    cluster.shutdown()


if __name__ == "__main__":
    main()