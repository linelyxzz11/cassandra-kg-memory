import argparse
import time

from cassandra.cluster import Cluster
from cassandra.query import BatchStatement, BatchType, SimpleStatement


SELECT_SRC_EDGES = """
SELECT graph_id, src_id, relation, dst_id, edge_id,
       src_type, dst_type, confidence, source, created_at
FROM kg_edges_by_src
"""

INSERT_SRC_RELATION_INDEX = """
INSERT INTO kg_edges_by_src_relation (
    graph_id,
    src_id,
    relation,
    dst_id,
    edge_id,
    src_type,
    dst_type,
    confidence,
    source,
    created_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def sync_graph(session, graph_id, batch_size, fetch_size):
    select_stmt = SimpleStatement(
        SELECT_SRC_EDGES,
        fetch_size=fetch_size,
    )

    insert_stmt = session.prepare(INSERT_SRC_RELATION_INDEX)

    scanned_rows = 0
    matched_rows = 0
    inserted_rows = 0
    batch = BatchStatement(batch_type=BatchType.UNLOGGED)

    start_time = time.perf_counter()

    for row in session.execute(select_stmt):
        scanned_rows += 1

        # This offline sync scans the source table and keeps only the target graph.
        if row.graph_id != graph_id:
            continue

        matched_rows += 1

        batch.add(
            insert_stmt,
            (
                row.graph_id,
                row.src_id,
                row.relation,
                row.dst_id,
                row.edge_id,
                row.src_type,
                row.dst_type,
                row.confidence,
                row.source,
                row.created_at,
            ),
        )

        if len(batch) >= batch_size:
            session.execute(batch)
            inserted_rows += len(batch)
            batch.clear()

            if inserted_rows % 1000 == 0:
                print(f"Inserted {inserted_rows} index rows...")

    if len(batch) > 0:
        session.execute(batch)
        inserted_rows += len(batch)

    end_time = time.perf_counter()

    print()
    print("Source relation index sync completed.")
    print("-------------------------------------")
    print(f"Graph id       : {graph_id}")
    print(f"Scanned rows   : {scanned_rows}")
    print(f"Matched rows   : {matched_rows}")
    print(f"Inserted rows  : {inserted_rows}")
    print(f"Elapsed time   : {(end_time - start_time):.3f} s")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill kg_edges_by_src_relation from kg_edges_by_src."
    )

    parser.add_argument(
        "--graph-id",
        required=True,
        help="Graph id to sync into kg_edges_by_src_relation.",
    )

    parser.add_argument(
        "--keyspace",
        default="ai_memory",
        help="Cassandra keyspace.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of rows per batch write.",
    )

    parser.add_argument(
        "--fetch-size",
        type=int,
        default=1000,
        help="Cassandra fetch size for scanning source rows.",
    )

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    sync_graph(
        session=session,
        graph_id=args.graph_id,
        batch_size=args.batch_size,
        fetch_size=args.fetch_size,
    )

    cluster.shutdown()


if __name__ == "__main__":
    main()