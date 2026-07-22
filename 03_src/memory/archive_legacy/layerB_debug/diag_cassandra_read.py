import sys
from cassandra.cluster import Cluster

KEYSPACE = "ai_memory"
HOSTS = ["127.0.0.1"]
PORT = 9042

cluster = Cluster(HOSTS, port=PORT)
session = cluster.connect(KEYSPACE)

print("=== 1. Table schema ===")
try:
    rows = session.execute("SELECT * FROM kg_edges_by_src LIMIT 1")
    cols = rows.column_names
    print(f"  Columns: {cols}")
    for r in rows:
        print(f"  Row: {r}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== 2. Count rows ===")
try:
    rows = session.execute("SELECT COUNT(*) FROM kg_edges_by_src")
    for r in rows:
        print(f"  kg_edges_by_src: {r.count} rows")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== 3. Test insert + read (10 edges) ===")
insert_src = session.prepare(
    "INSERT INTO kg_edges_by_src (graph_id, src_id, relation, dst_id, edge_id, src_type, dst_type, confidence, source, created_at) "
    "VALUES (?, ?, ?, ?, now(), ?, ?, ?, ?, toTimestamp(now()))"
)
for i in range(10):
    session.execute(insert_src, ("test_diag", f"e{i}", "likes", f"d{i}", "ETYPE", "ETYPE", 0.9, "test_src"))

rows = session.execute("SELECT src_id, relation, dst_id, source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", ("test_diag", "e0"))
results = list(rows)
print(f"  Inserted 10, query e0 returned {len(results)} rows")
if results:
    print(f"  First row: {results[0]}")

print("\n=== 4. Cleanup ===")
session.execute("DELETE FROM kg_edges_by_src WHERE graph_id=%s", ("test_diag",))
print("  Test rows deleted.")

cluster.shutdown()
print("\nDone.")