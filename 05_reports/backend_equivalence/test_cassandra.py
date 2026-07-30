#!/usr/bin/env python3
"""
Test Cassandra connection and verify imported data.
"""
from cassandra.cluster import Cluster

cluster = Cluster(["127.0.0.1"], port=9042)
session = cluster.connect()

# Test basic
rows = list(session.execute("SELECT now() FROM system.local"))
print(f"Cassandra connection OK | system.now() = {rows[0][0]}")

# Switch to kg_memory
try:
    session.set_keyspace("kg_memory")
    
    cnt = session.execute("SELECT COUNT(*) FROM kg_memory_table").one()[0]
    print(f"kg_memory_table rows: {cnt}")
    
    trip_cnt = session.execute("SELECT COUNT(*) FROM kg_triples").one()[0]
    print(f"kg_triples rows: {trip_cnt}")
    
    # Sample rows
    sample = session.execute("SELECT * FROM kg_memory_table LIMIT 2")
    for r in sample:
        print(f"  Sample: memory_id={r.memory_id}, raw_text[:60]={str(r.raw_text)[:60]}")
    
    print("Cassandra data OK" if cnt > 0 else "Cassandra: WARNING - no data found")
except Exception as e:
    print(f"Cassandra ERROR: {e}")

cluster.shutdown()
