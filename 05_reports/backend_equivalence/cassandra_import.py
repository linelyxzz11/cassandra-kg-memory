#!/usr/bin/env python3
"""Cassandra Import — fixed paths, fixed fields, no len() bug."""
import csv
from pathlib import Path

from cassandra.cluster import Cluster

cluster = Cluster(["127.0.0.1"], port=9042)
session = cluster.connect()

session.execute("""
    CREATE KEYSPACE IF NOT EXISTS kg_memory
    WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
""")
session.set_keyspace("kg_memory")

session.execute("""
    CREATE TABLE IF NOT EXISTS kg_memory_table (
        conversation_id text, memory_id text, raw_text text,
        entities text, relations text, keywords text, timestamp text, embedding_id text,
        PRIMARY KEY (conversation_id, memory_id)
    )
""")
session.execute("""
    CREATE TABLE IF NOT EXISTS kg_triples (
        scope_id text, src_memory_id text, src_entity text, relation text, tgt_memory_id text,
        PRIMARY KEY (scope_id, src_memory_id, tgt_memory_id)
    )
""")
print("Tables created OK")

# Load memories
BASE = Path("D:/memorytable/cassandra-kg-memory")
memories = {}
with open(BASE / "01_data" / "locomo_memory_records.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        mid = r["memory_id"].strip()
        memories[mid] = {
            "cid": mid.split("_session_")[0] if "_session_" in mid else mid,
            "text": r.get("text", "").strip(),
            "ts": r.get("timestamp", "").strip(),
        }
print(f"Loaded {len(memories)} memories")

# Load features
features = {}
feat_path = BASE / "02_artifacts" / "p3_memory_features.csv"
if feat_path.exists():
    with open(feat_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            mid = r["memory_id"].strip()
            features[mid] = {
                "entities": r.get("entities", "").strip(),
                "relations": r.get("relations", "").strip(),
                "keywords": r.get("keywords", "").strip(),
                "triples": r.get("triples", "").strip(),
            }
    print(f"Loaded {len(features)} feature rows")

# Insert memories
insert = session.prepare(
    "INSERT INTO kg_memory_table (conversation_id,memory_id,raw_text,entities,relations,keywords,timestamp,embedding_id) "
    "VALUES (?,?,?,?,?,?,?,?)"
)
total = 0
for mid, m in memories.items():
    f = features.get(mid, {})
    session.execute(insert, (
        m["cid"], mid, m["text"],
        f.get("entities", ""), f.get("relations", ""), f.get("keywords", ""),
        m["ts"], f"embed_{mid}"
    ))
    total += 1
    if total % 500 == 0:
        print(f"  ... {total}/{len(memories)}")
print(f"kg_memory_table: {total} rows")

# Insert triples
tins = session.prepare(
    "INSERT INTO kg_triples (scope_id,src_memory_id,src_entity,relation,tgt_memory_id) VALUES (?,?,?,?,?)"
)
tc = 0
for mid, f in features.items():
    triples_str = f.get("triples", "")
    if not triples_str:
        continue
    cid = mid.split("_session_")[0] if "_session_" in mid else mid
    # triples format: "entity,relation,target|entity,relation,target"
    for t in triples_str.split("|"):
        parts = t.split(",")
        if len(parts) >= 3:
            session.execute(tins, (cid, mid, parts[0].strip(), parts[1].strip(), parts[2].strip()))
            tc += 1

print(f"kg_triples: {tc} rows")

cluster.shutdown()
print("Cassandra import complete.")
