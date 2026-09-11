#!/usr/bin/env python3
"""Neo4j Import — LocoMoMemory / LocoMoEntity labels (don't touch old benchmark data)."""
import csv, os, re
from pathlib import Path
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "")))
BASE = Path("D:/memorytable/cassandra-kg-memory")
TRIPLE_RE = re.compile(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)")

# Step 1 — Constraints on new labels only
with driver.session() as s:
    s.run("CREATE CONSTRAINT locomo_mem_id IF NOT EXISTS FOR (m:LocoMoMemory) REQUIRE m.memory_id IS UNIQUE")
    s.run("CREATE CONSTRAINT locomo_ent_name IF NOT EXISTS FOR (e:LocoMoEntity) REQUIRE e.name IS UNIQUE")
    s.run("CREATE INDEX locomo_scope IF NOT EXISTS FOR (m:LocoMoMemory) ON (m.scope_id)")
print("Constraints OK")

# Step 2 — Load
memories = list(csv.DictReader(open(BASE / "01_data" / "locomo_memory_records.csv", encoding="utf-8-sig")))
features = {}
feat_path = BASE / "02_artifacts" / "p3_memory_features.csv"
if feat_path.exists():
    with open(feat_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            features[r["memory_id"].strip()] = r
print(f"Memories: {len(memories)}, Features: {len(features)}")

# Step 3 — Memory nodes
for i, m in enumerate(memories):
    mid = m["memory_id"].strip()
    f = features.get(mid, {})
    cid = mid.split("_session_")[0] if "_session_" in mid else mid
    with driver.session() as s:
        s.run("""
            MERGE (mem:LocoMoMemory {memory_id: $mid})
            SET mem.scope_id = $cid, mem.raw_text = $text,
                mem.entities = $ent, mem.relations = $rel, mem.keywords = $kw,
                mem.speaker = $speaker, mem.timestamp = $ts
        """, mid=mid, cid=cid, text=m.get("text","").strip(),
             ent=f.get("entities",""), rel=f.get("relations",""),
             kw=f.get("keywords",""), speaker=m.get("speaker","").strip(),
             ts=m.get("timestamp","").strip())
    if i % 1000 == 0:
        print(f"  Memory: {i}/{len(memories)}")
print(f"LocoMoMemory created: {len(memories)}")

# Step 4 — Entity nodes + HAS_ENTITY
ent_count, he_count = 0, 0
for mid, f in features.items():
    for en in (f.get("entities","") or "").split(";"):
        en = en.strip()
        if not en:
            continue
        with driver.session() as s:
            s.run("MERGE (e:LocoMoEntity {name: $n})", n=en)
            ent_count += 1
            s.run("""
                MATCH (mem:LocoMoMemory {memory_id: $mid}), (e:LocoMoEntity {name: $n})
                MERGE (mem)-[:HAS_ENTITY]->(e)
            """, mid=mid, n=en)
            he_count += 1
print(f"Entity/HAS_ENTITY: {ent_count} merges, {he_count} relationships")

# Step 5 — KG triples as Entity-to-Entity LOCORELATES
trip_count, rel_count = 0, 0
for mid, f in features.items():
    t_str = (f.get("triples") or "").strip()
    if not t_str:
        continue
    m = TRIPLE_RE.search(t_str)
    if not m:
        continue
    src, rel, tgt = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    trip_count += 1
    with driver.session() as s:
        s.run("MERGE (:LocoMoEntity {name: $s})", s=src)
        s.run("MERGE (:LocoMoEntity {name: $t})", t=tgt)
        s.run("""
            MATCH (a:LocoMoEntity {name: $src}), (b:LocoMoEntity {name: $tgt})
            MERGE (a)-[:LOCORELATES {type: $rel}]->(b)
        """, src=src, tgt=tgt, rel=rel)
        rel_count += 1
print(f"Triples: {trip_count} parsed, {rel_count} LOCORELATES")

# Step 6 — Verify
with driver.session() as s:
    mc = s.run("MATCH (m:LocoMoMemory) RETURN count(m) AS c").single()["c"]
    ec = s.run("MATCH (e:LocoMoEntity) RETURN count(e) AS c").single()["c"]
    rc = s.run("MATCH ()-[r:LOCORELATES]->() RETURN count(r) AS c").single()["c"]
    hc = s.run("MATCH ()-[r:HAS_ENTITY]->() RETURN count(r) AS c").single()["c"]
    print(f"Final: LocoMoMemory={mc}  LocoMoEntity={ec}  LOCORELATES={rc}  HAS_ENTITY={hc}")

driver.close()
print("Neo4j import complete — existing benchmark data untouched.")
