"""P5-1: Concurrent Update-to-Searchable. Thread-pool based, sync drivers."""
import csv, json, hashlib, random, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path("D:/memorytable/cassandra-kg-memory")
OUT = BASE / "reports/p5_minimal_core"
ART = OUT / "artifacts"

GRAPH_ID = "c3_scale_1M_seed42"
N_EVENTS = 20000  # reduced for feasibility
N_WARMUP = 2000
N_RUNS = 3  # reduced
CONCURRENCIES = [8, 32, 64]
SEED = 20260720
PASSWORD = "password123"

# ===================== DATA =====================
print("Loading 1M graph...", flush=True)
graph_edges = []
with open(BASE/"results/c3_source_scale_1M.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        graph_edges.append({"src_id": row["src_id"], "dst_id": row["dst_id"], "relation": row["relation"]})
print(f"  {len(graph_edges)} edges")

# Events
EVENT_FILE = ART / "frozen_update_events.jsonl"
if not EVENT_FILE.exists():
    print(f"  Generating events...", flush=True)
    rng = random.Random(SEED)
    events = []
    for i in range(N_WARMUP + N_EVENTS):
        e = rng.choice(graph_edges)
        mid = f"mem_{i:08d}"
        evt = {"update_id": f"evt_{i:08d}", "graph_id": GRAPH_ID, "memory_id": mid, "version": i,
               "operation": "insert_memory", "raw_text": f"Memory {i}", "entities": f"{e['src_id']},{e['dst_id']}",
               "relations": e["relation"], "keywords": f"{e['src_id']},{e['dst_id']}",
               "raw_erk_text": f"Memory {i}\nE: {e['src_id']},{e['dst_id']}\nR: {e['relation']}",
               "probe_query": f"{e['src_id']} {e['relation']}", "src_id": e["src_id"], "dst_id": e["dst_id"],
               "relation": e["relation"], "edge_id": f"{e['src_id']}_{e['relation']}_{e['dst_id']}"}
        events.append(evt)
    with EVENT_FILE.open("w") as f:
        for evt in events: f.write(json.dumps(evt)+"\n")
    print(f"  Done: {len(events)} events")

events = [json.loads(line) for line in EVENT_FILE.read_text().strip().split("\n")]
warmup = events[:N_WARMUP]
formal = events[N_WARMUP:N_WARMUP+N_EVENTS]
print(f"  Warmup: {len(warmup)}, Formal: {len(formal)}")

# ===================== CASSANDRA =====================
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel

def cass_connect():
    cluster = Cluster(["127.0.0.1"], protocol_version=4)
    session = cluster.connect()
    session.execute("CREATE KEYSPACE IF NOT EXISTS kg_memory_bench WITH replication={'class':'SimpleStrategy','replication_factor':1}")
    session.execute("USE kg_memory_bench")
    session.execute("CREATE TABLE IF NOT EXISTS memory_by_id (graph_id text, memory_id text, raw_text text, version int, updated_at double, PRIMARY KEY ((graph_id), memory_id))")
    session.execute("CREATE TABLE IF NOT EXISTS kg_edges_by_src (graph_id text, src_id text, edge_id text, dst_id text, relation text, memory_id text, version int, updated_at double, PRIMARY KEY ((graph_id, src_id), edge_id))")
    session.execute("CREATE TABLE IF NOT EXISTS raw_erk_view (graph_id text, memory_id text, text text, version int, updated_at double, PRIMARY KEY ((graph_id), memory_id))")
    p_mem = session.prepare("INSERT INTO memory_by_id (graph_id,memory_id,raw_text,version,updated_at) VALUES (?,?,?,?,?)")
    p_edge = session.prepare("INSERT INTO kg_edges_by_src (graph_id,src_id,edge_id,dst_id,relation,memory_id,version,updated_at) VALUES (?,?,?,?,?,?,?,?)")
    p_erk = session.prepare("INSERT INTO raw_erk_view (graph_id,memory_id,text,version,updated_at) VALUES (?,?,?,?,?)")
    p_probe = session.prepare("SELECT edge_id,version FROM kg_edges_by_src WHERE graph_id=? AND src_id=?")
    return cluster, session, p_mem, p_edge, p_erk, p_probe

# ===================== NEO4J =====================
from neo4j import GraphDatabase

def neo4j_connect():
    driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", PASSWORD), max_connection_pool_size=128)
    driver.verify_connectivity()
    with driver.session(database="neo4j") as s:
        s.run("CREATE CONSTRAINT memory_key IF NOT EXISTS FOR (m:Memory) REQUIRE (m.graph_id,m.memory_id) IS UNIQUE")
        s.run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE (e.graph_id,e.entity_id) IS UNIQUE")
        s.run("CREATE CONSTRAINT rawerk_key IF NOT EXISTS FOR (v:RawERKView) REQUIRE (v.graph_id,v.memory_id) IS UNIQUE")
    return driver

# ===================== WRITE + PROBE =====================
def cass_worker(evt, session, p_mem, p_edge, p_erk, p_probe):
    t0 = time.perf_counter()
    session.execute(p_mem, (evt["graph_id"],evt["memory_id"],evt["raw_text"],evt["version"],t0))
    session.execute(p_edge, (evt["graph_id"],evt["src_id"],evt["edge_id"],evt["dst_id"],evt["relation"],evt["memory_id"],evt["version"],t0))
    session.execute(p_erk, (evt["graph_id"],evt["memory_id"],evt["raw_erk_text"],evt["version"],t0))
    for attempt in range(120):
        rows = session.execute(p_probe, (evt["graph_id"],evt["src_id"]))
        for r in rows:
            if r.edge_id == evt["edge_id"] and r.version >= evt["version"]:
                return t0, time.perf_counter(), True
        time.sleep(0.05)
    return t0, time.perf_counter(), False

def neo4j_worker(evt, driver):
    t0 = time.perf_counter()
    with driver.session(database="neo4j") as s:
        s.execute_write(lambda tx: _neo4j_write_tx(tx, evt))
    for attempt in range(120):
        with driver.session(database="neo4j") as s:
            result = s.run("MATCH (e:Entity {graph_id:$g,entity_id:$s})-[r:KG_EDGE {edge_id:$e}]->() WHERE r.version>=$v RETURN 1",
                g=evt["graph_id"], s=evt["src_id"], e=evt["edge_id"], v=evt["version"])
            if result.peek():
                return t0, time.perf_counter(), True
        time.sleep(0.05)
    return t0, time.perf_counter(), False

def _neo4j_write_tx(tx, evt):
    ts = time.perf_counter()
    tx.run("MERGE (m:Memory {graph_id:$g,memory_id:$m}) ON CREATE SET m.raw_text=$t,m.version=$v,m.updated_at=$ts ON MATCH SET m.version=CASE WHEN coalesce(m.version,-1)<=$v THEN $v ELSE m.version END,m.raw_text=CASE WHEN coalesce(m.version,-1)<=$v THEN $t ELSE m.raw_text END",
        g=evt["graph_id"],m=evt["memory_id"],t=evt["raw_text"],v=evt["version"],ts=ts)
    tx.run("MERGE (se:Entity {graph_id:$g,entity_id:$sid}) ON CREATE SET se.name=$sid,se.version=$v MERGE (de:Entity {graph_id:$g,entity_id:$did}) ON CREATE SET de.name=$did,de.version=$v WITH se,de MATCH (m:Memory {graph_id:$g,memory_id:$m}) MERGE (m)-[:MENTIONS]->(se) MERGE (m)-[:MENTIONS]->(de) MERGE (se)-[r:KG_EDGE {graph_id:$g,edge_id:$e}]->(de) ON CREATE SET r.relation=$rel,r.source_memory_id=$m,r.version=$v,r.updated_at=$ts ON MATCH SET r.version=CASE WHEN coalesce(r.version,-1)<=$v THEN $v ELSE r.version END,r.relation=CASE WHEN coalesce(r.version,-1)<=$v THEN $rel ELSE r.relation END",
        g=evt["graph_id"],sid=evt["src_id"],did=evt["dst_id"],m=evt["memory_id"],e=evt["edge_id"],rel=evt["relation"],v=evt["version"],ts=ts)
    tx.run("MERGE (v:RawERKView {graph_id:$g,memory_id:$m}) ON CREATE SET v.text=$t,v.version=$v,v.updated_at=$ts ON MATCH SET v.version=CASE WHEN coalesce(v.version,-1)<=$v THEN $v ELSE v.version END",
        g=evt["graph_id"],m=evt["memory_id"],t=evt["raw_erk_text"],v=evt["version"],ts=ts)

# ===================== RUN =====================
all_results = []
for concurrency in CONCURRENCIES:
    for run_id in range(1, N_RUNS+1):
        order = [("cassandra",cass_connect),("neo4j",neo4j_connect)] if run_id%2==1 else [("neo4j",neo4j_connect),("cassandra",cass_connect)]
        for backend, connect_fn in order:
            label = f"P5-1_{backend}_c{concurrency}_r{run_id}"
            print(f"  {label} connecting...", end=" ", flush=True)
            
            if backend == "cassandra":
                cluster, session, p_mem, p_edge, p_erk, p_probe = connect_fn()
                if run_id == 1:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        fs = [pool.submit(cass_worker, evt, session, p_mem, p_edge, p_erk, p_probe) for evt in warmup]
                        [f.result() for f in fs]
                t_start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    fs = [pool.submit(cass_worker, evt, session, p_mem, p_edge, p_erk, p_probe) for evt in formal]
                    for f, evt in zip(fs, formal):
                        t0, t5, ok = f.result()
                        all_results.append({"backend":backend,"concurrency":concurrency,"run_id":run_id,
                            "update_id":evt["update_id"],"update_to_searchable_ms":round((t5-t0)*1000,3),
                            "kg_write_ms":round((t5-t0)*1000,3),"probe_ms":0,"status":"ok" if ok else "timeout"})
                session.shutdown(); cluster.shutdown()
            else:
                driver = connect_fn()
                if run_id == 1:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        fs = [pool.submit(neo4j_worker, evt, driver) for evt in warmup]
                        [f.result() for f in fs]
                t_start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    fs = [pool.submit(neo4j_worker, evt, driver) for evt in formal]
                    for f, evt in zip(fs, formal):
                        t0, t5, ok = f.result()
                        all_results.append({"backend":backend,"concurrency":concurrency,"run_id":run_id,
                            "update_id":evt["update_id"],"update_to_searchable_ms":round((t5-t0)*1000,3),
                            "kg_write_ms":round((t5-t0)*1000,3),"probe_ms":0,"status":"ok" if ok else "timeout"})
                driver.close()
            
            elapsed = time.perf_counter() - t_start
            n_ok = sum(1 for r in all_results[-N_EVENTS:] if r["status"]=="ok")
            lats = sorted(r["update_to_searchable_ms"] for r in all_results[-N_EVENTS:] if r["status"]=="ok")
            if lats:
                print(f"ok={n_ok}/{N_EVENTS} p50={lats[int(len(lats)*0.5)]:.0f}ms p95={lats[int(len(lats)*0.95)]:.0f}ms ({elapsed:.0f}s)", flush=True)
            else:
                print(f"ALL TIMEOUT ({elapsed:.0f}s)", flush=True)

# Save
with (OUT/"p5_1_per_event.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(all_results[0].keys())); w.writeheader(); w.writerows(all_results)
print(f"\nWrote {len(all_results)} events to p5_1_per_event.csv")
