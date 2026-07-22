"""
C2-0 pilot: mixed read/write correctness gate.
Cassandra parallel vs Neo4j, 10% and 30% write ratios, 8 clients.
"""
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cassandra.cluster import Cluster
from neo4j import GraphDatabase

GR = "synth_100000_1781447372"
FANOUT = 20
HOP = 2
FRONTIER_WORKERS = 16
CLIENTS = 8

PROJ = Path("D:/memorytable/cassandra-kg-memory")
MANIFEST = PROJ / "results/c1_manifest_100k_h2.jsonl"
OUT_DIR = PROJ / "reports/c2_mixed_rw_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEO4J_PWD = os.environ["NEO4J_PASSWORD"]
NEO_L = "C1KGNode"
NEO_R = "C1KG_EDGE"
NEO_WRITE_NODE = "C2WriteNode"
NEO_WRITE_REL = "C2_WRITE_EDGE"

_cass_cluster = None
_cass_session = None
_cass_executor = None
_neo_driver = None

# Cass prepared statements
_cass_insert_src = None
_cass_insert_dst = None
_cass_insert_bucket = None
_cass_insert_src_rel = None


def init():
    global _cass_cluster, _cass_session, _cass_executor, _neo_driver
    global _cass_insert_src, _cass_insert_dst, _cass_insert_bucket, _cass_insert_src_rel
    _cass_cluster = Cluster(["127.0.0.1"], port=9042)
    _cass_session = _cass_cluster.connect("ai_memory")
    _cass_executor = ThreadPoolExecutor(max_workers=FRONTIER_WORKERS)
    _cass_insert_src = _cass_session.prepare(
        "INSERT INTO kg_edges_by_src (graph_id,src_id,relation,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) "
        "VALUES (?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))"
    )
    _cass_insert_dst = _cass_session.prepare(
        "INSERT INTO kg_edges_by_dst (graph_id,dst_id,relation,src_id,edge_id,src_type,dst_type,confidence,source,created_at) "
        "VALUES (?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))"
    )
    _cass_insert_bucket = _cass_session.prepare(
        "INSERT INTO kg_edges_by_relation_bucket (graph_id,relation,bucket,src_id,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) "
        "VALUES (?,?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))"
    )
    _cass_insert_src_rel = _cass_session.prepare(
        "INSERT INTO kg_edges_by_src_relation (graph_id,src_id,relation,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) "
        "VALUES (?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))"
    )
    _neo_driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", NEO4J_PWD))
    with _neo_driver.session() as s:
        s.run(f"CREATE CONSTRAINT c2w_node IF NOT EXISTS FOR (n:{NEO_WRITE_NODE}) REQUIRE (n.graph_id, n.node_id) IS UNIQUE")


def shutdown():
    if _cass_executor:
        _cass_executor.shutdown(wait=True)
    if _cass_session:
        _cass_cluster.shutdown()
    if _neo_driver:
        _neo_driver.close()


def cass_fetch(src):
    rows = _cass_session.execute(
        "SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
    return [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]


def cass_parallel_hop(sources, rel):
    se = {}
    futures = {_cass_executor.submit(cass_fetch, src): src for src in sources}
    for f in as_completed(futures):
        src = futures[f]
        se[src] = [e for e in f.result() if e[1] == rel]
    return se


def cass_read(q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for relation in q["relation_path"]:
        sources = sorted({s[0] for s in frontier})
        se = cass_parallel_hop(sources, relation)
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FANOUT]:
                if e[2] not in np:
                    nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({s[2] for s in frontier}))


def cass_write(wgid, wid, wsrc, wdst, wrel, wsource):
    t0 = time.perf_counter()
    bkt = hash(wdst) % 64
    _cass_session.execute(_cass_insert_src, (wgid, wsrc, wrel, wdst, wsource))
    _cass_session.execute(_cass_insert_dst, (wgid, wdst, wrel, wsrc, wsource))
    _cass_session.execute(_cass_insert_bucket, (wgid, wrel, bkt, wsrc, wdst, wsource))
    _cass_session.execute(_cass_insert_src_rel, (wgid, wsrc, wrel, wdst, wsource))
    return (time.perf_counter() - t0) * 1000


def neo_read(q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for relation in q["relation_path"]:
        sources = sorted({s[0] for s in frontier})
        se = {}
        for src in sources:
            with _neo_driver.session() as s:
                rows = list(s.run(
                    f"MATCH (n:{NEO_L} {{graph_id: $g, node_id: $n}})"
                    f"-[r:{NEO_R} {{relation: $rel}}]->(m:{NEO_L} {{graph_id: $g}}) "
                    "RETURN n.node_id AS s,r.relation AS r,m.node_id AS d,coalesce(r.source,'') AS src ORDER BY r,d,src",
                    g=GR, n=src, rel=relation))
            se[src] = [(row["s"], row["r"], row["d"], row["src"]) for row in rows]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FANOUT]:
                if e[2] not in np: nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({s[2] for s in frontier}))


def neo_write(wgid, wid, wsrc, wdst, wrel, wsource):
    t0 = time.perf_counter()
    with _neo_driver.session() as s:
        s.run(
            f"MATCH (n:{NEO_WRITE_NODE} {{graph_id: $g, node_id: $s}}) "
            f"MATCH (m:{NEO_WRITE_NODE} {{graph_id: $g, node_id: $d}}) "
            f"CREATE (n)-[:{NEO_WRITE_REL} {{graph_id: $g, relation: $rel, source: $src, write_id: $wid}}]->(m)",
            g=wgid, s=wsrc, d=wdst, rel=wrel, src=wsource, wid=wid)
    return (time.perf_counter() - t0) * 1000


def precreate_neo_nodes(wgid, target_count):
    nodes = []
    for i in range(target_count):
        nodes.append({"g": wgid, "nid": f"c2_wsrc_{i}"})
        nodes.append({"g": wgid, "nid": f"c2_wdst_{i}"})
    for i in range(0, len(nodes), 5000):
        with _neo_driver.session() as s:
            s.run(f"UNWIND $rows AS r MERGE (n:{NEO_WRITE_NODE} {{graph_id: r.g, node_id: r.nid}})", rows=nodes[i:i+5000]).consume()
    print(f"  Precreated {target_count*2} nodes", flush=True)


def validate_writes_cass(wgid, writes):
    sample = random.sample(writes, min(20, len(writes)))
    ok = 0
    for w in sample:
        wid = w["write_id"]
        # Each table queried by its full partition key — source is NOT a key column
        c1 = list(_cass_session.execute(
            "SELECT count(*) FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
            (wgid, w["src_id"])))[0].count
        c2 = list(_cass_session.execute(
            "SELECT count(*) FROM kg_edges_by_dst WHERE graph_id=%s AND dst_id=%s",
            (wgid, w["dst_id"])))[0].count
        c3 = list(_cass_session.execute(
            "SELECT count(*) FROM kg_edges_by_relation_bucket WHERE graph_id=%s AND relation=%s AND bucket=%s",
            (wgid, w["relation"], hash(w["dst_id"]) % 64)))[0].count
        c4 = list(_cass_session.execute(
            "SELECT count(*) FROM kg_edges_by_src_relation WHERE graph_id=%s AND src_id=%s AND relation=%s",
            (wgid, w["src_id"], w["relation"])))[0].count
        if c1 >= 1 and c2 >= 1 and c3 >= 1 and c4 >= 1:
            ok += 1
    return ok, len(sample)


def validate_writes_neo(wgid, writes):
    sample = random.sample(writes, min(20, len(writes)))
    ok = 0
    for w in sample:
        with _neo_driver.session() as s:
            c = s.run(
                f"MATCH ()-[r:{NEO_WRITE_REL} {{write_id: $wid}}]->() RETURN count(r) AS cnt",
                wid=w["write_id"]).single()["cnt"]
            if c >= 1: ok += 1
    return ok, len(sample)


def run_pilot(system, write_ratio_pct):
    wgid = f"c2_write_sink_{system}_w{write_ratio_pct}_r1"
    tag = f"{system} wr={write_ratio_pct}%"
    print(f"\n{'='*60}", flush=True)
    print(f"  {tag}", flush=True)

    queries = []
    with MANIFEST.open() as f:
        for line in f:
            queries.append(json.loads(line))

    # Unique write pool
    write_seq = 0
    write_lock = threading.Lock()
    completed_writes = []
    write_rel = "talked_to"

    # Precreate Neo nodes
    if system == "neo4j":
        precreate_neo_nodes(wgid, 5000)

    # Per-client data
    per_thread_read_lat = [[] for _ in range(CLIENTS)]
    per_thread_write_lat = [[] for _ in range(CLIENTS)]
    per_thread_errors = [0] * CLIENTS
    stop = threading.Event()
    write_interval = 10  # 1 write in N ops

    def client_loop(ci):
        read_idx = ci % len(queries)
        ops_since_write = ci % write_interval  # stagger start
        rlat = per_thread_read_lat[ci]
        wlat = per_thread_write_lat[ci]
        local_write_list = []
        nonlocal write_seq

        while not stop.is_set():
            # Decide read vs write based on write_ratio
            do_write = False
            if write_ratio_pct == 10:
                do_write = (ops_since_write % 10 == 9)
            else:  # 30%
                do_write = (ops_since_write % 10 >= 7)

            if do_write:
                t0 = time.perf_counter()
                try:
                    with write_lock:
                        wid = write_seq; write_seq += 1
                    wsrc = f"c2_wsrc_{wid}"; wdst = f"c2_wdst_{wid}"; ws = "c2_mixed_write"
                    if system == "cassandra_parallel":
                        dt = cass_write(wgid, str(wid), wsrc, wdst, write_rel, ws)
                    else:
                        dt = neo_write(wgid, str(wid), wsrc, wdst, write_rel, ws)
                    wlat.append(dt)
                    local_write_list.append({"write_id": str(wid), "src_id": wsrc, "dst_id": wdst,
                                             "relation": write_rel})
                except Exception:
                    per_thread_errors[ci] += 1
            else:
                q = queries[read_idx]
                t0 = time.perf_counter()
                try:
                    if system == "cassandra_parallel":
                        cass_read(q)
                    else:
                        neo_read(q)
                    rlat.append((time.perf_counter() - t0) * 1000)
                except Exception:
                    per_thread_errors[ci] += 1
                read_idx = (read_idx + 1) % len(queries)

            ops_since_write += 1

        with write_lock:
            completed_writes.extend(local_write_list)

    # Warmup
    threads = [threading.Thread(target=client_loop, args=(i,), daemon=True) for i in range(CLIENTS)]
    for t in threads: t.start()
    time.sleep(10)

    # Measurement
    for l in per_thread_read_lat: l.clear()
    for l in per_thread_write_lat: l.clear()
    for i in range(CLIENTS): per_thread_errors[i] = 0
    completed_writes.clear()
    write_seq = 0

    meas_start = time.perf_counter()
    time.sleep(20)
    meas_end = time.perf_counter()
    stop.set()
    for t in threads: t.join(timeout=10)

    meas_s = meas_end - meas_start
    all_reads = [v for l in per_thread_read_lat for v in l]
    all_writes = [v for l in per_thread_write_lat for v in l]
    n_reads = len(all_reads)
    n_writes = len(all_writes)
    total_errors = sum(per_thread_errors)
    actual_wr = round(n_writes / max(n_reads + n_writes, 1) * 100, 1)

    all_reads.sort(); all_writes.sort()
    def pct(a, p): return a[int((len(a)-1)*p/100)] if a else 0

    # Spotcheck reads
    rng = random.Random(42)
    check_q = rng.sample(queries, 10)
    read_ok = 0
    for q in check_q:
        p = neo_read(q) if system == "neo4j" else cass_read(q)
        canon = sorted([list(p) for p in sorted(p)])
        h = hashlib.sha256(json.dumps(canon, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if h == q["expected_path_hash"]: read_ok += 1

    # Validate writes
    vok, vtotal = (validate_writes_neo(wgid, completed_writes) if system == "neo4j"
                   else validate_writes_cass(wgid, completed_writes))

    row = {
        "system": system, "clients": CLIENTS,
        "target_write_ratio": write_ratio_pct, "actual_write_ratio": actual_wr,
        "repeat": 1, "warmup_seconds": 10, "measurement_seconds": round(meas_s, 3),
        "completed_reads": n_reads, "completed_writes": n_writes,
        "read_QPS": round(n_reads/max(meas_s,0.001), 3),
        "write_QPS": round(n_writes/max(meas_s,0.001), 3),
        "total_ops_QPS": round((n_reads+n_writes)/max(meas_s,0.001), 3),
        "read_mean_ms": round(statistics.mean(all_reads),3) if all_reads else 0,
        "read_p50_ms": round(pct(all_reads,50),3),
        "read_p95_ms": round(pct(all_reads,95),3),
        "read_p99_ms": round(pct(all_reads,99),3),
        "write_mean_ms": round(statistics.mean(all_writes),3) if all_writes else 0,
        "write_p50_ms": round(pct(all_writes,50),3),
        "write_p95_ms": round(pct(all_writes,95),3),
        "write_p99_ms": round(pct(all_writes,99),3),
        "read_error_count": total_errors, "write_error_count": 0,
        "total_error_rate": round(total_errors/max(n_reads+n_writes,1),6),
        "read_hash_spotcheck_passed": read_ok == 10,
        "write_validation_passed": vok == vtotal,
        "logical_writes_validated": f"{vok}/{vtotal}",
        "physical_writes_per_logical_write": 4 if system == "cassandra_parallel" else 1,
        "frontier_workers": FRONTIER_WORKERS,
        "application_cache": "disabled",
        "backend_state": "warm",
        "write_graph_id": wgid,
    }

    print(f"  Reads: {n_reads}  Writes: {n_writes}  actual_wr={actual_wr}%", flush=True)
    print(f"  Read QPS={row['read_QPS']:.1f} mean={row['read_mean_ms']:.1f}ms p95={row['read_p95_ms']:.1f}ms", flush=True)
    print(f"  Write QPS={row['write_QPS']:.1f} mean={row['write_mean_ms']:.1f}ms", flush=True)
    print(f"  Read spotcheck: {read_ok}/10  Write validation: {vok}/{vtotal}  errors={total_errors}", flush=True)

    return row


def main():
    print("=== C2-0 Mixed R/W Pilot ===", flush=True)
    print(f"Systems: cassandra_parallel, neo4j", flush=True)
    print(f"Clients=8  Write ratios=10%,30%  Warmup=10s  Measure=20s", flush=True)

    init()

    env = {"os": platform.platform(), "python": sys.version, "cpu_logical": os.cpu_count(),
           "frontier_workers": FRONTIER_WORKERS, "graph_id": GR}
    with (OUT_DIR / "environment.json").open("w") as f:
        json.dump(env, f, indent=2)
    with (OUT_DIR / "run_config.json").open("w") as f:
        json.dump({"clients": CLIENTS, "write_ratios": [10,30], "warmup_s": 10, "measurement_s": 20,
                   "hop": HOP, "fanout": FANOUT, "cycle_policy": "path"}, f, indent=2)

    all_rows = []
    for wr in [10, 30]:
        for sys_name in ["cassandra_parallel", "neo4j"]:
            row = run_pilot(sys_name, wr)
            all_rows.append(row)

    shutdown()

    fields = list(all_rows[0].keys()) if all_rows else []
    with (OUT_DIR / "pilot_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)
    with (OUT_DIR / "pilot_summary.json").open("w") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    # Spotchecks and write validation as JSONL
    with (OUT_DIR / "pilot_spotchecks.jsonl").open("w") as f:
        for r in all_rows:
            f.write(json.dumps({"system": r["system"], "write_ratio": r["target_write_ratio"],
                                "read_hash_spotcheck_passed": r["read_hash_spotcheck_passed"],
                                "write_validation_passed": r["write_validation_passed"]}, ensure_ascii=False) + "\n")
    with (OUT_DIR / "pilot_write_validation.jsonl").open("w") as f:
        for r in all_rows:
            f.write(json.dumps({"system": r["system"], "write_ratio": r["target_write_ratio"],
                                "logical_writes_validated": r["logical_writes_validated"],
                                "physical_writes_per_logical_write": r["physical_writes_per_logical_write"]}, ensure_ascii=False) + "\n")

    print(f"\n=== PILOT SUMMARY ===", flush=True)
    for r in all_rows:
        ok = r["read_hash_spotcheck_passed"] and r["write_validation_passed"] and r["total_error_rate"] == 0
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {r['system']:20s} wr={r['actual_write_ratio']:4.1f}%  "
              f"readQPS={r['read_QPS']:6.1f}  writeQPS={r['write_QPS']:4.1f}  "
              f"read_hash={r['read_hash_spotcheck_passed']}  write_valid={r['write_validation_passed']}  "
              f"err={r['read_error_count']}", flush=True)

    print(f"\n--- Cassandra 4-table INSERT ---", flush=True)
    print("INSERT INTO kg_edges_by_src             (graph_id,src_id,relation,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) VALUES (?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))")
    print("INSERT INTO kg_edges_by_dst             (graph_id,dst_id,relation,src_id,edge_id,src_type,dst_type,confidence,source,created_at) VALUES (?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))")
    print("INSERT INTO kg_edges_by_relation_bucket  (graph_id,relation,bucket,src_id,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) VALUES (?,?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))")
    print("INSERT INTO kg_edges_by_src_relation     (graph_id,src_id,relation,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) VALUES (?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))")

    print(f"\n--- Neo4j write Cypher ---")
    print(f"MATCH (n:{NEO_WRITE_NODE} {{graph_id: $g, node_id: $s}})")
    print(f"MATCH (m:{NEO_WRITE_NODE} {{graph_id: $g, node_id: $d}})")
    print(f"CREATE (n)-[:{NEO_WRITE_REL} {{graph_id: $g, relation: $rel, source: $src, write_id: $wid}}]->(m)")

    print(f"\n--- Mixed scheduler ---")
    print("write_interval = 10")
    print("do_write = (ops_since_write % 10 >= 7)  # 30% ratio")
    print("do_write = (ops_since_write % 10 == 9)  # 10% ratio")
    print("stagger: ops_since_write = (ci % write_interval)  # per-client offset")


if __name__ == "__main__":
    main()
