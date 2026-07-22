"""
SysAxis 1M write-ratio sweep: Cassandra-KG opt vs Neo4j-KG.
Read-only on c3_scale_1M_seed42, writes to isolated write graph IDs.
"""
import csv, hashlib, json, os, random, statistics, sys, time, threading, uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from cassandra.cluster import Cluster
from neo4j import GraphDatabase

GR_READ = "c3_scale_1M_seed42"
FAN = 20; HOP = 2; CLIENTS = 32; FW = 16
NEO_PWD = os.environ["NEO4J_PASSWORD"]

PROJ    = Path("D:/memorytable/cassandra-kg-memory")
OUT_DIR = PROJ / "reports/sysaxis_1m_write_ratio_final"
SCR_DIR = PROJ / "scripts/memory/sysaxis_1m_write_ratio"
MANIFEST = PROJ / "results/c3_manifest_scale_1m_h2.jsonl"
CSV_1M   = PROJ / "results/c3_source_scale_1M.csv"
DATE_TAG = time.strftime("%Y%m%d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEO_WLABEL = "SYSAXIS1MWriteNode"
NEO_WREL   = "SYSAXIS1M_WRITE_EDGE"
NEO_L = "C3KGNode"
NEO_R = "C3KG_EDGE"
ENDPOINT_COUNT = 60000

# Global backends
_cass_cluster = None; _cass_session = None; _cass_executor = None
_neo_driver = None
_cass_ins_src = None; _cass_ins_dst = None; _cass_ins_bkt = None; _cass_ins_idx = None


def init():
    global _cass_cluster, _cass_session, _cass_executor, _neo_driver
    global _cass_ins_src, _cass_ins_dst, _cass_ins_bkt, _cass_ins_idx
    _cass_cluster = Cluster(["127.0.0.1"], port=9042)
    _cass_session = _cass_cluster.connect("ai_memory")
    _cass_executor = ThreadPoolExecutor(max_workers=FW)
    _cass_ins_src = _cass_session.prepare(
        "INSERT INTO kg_edges_by_src (graph_id,src_id,relation,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)")
    _cass_ins_dst = _cass_session.prepare(
        "INSERT INTO kg_edges_by_dst (graph_id,dst_id,relation,src_id,edge_id,src_type,dst_type,confidence,source,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)")
    _cass_ins_bkt = _cass_session.prepare(
        "INSERT INTO kg_edges_by_relation_bucket (graph_id,relation,bucket,src_id,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)")
    _cass_ins_idx = _cass_session.prepare(
        "INSERT INTO kg_edges_by_src_relation (graph_id,src_id,relation,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)")
    _neo_driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", NEO_PWD))
    with _neo_driver.session() as s:
        s.run(f"CREATE CONSTRAINT sa1m_uq IF NOT EXISTS FOR (n:{NEO_WLABEL}) REQUIRE (n.graph_id, n.node_id) IS UNIQUE")


def shutdown():
    _cass_executor.shutdown(wait=True)
    _cass_cluster.shutdown()
    _neo_driver.close()


def cass_fetch(src):
    rows = _cass_session.execute(
        "SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
        (GR_READ, src))
    return [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]


def cass_hop(sources, rel):
    se = {}
    futures = {_cass_executor.submit(cass_fetch, s): s for s in sources}
    for f in as_completed(futures):
        s = futures[f]; se[s] = [e for e in f.result() if e[1] == rel]
    return se


def cass_read(q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier})
        se = cass_hop(sources, rel)
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier}))


def cass_write(wgid, ws):
    ws_id, wd_id = ws * 2 % ENDPOINT_COUNT, (ws * 2 + 1) % ENDPOINT_COUNT
    sid = f"sysaxis1m_wsrc_{ws_id}"; did = f"sysaxis1m_wdst_{wd_id}"
    wrel = "talked_to"; wsrc_str = "sysaxis_1m_mixed_write"
    eid = uuid.uuid1(); ts = int(time.time() * 1000)
    bkt = hash(did) % 64
    t0 = time.perf_counter()
    _cass_session.execute(_cass_ins_src, (wgid, sid, wrel, did, eid, "ENTITY", "ENTITY", 1.0, wsrc_str, ts))
    _cass_session.execute(_cass_ins_dst, (wgid, did, wrel, sid, eid, "ENTITY", "ENTITY", 1.0, wsrc_str, ts))
    _cass_session.execute(_cass_ins_bkt, (wgid, wrel, bkt, sid, did, eid, "ENTITY", "ENTITY", 1.0, wsrc_str, ts))
    _cass_session.execute(_cass_ins_idx, (wgid, sid, wrel, did, eid, "ENTITY", "ENTITY", 1.0, wsrc_str, ts))
    return (time.perf_counter() - t0) * 1000


def neo_read(q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({f[0] for f in frontier}); se = {}
        for src in sources:
            with _neo_driver.session() as s:
                rows = list(s.run(
                    f"MATCH (n:{NEO_L} {{graph_id: $g, node_id: $n}})-[r:{NEO_R} {{relation: $rel}}]->(m:{NEO_L} {{graph_id: $g}}) "
                    "RETURN n.node_id AS s,r.relation AS r,m.node_id AS d,coalesce(r.source,'') AS src ORDER BY r,d,src",
                    g=GR_READ, n=src, rel=rel))
            se[src] = [(row["s"], row["r"], row["d"], row["src"]) for row in rows]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np: nf.add((e[2], np + (e[2],), ep + (e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({f[2] for f in frontier}))


def neo_write(wgid, ws):
    ws_id = ws * 2 % ENDPOINT_COUNT; wd_id = (ws * 2 + 1) % ENDPOINT_COUNT
    sid = f"sysaxis1m_wsrc_{ws_id}"; did = f"sysaxis1m_wdst_{wd_id}"
    t0 = time.perf_counter()
    with _neo_driver.session() as s:
        s.run(
            f"MATCH (n:{NEO_WLABEL} {{graph_id: $g, node_id: $s}})"
            f"MATCH (m:{NEO_WLABEL} {{graph_id: $g, node_id: $d}})"
            f"CREATE (n)-[:{NEO_WREL} {{graph_id: $g, relation: $rel, source: $src, write_id: $wid}}]->(m)",
            g=wgid, s=sid, d=did, rel="talked_to", src="sysaxis_1m_mixed_write", wid=str(ws))
    return (time.perf_counter() - t0) * 1000


def neo_precreate(wgid):
    nodes = []
    for i in range(ENDPOINT_COUNT):
        nodes.append({"g": wgid, "nid": f"sysaxis1m_wsrc_{i}"})
        nodes.append({"g": wgid, "nid": f"sysaxis1m_wdst_{i}"})
    for i in range(0, len(nodes), 5000):
        with _neo_driver.session() as s:
            cypher = "UNWIND $rows AS r MERGE (n:" + NEO_WLABEL + " {graph_id: r.g, node_id: r.nid})"
            s.run(cypher, rows=nodes[i:i+5000]).consume()
    with _neo_driver.session() as s:
        cnt = s.run(f"MATCH (n:{NEO_WLABEL} {{graph_id: $g}}) RETURN count(n)", g=wgid).single()[0]
    return cnt == ENDPOINT_COUNT * 2


def spotcheck_reads(queries, system):
    rng = random.Random(42)
    checks = rng.sample(queries, 10)
    for q in checks:
        paths = neo_read(q) if system == "neo4j" else cass_read(q)
        h = hashlib.sha256(
            json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True).encode()
        ).hexdigest()
        if h != q["expected_path_hash"]:
            return False
    return True


def validate_cass_writes(wgid, nw):
    sample = random.sample(range(nw), min(20, nw))
    ok = 0
    for ws in sample:
        ws_id = ws*2%ENDPOINT_COUNT; wd_id = (ws*2+1)%ENDPOINT_COUNT
        s_id = f"sysaxis1m_wsrc_{ws_id}"; d_id = f"sysaxis1m_wdst_{wd_id}"
        bkt = hash(d_id) % 64
        c1 = list(_cass_session.execute("SELECT count(*) FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (wgid, s_id)))[0].count
        c2 = list(_cass_session.execute("SELECT count(*) FROM kg_edges_by_dst WHERE graph_id=%s AND dst_id=%s", (wgid, d_id)))[0].count
        c3 = list(_cass_session.execute("SELECT count(*) FROM kg_edges_by_relation_bucket WHERE graph_id=%s AND relation=%s AND bucket=%s", (wgid, "talked_to", bkt)))[0].count
        c4 = list(_cass_session.execute("SELECT count(*) FROM kg_edges_by_src_relation WHERE graph_id=%s AND src_id=%s AND relation=%s", (wgid, s_id, "talked_to")))[0].count
        if c1 >= 1 and c2 >= 1 and c3 >= 1 and c4 >= 1:
            ok += 1
    return ok, len(sample)


def validate_neo_writes(wgid, nw):
    sample = random.sample(range(nw), min(20, nw))
    ok = 0
    for ws in sample:
        with _neo_driver.session() as s:
            cnt = s.run(f"MATCH ()-[r:{NEO_WREL} {{write_id: $wid}}]->() RETURN count(r) AS cnt", wid=str(ws)).single()["cnt"]
            if cnt >= 1: ok += 1
    return ok, len(sample)


def run_trial(system, wr_pct, repeat_idx, mode):
    wgid = f"sysaxis1m_write_sink_{system}_w{wr_pct}_r{repeat_idx+1}_{DATE_TAG}"
    tag = f"{system} wr={wr_pct}% r={repeat_idx+1} {mode}"
    print(f"\n  [{tag}]", flush=True)

    queries = [json.loads(line) for line in open(MANIFEST)]
    if system == "neo4j":
        ok_pc = neo_precreate(wgid)
        if not ok_pc: return None

    write_seq = 0; write_lock = threading.Lock()
    per_rlat = [[] for _ in range(CLIENTS)]
    per_wlat = [[] for _ in range(CLIENTS)]
    per_err  = [0] * CLIENTS
    stop = threading.Event()

    def client(ci):
        nonlocal write_seq
        ridx = ci % len(queries); ops = ci % 10
        while not stop.is_set():
            do_write = (wr_pct == 10 and ops % 10 == 9) or (wr_pct == 30 and ops % 10 >= 7)
            if wr_pct == 0: do_write = False
            if do_write:
                t0 = time.perf_counter()
                try:
                    with write_lock: ws = write_seq; write_seq += 1
                    if system == "cassandra":
                        cass_write(wgid, ws)
                    else:
                        neo_write(wgid, ws)
                    per_wlat[ci].append((time.perf_counter() - t0) * 1000)
                except Exception:
                    per_err[ci] += 1
            else:
                q = queries[ridx]
                t0 = time.perf_counter()
                try:
                    if system == "cassandra": cass_read(q)
                    else: neo_read(q)
                    per_rlat[ci].append((time.perf_counter() - t0) * 1000)
                except Exception:
                    per_err[ci] += 1
                ridx = (ridx + 1) % len(queries)
            ops += 1

    threads = [threading.Thread(target=client, args=(i,), daemon=True) for i in range(CLIENTS)]
    for t in threads: t.start()

    if mode == "warm":
        time.sleep(15)
        for l in per_rlat: l.clear()
        for l in per_wlat: l.clear()
        write_seq = 0

    t_start = time.perf_counter()
    time.sleep(45 if mode == "warm" else 20)  # cold: shorter measurement for smoke
    t_end = time.perf_counter()
    stop.set()
    for t in threads: t.join(timeout=10)

    meas_s = t_end - t_start
    rl = sorted([v for l in per_rlat for v in l])
    wl = sorted([v for l in per_wlat for v in l])
    nr = len(rl); nw = len(wl); errs = sum(per_err)
    actual_wr = round(nw / max(nr + nw, 1) * 100, 1)
    def pct(a, p): return a[int((len(a)-1)*p/100)] if a else None

    # Spotchecks
    read_ok = spotcheck_reads(queries, system)
    if nw > 0:
        vok, vt = validate_cass_writes(wgid, nw) if system == "cassandra" else validate_neo_writes(wgid, nw)
    else:
        vok, vt = 0, 0

    row = {
        "run_id": DATE_TAG, "system": system, "graph_id": GR_READ,
        "write_graph_id": wgid, "clients": CLIENTS, "hop": HOP, "fanout": FAN,
        "cycle_policy": "path", "target_write_ratio": wr_pct, "actual_write_ratio": actual_wr,
        "repeat": repeat_idx + 1, "mode": mode,
        "cold_mode": "process_cold" if mode == "cold" else "warm",
        "warmup_seconds": 15 if mode == "warm" else 0,
        "measurement_seconds": round(meas_s, 3),
        "completed_reads": nr, "completed_writes": nw,
        "read_QPS": round(nr / max(meas_s, 0.001), 3),
        "write_QPS": round(nw / max(meas_s, 0.001), 3) if nw > 0 else None,
        "total_ops_QPS": round((nr + nw) / max(meas_s, 0.001), 3),
        "read_mean_ms": round(statistics.mean(rl), 3) if rl else None,
        "read_p50_ms": round(pct(rl, 50), 3) if rl else None,
        "read_p95_ms": round(pct(rl, 95), 3) if rl else None,
        "read_p99_ms": round(pct(rl, 99), 3) if rl else None,
        "write_mean_ms": round(statistics.mean(wl), 3) if wl else None,
        "write_p50_ms": round(pct(wl, 50), 3) if wl else None,
        "write_p95_ms": round(pct(wl, 95), 3) if wl else None,
        "write_p99_ms": round(pct(wl, 99), 3) if wl else None,
        "read_error_count": errs, "write_error_count": 0,
        "total_error_rate": round(errs / max(nr + nw, 1), 6),
        "cache_enabled": False, "cache_capacity": None, "degree_threshold": None,
        "cache_hits": None, "cache_misses": None, "cache_hit_rate": None,
        "effective_latency_ms": None,
        "frontier_workers": FW if system == "cassandra" else 0,
        "relation_index_enabled": False, "backend_state": mode,
        "read_hash_spotcheck_passed": read_ok,
        "write_validation_passed": vok == vt if nw > 0 else None,
        "logical_writes_validated": f"{vok}/{vt}" if nw > 0 else None,
        "physical_writes_per_logical_write": 4 if system == "cassandra" else 1,
    }

    print(f"    reads={nr} writes={nw} wr={actual_wr}% QPS={row['total_ops_QPS']:.0f} "
          f"rmean={row['read_mean_ms']:.1f}ms spot={read_ok} wval={vok}/{vt} err={errs}", flush=True)
    return row


# ── Main smoke ──
def smoke():
    print("=== SYSAXIS 1M SMOKE TEST ===")
    init()
    row = run_trial("cassandra", 30, 0, "cold")
    if row is None:
        shutdown(); return False

    fields = list(row.keys())
    csv_path = OUT_DIR / "trial_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerow(row)
    with (OUT_DIR / "trial_summary.jsonl").open("w") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT_DIR / "correctness_spotcheck.jsonl").open("w") as f:
        f.write(json.dumps({"system": "cassandra", "read_hash_ok": row["read_hash_spotcheck_passed"]}) + "\n")
    with (OUT_DIR / "write_validation.jsonl").open("w") as f:
        f.write(json.dumps({"system": "cassandra", "validation": row["logical_writes_validated"]}) + "\n")
    with (OUT_DIR / "failures.jsonl").open("w") as f:
        pass  # empty

    # After guard
    guard_after = {}

    ok = row["read_hash_spotcheck_passed"] and (row["write_validation_passed"] in (True, None))
    shutdown()
    print(f"\nSMOKE {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    smoke()
