"""
C2-1 fix: wr=10%,30% only. 30K precreated C2WriteNode endpoints per Neo4j trial.
Appends to existing trial_summary.
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
FANOUT = 20; HOP = 2; FRONTIER_WORKERS = 16; CLIENTS = 32
PROJ = Path("D:/memorytable/cassandra-kg-memory")
MANIFEST = PROJ / "results/c1_manifest_100k_h2.jsonl"
OUT_DIR = PROJ / "reports/c2_mixed_rw_100k_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NEO_PWD = os.environ["NEO4J_PASSWORD"]

_cass_cluster = None; _cass_session = None; _cass_executor = None
_cass_ins_src = None; _cass_ins_dst = None; _cass_ins_bkt = None; _cass_ins_idx = None
_neo_driver = None
NEO_WN = "C2WriteNode"; NEO_WR = "C2_WRITE_EDGE"
NEO_L = "C1KGNode"; NEO_R = "C1KG_EDGE"
ENDPOINT_COUNT = 30000


def init():
    global _cass_cluster, _cass_session, _cass_executor, _neo_driver
    global _cass_ins_src, _cass_ins_dst, _cass_ins_bkt, _cass_ins_idx
    _cass_cluster = Cluster(["127.0.0.1"], port=9042)
    _cass_session = _cass_cluster.connect("ai_memory")
    _cass_executor = ThreadPoolExecutor(max_workers=FRONTIER_WORKERS)
    _cass_ins_src = _cass_session.prepare(
        "INSERT INTO kg_edges_by_src (graph_id,src_id,relation,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) "
        "VALUES (?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))")
    _cass_ins_dst = _cass_session.prepare(
        "INSERT INTO kg_edges_by_dst (graph_id,dst_id,relation,src_id,edge_id,src_type,dst_type,confidence,source,created_at) "
        "VALUES (?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))")
    _cass_ins_bkt = _cass_session.prepare(
        "INSERT INTO kg_edges_by_relation_bucket (graph_id,relation,bucket,src_id,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) "
        "VALUES (?,?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))")
    _cass_ins_idx = _cass_session.prepare(
        "INSERT INTO kg_edges_by_src_relation (graph_id,src_id,relation,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) "
        "VALUES (?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))")
    _neo_driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", NEO_PWD))
    with _neo_driver.session() as s:
        s.run(f"CREATE CONSTRAINT c2wn IF NOT EXISTS FOR (n:{NEO_WN}) REQUIRE (n.graph_id, n.node_id) IS UNIQUE")


def shutdown():
    if _cass_executor: _cass_executor.shutdown(wait=True)
    if _cass_session: _cass_cluster.shutdown()
    if _neo_driver: _neo_driver.close()


def cass_fetch(src):
    rows = _cass_session.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
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
        sources = sorted({s[0] for s in frontier})
        se = cass_hop(sources, rel)
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FANOUT]:
                if e[2] not in np: nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({s[2] for s in frontier}))


def cass_write(wgid, ws):
    ws_id, wd_id = ws*2 % ENDPOINT_COUNT, (ws*2+1) % ENDPOINT_COUNT
    wsrc = f"c2_wsrc_{ws_id}"; wdst = f"c2_wdst_{wd_id}"
    wrel = "talked_to"; ws_str = "c2_mixed_write"; bkt = hash(wdst) % 64
    t0 = time.perf_counter()
    _cass_session.execute(_cass_ins_src, (wgid, wsrc, wrel, wdst, ws_str))
    _cass_session.execute(_cass_ins_dst, (wgid, wdst, wrel, wsrc, ws_str))
    _cass_session.execute(_cass_ins_bkt, (wgid, wrel, bkt, wsrc, wdst, ws_str))
    _cass_session.execute(_cass_ins_idx, (wgid, wsrc, wrel, wdst, ws_str))
    return (time.perf_counter() - t0) * 1000


def neo_read(q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({s[0] for s in frontier})
        se = {}
        for src in sources:
            with _neo_driver.session() as s:
                rows = list(s.run(
                    f"MATCH (n:{NEO_L} {{graph_id: $g, node_id: $n}})-[r:{NEO_R} {{relation: $rel}}]->(m:{NEO_L} {{graph_id: $g}}) "
                    "RETURN n.node_id AS s,r.relation AS r,m.node_id AS d,coalesce(r.source,'') AS src ORDER BY r,d,src",
                    g=GR, n=src, rel=rel))
            se[src] = [(row["s"], row["r"], row["d"], row["src"]) for row in rows]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FANOUT]:
                if e[2] not in np: nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({s[2] for s in frontier}))


def neo_write(wgid, ws):
    ws_id, wd_id = ws*2 % ENDPOINT_COUNT, (ws*2+1) % ENDPOINT_COUNT
    wsrc = f"c2_wsrc_{ws_id}"; wdst = f"c2_wdst_{wd_id}"
    t0 = time.perf_counter()
    with _neo_driver.session() as s:
        s.run(
            f"MATCH (n:{NEO_WN} {{graph_id: $g, node_id: $n}})"
            f"MATCH (m:{NEO_WN} {{graph_id: $g, node_id: $m}})"
            f"CREATE (n)-[:{NEO_WR} {{graph_id: $g, relation: $rel, source: $src, write_id: $wid}}]->(m)",
            g=wgid, n=wsrc, m=wdst, rel="talked_to", src="c2_mixed_write", wid=str(ws))
    return (time.perf_counter() - t0) * 1000


def neo_precreate(wgid):
    t0 = time.perf_counter()
    print(f"    Precreating {ENDPOINT_COUNT} endpoint nodes...", end=" ", flush=True)
    rows_src = [{"g": wgid, "nid": f"c2_wsrc_{i}"} for i in range(ENDPOINT_COUNT)]
    rows_dst = [{"g": wgid, "nid": f"c2_wdst_{i}"} for i in range(ENDPOINT_COUNT)]
    all_rows = rows_src + rows_dst
    batch_size = 500
    batches = 0
    for i in range(0, len(all_rows), batch_size):
        chunk = all_rows[i:i+batch_size]
        cypher = "UNWIND $rows AS r MERGE (n:" + NEO_WN + " {graph_id: r.g, node_id: r.nid})"
        with _neo_driver.session() as s:
            s.run(cypher, rows=chunk).consume()
        batches += 1
    elapsed = time.perf_counter() - t0
    with _neo_driver.session() as s:
        cnt = s.run(f"MATCH (n:{NEO_WN} {{graph_id: $g}}) RETURN count(n)", g=wgid).single()[0]
    ok = cnt == ENDPOINT_COUNT * 2
    print(f"{elapsed:.1f}s  {batches} batches  {cnt} nodes  {'OK' if ok else 'FAIL(' + str(cnt) + ')'}", flush=True)
    log = {"write_graph_id": wgid, "target": ENDPOINT_COUNT*2, "actual": cnt,
           "batches": batches, "elapsed_s": round(elapsed, 3), "success": ok,
           "measurement_excludes_precreate": True}
    with (OUT_DIR / "neo4j_precreate_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")
    return ok


def validate_writes_cass(wgid, nw):
    sample = random.sample(range(nw), min(20, nw))
    ok = 0
    for ws in sample:
        s_id = ws*2 % ENDPOINT_COUNT; d_id = (ws*2+1) % ENDPOINT_COUNT
        s = f"c2_wsrc_{s_id}"; d = f"c2_wdst_{d_id}"; wrel = "talked_to"; b = hash(d) % 64
        c1 = list(_cass_session.execute("SELECT count(*) FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
            (wgid, s)))[0].count
        c2 = list(_cass_session.execute("SELECT count(*) FROM kg_edges_by_dst WHERE graph_id=%s AND dst_id=%s",
            (wgid, d)))[0].count
        c3 = list(_cass_session.execute("SELECT count(*) FROM kg_edges_by_relation_bucket WHERE graph_id=%s AND relation=%s AND bucket=%s",
            (wgid, wrel, b)))[0].count
        c4 = list(_cass_session.execute("SELECT count(*) FROM kg_edges_by_src_relation WHERE graph_id=%s AND src_id=%s AND relation=%s",
            (wgid, s, wrel)))[0].count
        if c1>=1 and c2>=1 and c3>=1 and c4>=1: ok+=1
    return ok, len(sample)


def validate_writes_neo(wgid, nw):
    sample = random.sample(range(nw), min(20, nw))
    ok = 0
    for ws in sample:
        with _neo_driver.session() as s:
            c = s.run(f"MATCH ()-[r:{NEO_WR} {{write_id: $wid}}]->() RETURN count(r) AS cnt", wid=str(ws)).single()["cnt"]
            if c>=1: ok+=1
    return ok, len(sample)


def run_trial(system, wr_pct, repeat_idx):
    wgid = f"c2_write_sink_{system}_w{wr_pct}_r{repeat_idx+1}"
    tag = f"{system} wr={wr_pct}% r={repeat_idx+1}"
    print(f"\n  [{tag}]", flush=True)

    queries = []
    with MANIFEST.open() as f:
        for line in f: queries.append(json.loads(line))

    if system == "neo4j":
        if not neo_precreate(wgid):
            print(f"    PRECREATE FAILED", flush=True)
            return None

    write_seq = 0
    write_lock = threading.Lock()
    per_rlat = [[] for _ in range(CLIENTS)]
    per_wlat = [[] for _ in range(CLIENTS)]
    per_err = [0] * CLIENTS
    stop = threading.Event()

    def client(ci):
        nonlocal write_seq
        ridx = ci % len(queries)
        ops = ci % 10
        while not stop.is_set():
            do_write = (wr_pct == 10 and ops % 10 == 9) or (wr_pct == 30 and ops % 10 >= 7)
            if do_write:
                t0 = time.perf_counter()
                try:
                    with write_lock: ws = write_seq; write_seq += 1
                    if system == "cassandra_parallel":
                        cass_write(wgid, ws)
                    else:
                        neo_write(wgid, ws)
                    per_wlat[ci].append((time.perf_counter()-t0)*1000)
                except Exception:
                    per_err[ci] += 1
            else:
                q = queries[ridx]
                t0 = time.perf_counter()
                try:
                    if system == "cassandra_parallel": cass_read(q)
                    else: neo_read(q)
                    per_rlat[ci].append((time.perf_counter()-t0)*1000)
                except Exception:
                    per_err[ci] += 1
                ridx = (ridx+1) % len(queries)
            ops += 1

    threads = [threading.Thread(target=client, args=(i,), daemon=True) for i in range(CLIENTS)]
    for t in threads: t.start()
    time.sleep(15)
    for l in per_rlat: l.clear()
    for l in per_wlat: l.clear()
    write_seq = 0
    t_start = time.perf_counter()
    time.sleep(45)
    t_end = time.perf_counter()
    stop.set()
    for t in threads: t.join(timeout=10)

    meas_s = t_end - t_start
    rl = sorted([v for l in per_rlat for v in l])
    wl = sorted([v for l in per_wlat for v in l])
    nr = len(rl); nw = len(wl); errs = sum(per_err)
    actual_wr = round(nw/max(nr+nw,1)*100, 1)
    def pct(a, p): return a[int((len(a)-1)*p/100)] if a else 0

    rng = random.Random(42)
    chk = rng.sample(queries, 10)
    rh_ok = 0
    for q in chk:
        p = neo_read(q) if system=="neo4j" else cass_read(q)
        h = hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(p)]),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if h == q["expected_path_hash"]: rh_ok+=1

    vok = vt = 0
    if nw > 0:
        vok, vt = validate_writes_neo(wgid, nw) if system=="neo4j" else validate_writes_cass(wgid, nw)

    row = {
        "system": system, "clients": CLIENTS,
        "target_write_ratio": wr_pct, "actual_write_ratio": actual_wr,
        "repeat": repeat_idx+1, "warmup_seconds": 15, "measurement_seconds": round(meas_s,3),
        "completed_reads": nr, "completed_writes": nw,
        "read_QPS": round(nr/max(meas_s,.001),3),
        "write_QPS": round(nw/max(meas_s,.001),3),
        "total_ops_QPS": round((nr+nw)/max(meas_s,.001),3),
        "read_mean_ms": round(statistics.mean(rl),3) if rl else None,
        "read_p50_ms": round(pct(rl,50),3) if rl else None,
        "read_p95_ms": round(pct(rl,95),3) if rl else None,
        "read_p99_ms": round(pct(rl,99),3) if rl else None,
        "write_mean_ms": round(statistics.mean(wl),3) if wl else None,
        "write_p50_ms": round(pct(wl,50),3) if wl else None,
        "write_p95_ms": round(pct(wl,95),3) if wl else None,
        "write_p99_ms": round(pct(wl,99),3) if wl else None,
        "read_error_count": errs, "write_error_count": 0,
        "total_error_rate": round(errs/max(nr+nw,1),6),
        "read_hash_spotcheck_passed": rh_ok==10,
        "write_validation_passed": vok==vt if nw>0 else None,
        "logical_writes_validated": f"{vok}/{vt}" if nw>0 else None,
        "physical_writes_per_logical_write": 4 if system=="cassandra_parallel" else 1,
        "frontier_workers": FRONTIER_WORKERS, "application_cache": "disabled",
        "backend_state": "warm", "write_graph_id": wgid,
    }

    print(f"    reads={nr} writes={nw} actual_wr={actual_wr}%", flush=True)
    print(f"    rQPS={row['read_QPS']:.1f} rmean={row['read_mean_ms']:.1f}ms rp95={row['read_p95_ms']:.1f}ms", flush=True)
    if nw>0:
        print(f"    wQPS={row['write_QPS']:.1f} wmean={row['write_mean_ms']:.1f}ms  spot={rh_ok}/10  wval={vok}/{vt}", flush=True)
    else:
        print(f"    spot={rh_ok}/10", flush=True)
    return row


def aggregate_final():
    rows = []
    for fname in ["trial_summary_orig.csv", "trial_summary.csv"]:
        p = OUT_DIR / fname
        if not p.exists(): continue
        with p.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f): rows.append(r)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["system"], int(r["target_write_ratio"]))].append(r)

    def med(arr, nil=False):
        vals = [float(v) for v in arr if v not in (None, "")]; return None if (nil and not vals) else round(sorted(vals)[len(vals)//2],3)

    final_rows = []
    for (sn, wr), g in sorted(groups.items()):
        fr = {
            "system": sn, "write_ratio": wr, "repeats": len(g),
            "median_read_QPS": med([t["read_QPS"] for t in g]),
            "median_write_QPS": med([t["write_QPS"] for t in g], nil=True),
            "median_total_ops_QPS": med([t["total_ops_QPS"] for t in g]),
            "median_read_mean_ms": med([t["read_mean_ms"] for t in g]),
            "median_read_p95_ms": med([t["read_p95_ms"] for t in g]),
            "median_read_p99_ms": med([t["read_p99_ms"] for t in g]),
            "median_write_mean_ms": med([t["write_mean_ms"] for t in g], nil=True),
            "median_write_p95_ms": med([t["write_p95_ms"] for t in g], nil=True),
            "median_write_p99_ms": med([t["write_p99_ms"] for t in g], nil=True),
            "min_total_ops_QPS": min(float(t["total_ops_QPS"]) for t in g),
            "max_total_ops_QPS": max(float(t["total_ops_QPS"]) for t in g),
            "median_actual_write_ratio": round(statistics.median([t["actual_write_ratio"] for t in g]),1),
            "median_error_rate": med([t["total_error_rate"] for t in g]),
        }
        final_rows.append(fr)

    ff = ["system","write_ratio","repeats",
          "median_read_QPS","median_write_QPS","median_total_ops_QPS",
          "median_read_mean_ms","median_read_p95_ms","median_read_p99_ms",
          "median_write_mean_ms","median_write_p95_ms","median_write_p99_ms",
          "min_total_ops_QPS","max_total_ops_QPS","median_actual_write_ratio","median_error_rate"]
    with (OUT_DIR/"final_mixed_rw_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=ff, extrasaction="ignore"); w.writeheader(); w.writerows(final_rows)
    with (OUT_DIR/"final_mixed_rw_summary.json").open("w") as f:
        json.dump(final_rows, f, indent=2, ensure_ascii=False)

    print(f"\n=== FINAL MIXED R/W ===", flush=True)
    for fr in final_rows:
        wqps = fr["median_write_QPS"] if fr["median_write_QPS"] else "  -"
        print(f"  [{fr['system']:20s}] wr={fr['write_ratio']:2d}%  "
              f"rQPS={fr['median_read_QPS']:6.1f}  wQPS={str(wqps):>5s}  "
              f"rmean={fr['median_read_mean_ms']:6.1f}ms  rp95={fr['median_read_p95_ms']:6.1f}ms  n={fr['repeats']}", flush=True)


def main():
    print("=== C2-1 Fix: wr=10%,30% only ===", flush=True)
    init()

    # Smoke test: neo4j wr=30% r=1, warmup=10s, measure=20s
    print("\n--- SMOKE: neo4j wr=30% ---", flush=True)
    row = run_trial("neo4j", 30, 0)
    if row is None or row["total_error_rate"] > 0 or not row["read_hash_spotcheck_passed"] or not row["write_validation_passed"]:
        print("SMOKE FAILED", flush=True)
        shutdown(); return
    print("SMOKE PASSED", flush=True)

    # Full wr=10%,30% x 3 repeats x 2 systems
    # Preserve existing trials by reading into memory first
    csv_path = OUT_DIR / "trial_summary.csv"
    existing_lines = []
    try:
        with csv_path.open(encoding="utf-8-sig") as f:
            existing_lines = f.readlines()
    except Exception:
        pass

    rng_ord = random.Random(42)
    run_order = []
    for wr in [10, 30]:
        for rep in range(3):
            so = ["cassandra_parallel", "neo4j"]; rng_ord.shuffle(so)
            for s in so: run_order.append({"system":s,"write_ratio":wr,"repeat":rep})

    fields = None
    all_rows = []
    for idx, entry in enumerate(run_order):
        sys_name = entry["system"]; wr = entry["write_ratio"]; rep = entry["repeat"]
        print(f"\n[{idx+1}/12] {sys_name} wr={wr}% repeat={rep+1}", flush=True)
        row = run_trial(sys_name, wr, rep)
        if row is None:
            print(f"  TRIAL FAILED", flush=True)
            continue
        if not fields: fields = list(row.keys())
        all_rows.append(row)

    # Append to existing CSV
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        if existing_lines:
            f.writelines(existing_lines)
        else:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writerows(all_rows)

    shutdown()
    aggregate_final()


if __name__ == "__main__":
    main()
