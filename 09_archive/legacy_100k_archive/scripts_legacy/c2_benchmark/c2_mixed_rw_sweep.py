"""
C2-1: full read/write ratio sweep. Cassandra parallel vs Neo4j.
3 write ratios x 3 repeats, 32 clients, warmup=15s, measure=45s.
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
CLIENTS = 32

PROJ = Path("D:/memorytable/cassandra-kg-memory")
MANIFEST = PROJ / "results/c1_manifest_100k_h2.jsonl"
OUT_DIR = PROJ / "reports/c2_mixed_rw_100k_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEO4J_PWD = os.environ["NEO4J_PASSWORD"]
NEO_L = "C1KGNode"
NEO_R = "C1KG_EDGE"
NEO_WNODE = "C2WriteNode"
NEO_WREL = "C2_WRITE_EDGE"

_cass_cluster = None
_cass_session = None
_cass_executor = None
_cass_ins_src = None
_cass_ins_dst = None
_cass_ins_bkt = None
_cass_ins_idx = None
_neo_driver = None


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
    _neo_driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", NEO4J_PWD))
    with _neo_driver.session() as s:
        s.run(f"CREATE CONSTRAINT c2w_node IF NOT EXISTS FOR (n:{NEO_WNODE}) REQUIRE (n.graph_id, n.node_id) IS UNIQUE")


def shutdown():
    if _cass_executor: _cass_executor.shutdown(wait=True)
    if _cass_session: _cass_cluster.shutdown()
    if _neo_driver: _neo_driver.close()


def cass_fetch(src):
    rows = _cass_session.execute(
        "SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (GR, src))
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


def cass_write(wgid, ws):  # ws = write_seq int
    wsrc = f"c2_wsrc_{ws}"; wdst = f"c2_wdst_{ws}"
    wrel = "talked_to"; wsrc_str = "c2_mixed_write"
    bkt = hash(wdst) % 64
    t0 = time.perf_counter()
    _cass_session.execute(_cass_ins_src, (wgid, wsrc, wrel, wdst, wsrc_str))
    _cass_session.execute(_cass_ins_dst, (wgid, wdst, wrel, wsrc, wsrc_str))
    _cass_session.execute(_cass_ins_bkt, (wgid, wrel, bkt, wsrc, wdst, wsrc_str))
    _cass_session.execute(_cass_ins_idx, (wgid, wsrc, wrel, wdst, wsrc_str))
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
    wsrc = f"c2_wsrc_{ws}"; wdst = f"c2_wdst_{ws}"; wrel = "talked_to"
    t0 = time.perf_counter()
    with _neo_driver.session() as s:
        s.run(
            f"MATCH (n:{NEO_WNODE} {{graph_id: $g, node_id: $s}})"
            f"MATCH (m:{NEO_WNODE} {{graph_id: $g, node_id: $d}})"
            f"CREATE (n)-[:{NEO_WREL} {{graph_id: $g, relation: $rel, source: $src, write_id: $wid}}]->(m)",
            g=wgid, s=wsrc, d=wdst, rel=wrel, src="c2_mixed_write", wid=str(ws))
    return (time.perf_counter() - t0) * 1000


def precreate_neo_nodes(wgid):
    nodes = []
    for i in range(10000):
        nodes.append({"g": wgid, "nid": f"c2_wsrc_{i}"})
        nodes.append({"g": wgid, "nid": f"c2_wdst_{i}"})
    for i in range(0, len(nodes), 5000):
        with _neo_driver.session() as s:
            s.run(f"UNWIND $rows AS r MERGE (n:{NEO_WNODE} {{graph_id: r.g, node_id: r.nid}})", rows=nodes[i:i+5000]).consume()


def validate_writes_cass(wgid, total_writes):
    sample = random.sample(range(total_writes), min(20, total_writes))
    ok = 0
    for ws in sample:
        s = f"c2_wsrc_{ws}"; d = f"c2_wdst_{ws}"; wrel = "talked_to"; b = hash(d) % 64
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


def validate_writes_neo(wgid, total_writes):
    sample = random.sample(range(total_writes), min(20, total_writes))
    ok = 0
    for ws in sample:
        with _neo_driver.session() as s:
            c = s.run(f"MATCH ()-[r:{NEO_WREL} {{write_id: $wid}}]->() RETURN count(r)", wid=str(ws)).single()["cnt"]
            if c>=1: ok+=1
    return ok, len(sample)


def run_trial(system, wr_pct, repeat_idx):
    wgid = f"c2_wr{wr_pct}_r{repeat_idx+1}"
    tag = f"{system} wr={wr_pct}% r={repeat_idx+1}"
    print(f"\n  [{tag}]", flush=True)

    queries = []
    with MANIFEST.open() as f:
        for line in f: queries.append(json.loads(line))

    if system == "neo4j":
        precreate_neo_nodes(wgid)

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
            if wr_pct == 0:
                do_write = False
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
                    if system == "cassandra_parallel":
                        cass_read(q)
                    else:
                        neo_read(q)
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

    # Spotcheck reads
    rng = random.Random(42)
    chk = rng.sample(queries, 10)
    rh_ok = 0
    for q in chk:
        p = neo_read(q) if system=="neo4j" else cass_read(q)
        h = hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(p)]),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if h == q["expected_path_hash"]: rh_ok+=1

    # Write validation
    if nw > 0:
        vok, vt = (validate_writes_neo(wgid, nw) if system=="neo4j" else validate_writes_cass(wgid, nw))
    else:
        vok, vt = 0, 0

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


def main():
    print("=== C2-1 Mixed R/W Sweep ===", flush=True)
    print(f"Systems: cassandra_parallel, neo4j  Clients={CLIENTS}", flush=True)
    print(f"Write ratios: 0%/10%/30% x 3 repeats", flush=True)
    print(f"Warmup=15s  Measure=45s  18 trials total", flush=True)

    init()

    env = {"os": platform.platform(), "python": sys.version, "cpu_logical": os.cpu_count(),
           "frontier_workers": FRONTIER_WORKERS, "graph_id": GR}
    with (OUT_DIR/"environment.json").open("w") as f: json.dump(env, f, indent=2)
    with (OUT_DIR/"run_config.json").open("w") as f:
        json.dump({"clients":CLIENTS,"write_ratios":[0,10,30],"repeats":3,"warmup_s":15,"measurement_s":45,
                   "hop":HOP,"fanout":FANOUT,"cycle_policy":"path"}, f, indent=2)

    # Generate run order
    rng_ord = random.Random(42)
    run_order = []
    for wr in [0, 10, 30]:
        for rep in range(3):
            so = ["cassandra_parallel", "neo4j"]; rng_ord.shuffle(so)
            for s in so: run_order.append({"system":s,"write_ratio":wr,"repeat":rep})
    with (OUT_DIR/"run_order.jsonl").open("w") as f:
        for e in run_order: f.write(json.dumps(e, ensure_ascii=False)+"\n")

    csv_path = OUT_DIR / "trial_summary.csv"
    jsl_path = OUT_DIR / "trial_summary.jsonl"
    failures_path = OUT_DIR / "failures.jsonl"
    spot_path = OUT_DIR / "correctness_spotcheck.jsonl"
    wval_path = OUT_DIR / "write_validation.jsonl"

    for fn in [csv_path, jsl_path, failures_path]: fn.unlink(missing_ok=True)

    all_rows = []
    all_ok = True
    total = len(run_order)
    fields = None

    for idx, entry in enumerate(run_order):
        sys_name = entry["system"]
        wr = entry["write_ratio"]
        rep = entry["repeat"]
        print(f"\n[{idx+1}/{total}] {sys_name} wr={wr}% repeat={rep+1}", flush=True)

        row = run_trial(sys_name, wr, rep)
        all_rows.append(row)
        if not fields: fields = list(row.keys())

        write_hdr = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if write_hdr: w.writeheader()
            w.writerow(row)
        with jsl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")

        with spot_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"system":sys_name,"wr":wr,"repeat":rep+1,
                                "read_hash_pass":row["read_hash_spotcheck_passed"]}, ensure_ascii=False)+"\n")

        if row["write_validation_passed"] is not None:
            with wval_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"system":sys_name,"wr":wr,"repeat":rep+1,
                                    "validation":row["logical_writes_validated"],
                                    "physical_writes":row["physical_writes_per_logical_write"]}, ensure_ascii=False)+"\n")

        if not row["read_hash_spotcheck_passed"]:
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"trial":idx+1,"system":sys_name,"wr":wr,"rep":rep+1,
                                    "reason":"hash_spotcheck_failed"}, ensure_ascii=False)+"\n")
            all_ok = False
        if row["write_validation_passed"] is False:
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"trial":idx+1,"system":sys_name,"wr":wr,"rep":rep+1,
                                    "reason":"write_validation_failed"}, ensure_ascii=False)+"\n")
            all_ok = False
        if row["total_error_rate"] > 0:
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"trial":idx+1,"system":sys_name,"wr":wr,"rep":rep+1,
                                    "reason":f"error_rate={row['total_error_rate']}"}, ensure_ascii=False)+"\n")
            all_ok = False

    shutdown()

    # Aggregate final summary
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["system"], r["target_write_ratio"])].append(r)

    final_rows = []
    for (sn, wr), g in sorted(groups.items()):
        nr = len(g)
        def med(arr, nil=False):
            vals = [float(v) for v in arr if v is not None]
            if not vals: return None if nil else 0
            return round(sorted(vals)[len(vals)//2], 3)
        fr = {
            "system": sn, "write_ratio": wr, "repeats": nr,
            "median_read_QPS": med([r["read_QPS"] for r in g]),
            "median_write_QPS": med([r["write_QPS"] for r in g], nil=True),
            "median_total_ops_QPS": med([r["total_ops_QPS"] for r in g]),
            "median_read_mean_ms": med([r["read_mean_ms"] for r in g]),
            "median_read_p95_ms": med([r["read_p95_ms"] for r in g]),
            "median_read_p99_ms": med([r["read_p99_ms"] for r in g]),
            "median_write_mean_ms": med([r["write_mean_ms"] for r in g], nil=True),
            "median_write_p95_ms": med([r["write_p95_ms"] for r in g], nil=True),
            "median_write_p99_ms": med([r["write_p99_ms"] for r in g], nil=True),
            "min_total_ops_QPS": min(float(r["total_ops_QPS"]) for r in g),
            "max_total_ops_QPS": max(float(r["total_ops_QPS"]) for r in g),
            "median_actual_write_ratio": round(statistics.median([r["actual_write_ratio"] for r in g]),1),
            "median_error_rate": med([r["total_error_rate"] for r in g]),
        }
        final_rows.append(fr)

    ff = ["system","write_ratio","repeats",
          "median_read_QPS","median_write_QPS","median_total_ops_QPS",
          "median_read_mean_ms","median_read_p95_ms","median_read_p99_ms",
          "median_write_mean_ms","median_write_p95_ms","median_write_p99_ms",
          "min_total_ops_QPS","max_total_ops_QPS",
          "median_actual_write_ratio","median_error_rate"]
    with (OUT_DIR/"final_mixed_rw_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=ff, extrasaction="ignore")
        w.writeheader(); w.writerows(final_rows)
    with (OUT_DIR/"final_mixed_rw_summary.json").open("w") as f:
        json.dump(final_rows, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}", flush=True)
    print("FINAL MIXED R/W SUMMARY", flush=True)
    for fr in final_rows:
        print(f"  [{fr['system']:20s}] wr={fr['write_ratio']:2d}%  "
              f"rQPS={fr['median_read_QPS']:6.1f}  wQPS={fr['median_write_QPS'] if fr['median_write_QPS'] else '  -  '}  "
              f"rmean={fr['median_read_mean_ms']:6.1f}ms  rp95={fr['median_read_p95_ms']:6.1f}ms  "
              f"err={fr['median_error_rate']}  n={fr['repeats']}", flush=True)
    print(f"\n{'ALL PASS' if all_ok else 'FAILURES — check failures.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
