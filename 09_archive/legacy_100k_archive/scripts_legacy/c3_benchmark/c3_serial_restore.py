"""
Serial restore: TRUNCATE → serial import 100K → verify → hash gate → serial import 1M → verify → hash gate.
"""
import csv, hashlib, json, time
from pathlib import Path
from collections import Counter
from cassandra.cluster import Cluster

PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT  = PROJ / "reports/data_recovery"
OUT.mkdir(parents=True, exist_ok=True)
GR_100K = "synth_100000_1781447372"
GR_1M   = "c3_scale_1M_seed42"

c = Cluster(["127.0.0.1"], port=9042)
s = c.connect("ai_memory")

# ── TRUNCATE ──
print("[0] TRUNCATE kg_edges_by_src")
s.execute("TRUNCATE ai_memory.kg_edges_by_src")
print("    Done")

ins = s.prepare(
    "INSERT INTO kg_edges_by_src (graph_id,src_id,relation,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) "
    "VALUES (?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))"
)


def serial_import(csv_path, gid):
    edges = []
    with csv_path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            edges.append((r["graph_id"], r["src_id"], r["relation"], r["dst_id"], r["source"]))
    n = len(edges)
    print(f"    CSV: {n} edges")

    t0 = time.perf_counter()
    for i, (g, sid, rel, d, src) in enumerate(edges):
        s.execute(ins, (g, sid, rel, d, src))
        if (i+1) % 50000 == 0:
            print(f"    {i+1}/{n} ({time.perf_counter()-t0:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"    Done: {n} in {elapsed:.0f}s ({int(n/max(elapsed,.001))} e/s)", flush=True)
    return edges, elapsed


def verify(edges, gid):
    csv_set = set((e[1], e[2], e[3], e[4]) for e in edges)
    srcs = sorted(set(e[1] for e in edges))
    cass_set = set()
    cass_raw = 0
    for src in srcs:
        rows = s.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (gid, src))
        for r in rows:
            cass_set.add((str(r.src_id), str(r.relation), str(r.dst_id), str(r.source or "")))
            cass_raw += 1
    dup = cass_raw - len(cass_set)
    miss = len(csv_set - cass_set)
    extra = len(cass_set - csv_set)
    sm = {"csv": len(csv_set), "raw": cass_raw, "distinct": len(cass_set),
          "duplicates": dup, "missing": miss, "extra": extra}
    print(f"    CSV={len(csv_set)} raw={cass_raw} distinct={len(cass_set)} dup={dup} miss={miss} extra={extra}")
    return sm, dup == 0 and miss == 0


def hash_gate(gid, manifest_path, nq):
    queries = [json.loads(line) for line in open(manifest_path)][:nq]
    empty = 0; mm = 0
    for i, q in enumerate(queries):
        frontier = {(q["seed_id"], (q["seed_id"],), ())}
        for rel in q["relation_path"]:
            sources = sorted({f[0] for f in frontier}); se = {}
            for src in sources:
                rows = s.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (gid, src))
                se[src] = [(str(r.src_id), str(r.relation), str(r.dst_id), str(r.source or "")) for r in rows if r.relation == rel]
            nf = set()
            for src, np, ep in frontier:
                for e in se.get(src, [])[:20]:
                    if e[2] not in np: nf.add((e[2], np+(e[2],), ep+(e[3],)))
            frontier = nf
        paths = tuple(sorted({f[2] for f in frontier}))
        if len(paths) == 0:
            empty += 1
        else:
            h = hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True).encode()).hexdigest()
            if h != q["expected_path_hash"]: mm += 1
        if (i+1) % 64 == 0:
            print(f"    {i+1}/{nq} empty={empty} mismatch={mm}", flush=True)
    r = {"checked": nq, "empty": empty, "mismatch": mm, "all_pass": empty == 0 and mm == 0}
    print(f"    Hash gate: empty={empty} mismatch={mm} {'PASS' if r['all_pass'] else 'FAIL'}")
    return r


# ── 100K ──
print("\n[1] Serial import 100K")
edges_100k, t_100k = serial_import(PROJ / "results/c1_source_100k.csv", GR_100K)

print("\n[2] Verify 100K")
sm_100k, clean_100k = verify(edges_100k, GR_100K)
with (OUT / "by_src_restore_100k_summary.json").open("w") as f:
    json.dump(sm_100k, f, indent=2)

if not clean_100k:
    print("ABORT: 100K not clean"); c.shutdown(); exit(1)

print("\n[3] C1 hash gate (256)")
hg_100k = hash_gate(GR_100K, PROJ / "results/c1_manifest_100k_h2.jsonl", 256)
with (OUT / "c1_post_restore_hash_gate.json").open("w") as f:
    json.dump(hg_100k, f, indent=2)

if not hg_100k["all_pass"]:
    print("ABORT: C1 hash gate failed"); c.shutdown(); exit(1)

# ── 1M ──
# Pre-check 1M CSV is scale-controlled
print("\n[4] Pre-check 1M CSV")
edges_1m_check = []
with (PROJ / "results/c3_source_scale_1M.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        edges_1m_check.append(r)
srcs = Counter(e["src_id"] for e in edges_1m_check)
deg = sorted(srcs.values())
assert len(edges_1m_check) >= 900000
assert deg[len(deg)//2] <= 25, f"p50 outdegree expected ~20, got {deg[len(deg)//2]}"
assert deg[-1] <= 500, f"max outdegree expected ~400, got {deg[-1]}"
assert len(srcs) > 10000
print(f"    OK: {len(edges_1m_check)} edges, {len(srcs)} sources, outdegree {deg[0]}-{deg[-1]} p50={deg[len(deg)//2]}")

print("\n[5] Serial import 1M (est. ~30 min)")
edges_1m, t_1m = serial_import(PROJ / "results/c3_source_scale_1M.csv", GR_1M)

print("\n[6] Verify 1M")
sm_1m, clean_1m = verify(edges_1m, GR_1M)
with (OUT / "by_src_restore_1m_summary.json").open("w") as f:
    json.dump(sm_1m, f, indent=2)

if not clean_1m:
    print("ABORT: 1M not clean"); c.shutdown(); exit(1)

print("\n[7] C3 hash gate (64)")
hg_1m = hash_gate(GR_1M, PROJ / "results/c3_manifest_scale_1m_h2.jsonl", 64)
with (OUT / "c3_post_restore_hash_gate.json").open("w") as f:
    json.dump(hg_1m, f, indent=2)

# ── Log ──
log = f"""# kg_edges_by_src Recovery Log (Serial)
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 100K
- CSV: c1_source_100k.csv → {GR_100K}
- Method: serial, single connection, prepared statement
- Imported: {len(edges_100k)} edges in {t_100k:.0f}s ({int(len(edges_100k)/max(t_100k,.001))} e/s)
- Verify: raw={sm_100k['raw']} distinct={sm_100k['distinct']} dup={sm_100k['duplicates']} miss={sm_100k['missing']}
- C1 gate: {hg_100k['checked']} checked, empty={hg_100k['empty']}, mismatch={hg_100k['mismatch']}

## 1M
- CSV: c3_source_scale_1M.csv → {GR_1M}
- Method: serial, single connection, prepared statement
- Imported: {len(edges_1m)} edges in {t_1m:.0f}s ({int(len(edges_1m)/max(t_1m,.001))} e/s)
- Verify: raw={sm_1m['raw']} distinct={sm_1m['distinct']} dup={sm_1m['duplicates']} miss={sm_1m['missing']}
- C3 gate: {hg_1m['checked']} checked, empty={hg_1m['empty']}, mismatch={hg_1m['mismatch']}
"""
with (OUT / "by_src_restore_log.md").open("w") as f:
    f.write(log)

c.shutdown()
print(f"\n{'='*50}")
print("ALL DONE")
print(f"100K: {'PASS' if clean_100k and hg_100k['all_pass'] else 'FAIL'}")
print(f"1M:   {'PASS' if clean_1m and hg_1m['all_pass'] else 'FAIL'}")
