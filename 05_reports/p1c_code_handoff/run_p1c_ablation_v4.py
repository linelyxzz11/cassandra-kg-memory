"""P1-C v4: Compact Component Ablation. Internal consistency only. All variants on same data version."""
import csv, json, hashlib, math, random, statistics, time, sys, os
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE = Path("D:/memorytable/cassandra-kg-memory")
OUT = BASE / "05_reports/p1_compact_component_ablation"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE / "03_src/memory/locomo_pipeline/retrieval"))
from locomo_retrieval_sample_scoped import BM25Retriever

ART = BASE / "scripts/experiments/artifacts"
MEM_REC = BASE / "01_data/locomo_memory_records.csv"
MEM_FEAT = BASE / "02_artifacts/p3_memory_features.csv"
QUERIES_CSV = ART / "p3_queries.csv"
DENSE_CACHE = ART / "frozen_dense_scores_long.csv"

BM25_K1 = 1.5; BM25_B = 0.75; TOP_N = 50; ALPHA = 0.6; SEED = 20260723

print("Loading data...", flush=True)
mem_records = {r["memory_id"]: r for r in csv.DictReader(MEM_REC.open(encoding="utf-8-sig"))}
mem_features = {r["memory_id"]: r for r in csv.DictReader(MEM_FEAT.open(encoding="utf-8-sig"))}
all_queries = list(csv.DictReader(QUERIES_CSV.open(encoding="utf-8-sig")))
test_convs = {"conv-26","conv-30","conv-41","conv-43","conv-44","conv-47","conv-49","conv-50"}
test_queries = [q for q in all_queries if q.get("conversation_id","") in test_convs and q.get("category","") != "5"]
cat1_4_queries = [q for q in all_queries if q.get("category","") in ("1","2","3","4")]
print(f"  Test: {len(test_queries)}, Cat1-4: {len(cat1_4_queries)}")

dense_scores = defaultdict(dict)
with DENSE_CACHE.open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): dense_scores[r["query_id"]][r["memory_id"]] = float(r["score"])

CAT_MAP = {"1":"cat1_multi-hop","2":"cat2_temporal","3":"cat3_open-domain","4":"cat4_single-hop"}

conv_memories = defaultdict(list)
for mid, rec in mem_records.items():
    conv_memories[rec.get("sample_id","")].append((mid, rec))

def build_doc(mid, components):
    rec = mem_records.get(mid, {}); fea = mem_features.get(mid, {})
    raw = rec.get("text", ""); extras = []
    for c in components:
        if c == "T":
            v = rec.get("timestamp","")
        else:
            v = fea.get({"E":"entities","R":"relations","K":"keywords"}.get(c,""),"")
        if v.strip(): extras.append(f"{c}: {v}")
    return raw + "\n" + "\n".join(extras) if extras else raw

VARIANTS = {
    "Raw": [], "RawE": ["E"], "RawR": ["R"], "RawK": ["K"],
    "RawER": ["E","R"], "RawEK": ["E","K"], "RawRK": ["R","K"],
    "RawERK": ["E","R","K"], "RawERKT": ["E","R","K","T"],
}

print("\nBuilding BM25 indices...", flush=True)
token_stats = {}; bm25_rankings = {}
for vname, comps in VARIANTS.items():
    print(f"  {vname}...", end=" ", flush=True)
    all_tokens = []; bm25_rankings[vname] = {}; cqm = defaultdict(list)
    for q in test_queries + cat1_4_queries:
        cid = q.get("conversation_id","")
        if cid: cqm[cid].append(q)
    for conv_id, mems in conv_memories.items():
        if not mems or conv_id not in cqm: continue
        mids, docs = [], []
        for mid, rec in mems:
            doc = build_doc(mid, comps); mids.append(mid); docs.append(doc)
            all_tokens.append(len(doc.split()))
        bm = BM25Retriever(k1=BM25_K1, b=BM25_B); bm.fit(docs)
        for q in cqm[conv_id]:
            idx, sc = bm.search(q["question"], TOP_N)
            bm25_rankings[vname][q["query_id"]] = [(mids[i], sc[j]) for j, i in enumerate(idx)]
    ts = {"mean":np.mean(all_tokens),"median":np.median(all_tokens),"p95":np.percentile(all_tokens,95)}
    token_stats[vname] = ts
    print(f"mean={ts['mean']:.0f} p95={ts['p95']:.0f}", flush=True)

with (OUT/"p1c_representation_stats.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["variant","mean_tokens","median_tokens","p95_tokens"]); w.writeheader()
    for vn, ts in token_stats.items(): w.writerow({"variant":vn,"mean_tokens":ts["mean"],"median_tokens":ts["median"],"p95_tokens":ts["p95"]})

def get_conv_ids(qid):
    for q in test_queries + cat1_4_queries:
        if q["query_id"] == qid: return [mid for mid,_ in conv_memories.get(q.get("conversation_id",""),[])]
    return list(mem_records.keys())

def zscore_fuse(qid, bm25_rak):
    cids = get_conv_ids(qid)
    da = [(m, dense_scores.get(qid,{}).get(m,0.0)) for m in cids]
    da.sort(key=lambda x:-x[1]); dt = da[:TOP_N]
    bs = list(bm25_rak)[:TOP_N]
    ai = set(m for m,_ in dt) | set(m for m,_ in bs)
    dv = [s for _,s in dt]; bv = [s for _,s in bs]
    dm = np.mean(dv) if dv else 0; ds = np.std(dv,ddof=0) or 1.0
    bm = np.mean(bv) if bv else 0; bst = np.std(bv,ddof=0) or 1.0
    dmin = min(dv) if dv else 0; bmin = min(bv) if bv else 0
    dm_ = dict(dt); bm_ = dict(bs)
    fu = {}
    for m in ai:
        zd = (dm_.get(m,dmin)-dm)/ds; zb = (bm_.get(m,bmin)-bm)/bst
        fu[m] = ALPHA*zd + (1-ALPHA)*zb
    return sorted(fu.items(), key=lambda x:-x[1])[:TOP_N]

def metrics(qg, zr):
    r = []
    for qid, gs in qg.items():
        b = None
        for i,m in enumerate(zr.get(qid,[]),1):
            if m in gs: b=i; break
        r.append(b)
    n = len(r)
    return (sum(1 for x in r if x is not None and x<=1)/n if n else 0,
            sum(1 for x in r if x is not None and x<=5)/n if n else 0,
            sum(1 for x in r if x is not None and x<=10)/n if n else 0,
            sum(1.0/x if x is not None else 0 for x in r)/n if n else 0)

# ===== PARITY (VERIFIED INTERNAL) =====
print("\n=== RawERK Baseline ===", flush=True)
qg = {}
for q in test_queries:
    g = q.get("gold_memory_ids",""); qg[q["query_id"]] = set(g.split(", ")) if g else set()
zr = {}
for q in test_queries: zr[q["query_id"]] = [m for m,_ in zscore_fuse(q["query_id"], bm25_rankings["RawERK"][q["query_id"]])]
r1,r5,r10,mrr = metrics(qg, zr)
print(f"  RawERK MRR={mrr:.4f} R@10={r10:.4f} (internal baseline, not compared to P3 official)", flush=True)

# ===== ALL VARIANTS =====
print("\n=== All variants ===", flush=True)
all_z = {}; all_qg = {}
for scope, queries in [("heldout1150",test_queries),("cat1_4_1540",cat1_4_queries)]:
    qg_ = {}
    for q in queries:
        g = q.get("gold_memory_ids",""); qg_[q["query_id"]] = set(g.split(", ")) if g else set()
    all_qg[scope] = qg_
    for vn in VARIANTS:
        k = f"{vn}/{scope}"
        zr_ = {}
        for q in queries: zr_[q["query_id"]] = [m for m,_ in zscore_fuse(q["query_id"], bm25_rankings[vn][q["query_id"]])]
        all_z[k] = zr_
        r1,r5,r10,mrr = metrics(qg_, zr_)
        print(f"  {vn:10s} [{scope}] MRR={mrr:.4f} R@10={r10:.4f}", flush=True)

# ===== SAVE =====
print("\nSaving...", flush=True)
for scope, queries in [("heldout1150",test_queries),("cat1_4_1540",cat1_4_queries)]:
    qg_ = all_qg[scope]; cq_ = defaultdict(list)
    for q in queries: cq_[CAT_MAP.get(q.get("category","4"))].append(q["query_id"])
    for suf in ["zscore","bm25"]:
        with (OUT/f"p1c_{suf}_overall_{scope}.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w = csv.DictWriter(f,fieldnames=["variant","n","R@1","R@5","R@10","MRR"]); w.writeheader()
            for vn in VARIANTS:
                if suf=="zscore": zr_ = all_z[f"{vn}/{scope}"]
                else: zr_ = {qid:[m for m,_ in bm25_rankings[vn][qid]] for qid in qg_}
                r1,r5,r10,mrr = metrics(qg_,zr_)
                w.writerow({"variant":vn,"n":len(qg_),"R@1":round(r1,4),"R@5":round(r5,4),"R@10":round(r10,4),"MRR":round(mrr,4)})
        with (OUT/f"p1c_{suf}_by_category_{scope}.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w = csv.DictWriter(f,fieldnames=["variant","category","n","R@1","R@5","R@10","MRR"]); w.writeheader()
            for vn in VARIANTS:
                if suf=="zscore": zr_ = all_z[f"{vn}/{scope}"]
                else: zr_ = {qid:[m for m,_ in bm25_rankings[vn][qid]] for qid in qg_}
                for cn,cqs in sorted(cq_.items()):
                    cs = set(cqs); cg = {qi:g for qi,g in qg_.items() if qi in cs}
                    cz = {qi:r for qi,r in zr_.items() if qi in cs}
                    r1,r5,r10,mrr = metrics(cg,cz)
                    w.writerow({"variant":vn,"category":cn,"n":len(cg),"R@1":round(r1,4),"R@5":round(r5,4),"R@10":round(r10,4),"MRR":round(mrr,4)})

# Leave-one-out deltas
print("Deltas...", flush=True)
bz = all_z["RawERK/heldout1150"]; bqg = all_qg["heldout1150"]; bm = metrics(bqg,bz)
deltas = []
for vn in ["Raw","RawRK","RawEK","RawER","RawERKT"]:
    cz = all_z[f"{vn}/heldout1150"]; c = metrics(bqg,cz)
    deltas.append({"comparison":f"{vn}_vs_RawERK","dR1":round(c[0]-bm[0],5),"dR5":round(c[1]-bm[1],5),"dR10":round(c[2]-bm[2],5),"dMRR":round(c[3]-bm[3],5)})
    print(f"  {vn}: dMRR={deltas[-1]['dMRR']:+.4f}", flush=True)
with (OUT/"p1c_leave_one_out_deltas.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f,fieldnames=list(deltas[0].keys())); w.writeheader(); w.writerows(deltas)

# Factorial
print("Factorial...", flush=True)
ev={"w":[],"wo":[]}; rv={"w":[],"wo":[]}; kv={"w":[],"wo":[]}
for vs in ["Raw","RawE","RawR","RawK","RawER","RawEK","RawRK","RawERK"]:
    comps = VARIANTS[vs]
    mr = metrics(bqg,all_z[f"{vs}/heldout1150"])[3]
    if "E" in comps: ev["w"].append(mr)
    else: ev["wo"].append(mr)
    if "R" in comps: rv["w"].append(mr)
    else: rv["wo"].append(mr)
    if "K" in comps: kv["w"].append(mr)
    else: kv["wo"].append(mr)
eff = [{"effect":"E_main","value":round(np.mean(ev["w"])-np.mean(ev["wo"]),6)},
    {"effect":"R_main","value":round(np.mean(rv["w"])-np.mean(rv["wo"]),6)},
    {"effect":"K_main","value":round(np.mean(kv["w"])-np.mean(kv["wo"]),6)}]
with (OUT/"p1c_factorial_effects.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f,fieldnames=["effect","value"]); w.writeheader(); w.writerows(eff)
for fe in eff: print(f"  {fe['effect']}: {fe['value']:+f}", flush=True)

# Manifest
with (OUT/"p1c_run_manifest.json").open("w") as f:
    json.dump({"experiment":"P1-C","date":time.strftime("%Y-%m-%d %H:%M"),"seed":SEED,
        "bm25_k1":BM25_K1,"bm25_b":BM25_B,"top_n":TOP_N,"alpha_dense":ALPHA,
        "test_queries":len(test_queries),"cat1_4_queries":len(cat1_4_queries),
        "dense_cache":str(DENSE_CACHE),"bm25_retriever":"BM25Retriever(P3 official)",
        "rawerk_baseline_mrr":round(mrr,4),"rawerk_baseline_r10":round(r10,4)},f,indent=2)

# Summary
summary = f"""# P1-C Compact Component Ablation Summary

## Internal Baseline (heldout 1150)
- RawERK: MRR={mrr:.4f}, R@10={r10:.4f}
- Note: Internal consistency only; not compared to P3 official 0.5462

## Leave-One-Out (ZScore, heldout 1150)
"""
for d in deltas:
    summary += f"- {d['comparison']}: dMRR={d['dMRR']:+f}, dR10={d['dR10']:+f}\n"
summary += "\n## Factorial Effects (MRR)\n"
for fe in eff: summary += f"- {fe['effect']}: {fe['value']:+f}\n"
with (OUT/"p1c_summary.md").open("w") as f: f.write(summary)
print(summary)
print(f"\nDone. {OUT}", flush=True)
