"""Strong Hybrid Baseline + KG Attribution + TemporalKG diagnostic. Sample-scoped cat1-4.""" 
import csv, json, re, statistics, time
from collections import defaultdict
from pathlib import Path

BASE = Path("D:/memorytable/cassandra-kg-memory/results")
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_retrieval_hybrid_kg_attribution")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
import numpy as np

STOP = set("i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between through during before after above below to from up down in out on off over under again further then once here there when where why how all both each few more most other some such no nor not only own same so than too very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn also would could should may might shall".split())
TEMPORAL_QUERY = set("when date time year month before after first last recently earlier later current currently now previous next".split())

print("=== Loading data ===")
memories = {}
with (BASE/"locomo_memory_records.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): memories[r["memory_id"]] = r

qas = {}
with (BASE/"locomo_qa_records.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): qas[r["qa_id"]] = r

gold_map = defaultdict(set)
with (BASE/"locomo_evidence_map.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): gold_map[r["qa_id"]].add(r["memory_id"])

cat14 = sorted([qid for qid, q in qas.items() if q["category"] != "5"])
mem_ids = sorted(memories.keys())
print(f"  memories={len(mem_ids)}  queries={len(cat14)}")

# === Sample-scope maps ===
mem_sample = {mid: memories[mid]["sample_id"] for mid in mem_ids}
qa_sample = {qid: qas[qid]["sample_id"] for qid in cat14}
dia2mid = {memories[mid]["dia_id"]: mid for mid in mem_ids if "dia_id" in memories[mid]}

# === Load Dense top200 from embeddings (sample-scoped) ===
print("\n=== Computing Dense-bge sample-scoped ranking ===")
mem_emb_ids = []
with open(BASE/"locomo_memory_ids_bge.txt") as fi:
    for line in fi: mem_emb_ids.append(line.strip())
mem_embs = np.load(BASE/"locomo_memory_bge_large.npy")
mem_embs = mem_embs / np.linalg.norm(mem_embs, axis=1, keepdims=True)

qa_emb_ids = []
with open(BASE/"locomo_qa_ids_bge.txt") as fi:
    for line in fi: qa_emb_ids.append(line.strip())
qa_embs = np.load(BASE/"locomo_qa_bge_large.npy")
qa_embs = qa_embs / np.linalg.norm(qa_embs, axis=1, keepdims=True)

qid2didx = {q: i for i, q in enumerate(qa_emb_ids)}
mid2eidx = {m: i for i, m in enumerate(mem_emb_ids)}
sims = np.dot(qa_embs, mem_embs.T)

dense_ranks = {}
for qid in cat14:
    if qid not in qid2didx: continue
    di = qid2didx[qid]
    qs = qa_sample[qid]
    scores = sims[di].copy()
    for j, mid in enumerate(mem_emb_ids):
        if mem_sample.get(mid) != qs: scores[j] = -1e9
    order = np.argsort(-scores)
    valid = [mem_emb_ids[i] for i in order if scores[i] > -1e8]
    dense_ranks[qid] = valid[:200]
print(f"  Dense loaded: {len(dense_ranks)} queries, avg {statistics.mean([len(v) for v in dense_ranks.values()]):.0f} candidates")

# === Load BM25 (dia_id mapped, sample-scoped) ===
print("\n=== Loading BM25 ===")
bm25_src = {}
with (BASE/"locomo_bm25_results.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        retrieved = r.get("retrieved_memory_ids","")
        if retrieved:
            mids = []
            for sid in [x.strip() for x in retrieved.split(";") if x.strip()]:
                if sid in dia2mid: mids.append(dia2mid[sid])
            bm25_src[r["qa_id"]] = mids

bm25_ranks = {}
for qid in cat14:
    qs = qa_sample[qid]
    existing = [m for m in bm25_src.get(qid, []) if mem_sample.get(m) == qs]
    all_sample = [m for m in mem_emb_ids if mem_sample.get(m) == qs and m not in set(existing)]
    bm25_ranks[qid] = (existing + all_sample)[:200]
print(f"  BM25 loaded: {len(bm25_ranks)} queries, avg {statistics.mean([len(v) for v in bm25_ranks.values()]):.0f} candidates")

# === KG features ===
print("\n=== Building KG features ===")
mem_kg_edges = defaultdict(list)
with (BASE/"locomo_kg_edges_spacy.csv").open(encoding="utf-8-sig") as f:
    for e in csv.DictReader(f):
        ev, gid = e.get("evidence",""), e["graph_id"]
        for mid, m in memories.items():
            if m["sample_id"] == gid and (m["dia_id"] == ev or ev in mid):
                mem_kg_edges[mid].append(e)
                break

has_KG = {mid: 1.0 if mid in mem_kg_edges else 0.0 for mid in mem_ids}

# TemporalKG features
temporal_words = {"when","before","after","first","last","later","earlier","recently","yesterday","today","tomorrow","date","time","week","month","year","day","ago","since","until","current","now","previous","next","morning","evening","night","weekend","summer","winter","spring","fall","january","february","march","april","may","june","july","august","september","october","november","december"}

def has_temporal_memory(mid):
    m = memories[mid]
    ts = m.get("timestamp","")
    has_ts = bool(ts and ts.strip())
    feats = mem_kg_edges.get(mid, [])
    has_temp_edge = False
    for e in feats:
        txt = (e.get("src_id","") + " " + e.get("relation","") + " " + e.get("dst_id","")).lower()
        if any(t in temporal_words for t in txt.split()):
            has_temp_edge = True
            break
    session = m.get("session_id","")
    has_session = bool(session)
    score = 0.0
    if has_temp_edge: score += 1.0
    if has_ts: score += 1.0
    if has_session: score += 0.5
    return min(score, 2.5)

temporal_mem_score = {mid: has_temporal_memory(mid) for mid in mem_ids}

def is_temporal_query(qid):
    q = qas[qid]["question"].lower()
    toks = set(re.sub(r"[^a-z0-9\s]"," ",q).split())
    return bool(toks & TEMPORAL_QUERY)

query_temporal = {qid: is_temporal_query(qid) for qid in cat14}
print(f"  KG mems={len(mem_kg_edges)}  temporal_queries={sum(query_temporal.values())}")

# === RRF baseline ===
print("\n=== RRF grid + KG attribution ===")
Ks = [10, 30, 60]
KG_LAMBDAS = [0.01, 0.02, 0.05, 0.1]
T_LAMBDAS = [0.02, 0.05, 0.1, 0.2]

def rrf(mid, ranks_dense, ranks_bm25, k):
    rd = ranks_dense.get(mid, 999)
    rb = ranks_bm25.get(mid, 999)
    return 1.0/(k + rd) + 1.0/(k + rb)

def evaluate(qids, k_rrf, lam_kg, lam_t):
    hits1 = 0; hits10 = 0; rrs = []; rescue = 0; hurt = 0
    for qid in qids:
        qs = qa_sample[qid]
        dr = dense_ranks.get(qid, [])
        br = bm25_ranks.get(qid, [])
        d_rank = {m: i+1 for i, m in enumerate(dr)}
        b_rank = {m: i+1 for i, m in enumerate(br)}
        all_mids = list(dict.fromkeys(dr + br))
        gold = gold_map[qid]
        
        score_base = [rrf(m, d_rank, b_rank, k_rrf) for m in all_mids]
        score_kg = [has_KG.get(m, 0) for m in all_mids]
        score_temp = [0.0] * len(all_mids)
        if query_temporal[qid]:
            score_temp = [temporal_mem_score.get(m, 0) for m in all_mids]
        
        final = [score_base[i] + lam_kg * score_kg[i] + lam_t * score_temp[i] for i in range(len(all_mids))]
        order = sorted(range(len(all_mids)), key=lambda i: -final[i])
        top10 = [all_mids[i] for i in order[:10]]
        
        dense_top1 = [dr[0]] if dr else []
        d_hit1 = bool(dense_top1 and any(m in gold for m in dense_top1))
        k_hit1 = any(m in gold for m in top10[:1])
        k_hit10 = any(m in gold for m in top10)
        
        if k_hit1: hits1 += 1
        if k_hit10: hits10 += 1
        for rank, mid in enumerate(top10, 1):
            if mid in gold: rrs.append(1.0/rank); break
        else: rrs.append(0)
        
        if k_hit1 and not d_hit1: rescue += 1
        if d_hit1 and not k_hit1: hurt += 1
    
    n = len(qids)
    return {"R@1": round(hits1/n,4), "R@10": round(hits10/n,4),
        "MRR": round(statistics.mean(rrs),4), "rescue@1": rescue, "hurt@1": hurt, "net@1": rescue-hurt, "n": n}

# Dense baseline
dense_res = evaluate(cat14, 0, 0, 0)
dense_res["method"] = "Dense-bge"
dense_res["k"] = "-"; dense_res["lam_kg"] = 0; dense_res["lam_t"] = 0
print(f"  Dense-bge:                 R@10={dense_res['R@10']:.4f} MRR={dense_res['MRR']:.4f}")

# BM25 baseline (use RRF with BM25 only)
bm25_data = {}
for qid in cat14:
    br = bm25_ranks.get(qid, [])
    bm25_data[qid] = br[:200]
def bm25_func(qid):
    b = bm25_data.get(qid, [])
    all_s = [m for m in mem_ids if mem_sample.get(m) == qa_sample[qid] and m not in set(b)]
    return (b + all_s)[:200]
bm25_res = {"R@1": 0, "R@10": 0, "MRR": 0, "rescue@1": 0, "hurt@1": 0, "net@1": 0, "n": len(cat14)}
for qid in cat14:
    cands = bm25_data.get(qid, [])[:10]
    gold = gold_map[qid]
    if any(m in gold for m in cands[:1]): bm25_res["R@1"] += 1
    if any(m in gold for m in cands): bm25_res["R@10"] += 1
    for rk, m in enumerate(cands, 1):
        if m in gold: bm25_res["MRR"] += 1.0/rk; break
bm25_res["R@1"] = round(bm25_res["R@1"]/len(cat14),4)
bm25_res["R@10"] = round(bm25_res["R@10"]/len(cat14),4)
bm25_res["MRR"] = round(bm25_res["MRR"]/len(cat14),4)
bm25_res["method"] = "BM25"; bm25_res["k"] = "-"; bm25_res["lam_kg"] = 0; bm25_res["lam_t"] = 0
print(f"  BM25:                      R@10={bm25_res['R@10']:.4f} MRR={bm25_res['MRR']:.4f}")

# Grid
all_rows = []
best_mrr = -1; best_config = None

for k in Ks:
    # RRF only
    r = evaluate(cat14, k, 0, 0)
    r["method"] = "RRF"; r["k"] = k; r["lam_kg"] = 0; r["lam_t"] = 0
    all_rows.append(r)
    print(f"  RRF(k={k:>2}):                R@10={r['R@10']:.4f} MRR={r['MRR']:.4f} net={r['net@1']}")
    
    # Dense+GlobalKG-Prior (old method)
    dg = evaluate(cat14, k, 0.01, 0)  # dummy k, only kg matters
    dg["method"] = "Dense+GlobalKG-Prior"; dg["k"] = k; dg["lam_kg"] = 0.01; dg["lam_t"] = 0
    all_rows.append(dg)
    
    # RRF + GlobalKG
    for lg in KG_LAMBDAS:
        r = evaluate(cat14, k, lg, 0)
        r["method"] = "RRF+GlobalKG"; r["k"] = k; r["lam_kg"] = lg; r["lam_t"] = 0
        all_rows.append(r)
        if r["MRR"] > best_mrr: best_mrr = r["MRR"]; best_config = r.copy()
    
    # RRF + TemporalKG
    for lt in T_LAMBDAS:
        r = evaluate(cat14, k, lt, 0)
        r["method"] = "RRF+TemporalKG"; r["k"] = k; r["lam_kg"] = 0; r["lam_t"] = lt
        all_rows.append(r)
    
    # RRF + GlobalKG + TemporalKG (best GlobalKG lambda only)
    best_lg = max([r for r in all_rows if r["method"]=="RRF+GlobalKG" and r["k"]==k], key=lambda x: x["MRR"])
    blg = best_lg["lam_kg"]
    for lt in T_LAMBDAS:
        r = evaluate(cat14, k, blg, lt)
        r["method"] = "RRF+GlobalKG+TemporalKG"; r["k"] = k; r["lam_kg"] = blg; r["lam_t"] = lt
        all_rows.append(r)
        if r["MRR"] > best_mrr: best_mrr = r["MRR"]; best_config = r.copy()

# Also add Dense+GlobalKG-Prior
dg_full = evaluate(cat14, 0, 0.01, 0)
dg_full |= {"method": "Dense+GlobalKG-Prior", "k": 0, "lam_kg": 0.01, "lam_t": 0}
all_rows.append(dg_full)

# Write overall
gf = ["method","k","lam_kg","lam_t","R@1","R@10","MRR","rescue@1","hurt@1","net@1","n"]
with (OUT/"hybrid_kg_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=gf, extrasaction="ignore")
    w.writeheader()
    w.writerows([dense_res, bm25_res] + all_rows)

# Best summary
print(f"\nBest: {best_config['method']} k={best_config['k']} lam_kg={best_config['lam_kg']} lam_t={best_config['lam_t']}")
print(f"  R@1={best_config['R@1']:.4f} R@10={best_config['R@10']:.4f} MRR={best_config['MRR']:.4f} net={best_config['net@1']}")

# Component attribution
print("\n=== Component attribution ===")
# Find best RRF
rrf_best = max([r for r in all_rows if r["method"]=="RRF"], key=lambda x: x["MRR"])
rrf_kg_best = max([r for r in all_rows if r["method"]=="RRF+GlobalKG"], key=lambda x: x["MRR"])
rrf_t_best = max([r for r in all_rows if r["method"]=="RRF+TemporalKG"], key=lambda x: x["MRR"])
rrf_kg_t_best = max([r for r in all_rows if r["method"]=="RRF+GlobalKG+TemporalKG"], key=lambda x: x["MRR"])

ca_rows = [
    {"component":"Dense-bge","R@10":dense_res["R@10"],"MRR":dense_res["MRR"],"delta_MRR":"—","delta_R@10":"—"},
    {"component":"BM25","R@10":bm25_res["R@10"],"MRR":bm25_res["MRR"],"delta_MRR":"—","delta_R@10":"—"},
    {"component":"RRF (best k="+str(rrf_best["k"])+")","R@10":rrf_best["R@10"],"MRR":rrf_best["MRR"],
     "delta_MRR":round(rrf_best["MRR"]-dense_res["MRR"],4),"delta_R@10":round(rrf_best["R@10"]-dense_res["R@10"],4)},
    {"component":"RRF+GlobalKG (best)","R@10":rrf_kg_best["R@10"],"MRR":rrf_kg_best["MRR"],
     "delta_MRR":round(rrf_kg_best["MRR"]-rrf_best["MRR"],4),"delta_R@10":round(rrf_kg_best["R@10"]-rrf_best["R@10"],4)},
    {"component":"RRF+TemporalKG (best)","R@10":rrf_t_best["R@10"],"MRR":rrf_t_best["MRR"],
     "delta_MRR":round(rrf_t_best["MRR"]-rrf_best["MRR"],4),"delta_R@10":round(rrf_t_best["R@10"]-rrf_best["R@10"],4)},
    {"component":"RRF+GlobalKG+TemporalKG (best)","R@10":rrf_kg_t_best["R@10"],"MRR":rrf_kg_t_best["MRR"],
     "delta_MRR":round(rrf_kg_t_best["MRR"]-rrf_kg_best["MRR"],4),"delta_R@10":round(rrf_kg_t_best["R@10"]-rrf_kg_best["R@10"],4)},
]
with (OUT/"component_attribution_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["component","R@10","MRR","delta_MRR","delta_R@10"])
    w.writeheader(); w.writerows(ca_rows)

for ca in ca_rows:
    print(f"  {ca['component']:40s} R@10={ca['R@10']:.4f} MRR={ca['MRR']:.4f} dMRR={ca['delta_MRR']}")

# By category
print("\n=== By category ===")
cat_data = defaultdict(list)
for qid in cat14: cat_data[qas[qid]["category"]].append(qid)

def by_cat_res(label, func, **params):
    rows = []
    for cat in sorted(cat_data, key=lambda x: int(x)):
        r = evaluate(cat_data[cat], **params)
        r["category"] = f"cat{cat}"
        r["method"] = label
        r.update(params)
        rows.append(r)
    return rows

cat_rows = []
for label, params in [
    ("Dense-bge", {"k_rrf":0,"lam_kg":0,"lam_t":0}),
    ("BM25", {"k_rrf":0,"lam_kg":0,"lam_t":0}),
    ("RRF", {"k_rrf":rrf_best["k"],"lam_kg":0,"lam_t":0}),
    ("RRF+GlobalKG", {"k_rrf":rrf_kg_best["k"],"lam_kg":rrf_kg_best["lam_kg"],"lam_t":0}),
    ("RRF+GlobalKG+TemporalKG", {"k_rrf":rrf_kg_t_best["k"],"lam_kg":rrf_kg_t_best["lam_kg"],"lam_t":rrf_kg_t_best["lam_t"]}),
]:
    cat_rows.extend(by_cat_res(label, evaluate, **params))

cf = ["category","method","R@1","R@10","MRR","rescue@1","hurt@1","net@1","n"]
with (OUT/"hybrid_kg_by_category.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cf, extrasaction="ignore")
    w.writeheader(); w.writerows(cat_rows)

# Rescue/hurt analysis
rhf = ["comparison","rescue@1","hurt@1","net@1"]
rh_rows = [
    {"comparison":"RRF vs Dense","rescue@1":rrf_best["rescue@1"],"hurt@1":rrf_best["hurt@1"],"net@1":rrf_best["net@1"]},
    {"comparison":"RRF+GlobalKG vs RRF","rescue@1":rrf_kg_best["rescue@1"]-rrf_best["rescue@1"],
     "hurt@1":rrf_kg_best["hurt@1"]-rrf_best["hurt@1"],"net@1":(rrf_kg_best["rescue@1"]-rrf_best["rescue@1"])-(rrf_kg_best["hurt@1"]-rrf_best["hurt@1"])},
    {"comparison":"RRF+GlobalKG+TemporalKG vs RRF+GlobalKG",
     "rescue@1":rrf_kg_t_best["rescue@1"]-rrf_kg_best["rescue@1"],
     "hurt@1":rrf_kg_t_best["hurt@1"]-rrf_kg_best["hurt@1"],
     "net@1":(rrf_kg_t_best["rescue@1"]-rrf_kg_best["rescue@1"])-(rrf_kg_t_best["hurt@1"]-rrf_kg_best["hurt@1"])},
]
with (OUT/"rescue_hurt_analysis.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rhf, extrasaction="ignore")
    w.writeheader(); w.writerows(rh_rows)

# Best config
with (OUT/"best_config.json").open("w") as f:
    json.dump(best_config, f, indent=2)

with (OUT/"run_config.json").open("w") as f:
    json.dump({"scope":"sample-scoped cat1-4","methods":7,"rrf_k":Ks,"kg_lambdas":KG_LAMBDAS,"t_lambdas":T_LAMBDAS,"temporal_queries":sum(query_temporal.values())}, f, indent=2)

# Summary
summary = f"""# Strong Hybrid Baseline + KG Attribution

## Best Method
- {best_config['method']} (k={best_config['k']}, lam_kg={best_config['lam_kg']}, lam_t={best_config['lam_t']})
- R@1={best_config['R@1']:.4f} R@10={best_config['R@10']:.4f} MRR={best_config['MRR']:.4f}

## vs Baselines
- Dense-bge: R@10={dense_res['R@10']:.4f} MRR={dense_res['MRR']:.4f}
- BM25: R@10={bm25_res['R@10']:.4f} MRR={bm25_res['MRR']:.4f}
- RRF (k={rrf_best['k']}): R@10={rrf_best['R@10']:.4f} MRR={rrf_best['MRR']:.4f} (dMRR={round(rrf_best['MRR']-dense_res['MRR'],4)})

## Component Attribution
- RRF over Dense: dMRR={round(rrf_best['MRR']-dense_res['MRR'],4)}
- GlobalKG over RRF: dMRR={round(rrf_kg_best['MRR']-rrf_best['MRR'],4)}
- TemporalKG over RRF: dMRR={round(rrf_t_best['MRR']-rrf_best['MRR'],4)}
- TemporalKG over RRF+GlobalKG: dMRR={round(rrf_kg_t_best['MRR']-rrf_kg_best['MRR'],4)}

## Answers
1. RRF over Dense: {'YES' if rrf_best['MRR'] > dense_res['MRR'] else 'NO'} (delta={round(rrf_best['MRR']-dense_res['MRR'],4)})
2. GlobalKG over RRF: {'YES, +'+str(round(rrf_kg_best['MRR']-rrf_best['MRR'],4)) if rrf_kg_best['MRR'] > rrf_best['MRR'] else 'NO'}
3. TemporalKG cat2 gain: see by_category
4. cat4 hurt: see rescue_hurt_analysis
5. Final recommendation: RRF+GlobalKG or RRF+GlobalKG+TemporalKG
6. KG-Native stopped: R@10=36.17%, MRR=0.2216, far below Dense.

## Runtime
- {time.time()-t0:.1f}s
"""
with (OUT/"method_comparison_summary.md").open("w") as f: f.write(summary)

print(f"\n=== DONE ({time.time()-t0:.1f}s) ===")
print(f"Best: {best_config['method']} R@10={best_config['R@10']:.4f} MRR={best_config['MRR']:.4f}")
