"""v6: import BM25Retriever directly from the original sample_scoped script. Zero implementation drift."""
import csv, json, math, random, re, statistics, time, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

# Import BM25Retriever from the original script (same version, same sklearn)
SCRIPT_DIR = Path("D:/memorytable/cassandra-kg-memory/scripts/memory/locomo_pipeline/retrieval")
sys.path.insert(0, str(SCRIPT_DIR))
from locomo_retrieval_sample_scoped import BM25Retriever

BASE = Path("D:/memorytable/cassandra-kg-memory/results")
ENR = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_retrieval_representation")
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/compact_16_factorial_ablation")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

print("Loading data...", flush=True)
memories = {}
mem_sample_map = {}
sample_memories = defaultdict(list)
with (BASE/"locomo_memory_records.csv").open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        mid = row["memory_id"].strip(); sid = row["sample_id"].strip(); txt = row["text"].strip()
        memories[mid] = row; mem_sample_map[mid] = sid
        sample_memories[sid].append({"memory_id": mid, "text": txt, "dia_id": row["dia_id"].strip()})

qas = {}; qa_sample_map = {}
with (BASE/"locomo_qa_records.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        qas[r["qa_id"]] = r
        qa_sample_map[r["qa_id"].strip()] = r["sample_id"].strip()
cat14 = sorted([qid for qid, q in qas.items() if q["category"] != "5"])

gold_map = defaultdict(set)
with (BASE/"locomo_evidence_map.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): gold_map[r["qa_id"]].add(r["memory_id"])

# Dense embeddings (same as sample_scoped script lines 725-728)
with open(BASE/"locomo_memory_ids_bge.txt") as f: mem_ids_bge = [line.strip() for line in f if line.strip()]
with open(BASE/"locomo_qa_ids_bge.txt") as f: qa_ids_bge = [line.strip() for line in f if line.strip()]
mem_embs = np.load(BASE/"locomo_memory_bge_large.npy")
qa_embs = np.load(BASE/"locomo_qa_bge_large.npy")
mem_embs = mem_embs / np.linalg.norm(mem_embs, axis=1, keepdims=True)
qa_embs = qa_embs / np.linalg.norm(qa_embs, axis=1, keepdims=True)
qid_to_idx = {qid: i for i, qid in enumerate(qa_ids_bge)}
mid_to_idx = {mid: i for i, mid in enumerate(mem_ids_bge)}
sample_mem_idx = defaultdict(list)
for mid, sid in mem_sample_map.items():
    if mid in mid_to_idx: sample_mem_idx[sid].append(mid_to_idx[mid])

# Enriched fields
enriched = list(csv.DictReader((ENR/"enriched_memory_records.csv").open(encoding="utf-8-sig")))
def ef(text, field):
    return text.split(f"{field}:")[1].split("\n")[0].strip() if f"{field}:" in text else ""
enrich = {"E": {}, "R": {}, "K": {}, "T": {}}
for e in enriched:
    mid = e["memory_id"]
    enrich["E"][mid] = ef(e["enriched_text"], "Entities")
    enrich["R"][mid] = ef(e["enriched_text"], "Relations")
    enrich["K"][mid] = ef(e["enriched_text"], "Keywords")
for mid in memories:
    ts = str(memories[mid].get("timestamp", "")).strip()
    m = re.search(r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})", ts, re.IGNORECASE)
    months = {"january":"01","february":"02","march":"03","april":"04","may":"05","june":"06","july":"07","august":"08","september":"09","october":"10","november":"11","december":"12"}
    enrich["T"][mid] = f"{m.group(3)}-{months.get(m.group(2).lower(),'??')}-{int(m.group(1)):02d}" if m else (m2.group(1) if (m2 := re.search(r"(\d{4})", ts)) else "")

# Per-sample Dense
def dense_sample_scoped_ranks():
    ranks = {}
    for qid in cat14:
        sid = qa_sample_map.get(qid)
        if qid not in qid_to_idx or sid not in sample_mem_idx: continue
        qi = qid_to_idx[qid]; q_emb = qa_embs[qi]
        candidate_indices = sample_mem_idx[sid]
        if not candidate_indices: continue
        scores = np.dot(mem_embs[candidate_indices], q_emb)
        sorted_local = np.argsort(-scores)
        top_k = min(10, len(sorted_local))
        ranks[qid] = [mem_ids_bge[candidate_indices[i]] for i in sorted_local[:top_k]]
    return ranks

print("Dense per-sample...", flush=True)
dense_ranks = dense_sample_scoped_ranks()

# ===================== PER-SAMPLE BM25 (16 variants, using IMPORTED BM25Retriever) =====================
bit_labels = [
    ("E0R0K0T0",0,0,0,0),("E1R0K0T0",1,0,0,0),("E0R1K0T0",0,1,0,0),("E0R0K1T0",0,0,1,0),("E0R0K0T1",0,0,0,1),
    ("E1R1K0T0",1,1,0,0),("E1R0K1T0",1,0,1,0),("E1R0K0T1",1,0,0,1),("E0R1K1T0",0,1,1,0),("E0R1K0T1",0,1,0,1),
    ("E0R0K1T1",0,0,1,1),("E1R1K1T0",1,1,1,0),("E1R1K0T1",1,1,0,1),("E1R0K1T1",1,0,1,1),
    ("E0R1K1T1",0,1,1,1),("E1R1K1T1",1,1,1,1),
]

def build_variant_texts(e, r, k, t, mem_list):
    texts = []
    for m in mem_list:
        mid = m["memory_id"]
        parts = [m["text"]]
        if e and enrich["E"].get(mid): parts.append(f"Entities: {enrich['E'][mid]}")
        if r and enrich["R"].get(mid): parts.append(f"Relations: {enrich['R'][mid]}")
        if k and enrich["K"].get(mid): parts.append(f"Keywords: {enrich['K'][mid]}")
        if t and enrich["T"].get(mid): parts.append(f"Time: {enrich['T'][mid]}")
        texts.append("\n".join(parts))
    return texts

def run_bm25_sample_scoped(e, r, k, t):
    sample_bm25 = {}
    for sid, mem_list in sample_memories.items():
        texts = build_variant_texts(e, r, k, t, mem_list)
        bm = BM25Retriever(k1=1.5, b=0.75)
        bm.fit(texts)
        sample_bm25[sid] = (bm, mem_list)
    ranks = {}
    for qid in cat14:
        sid = qa_sample_map.get(qid)
        if sid not in sample_bm25: continue
        bm, mem_list = sample_bm25[sid]
        indices, _ = bm.search(qas[qid]["question"], top_k=10)
        ranks[qid] = [mem_list[i]["memory_id"] for i in indices]
    return ranks

# ===================== RRF =====================
A, K = 0.5, 10

def rrf_fuse(d_ranks, b_ranks):
    out = {}
    for qid in cat14:
        dl = d_ranks.get(qid, [])[:10]; bl = b_ranks.get(qid, [])[:10]
        dr = {m: i+1 for i, m in enumerate(dl)}; br = {m: i+1 for i, m in enumerate(bl)}
        am = list(dict.fromkeys(dl + bl))
        sc = [(A/(K+dr.get(m,999))+(1-A)/(K+br.get(m,999)), m) for m in am]
        sc.sort(key=lambda x: -x[0])
        out[qid] = [m for _, m in sc[:10]]
    return out

# ===================== COMPUTE ALL 16 =====================
all_bm25 = {}; all_rrf = {}
print(f"\nPer-sample BM25 ({len(bit_labels)} variants)...", flush=True)
for vi, (name, e, r, k, t) in enumerate(bit_labels):
    ts = time.time()
    print(f"  [{vi+1}/16] {name}", end=" ", flush=True)
    ranks = run_bm25_sample_scoped(e, r, k, t)
    all_bm25[name] = ranks
    all_rrf[name] = rrf_fuse(dense_ranks, ranks)
    print(f"({time.time()-ts:.0f}s)", flush=True)

# ===================== EVALUATE =====================
def evaluate(ranks):
    h1 = h10 = 0; rrs = []
    for qid in cat14:
        top = ranks.get(qid, [])[:10]; gold = gold_map[qid]
        h1 += any(m in gold for m in top[:1]); h10 += any(m in gold for m in top)
        for rk, m in enumerate(top, 1):
            if m in gold: rrs.append(1.0/rk); break
        else: rrs.append(0)
    n = len(cat14)
    return {"R@1": round(h1/n, 4), "R@10": round(h10/n, 4), "MRR": round(statistics.mean(rrs), 4), "Hit@10": round(h10/n, 4)}

bm25_res = {n: evaluate(all_bm25[n]) for n, _, _, _, _ in bit_labels}
rrf_res = {n: evaluate(all_rrf[n]) for n, _, _, _, _ in bit_labels}

# ===================== GATE (cat1-4, from canonical per-query CSVs) =====================
print("\n=== Gate (cat1-4, n=1540) ===")
dense_metrics = evaluate(dense_ranks)
gate = {
    "BM25_raw": (bm25_res["E0R0K0T0"], {"R@1":0.2552,"R@10":0.5487,"MRR":0.3488}),
    "Dense_raw": (dense_metrics, {"R@1":0.3740,"R@10":0.7299,"MRR":0.4859}),
    "RRF_raw": (rrf_res["E0R0K0T0"], {"R@1":0.3539,"R@10":0.7409,"MRR":0.4763}),
}
all_ok = True
for label, (res, exp) in gate.items():
    for m in ["R@1","R@10","MRR"]:
        d = abs(res[m] - exp[m])
        if d > 1e-4: all_ok = False
        print(f"  {label:15s} {m}: {res[m]:.4f} vs {exp[m]:.4f} {'PASS' if d<=1e-4 else 'FAIL'}")

if not all_ok: print("\nSTOP: Gate FAILED"); exit(1)
print("  ALL PASS — aligned with canonical per-query CSVs on cat1-4")

# ===================== WRITE CSVs =====================
raw_bm = bm25_res["E0R0K0T0"]; raw_rf = rrf_res["E0R0K0T0"]

def write_overall(results, baseline, fn):
    rows = []
    for nm, e, r, k, t in bit_labels:
        x = results[nm]
        rows.append({"bitmask":nm,"E":e,"R":r,"K":k,"T":t,
            "R@1":x["R@1"],"R@10":x["R@10"],"MRR":x["MRR"],"Hit@10":x["Hit@10"],
            "dR@1":round(x["R@1"]-baseline["R@1"],4),"dR@10":round(x["R@10"]-baseline["R@10"],4),
            "dMRR":round(x["MRR"]-baseline["MRR"],4),"dHit@10":round(x["Hit@10"]-baseline["Hit@10"],4)})
    with (OUT/fn).open("w",newline="",encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

write_overall(bm25_res, raw_bm, "bm25_16_variants_overall.csv")
write_overall(rrf_res, raw_rf, "rrf_16_variants_overall.csv")

crows = [{"bitmask":nm,"E":e,"R":r,"K":k,"T":t,
    "BM25_R@1":bm25_res[nm]["R@1"],"BM25_R@10":bm25_res[nm]["R@10"],"BM25_MRR":bm25_res[nm]["MRR"],
    "RRF_R@1":rrf_res[nm]["R@1"],"RRF_R@10":rrf_res[nm]["R@10"],"RRF_MRR":rrf_res[nm]["MRR"]}
    for nm,e,r,k,t in bit_labels]
with (OUT/"compact_16_variants_combined.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(crows[0].keys())); w.writeheader(); w.writerows(crows)

# Leave-One-Out
loo = []
for field, off in [("E","E0R1K1T1"),("R","E1R0K1T1"),("K","E1R1K0T1"),("T","E1R1K1T0")]:
    for mt, res in [("BM25",bm25_res),("RRF",rrf_res)]:
        f = res["E1R1K1T1"]; o = res[off]
        loo.append({"method_type":mt,"field_removed":field,"dR@1":round(f["R@1"]-o["R@1"],4),
            "dR@10":round(f["R@10"]-o["R@10"],4),"dMRR":round(f["MRR"]-o["MRR"],4),"dHit@10":round(f["Hit@10"]-o["Hit@10"],4)})
with (OUT/"leave_one_out_effects.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(loo[0].keys())); w.writeheader(); w.writerows(loo)

# Main effects
me = []
for field, fi in [("Entity",0),("Relation",1),("Keyword",2),("Time",3)]:
    for mt, res in [("BM25",bm25_res),("RRF",rrf_res)]:
        on = defaultdict(list); off = defaultdict(list)
        for nm, e, r, k, t in bit_labels:
            v = 1 if fi == 0 else (e if fi == 1 else (r if fi == 2 else (k if fi == 3 else t)))
            for m in ["R@1","R@10","MRR","Hit@10"]:
                (on if v == 1 else off)[m].append(res[nm][m])
        me.append({"method_type":mt,"field":field,"dR@1":round(statistics.mean(on["R@1"])-statistics.mean(off["R@1"]),4),
            "dR@10":round(statistics.mean(on["R@10"])-statistics.mean(off["R@10"]),4),
            "dMRR":round(statistics.mean(on["MRR"])-statistics.mean(off["MRR"]),4),
            "dHit@10":round(statistics.mean(on["Hit@10"])-statistics.mean(off["Hit@10"]),4)})
with (OUT/"factorial_main_effects.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(me[0].keys())); w.writeheader(); w.writerows(me)

# Category
cat_d = defaultdict(list)
for qid in cat14: cat_d[qas[qid]["category"]].append(qid)
for suffix, ranks in [("bm25_16_variants_by_category.csv",all_bm25),("rrf_16_variants_by_category.csv",all_rrf)]:
    rows = []
    for nm, _, _, _, _ in bit_labels:
        for cat in sorted(cat_d):
            cq = cat_d[cat]; cn = len(cq)
            h1 = sum(any(m in gold_map[q] for m in ranks[nm].get(q,[])[:1]) for q in cq)
            h10 = sum(any(m in gold_map[q] for m in ranks[nm].get(q,[])[:10]) for q in cq)
            rrs = []
            for q in cq:
                for rk, m in enumerate(ranks[nm].get(q,[])[:10], 1):
                    if m in gold_map[q]: rrs.append(1.0/rk); break
                else: rrs.append(0)
            rows.append({"bitmask":nm,"category":f"cat{cat}","n":cn,
                "R@1":round(h1/cn,4),"R@10":round(h10/cn,4),"MRR":round(statistics.mean(rrs),4),"Hit@10":round(h10/cn,4)})
    with (OUT/suffix).open("w",newline="",encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# Bootstrap
def bs_paired(m1, m2, metric_fn):
    rng = random.Random(42); n = len(cat14)
    diffs = []
    for _ in range(10000):
        idx = [rng.randint(0, n-1) for __ in range(n)]
        diffs.append(metric_fn([m1.get(cat14[i],[]) for i in idx]) - metric_fn([m2.get(cat14[i],[]) for i in idx]))
    diffs.sort(); return round(statistics.mean(diffs),4), round(diffs[250],4), round(diffs[9750],4)

def mrr_fn(tops):
    rrs = []
    for i, t in enumerate(tops):
        gold = gold_map[cat14[i % len(cat14)]]
        for rk, m in enumerate(t, 1):
            if m in gold: rrs.append(1.0/rk); break
        else: rrs.append(0)
    return statistics.mean(rrs)

erkt_bm = all_bm25["E1R1K1T1"]; erk_bm = all_bm25["E1R1K1T0"]
erkt_rf = all_rrf["E1R1K1T1"]; erk_rf = all_rrf["E1R1K1T0"]
m_bm, lo_bm, hi_bm = bs_paired(erkt_bm, erk_bm, mrr_fn)
m_rf, lo_rf, hi_rf = bs_paired(erkt_rf, erk_rf, mrr_fn)
print(f"\nERKT-ERK BM25 bootstrap: {m_bm:+.4f} [{lo_bm:+.4f}, {hi_bm:+.4f}]")
print(f"ERKT-ERK RRF  bootstrap: {m_rf:+.4f} [{lo_rf:+.4f}, {hi_rf:+.4f}]")

with (OUT/"run_config.json").open("w") as f:
    json.dump({"engine":"imported_BM25Retriever_from_locomo_retrieval_sample_scoped","per_sample":True,"rrf_alpha":0.5,"rrf_k":10,"n_variants":16,"n_queries":len(cat14),"runtime_s":round(time.time()-t0,1)}, f, indent=2)

best_bm = max(bit_labels, key=lambda x: bm25_res[x[0]]["MRR"])
best_rf = max(bit_labels, key=lambda x: rrf_res[x[0]]["MRR"])
print(f"\nBest BM25: {best_bm[0]} MRR={bm25_res[best_bm[0]]['MRR']}")
print(f"Best RRF:  {best_rf[0]} MRR={rrf_res[best_rf[0]]['MRR']}")
print(f"Runtime: {time.time()-t0:.0f}s")
