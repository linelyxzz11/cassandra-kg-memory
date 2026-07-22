"""P3: Memory Representation Comparison. Reuses P2's exact Dense + per-sample BM25 + ZScore pipeline.
Only changes: the BM25 document text per representation. No new retrieval engine."""
import csv, json, random, statistics, time, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path("D:/memorytable/cassandra-kg-memory/scripts/memory/locomo_pipeline/retrieval")
sys.path.insert(0, str(SCRIPT_DIR))
from locomo_retrieval_sample_scoped import BM25Retriever

BASE = Path("D:/memorytable/cassandra-kg-memory/results")
ART = Path("D:/memorytable/cassandra-kg-memory/scripts/experiments/artifacts")
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/p3_memory_representation_comparison")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

# ===================== DATA =====================
print("Loading data...", flush=True)
memories = {}; mem_sample_map = {}; sample_memories = defaultdict(list)
with (BASE/"locomo_memory_records.csv").open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        mid = row["memory_id"].strip(); sid = row["sample_id"].strip()
        memories[mid] = row; mem_sample_map[mid] = sid
        sample_memories[sid].append({"memory_id": mid, "text": row["text"].strip()})

# Load P3 features (summary, triples, ERK)
p3_feats = {}
with (ART/"p3_memory_features.csv").open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        p3_feats[row["memory_id"]] = row

qas = {}; qa_sample_map = {}
with (BASE/"locomo_qa_records.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): qas[r["qa_id"]] = r; qa_sample_map[r["qa_id"].strip()] = r["sample_id"].strip()

cat14 = sorted([q for q in qas if qas[q]["category"] != "5"])
test_convs = ["conv-26","conv-30","conv-41","conv-43","conv-44","conv-47","conv-49","conv-50"]
test_qids = sorted([q for q in cat14 if qa_sample_map[q] in test_convs])
print(f"  test queries: {len(test_qids)}")

gold_map = defaultdict(set)
with (BASE/"locomo_evidence_map.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): gold_map[r["qa_id"]].add(r["memory_id"])

# Dense embeddings
with open(BASE/"locomo_memory_ids_bge.txt") as f: mem_ids_bge = [line.strip() for line in f if line.strip()]
with open(BASE/"locomo_qa_ids_bge.txt") as f: qa_ids_bge = [line.strip() for line in f if line.strip()]
mem_embs = np.load(BASE/"locomo_memory_bge_large.npy"); mem_embs = mem_embs / np.linalg.norm(mem_embs, axis=1, keepdims=True)
qa_embs = np.load(BASE/"locomo_qa_bge_large.npy"); qa_embs = qa_embs / np.linalg.norm(qa_embs, axis=1, keepdims=True)
qid_to_idx = {qid: i for i, qid in enumerate(qa_ids_bge)}
mid_to_idx = {mid: i for i, mid in enumerate(mem_ids_bge)}
sample_mem_idx = defaultdict(list)
for mid, sid in mem_sample_map.items():
    if mid in mid_to_idx: sample_mem_idx[sid].append(mid_to_idx[mid])

# ===================== DENSE (ALWAYS raw text, computed once) =====================
print("Dense per-sample...", flush=True)
dense_o50 = {}; dense_s50 = {}
for qid in test_qids:
    sid = qa_sample_map.get(qid)
    if qid not in qid_to_idx or sid not in sample_mem_idx: continue
    qi = qid_to_idx[qid]; ci = sample_mem_idx[sid]
    scores = np.dot(mem_embs[ci], qa_embs[qi])
    order = np.argsort(-scores)
    dense_o50[qid] = [mem_ids_bge[ci[i]] for i in order[:50]]
    dense_s50[qid] = [float(scores[i]) for i in order[:50]]

# ===================== REPRESENTATION TEXT BUILDERS =====================
def format_erk(ents, rels, kws):
    parts = []
    if ents: parts.append(f"E: {ents}")
    if rels: parts.append(f"R: {rels}")
    if kws: parts.append(f"K: {kws}")
    return "\n".join(parts)

def format_triples(triples_text):
    return triples_text if triples_text else ""

def build_bm25_text(mid, rep):
    feats = p3_feats.get(mid, {})
    raw = memories[mid]["text"]
    
    if rep == "raw": return raw
    if rep == "summary": return f"Summary: {feats.get('summary','')}"
    if rep == "raw_summary": return f"{raw}\nSummary: {feats.get('summary','')}"
    if rep == "triples":
        tr = format_triples(feats.get('triples',''))
        return tr if tr else raw
    if rep == "raw_triples":
        tr = format_triples(feats.get('triples',''))
        return f"{raw}\n{tr}" if tr else raw
    if rep == "erk_only":
        return format_erk(feats.get('entities',''), feats.get('relations',''), feats.get('keywords',''))
    if rep == "raw_erk":
        return f"{raw}\n{format_erk(feats.get('entities',''), feats.get('relations',''), feats.get('keywords',''))}"
    return raw

# ===================== BM25 PER-SAMPLE (Reuses P2's exact BM25Retriever) =====================
def run_bm25(rep):
    """Per-sample BM25 with given representation text."""
    sample_bm = {}
    for sid, mem_list in sample_memories.items():
        texts = [build_bm25_text(m["memory_id"], rep) for m in mem_list]
        bm = BM25Retriever(k1=1.5, b=0.75)
        bm.fit(texts)
        sample_bm[sid] = (bm, mem_list)
    
    bm25_o50 = {}; bm25_s50 = {}
    for qid in test_qids:
        sid = qa_sample_map.get(qid)
        if sid not in sample_bm: continue
        bm, mem_list = sample_bm[sid]
        indices, svals = bm.search(qas[qid]["question"], top_k=50)
        bm25_o50[qid] = [mem_list[i]["memory_id"] for i in indices]
        bm25_s50[qid] = [float(svals[j]) for j in range(len(indices))]
    return bm25_o50, bm25_s50

# ===================== ZSCORE FUSION (Exact P2 implementation) =====================
def zscore_fuse(bm25_o50, bm25_s50, alpha=0.6, dense_o=None, dense_s=None):
    ranks = {}
    if dense_o is None: dense_o = dense_o50
    if dense_s is None: dense_s = dense_s50
    for qid in test_qids:
        d_ids = dense_o.get(qid, []); d_vals = dense_s.get(qid, [])
        b_ids = bm25_o50.get(qid, []); b_vals = bm25_s50.get(qid, [])
        all_ids = list(dict.fromkeys(d_ids + b_ids))
        d_m = {m: s for m, s in zip(d_ids, d_vals)}; b_m = {m: s for m, s in zip(b_ids, b_vals)}
        dm = statistics.mean(d_vals) if d_vals else 0; ds = max(statistics.stdev(d_vals), 1e-9) if len(d_vals) > 1 else 1
        bm2 = statistics.mean(b_vals) if b_vals else 0; bs = max(statistics.stdev(b_vals), 1e-9) if len(b_vals) > 1 else 1
        d_min = min(d_vals) if d_vals else 0; b_min = min(b_vals) if b_vals else 0
        sc = []
        for m in all_ids:
            dv = d_m.get(m, d_min); bv = b_m.get(m, b_min)
            sc.append((alpha * (dv - dm) / ds + (1 - alpha) * (bv - bm2) / bs, m))
        sc.sort(key=lambda x: (-x[0], x[1]))
        ranks[qid] = [m for _, m in sc[:10]]
    return ranks

def evaluate(ranks, qids):
    h1 = h10 = 0; rrs = []
    for qid in qids:
        top = ranks.get(qid, [])[:10]; gold = gold_map[qid]
        h1 += any(m in gold for m in top[:1]); h10 += any(m in gold for m in top)
        for rk, m in enumerate(top, 1):
            if m in gold: rrs.append(1.0 / rk); break
        else: rrs.append(0)
    n = len(qids)
    return {"R@1": round(h1/n, 4), "R@10": round(h10/n, 4), "MRR": round(statistics.mean(rrs), 4), "Hit@10": round(h10/n, 4), "n": n}

# ===================== GATE: raw_erk must match per-query =====================
print("\n=== GATE: raw_erk must match P2 ZScore_RawERK exactly ===", flush=True)
# Load canonical ZScore rankings
canonical = {}
with open("D:/memorytable/cassandra-kg-memory/scripts/memory/__pycache__/zscore_alpha0.6_test_rankings.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        qid = r["query_id"]
        if qid not in canonical: canonical[qid] = []
        canonical[qid].append(r["memory_id"])

# Compute raw_erk
bm25_o, bm25_s = run_bm25("raw_erk")
zs_ranks = zscore_fuse(bm25_o, bm25_s)

# Per-query check
per_query_mismatches = 0
for qid in test_qids:
    c10 = canonical.get(qid, [])[:10]
    z10 = zs_ranks.get(qid, [])[:10]
    if c10 != z10:
        per_query_mismatches += 1

zs_res = evaluate(zs_ranks, test_qids)
can_res = evaluate(canonical, test_qids)

print(f"  Canonical   MRR={can_res['MRR']:.4f} R@10={can_res['R@10']:.4f}")
print(f"  P3 raw_erk  MRR={zs_res['MRR']:.4f} R@10={zs_res['R@10']:.4f}")
print(f"  Per-query top10 mismatches: {per_query_mismatches}/{len(test_qids)}")

mrr_ok = abs(zs_res['MRR'] - 0.5462) <= 1e-4
r10_ok = abs(zs_res['R@10'] - 0.8070) <= 1e-4
pq_ok = per_query_mismatches == 0
gate_ok = mrr_ok and r10_ok and pq_ok
print(f"  MRR gate: {'PASS' if mrr_ok else 'FAIL'}  R@10 gate: {'PASS' if r10_ok else 'FAIL'}  Per-query: {'PASS' if pq_ok else 'FAIL'}")
if not gate_ok: print("GATE FAILED - stopping"); exit(1)
print("  GATE PASSED")

# ===================== RUN ALL 7 REPRESENTATIONS =====================
reps = ["raw", "summary", "raw_summary", "triples", "raw_triples", "erk_only", "raw_erk"]
print(f"\nRunning {len(reps)} representations...", flush=True)
results = {}
for rep in reps:
    ts = time.time()
    print(f"  {rep}...", end=" ", flush=True)
    bm25_o, bm25_s = run_bm25(rep)
    zs = zscore_fuse(bm25_o, bm25_s)
    r = evaluate(zs, test_qids)
    results[rep] = r
    print(f"MRR={r['MRR']:.4f} R@10={r['R@10']:.4f} ({time.time()-ts:.0f}s)", flush=True)

# ===================== WRITE RESULTS =====================
raw_res = results["raw"]
erk_res = results["raw_erk"]
print(f"\n=== Final Comparison ===")
for rep in reps:
    r = results[rep]
    d = r["MRR"] - raw_res["MRR"]
    print(f"  ZScore_{rep:15s} MRR={r['MRR']:.4f}  dMRR={d:+.4f}  R@10={r['R@10']:.4f}")

# Overall CSV
with (OUT/"test_representation_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["method","representation","n","R@1","R@10","MRR","Hit@10",
        "dMRR_vs_raw","dR@10_vs_raw","dMRR_vs_rawerk","dR@10_vs_rawerk"])
    w.writeheader()
    for rep in reps:
        r = results[rep]; m = f"ZScore_{rep}"
        w.writerow({"method": m, "representation": rep, "n": r["n"],
            "R@1": r["R@1"], "R@10": r["R@10"], "MRR": r["MRR"], "Hit@10": r["Hit@10"],
            "dMRR_vs_raw": round(r["MRR"] - raw_res["MRR"], 4),
            "dR@10_vs_raw": round(r["R@10"] - raw_res["R@10"], 4),
            "dMRR_vs_rawerk": round(r["MRR"] - erk_res["MRR"], 4),
            "dR@10_vs_rawerk": round(r["R@10"] - erk_res["R@10"], 4)})

print(f"\nWrote core results to: {OUT / 'test_representation_results.csv'}")
print(f"Best overall: {max(reps, key=lambda r: results[r]['MRR'])} at MRR={results[max(reps, key=lambda r: results[r]['MRR'])]['MRR']:.4f}")

# ===================== SAVE ZSCORE RANKS FOR BOOTSTRAP/DEV/READER =====================
zs_ranks = {}
for rep in reps:
    bm25_o, bm25_s = run_bm25(rep)
    zs_ranks[rep] = zscore_fuse(bm25_o, bm25_s)

# ===================== B. BOOTSTRAP =====================
print("\n=== B. Bootstrap ===", flush=True)
def bootstrap_paired(m1, m2, qids, n_reps=10000):
    rng = random.Random(42); n = len(qids)
    rrs1 = []; rrs2 = []
    for qid in qids:
        top1 = m1.get(qid, [])[:10]; top2 = m2.get(qid, [])[:10]; gold = gold_map[qid]
        rr1 = 0; rr2 = 0
        for rk, m in enumerate(top1, 1):
            if m in gold: rr1 = 1.0/rk; break
        for rk, m in enumerate(top2, 1):
            if m in gold: rr2 = 1.0/rk; break
        rrs1.append(rr1); rrs2.append(rr2)
    diffs = [rrs1[i] - rrs2[i] for i in range(n)]
    means = []
    for _ in range(n_reps):
        idx = [rng.randint(0, n-1) for __ in range(n)]
        means.append(statistics.mean([diffs[i] for i in idx]))
    means.sort()
    return round(statistics.mean(means), 4), round(means[250], 4), round(means[9750], 4)

comparisons = [
    ("ERKOnly_vs_RawERK", "erk_only", "raw_erk"),
    ("ERKOnly_vs_Raw", "erk_only", "raw"),
    ("ERKOnly_vs_RawTriples", "erk_only", "raw_triples"),
    ("ERKOnly_vs_StrongestSummary", "erk_only", "raw_summary" if results["raw_summary"]["MRR"] > results["summary"]["MRR"] else "summary"),
    ("RawERK_vs_Raw", "raw_erk", "raw"),
    ("RawERK_vs_RawTriples", "raw_erk", "raw_triples"),
    ("RawERK_vs_RawSummary", "raw_erk", "raw_summary"),
]
bs_rows = []
for label, m1n, m2n in comparisons:
    mq, lq, hq = bootstrap_paired(zs_ranks[m1n], zs_ranks[m2n], test_qids)
    bs_rows.append({"comparison": label, "m1": m1n, "m2": m2n,
        "query_mrr_delta": mq, "query_ci_lo": lq, "query_ci_hi": hq})

with (OUT/"p3_representation_bootstrap.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(bs_rows[0].keys())); w.writeheader(); w.writerows(bs_rows)

# ===================== C. DEV CONSISTENCY =====================
print("=== C. Dev Consistency ===", flush=True)
dev_qids = sorted([q for q in cat14 if qa_sample_map[q] in ["conv-42","conv-48"]])
do2, ds2 = {}, {}
for qid in dev_qids:
    sid = qa_sample_map.get(qid)
    if qid not in qid_to_idx or sid not in sample_mem_idx: continue
    qi = qid_to_idx[qid]; ci = sample_mem_idx[sid]
    scores = np.dot(mem_embs[ci], qa_embs[qi]); order = np.argsort(-scores)
    do2[qid] = [mem_ids_bge[ci[i]] for i in order[:50]]
    ds2[qid] = [float(scores[i]) for i in order[:50]]

dev_results = {}
for rep in reps:
    sample_bm = {}
    for sid, mem_list in sample_memories.items():
        texts = [build_bm25_text(m["memory_id"], rep) for m in mem_list]
        bm = BM25Retriever(k1=1.5, b=0.75); bm.fit(texts)
        sample_bm[sid] = (bm, mem_list)
    bo, bs_f = {}, {}
    for qid in dev_qids:
        sid = qa_sample_map.get(qid)
        if sid not in sample_bm: continue
        bm, mem_list = sample_bm[sid]
        indices, svals = bm.search(qas[qid]["question"], top_k=50)
        bo[qid] = [mem_list[i]["memory_id"] for i in indices]
        bs_f[qid] = [float(svals[j]) for j in range(len(indices))]
    # Inline ZScore fusion with dev_dense + dev_qids
    ranks = {}
    for qid in dev_qids:
        d_ids = do2.get(qid, []); d_vals = ds2.get(qid, [])
        b_ids = bo.get(qid, []); b_vals = bs_f.get(qid, [])
        all_ids = list(dict.fromkeys(d_ids + b_ids))
        d_m = {m: s for m, s in zip(d_ids, d_vals)}; b_m = {m: s for m, s in zip(b_ids, b_vals)}
        dm = statistics.mean(d_vals) if d_vals else 0; dsd = max(statistics.stdev(d_vals), 1e-9) if len(d_vals) > 1 else 1
        bm2 = statistics.mean(b_vals) if b_vals else 0; bsd = max(statistics.stdev(b_vals), 1e-9) if len(b_vals) > 1 else 1
        d_min = min(d_vals) if d_vals else 0; b_min = min(b_vals) if b_vals else 0
        sc = []
        for m in all_ids:
            dv = d_m.get(m, d_min); bv = b_m.get(m, b_min)
            sc.append((0.6*(dv-dm)/dsd + 0.4*(bv-bm2)/bsd, m))
        sc.sort(key=lambda x: (-x[0], x[1]))
        ranks[qid] = [m for _, m in sc[:10]]
    dev_results[rep] = evaluate(ranks, dev_qids)

with (OUT/"p3_dev_test_consistency.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["representation","dev_MRR","dev_R@10","test_MRR","test_R@10","dMRR_dev_test"])
    w.writeheader()
    for rep in reps:
        w.writerow({"representation": rep, "dev_MRR": dev_results[rep]["MRR"], "dev_R@10": dev_results[rep]["R@10"],
            "test_MRR": results[rep]["MRR"], "test_R@10": results[rep]["R@10"],
            "dMRR_dev_test": round(dev_results[rep]["MRR"] - results[rep]["MRR"], 4)})
    print(f"  ERKOnly dev={dev_results['erk_only']['MRR']:.4f} test={results['erk_only']['MRR']:.4f} diff={dev_results['erk_only']['MRR']-results['erk_only']['MRR']:+.4f}")

# ===================== D. READER CANDIDATES =====================
print("=== D. Reader Candidates ===", flush=True)
best_sum = max(["summary","raw_summary"], key=lambda r: results[r]["MRR"])
reader_reps = ["raw", "raw_erk", "erk_only", "raw_triples", best_sum]
reader_rows = []
for rep in reader_reps:
    zs = zs_ranks[rep]
    for qid in test_qids:
        top = zs.get(qid, [])[:10]
        reader_rows.append({"method": f"ZScore_{rep}", "qa_id": qid, "top10": ";".join(top)})
with (OUT/"p3_reader_candidate_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["method","qa_id","top10"]); w.writeheader(); w.writerows(reader_rows)

# ===================== FINAL DECISION =====================
best_r = max(reps, key=lambda r: results[r]["MRR"])
d = f"""# P3 Final Representation Decision

## Summary: 5882/5882 generated, 0 empty

## Best Representation: {best_r} (MRR={results[best_r]['MRR']:.4f}, R@10={results[best_r]['R@10']:.4f})

## Key MRR Values
- ERKOnly: {results['erk_only']['MRR']:.4f}
- RawERK: {results['raw_erk']['MRR']:.4f}
- RawTriples: {results['raw_triples']['MRR']:.4f}
- RawSummary: {results['raw_summary']['MRR']:.4f}
- Summary: {results['summary']['MRR']:.4f}
- Raw: {results['raw']['MRR']:.4f}

## ERKOnly vs RawERK Bootstrap
- Query CI: [{bs_rows[0]['query_ci_lo']:+.4f}, {bs_rows[0]['query_ci_hi']:+.4f}]

## Dev Consistency
- ERKOnly dev: {dev_results['erk_only']['MRR']:.4f} | test: {results['erk_only']['MRR']:.4f}

## Reader Candidates: {reader_reps}
"""
with (OUT/"p3_final_representation_decision.md").open("w") as f: f.write(d)
print(d)
print(f"Total runtime: {time.time()-t0:.0f}s")
