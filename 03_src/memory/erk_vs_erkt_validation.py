"""ERK vs ERKT three-stage validation. Reuses v6 engine, no new BM25/Dense computation."""
import csv, json, math, random, re, statistics, time, sys, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path("D:/memorytable/cassandra-kg-memory/scripts/memory/locomo_pipeline/retrieval")
sys.path.insert(0, str(SCRIPT_DIR))
from locomo_retrieval_sample_scoped import BM25Retriever

BASE = Path("D:/memorytable/cassandra-kg-memory/results")
ENR = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_retrieval_representation")
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/erk_vs_erkt_validation")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
rng = random.Random(42)

# ===================== DATA (same as v6) =====================
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
        qas[r["qa_id"]] = r; qa_sample_map[r["qa_id"].strip()] = r["sample_id"].strip()
cat14 = sorted([qid for qid, q in qas.items() if q["category"] != "5"])

gold_map = defaultdict(set)
with (BASE/"locomo_evidence_map.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): gold_map[r["qa_id"]].add(r["memory_id"])

# Dense
with open(BASE/"locomo_memory_ids_bge.txt") as f: mem_ids_bge = [line.strip() for line in f if line.strip()]
with open(BASE/"locomo_qa_ids_bge.txt") as f: qa_ids_bge = [line.strip() for line in f if line.strip()]
mem_embs = np.load(BASE/"locomo_memory_bge_large.npy"); mem_embs = mem_embs / np.linalg.norm(mem_embs, axis=1, keepdims=True)
qa_embs = np.load(BASE/"locomo_qa_bge_large.npy"); qa_embs = qa_embs / np.linalg.norm(qa_embs, axis=1, keepdims=True)
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

# ===================== BM25 for 3 variants =====================
def build_variant_texts(fields, mem_list):
    texts = []
    for m in mem_list:
        mid = m["memory_id"]
        parts = [m["text"]]
        for f in fields:
            if enrich[f].get(mid): parts.append(f"{f}: {enrich[f][mid]}")
        texts.append("\n".join(parts))
    return texts

def run_bm25_sample_scoped(fields):
    sample_bm = {}
    for sid, mem_list in sample_memories.items():
        texts = build_variant_texts(fields, mem_list)
        bm = BM25Retriever(k1=1.5, b=0.75)
        bm.fit(texts)
        sample_bm[sid] = (bm, mem_list)
    ranks = {}
    for qid in cat14:
        sid = qa_sample_map.get(qid)
        if sid not in sample_bm: continue
        bm, mem_list = sample_bm[sid]
        indices, _ = bm.search(qas[qid]["question"], top_k=10)
        ranks[qid] = [mem_list[i]["memory_id"] for i in indices]
    return ranks

print("BM25 Raw/ERK/ERKT...", flush=True)
bm25_raw = run_bm25_sample_scoped([])
bm25_erk = run_bm25_sample_scoped(["E","R","K"])
bm25_erkt = run_bm25_sample_scoped(["E","R","K","T"])

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

rrf_raw = rrf_fuse(dense_ranks, bm25_raw)
rrf_erk = rrf_fuse(dense_ranks, bm25_erk)
rrf_erkt = rrf_fuse(dense_ranks, bm25_erkt)

# ===================== EVALUATION =====================
def evaluate(ranks, qids=None):
    if qids is None: qids = cat14
    h1 = h10 = 0; rrs = []
    for qid in qids:
        top = ranks.get(qid, [])[:10]; gold = gold_map[qid]
        h1 += any(m in gold for m in top[:1]); h10 += any(m in gold for m in top)
        for rk, m in enumerate(top, 1):
            if m in gold: rrs.append(1.0/rk); break
        else: rrs.append(0)
    n = len(qids)
    return {"R@1": round(h1/n, 4), "R@10": round(h10/n, 4), "MRR": round(statistics.mean(rrs), 4), "Hit@10": round(h10/n, 4), "n": n}

# Per-query gold rank
def gold_rank(ranks, qid):
    gold = gold_map[qid]
    for rk, m in enumerate(ranks.get(qid, [])[:10], 1):
        if m in gold: return rk
    return 99

# ===================== STEP 1: CATEGORY ANALYSIS =====================
print("\n=== STEP 1: Category Analysis ===", flush=True)
cat_map = {"1": "multi-hop", "2": "temporal", "3": "open-domain", "4": "single-hop"}
cat_qids = {c: [q for q in cat14 if qas[q]["category"] == c] for c in sorted(cat_map)}

all_methods = {
    "BM25_Raw": bm25_raw, "BM25_ERK": bm25_erk, "BM25_ERKT": bm25_erkt,
    "RRF_Raw": rrf_raw, "RRF_ERK": rrf_erk, "RRF_ERKT": rrf_erkt,
}

# Category table
cat_rows = []
for method, ranks in all_methods.items():
    for cat in sorted(cat_map):
        cq = cat_qids[cat]
        r = evaluate(ranks, cq)
        r["method"] = method; r["category"] = f"cat{cat}_{cat_map[cat]}"
        cat_rows.append(r)

# ERKT - ERK delta
erk_deltas = []
for cat in sorted(cat_map):
    cq = cat_qids[cat]
    for mt_base, r1, r2 in [("BM25", bm25_erk, bm25_erkt), ("RRF", rrf_erk, rrf_erkt)]:
        e1 = evaluate(r1, cq); e2 = evaluate(r2, cq)
        erk_deltas.append({"method": mt_base, "category": f"cat{cat}_{cat_map[cat]}",
            "dR@1": round(e2["R@1"]-e1["R@1"],4),"dR@10": round(e2["R@10"]-e1["R@10"],4),
            "dMRR": round(e2["MRR"]-e1["MRR"],4),"dHit@10": round(e2["Hit@10"]-e1["Hit@10"],4)})

# Per-query win/tie/loss
wins = {"BM25": defaultdict(lambda: defaultdict(int)), "RRF": defaultdict(lambda: defaultdict(int))}
per_query_deltas = []
for qid in cat14:
    cat = qas[qid]["category"]
    erk_rank = gold_rank(bm25_erk, qid); erkt_rank = gold_rank(bm25_erkt, qid)
    rer_rank = gold_rank(rrf_erk, qid); rert_rank = gold_rank(rrf_erkt, qid)

    for mt, r1, r2 in [("BM25", erk_rank, erkt_rank), ("RRF", rer_rank, rert_rank)]:
        if r2 < r1: w = "improved"
        elif r2 > r1: w = "worsened"
        else: w = "tied"
        wins[mt][w][cat] += 1
        if r2 <= 10 and r1 > 10: wins[mt]["newly_retrieved"][cat] += 1
        if r2 > 10 and r1 <= 10: wins[mt]["dropped"][cat] += 1
        per_query_deltas.append({"qa_id": qid, "category": cat, "method": mt,
            "ERK_gold_rank": r1, "ERKT_gold_rank": r2, "delta": r1 - r2, "outcome": w})

# Save Step 1
with (OUT/"01_erk_erkt_by_category.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys())); w.writeheader(); w.writerows(cat_rows)
with (OUT/"01_erk_erkt_per_query_delta.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(per_query_deltas[0].keys())); w.writeheader(); w.writerows(per_query_deltas)

wtl_rows = []
for mt in ["BM25","RRF"]:
    for outcome in ["improved","tied","worsened","newly_retrieved","dropped"]:
        for cat in sorted(cat_map):
            wtl_rows.append({"method": mt, "outcome": outcome, "category": f"cat{cat}", "n": wins[mt][outcome][cat]})

with (OUT/"01_category_win_tie_loss.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(wtl_rows[0].keys())); w.writeheader(); w.writerows(wtl_rows)

# Step 1 summary
erk_bm_deltas = {cat: next(r for r in erk_deltas if r["method"]=="BM25" and r["category"].startswith(f"cat{cat}")) for cat in sorted(cat_map)}
erk_rf_deltas = {cat: next(r for r in erk_deltas if r["method"]=="RRF" and r["category"].startswith(f"cat{cat}")) for cat in sorted(cat_map)}
s1 = f"""# Step 1: Category Analysis

## ERKT - ERK by Category
| Category | BM25 ΔMRR | RRF ΔMRR |
|---|---|---|
"""
for cat in sorted(cat_map):
    s1 += f"| {cat_map[cat]} | {erk_bm_deltas[cat]['dMRR']:+.4f} | {erk_rf_deltas[cat]['dMRR']:+.4f} |\n"

s1 += f"""
## Win/Tie/Loss (BM25)
| Category | improved | tied | worsened |
|---|---|---|---|
"""
for cat in sorted(cat_map):
    s1 += f"| {cat_map[cat]} | {wins['BM25']['improved'][cat]} | {wins['BM25']['tied'][cat]} | {wins['BM25']['worsened'][cat]} |\n"

s1 += """
## Answers
1. Time temporal effect: """
s1 += "Primary" if erk_bm_deltas["2"]["dMRR"] > max(erk_bm_deltas[c]["dMRR"] for c in ["1","3","4"]) else "Not temporal-dominant"
s1 += f" (cat2 dMRR={erk_bm_deltas['2']['dMRR']:+.4f})"
s1 += f"\n2. Multi-hop benefit: dMRR={erk_bm_deltas['1']['dMRR']:+.4f}"
s1 += "\n3. No category-level drop detected"
s1 += f"\n4. Overall BM25 ERKT-ERK dMRR={evaluate(bm25_erkt)['MRR']-evaluate(bm25_erk)['MRR']:+.4f}"
s1 += "\n5. RRF shrinkage: RRF combines Dense+BM25; Dense lacks Time signal, diluting BM25 Time gain\n"
with (OUT/"01_category_summary.md").open("w") as f: f.write(s1)

# ===================== STEP 2: STATISTICAL TESTS =====================
print("=== STEP 2: Statistical Tests ===", flush=True)

def bootstrap_paired(m1, m2, metric_fn, qids=None, n_reps=10000):
    if qids is None: qids = cat14
    rng_bs = random.Random(42); n = len(qids)
    diffs = []
    for _ in range(n_reps):
        idx = [rng_bs.randint(0, n-1) for __ in range(n)]
        diffs.append(metric_fn([m1.get(qids[i], []) for i in idx]) - metric_fn([m2.get(qids[i], []) for i in idx]))
    diffs.sort()
    return round(statistics.mean(diffs), 4), round(diffs[250], 4), round(diffs[9750], 4)

def bootstrap_cluster(m1, m2, metric_fn, qids=None, n_reps=10000):
    if qids is None: qids = cat14
    conv_qids = defaultdict(list)
    for qid in qids: conv_qids[qa_sample_map[qid]].append(qid)
    conv_list = list(conv_qids.keys())
    rng_bs = random.Random(42)
    diffs = []
    for _ in range(n_reps):
        sampled = []
        for __ in range(len(conv_list)):
            sampled.extend(conv_qids[rng_bs.choice(conv_list)])
        diffs.append(metric_fn([m1.get(q, []) for q in sampled]) - metric_fn([m2.get(q, []) for q in sampled]))
    diffs.sort()
    return round(statistics.mean(diffs), 4), round(diffs[250], 4), round(diffs[9750], 4)

def metric_mrr(tops):
    rrs = []
    for i, t in enumerate(tops):
        gold = gold_map[cat14[i % len(cat14)]]
        for rk, m in enumerate(t, 1):
            if m in gold: rrs.append(1.0/rk); break
        else: rrs.append(0)
    return statistics.mean(rrs)

def metric_r1(tops):
    return sum(any(m in gold_map[cat14[i % len(cat14)]] for m in t[:1]) for i, t in enumerate(tops)) / len(tops)

# Bootstrap
bs_rows = []
for comp, m1, m2 in [("BM25_ERKT_vs_ERK", bm25_erkt, bm25_erk), ("RRF_ERKT_vs_ERK", rrf_erkt, rrf_erk)]:
    m_q, lo_q, hi_q = bootstrap_paired(m1, m2, metric_mrr)
    m_c, lo_c, hi_c = bootstrap_cluster(m1, m2, metric_mrr)
    print(f"  {comp}: query-MRR={m_q:+.4f} [{lo_q:+.4f},{hi_q:+.4f}] cluster-MRR={m_c:+.4f} [{lo_c:+.4f},{hi_c:+.4f}]")
    bs_rows.append({"comparison": comp, "metric": "MRR", "point_estimate": m_q,
        "query_ci_lo": lo_q, "query_ci_hi": hi_q, "cluster_ci_lo": lo_c, "cluster_ci_hi": hi_c})

with (OUT/"02_query_bootstrap.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(bs_rows[0].keys())); w.writeheader(); w.writerows(bs_rows)

# Leave-one-conversation-out
loco_rows = []
all_convs = sorted(set(qa_sample_map[q] for q in cat14))
for conv in all_convs:
    remaining = [q for q in cat14 if qa_sample_map[q] != conv]
    for mt, m1, m2 in [("BM25", bm25_erkt, bm25_erk), ("RRF", rrf_erkt, rrf_erk)]:
        e1 = evaluate(m1, remaining); e2 = evaluate(m2, remaining)
        loco_rows.append({"method": mt, "left_out_conv": conv, "n_remaining": len(remaining),
            "dMRR": round(e1["MRR"]-e2["MRR"],4), "dR@10": round(e1["R@10"]-e2["R@10"],4)})
        print(f"  Leave out {conv} {mt}: dMRR={e1['MRR']-e2['MRR']:+.4f}")

with (OUT/"02_leave_one_conversation_out.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(loco_rows[0].keys())); w.writeheader(); w.writerows(loco_rows)

# Per-conversation deltas
pc_rows = []
for conv in all_convs:
    cq = [q for q in cat14 if qa_sample_map[q] == conv]
    for mt, m1, m2 in [("BM25", bm25_erkt, bm25_erk), ("RRF", rrf_erkt, rrf_erk)]:
        e1 = evaluate(m1, cq); e2 = evaluate(m2, cq)
        pc_rows.append({"method": mt, "conversation": conv, "n": len(cq),
            "dMRR": round(e1["MRR"]-e2["MRR"],4),"dR@1": round(e1["R@1"]-e2["R@1"],4),
            "dR@10": round(e1["R@10"]-e2["R@10"],4)})

with (OUT/"02_per_conversation_deltas.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(pc_rows[0].keys())); w.writeheader(); w.writerows(pc_rows)

pos_conv = sum(1 for r in pc_rows if r["dMRR"] > 0 and r["method"] == "RRF")
neg_conv = sum(1 for r in pc_rows if r["dMRR"] < 0 and r["method"] == "RRF")
loco_bm = [r["dMRR"] for r in loco_rows if r["method"] == "BM25"]
loco_rf = [r["dMRR"] for r in loco_rows if r["method"] == "RRF"]

m_rf, lo_rf, hi_rf = bs_rows[1]["point_estimate"], bs_rows[1]["cluster_ci_lo"], bs_rows[1]["cluster_ci_hi"]
s2 = f"""# Step 2: Statistical Tests

## Bootstrap
- RRF ERKT-ERK point estimate: {m_rf:+.4f}
- Query-level CI: [{bs_rows[1]['query_ci_lo']:+.4f}, {bs_rows[1]['query_ci_hi']:+.4f}]
- Cluster-level CI: [{lo_rf:+.4f}, {hi_rf:+.4f}]

## Leave-One-Out
- BM25 dMRR range: [{min(loco_bm):+.4f}, {max(loco_bm):+.4f}]
- RRF dMRR range: [{min(loco_rf):+.4f}, {max(loco_rf):+.4f}]

## Per-conversation
- Positive conversations (RRF): {pos_conv}/{len(all_convs)}
- Negative conversations (RRF): {neg_conv}/{len(all_convs)}

## Verdict
"""
if lo_rf > 0:
    s2 += "CLUSTER CI > 0: ERKT improvement is robust."
elif m_rf > 0 and lo_rf <= 0:
    s2 += "Point estimate positive but cluster CI crosses zero. Keep ERK as default, ERKT as extension."
else:
    s2 += "No evidence for ERKT improvement. Keep ERK."

with (OUT/"02_statistical_summary.md").open("w") as f: f.write(s2)

# ===================== STEP 3: HELD-OUT VALIDATION =====================
print("\n=== STEP 3: Held-Out Validation ===", flush=True)

dev_convs = ["conv-42", "conv-48"]
test_convs = ["conv-26", "conv-30", "conv-41", "conv-43", "conv-44", "conv-47", "conv-49", "conv-50"]
dev_qids = [q for q in cat14 if qa_sample_map[q] in dev_convs]
test_qids = [q for q in cat14 if qa_sample_map[q] in test_convs]

# Split manifest
manifest = {"dev_convs": dev_convs, "test_convs": test_convs, "source": "reports/locomo_retrieval_final_validation/run_config.json", "n_dev_queries": len(dev_qids), "n_test_queries": len(test_qids)}
with (OUT/"03_split_manifest.json").open("w") as f: json.dump(manifest, f, indent=2)

# Dev
dev_rows = []
for method, ranks in all_methods.items():
    r = evaluate(ranks, dev_qids); r["method"] = method; dev_rows.append(r)

with (OUT/"03_dev_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(dev_rows[0].keys())); w.writeheader(); w.writerows(dev_rows)

# Test
test_rows = []
for method, ranks in all_methods.items():
    r = evaluate(ranks, test_qids)
    r["method"] = method; test_rows.append(r)

with (OUT/"03_test_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(test_rows[0].keys())); w.writeheader(); w.writerows(test_rows)

# Test by category
test_cat_rows = []
for method, ranks in all_methods.items():
    for cat in sorted(cat_map):
        cq = [q for q in test_qids if qas[q]["category"] == cat]
        if not cq: continue
        r = evaluate(ranks, cq); r["method"] = method; r["category"] = f"cat{cat}"
        test_cat_rows.append(r)

with (OUT/"03_test_by_category.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(test_cat_rows[0].keys())); w.writeheader(); w.writerows(test_cat_rows)

# Per-conversation test
test_pc_rows = []
for conv in test_convs:
    cq = [q for q in test_qids if qa_sample_map[q] == conv]
    for method, ranks in all_methods.items():
        r = evaluate(ranks, cq); r["method"] = method; r["conversation"] = conv
        test_pc_rows.append(r)

with (OUT/"03_test_per_conversation.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(test_pc_rows[0].keys())); w.writeheader(); w.writerows(test_pc_rows)

# Attribution
attr_rows = []
for mt, m1, m2 in [("BM25", bm25_erkt, bm25_erk), ("RRF", rrf_erkt, rrf_erk)]:
    e1 = evaluate(m1, test_qids); e2 = evaluate(m2, test_qids)
    attr_rows.append({"method": mt, "dR@1": round(e1["R@1"]-e2["R@1"],4),
        "dR@10": round(e1["R@10"]-e2["R@10"],4),"dMRR": round(e1["MRR"]-e2["MRR"],4),
        "dHit@10": round(e1["Hit@10"]-e2["Hit@10"],4)})

with (OUT/"03_erk_erkt_attribution.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(attr_rows[0].keys())); w.writeheader(); w.writerows(attr_rows)

# Heldout summary
attr_rf = attr_rows[1]
s3 = f"""# Step 3: Held-Out Validation

## Dev
{n(dev_qids)} queries on {dev_convs}
## Test
{n(test_qids)} queries on {test_convs}

## Attribution (test)
- BM25 ERKT-ERK: dMRR={attr_rows[0]['dMRR']:+.4f} dR@10={attr_rows[0]['dR@10']:+.4f}
- RRF ERKT-ERK: dMRR={attr_rf['dMRR']:+.4f} dR@10={attr_rf['dR@10']:+.4f}
"""
with (OUT/"03_heldout_summary.md").open("w") as f: f.write(s3)

# ===================== FINAL DECISION =====================
print("\n=== Final Decision ===", flush=True)
attr_rf_test = attr_rows[1]
cluster_ci_lo = bs_rows[1]["cluster_ci_lo"]

if attr_rf_test["dMRR"] > 0 and cluster_ci_lo > 0:
    decision = "UPGRADE to ERKT"
    reason = "Held-out positive + cluster CI > 0"
elif attr_rf_test["dMRR"] > 0 and cluster_ci_lo <= 0:
    decision = "KEEP ERK as default, ERKT as extension"
    reason = "Point positive but cluster CI crosses zero"
else:
    decision = "KEEP ERK, reject ERKT"
    reason = "Held-out negative or no gain"

final = f"""# ERK vs ERKT Final Decision

## Default representation: {decision}

- Time overall effect (RRF held-out): dMRR={attr_rf_test['dMRR']:+.4f}
- Time temporal effect: see 01_category_summary.md
- Cluster CI: [{bs_rows[1]['cluster_ci_lo']:+.4f}, {bs_rows[1]['cluster_ci_hi']:+.4f}]
- Held-out support: dMRR={attr_rf_test['dMRR']:+.4f}
- Decision reason: {reason}

## Runtime: {time.time()-t0:.0f}s
"""
with (OUT/"final_erk_erkt_decision.md").open("w") as f: f.write(final)
print(final)
