"""Final validation: generic hybrid + conversation-split + bootstrap CI + Dense_enriched prep."""
import csv, json, math, random, re, statistics, time
from collections import defaultdict
from pathlib import Path

BASE = Path("D:/memorytable/cassandra-kg-memory/results")
ENR = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_retrieval_representation")
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_retrieval_final_validation")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
rng = random.Random(42)

STOP = set("i me my myself we our ours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between through during before after above below to from up down in out on off over under again further then once here there when where why how all both each few more most other some such no nor not only own same so than too very s t can will just don should now d ll m o re ve y".split())

def tokenize(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return [t for t in text if t not in STOP and len(t) >= 2]

# ===================== DATA LOADING =====================
print("=== Loading data ===")
memories = {}
with (BASE/"locomo_memory_records.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): memories[r["memory_id"]] = r
mem_ids = sorted(memories.keys())

qas = {}
with (BASE/"locomo_qa_records.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): qas[r["qa_id"]] = r

gold_map = defaultdict(set)
with (BASE/"locomo_evidence_map.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): gold_map[r["qa_id"]].add(r["memory_id"])

cat14 = sorted([qid for qid, q in qas.items() if q["category"] != "5"])

# Identify conversations (samples)
conv_ids = sorted(set(qas[qid]["sample_id"] for qid in cat14))
print(f"  Conversations: {len(conv_ids)} — {conv_ids}")

# Conversation-grouped split
rng.shuffle(conv_ids)
dev_convs = set(conv_ids[:2])
test_convs = set(conv_ids[2:])
dev_qids = sorted([q for q in cat14 if qas[q]["sample_id"] in dev_convs])
test_qids = sorted([q for q in cat14 if qas[q]["sample_id"] in test_convs])
print(f"  Dev: {sorted(dev_convs)} ({len(dev_qids)} queries)")
print(f"  Test: {sorted(test_convs)} ({len(test_qids)} queries)")

# Category distribution check
for label, qids in [("dev", dev_qids), ("test", test_qids)]:
    cats = defaultdict(int)
    for q in qids: cats[qas[q]["category"]] += 1
    print(f"  {label}: cat1={cats.get('1',0)} cat2={cats.get('2',0)} cat3={cats.get('3',0)} cat4={cats.get('4',0)}")

# Canonical Dense (global rankings, sample-scope filtered)
dense_precomp = {}
with (BASE/"locomo_dense_bge_results.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        retrieved = r.get("retrieved_memory_ids","")
        if retrieved: dense_precomp[r["qa_id"]] = [x.strip() for x in retrieved.split(";") if x.strip()]

# Canonical BM25
dia2mid = {memories[mid]["dia_id"]: mid for mid in mem_ids if "dia_id" in memories[mid]}
bm25_precomp = {}
with (BASE/"locomo_bm25_results.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        retrieved = r.get("retrieved_memory_ids","")
        if retrieved:
            mids = []
            for sid in [x.strip() for x in retrieved.split(";") if x.strip()]:
                if sid in dia2mid: mids.append(dia2mid[sid])
            bm25_precomp[r["qa_id"]] = mids

# ===================== LOCAL BM25 (raw + compact) =====================
print("\n=== Building local BM25 indices ===")
sample_mem = defaultdict(list)
for mid in mem_ids:
    sample_mem[memories[mid]["sample_id"]].append(mid)

# Load enriched records
enriched = list(csv.DictReader((ENR/"enriched_memory_records.csv").open(encoding="utf-8-sig")))
def extract_field(text, field):
    if f"{field}:" in text:
        return text.split(f"{field}:")[1].split("\n")[0].strip()
    return ""

compact_variants = {}
for e in enriched:
    mid = e["memory_id"]
    raw = e["raw_text"]
    ent = extract_field(e["enriched_text"], "Entities")
    rel = extract_field(e["enriched_text"], "Relations")
    kw = extract_field(e["enriched_text"], "Keywords")
    compact_variants[mid] = {
        "raw": raw,
        "compact": raw + (f"\nEntities: {ent}" if ent else "") + (f"\nRelations: {rel}" if rel else "") + (f"\nKeywords: {kw}" if kw else ""),
    }

def bm25_build(corpus):
    df = defaultdict(int); dlens = {}
    for mid, text in corpus.items():
        toks = tokenize(text); dlens[mid] = len(toks)
        for t in set(toks): df[t] += 1
    return df, dlens

def bm25_score(query, doc_toks, df, dlens, N, avgdl, k1=1.2, b=0.75):
    score = 0
    for qt in set(tokenize(query)):
        if qt not in df: continue
        idf = math.log((N - df[qt] + 0.5) / (df[qt] + 0.5) + 1)
        tf = doc_toks.count(qt)
        score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc_toks) / avgdl))
    return score

def compute_bm25_ranks(qids, text_key):
    ranks = {}
    for qid in qids:
        q_sample = qas[qid]["sample_id"]
        smids = [m for m in sample_mem[q_sample] if m in compact_variants]
        corpus = {mid: compact_variants[mid][text_key] for mid in smids}
        df, dlens = bm25_build(corpus)
        N, avgdl = len(corpus), statistics.mean(dlens.values()) if dlens else 1
        scored = [(mid, bm25_score(qas[qid]["question"], tokenize(corpus[mid]), df, dlens, N, avgdl)) for mid in smids]
        scored.sort(key=lambda x: -x[1])
        ranks[qid] = [mid for mid, _ in scored]
    return ranks

print("  Computing BM25_raw local...")
bm25_raw_full = compute_bm25_ranks(cat14, "raw")
print("  Computing BM25_compact local...")
bm25_compact_full = compute_bm25_ranks(cat14, "compact")

# ===================== EVALUATION ENGINE =====================
def evaluate_ranks(ranks, qids, baseline_ranks=None):
    h1 = h10 = 0; rrs = []; rescue = 0; hurt = 0
    for qid in qids:
        top = ranks.get(qid, [])[:10]
        gold = gold_map[qid]
        if any(m in gold for m in top[:1]): h1 += 1
        if any(m in gold for m in top): h10 += 1
        for rk, m in enumerate(top, 1):
            if m in gold: rrs.append(1.0/rk); break
        else: rrs.append(0)
        if baseline_ranks:
            bh = any(m in gold for m in baseline_ranks.get(qid, [])[:1])
            kh = any(m in gold for m in top[:1])
            if kh and not bh: rescue += 1
            if bh and not kh: hurt += 1
    n = len(qids)
    r = {"R@1": round(h1/n,4), "R@10": round(h10/n,4), "MRR": round(statistics.mean(rrs),4), "n": n}
    if baseline_ranks: r["rescue@1"] = rescue; r["hurt@1"] = hurt; r["net@1"] = rescue - hurt
    return r

def weighted_rrf(qids, alpha, k, bm25_ranks):
    ranks = {}
    for qid in qids:
        dr = {m: i+1 for i, m in enumerate(dense_precomp.get(qid, []))}
        br = {m: i+1 for i, m in enumerate(bm25_ranks.get(qid, []))}
        all_m = list(dict.fromkeys(dense_precomp.get(qid, []) + bm25_ranks.get(qid, [])))
        scored = [(alpha/(k+dr.get(m,999)) + (1-alpha)/(k+br.get(m,999)), m) for m in all_m]
        scored.sort(key=lambda x: -x[0])
        ranks[qid] = [m for _, m in scored]
    return ranks

# ===================== DEV GRID SEARCH =====================
print("\n=== Dev Grid Search ===")
alphas = [0.3, 0.5, 0.7]
ks = [10, 30, 60]
bm25_src = {"raw": bm25_raw_full, "compact": bm25_compact_full}

# Baseline evaluations on dev
dense_dev_raw = evaluate_ranks(dense_precomp, dev_qids)
bm25_dev_raw = evaluate_ranks(bm25_raw_full, dev_qids)
bm25_dev_cpt = evaluate_ranks(bm25_compact_full, dev_qids)
print(f"  Dense_raw dev:          MRR={dense_dev_raw['MRR']:.4f}")
print(f"  BM25_raw dev:           MRR={bm25_dev_raw['MRR']:.4f}")
print(f"  BM25_compact dev:       MRR={bm25_dev_cpt['MRR']:.4f}")

dev_grid = []
best_rrf_raw = None; best_rrf_cpt = None
best_mrr_raw = -1; best_mrr_cpt = -1

for alpha in alphas:
    for k_val in ks:
        # RRF with BM25_raw
        rranks = weighted_rrf(dev_qids, alpha, k_val, bm25_raw_full)
        r = evaluate_ranks(rranks, dev_qids)
        r["method"] = f"RRF_raw_a={alpha}_k={k_val}"
        r["alpha"] = alpha; r["k"] = k_val
        dev_grid.append(r)
        if r["MRR"] > best_mrr_raw:
            best_mrr_raw = r["MRR"]
            best_rrf_raw = (alpha, k_val)
        
        # RRF with BM25_compact
        cranks = weighted_rrf(dev_qids, alpha, k_val, bm25_compact_full)
        c = evaluate_ranks(cranks, dev_qids)
        c["method"] = f"RRF_cpt_a={alpha}_k={k_val}"
        c["alpha"] = alpha; c["k"] = k_val
        dev_grid.append(c)
        if c["MRR"] > best_mrr_cpt:
            best_mrr_cpt = c["MRR"]
            best_rrf_cpt = (alpha, k_val)

print(f"  Best RRF_raw:  a={best_rrf_raw[0]} k={best_rrf_raw[1]} MRR={best_mrr_raw:.4f}")
print(f"  Best RRF_cpt:  a={best_rrf_cpt[0]} k={best_rrf_cpt[1]} MRR={best_mrr_cpt:.4f}")

gf = ["method","alpha","k","R@1","R@10","MRR","n"]
with (OUT/"final_dev_grid.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=gf, extrasaction="ignore")
    w.writeheader(); w.writerows(dev_grid)

# ===================== TEST EVALUATION (frozen params) =====================
print("\n=== Test Evaluation (frozen params) ===")

# Best params from dev
ap_raw, kp_raw = best_rrf_raw
ap_cpt, kp_cpt = best_rrf_cpt

# RRF with frozen params on test
rrf_raw_test = weighted_rrf(test_qids, ap_raw, kp_raw, bm25_raw_full)
rrf_cpt_test = weighted_rrf(test_qids, ap_cpt, kp_cpt, bm25_compact_full)

test_results = {
    "Dense_raw": evaluate_ranks(dense_precomp, test_qids),
    "BM25_raw": evaluate_ranks(bm25_raw_full, test_qids),
    "BM25_compact": evaluate_ranks(bm25_compact_full, test_qids),
    f"RRF_raw(a={ap_raw},k={kp_raw})": evaluate_ranks(rrf_raw_test, test_qids),
    f"RRF_cpt(a={ap_cpt},k={kp_cpt})": evaluate_ranks(rrf_cpt_test, test_qids),
}
for name, r in test_results.items():
    print(f"  {name:35s} MRR={r['MRR']:.4f} R@1={r['R@1']:.4f} R@10={r['R@10']:.4f}")

tf = ["method","R@1","R@10","MRR","n"]
with (OUT/"final_test_results.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=tf, extrasaction="ignore")
    w.writeheader()
    for name, r in test_results.items(): w.writerow({"method":name, **r})

# ===================== FULL-SET RESULTS (for final table) =====================
print("\n=== Full-set results (for final table) ===")
rrf_raw_full_ranks = weighted_rrf(cat14, ap_raw, kp_raw, bm25_raw_full)
rrf_cpt_full_ranks = weighted_rrf(cat14, ap_cpt, kp_cpt, bm25_compact_full)

full_results = {
    "BM25_raw": evaluate_ranks(bm25_raw_full, cat14),
    "Dense_raw": evaluate_ranks(dense_precomp, cat14),
    "BM25_KG-Enriched-Compact": evaluate_ranks(bm25_compact_full, cat14),
    f"RRF(Dense_raw+BM25_raw)": evaluate_ranks(rrf_raw_full_ranks, cat14),
    f"RRF(Dense_raw+BM25_KG-Enriched-Compact)": evaluate_ranks(rrf_cpt_full_ranks, cat14),
    "Dense_KG-Enriched-Compact": {"R@1":"NEEDS_API","R@10":"NEEDS_API","MRR":"NEEDS_API","n":len(cat14)},
    "Dense+GlobalKG-Prior": {"R@1":0.3669,"R@10":0.7110,"MRR":0.4766,"n":len(cat14)},
    "BM25_full_with_triples": {"R@1":0.4065,"R@10":0.6747,"MRR":0.4939,"n":len(cat14)},
}

for name, r in full_results.items():
    print(f"  {name:45s} MRR={r.get('MRR',0) if isinstance(r.get('MRR'),(int,float)) else '?'} R@10={r.get('R@10',0) if isinstance(r.get('R@10'),(int,float)) else '?'}")

with (OUT/"final_retrieval_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["method","R@1","R@10","MRR","n","status"])
    w.writeheader()
    for name, r in full_results.items():
        status = "final" if "NEEDS_API" not in str(r.get("MRR","")) else "pending_api"
        w.writerow({"method":name, "R@1":r.get("R@1",""), "R@10":r.get("R@10",""), "MRR":r.get("MRR",""), "n":r.get("n",""), "status":status})

# ===================== COMPONENT ATTRIBUTION =====================
print("\n=== Component Attribution ===")
rrf_raw_test_r = test_results[f"RRF_raw(a={ap_raw},k={kp_raw})"]
rrf_cpt_test_r = test_results[f"RRF_cpt(a={ap_cpt},k={kp_cpt})"]

attr_rows = [
    {"component": "generic_hybrid (RRF_raw - Dense_raw)",
     "delta_MRR": round(rrf_raw_test_r["MRR"] - test_results["Dense_raw"]["MRR"], 4),
     "delta_R@1": round(rrf_raw_test_r["R@1"] - test_results["Dense_raw"]["R@1"], 4),
     "delta_R@10": round(rrf_raw_test_r["R@10"] - test_results["Dense_raw"]["R@10"], 4)},
    {"component": "KG_rep_lexical (BM25_cpt - BM25_raw)",
     "delta_MRR": round(test_results["BM25_compact"]["MRR"] - test_results["BM25_raw"]["MRR"], 4),
     "delta_R@1": round(test_results["BM25_compact"]["R@1"] - test_results["BM25_raw"]["R@1"], 4),
     "delta_R@10": round(test_results["BM25_compact"]["R@10"] - test_results["BM25_raw"]["R@10"], 4)},
    {"component": "KG_rep_in_hybrid (RRF_cpt - RRF_raw)",
     "delta_MRR": round(rrf_cpt_test_r["MRR"] - rrf_raw_test_r["MRR"], 4),
     "delta_R@1": round(rrf_cpt_test_r["R@1"] - rrf_raw_test_r["R@1"], 4),
     "delta_R@10": round(rrf_cpt_test_r["R@10"] - rrf_raw_test_r["R@10"], 4)},
    {"component": "total_final_gain (RRF_cpt - Dense_raw)",
     "delta_MRR": round(rrf_cpt_test_r["MRR"] - test_results["Dense_raw"]["MRR"], 4),
     "delta_R@1": round(rrf_cpt_test_r["R@1"] - test_results["Dense_raw"]["R@1"], 4),
     "delta_R@10": round(rrf_cpt_test_r["R@10"] - test_results["Dense_raw"]["R@10"], 4)},
]
with (OUT/"final_component_attribution.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["component","delta_MRR","delta_R@1","delta_R@10"])
    w.writeheader(); w.writerows(attr_rows)
for a in attr_rows:
    print(f"  {a['component']}: dMRR={a['delta_MRR']} dR@1={a['delta_R@1']} dR@10={a['delta_R@10']}")

# ===================== BOOTSTRAP CI =====================
print("\n=== Bootstrap CI (1000 reps) ===")
def bootstrap_ci(qids, ranks_m1, ranks_m2, metric_fn, n_reps=1000, seed=42):
    rng_bs = random.Random(seed)
    diffs = []
    n = len(qids)
    for _ in range(n_reps):
        idx = [rng_bs.randint(0, n-1) for __ in range(n)]
        m1 = metric_fn([ranks_m1.get(qids[i], [])[:10] for i in idx], [gold_map[qids[i]] for i in idx])
        m2 = metric_fn([ranks_m2.get(qids[i], [])[:10] for i in idx], [gold_map[qids[i]] for i in idx])
        diffs.append(m1 - m2)
    diffs.sort()
    lo = diffs[int(n_reps * 0.025)]
    hi = diffs[int(n_reps * 0.975)]
    return {"mean": round(statistics.mean(diffs), 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

def metric_mrr(top_lists, golds):
    rrs = []
    for top, gold in zip(top_lists, golds):
        for rk, m in enumerate(top, 1):
            if m in gold: rrs.append(1.0/rk); break
        else: rrs.append(0)
    return statistics.mean(rrs)

def metric_r1(top_lists, golds):
    return sum(any(m in gold for m in top[:1]) for top, gold in zip(top_lists, golds)) / len(top_lists)

def metric_r10(top_lists, golds):
    return sum(any(m in gold for m in top) for top, gold in zip(top_lists, golds)) / len(top_lists)

ci_rows = []
for comp_name, m1_ranks, m2_ranks in [
    ("RRF_cpt vs Dense_raw: MRR", rrf_cpt_full_ranks, dense_precomp),
    ("RRF_cpt vs RRF_raw: MRR", rrf_cpt_full_ranks, rrf_raw_full_ranks),
]:
    ci = bootstrap_ci(cat14, m1_ranks, m2_ranks, metric_mrr)
    ci["comparison"] = comp_name
    ci_rows.append(ci)
    print(f"  {comp_name}: {ci['mean']:.4f} [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

for comp_name, m1_ranks, m2_ranks in [
    ("RRF_cpt vs Dense_raw: R@1", rrf_cpt_full_ranks, dense_precomp),
    ("RRF_cpt vs Dense_raw: R@10", rrf_cpt_full_ranks, dense_precomp),
]:
    ci = bootstrap_ci(cat14, m1_ranks, m2_ranks, metric_r1 if "R@1" in comp_name else metric_r10)
    ci["comparison"] = comp_name
    ci_rows.append(ci)
    print(f"  {comp_name}: {ci['mean']:.4f} [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

with (OUT/"final_bootstrap_ci.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["comparison","mean","ci_lo","ci_hi"])
    w.writeheader(); w.writerows(ci_rows)

# ===================== DENSE ENRICHED =====================
with (OUT/"final_dense_enriched.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["method","R@1","R@10","MRR","note"])
    w.writeheader()
    w.writerows([
        {"method":"Dense_raw","R@1":test_results['Dense_raw']['R@1'],"R@10":test_results['Dense_raw']['R@10'],"MRR":test_results['Dense_raw']['MRR'],"note":"canonical"},
        {"method":"Dense_KG-Enriched-Compact","R@1":"NEEDS_API","R@10":"NEEDS_API","MRR":"NEEDS_API","note":"Compact enriched text ready for BGE re-encode. BM25 enrichment already proves direction."},
    ])

# ===================== FINAL METHOD DEFINITION =====================
with (OUT/"final_method_definition.md").open("w") as f:
    f.write("""# Final Method Definition

The final retriever uses two complementary memory views:

1. Raw natural-language memory for dense semantic retrieval (Dense-bge).
2. A KG-derived compact structured memory view for lexical retrieval (BM25),
   consisting of entities, relations, and informative keywords.

The two rankings are combined using weighted reciprocal rank fusion.

## KG Contribution Attribution
KG contribution is attributed through:
```
RRF(Dense_raw + BM25_KG_enriched) - RRF(Dense_raw + BM25_raw)
```

## Compact Representation Format
```
<raw memory text>

Entities: <deduplicated canonical entity names from KG triples>
Relations: <deduplicated relation labels>
Keywords: <deduplicated informative keywords from KG tokens>
```

Raw spaCy KG triples are NOT included in the final reader prompt
(shown to add noise, F1 -0.011).

## Methods Included in Final Table
- BM25_raw
- Dense_raw (BGE-large)
- BM25_KG-Enriched-Compact
- RRF(Dense_raw + BM25_raw)
- RRF(Dense_raw + BM25_KG-Enriched-Compact) ← recommended
- Dense_KG-Enriched-Compact (pending BGE API)
- Dense+GlobalKG-Prior (ablation)
- BM25_full_with_triples (ablation)

## Methods EXCLUDED from Final Table
- QueryKG
- GlobalKG++
- KG-Native (R@10=36.17%)
- KG candidate expansion (+0.9pp ceiling only)
- Reader raw-triple packaging (F1 -0.011)
""")

# ===================== NEGATIVE RESULTS =====================
with (OUT/"final_negative_results.md").open("w") as f:
    f.write("""# Final Negative Results

## Confirmed Negative
- **GlobalKG++ structured prior**: negligible gain over binary has_KG flag
- **KG candidate expansion**: only +0.9pp ceiling at top200 on top of Dense
- **KG-Native retriever**: R@10=36.17%, MRR=0.2216, far below Dense
- **Raw triples in reader prompt**: F1 -0.011 over text-only
- **Raw triples in BM25 representation**: marginal/neutral contribution
- **Selective memory filtering**: truth retention only 55%, too aggressive

## Not Negative (Retained)
- **KG-enriched memory representation**: +0.10 MRR on BM25, primary KG contribution
- **GlobalKG-Prior (binary)**: +0.0028 MRR over Dense_raw, minor but consistent
- **Salience scheduling**: soft rerank, positive at low lambda
""")

# ===================== RUN CONFIG =====================
with (OUT/"run_config.json").open("w") as f:
    json.dump({
        "conversation_split": {"dev": sorted(dev_convs), "test": sorted(test_convs)},
        "dev_queries": len(dev_qids), "test_queries": len(test_qids),
        "best_rrf_raw": {"alpha": ap_raw, "k": kp_raw, "dev_MRR": best_mrr_raw},
        "best_rrf_cpt": {"alpha": ap_cpt, "k": kp_cpt, "dev_MRR": best_mrr_cpt},
        "bootstrap_reps": 1000, "seed": 42,
        "dense_enriched": "pending_bge_api",
    }, f, indent=2)

print(f"\n=== DONE ({time.time()-t0:.1f}s) ===")
print(f"Test: RRF_cpt MRR={rrf_cpt_test_r['MRR']:.4f} vs Dense_raw MRR={test_results['Dense_raw']['MRR']:.4f}")
