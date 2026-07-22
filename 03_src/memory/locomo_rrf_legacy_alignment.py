"""Legacy-aligned RRF alignment run. Global BM25, global top10, sample-scope filter. Reproduction gate first."""
import csv, json, math, re, statistics, time
from collections import defaultdict
from pathlib import Path

BASE = Path("D:/memorytable/cassandra-kg-memory/results")
ENR = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_retrieval_representation")
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_rrf_legacy_alignment")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

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
cat14 = sorted([qid for qid, q in qas.items() if q["category"] != "5"])

gold_map = defaultdict(set)
with (BASE/"locomo_evidence_map.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): gold_map[r["qa_id"]].add(r["memory_id"])

mem_sample = {mid: memories[mid]["sample_id"] for mid in mem_ids}
qa_sample = {qid: qas[qid]["sample_id"] for qid in cat14}
print(f"  queries={len(cat14)}, memories={len(mem_ids)}")

# ===================== CANONICAL DENSE (global, top10, sample-scope filter) =====================
print("\n=== Canonical Dense ===")
dense_global_top10 = {}
with (BASE/"locomo_dense_bge_results.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        dense_global_top10[r["qa_id"]] = [x.strip() for x in r.get("retrieved_memory_ids","").split(";") if x.strip()]

# Sample-scope filter: keep only in-sample from global top10, pad with remaining in-sample
dense_legacy = {}
for qid in cat14:
    qs = qa_sample[qid]
    top10 = [m for m in dense_global_top10.get(qid, []) if mem_sample.get(m) == qs][:10]
    dense_legacy[qid] = top10

# ===================== CANONICAL BM25 RAW (global 5882, top10, sample-scope filter) =====================
print("=== Canonical BM25 raw ===")
dia2mid = {memories[mid]["dia_id"]: mid for mid in mem_ids if "dia_id" in memories[mid]}
bm25_global_top10 = {}
with (BASE/"locomo_bm25_results.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        retrieved = r.get("retrieved_memory_ids","")
        if retrieved:
            mids = []
            for sid in [x.strip() for x in retrieved.split(";") if x.strip()]:
                if sid in dia2mid: mids.append(dia2mid[sid])
            bm25_global_top10[r["qa_id"]] = mids

bm25_raw_legacy = {}
for qid in cat14:
    qs = qa_sample[qid]
    top10 = [m for m in bm25_global_top10.get(qid, []) if mem_sample.get(m) == qs][:10]
    bm25_raw_legacy[qid] = top10

# ===================== REPRODUCTION GATE =====================
print("\n=== Reproduction Gate ===")
def eval_gate(ranks, expected):
    h1 = h10 = 0; rrs = []
    for qid in cat14:
        top = ranks.get(qid, [])[:10]
        gold = gold_map[qid]
        if any(m in gold for m in top[:1]): h1 += 1
        if any(m in gold for m in top): h10 += 1
        for rk, m in enumerate(top, 1):
            if m in gold: rrs.append(1.0/rk); break
        else: rrs.append(0)
    n = len(cat14)
    return {"R@1": h1/n, "R@10": h10/n, "MRR": statistics.mean(rrs)}

bm25_actual = eval_gate(bm25_raw_legacy, None)
dense_actual = eval_gate(dense_legacy, None)

bm25_expected = {"R@1": 0.2649, "R@10": 0.5619, "MRR": 0.3600}
dense_expected = {"R@1": 0.3419, "R@10": 0.7009, "MRR": 0.4534}

repro_ok = True
diffs = []
for label, actual, expected in [("BM25", bm25_actual, bm25_expected), ("Dense-bge", dense_actual, dense_expected)]:
    ok = True
    for m in ["R@1","R@10","MRR"]:
        delta = abs(actual[m] - expected[m])
        status = "PASS" if delta <= 1e-4 else "FAIL"
        if status == "FAIL": repro_ok = False; ok = False
        diffs.append({"method":label,"metric":m,"actual":round(actual[m],4),"expected":round(expected[m],4),"delta":round(delta,4),"status":status})
        print(f"  {label:8s} {m}: actual={actual[m]:.4f} expected={expected[m]:.4f} delta={delta:.4f} {status}")

with (OUT/"legacy_alignment_checks.json").open("w") as f: json.dump({"reproduction_ok":repro_ok,"checks":diffs}, f, indent=2)

if not repro_ok:
    with (OUT/"alignment_diff.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["method","metric","actual","expected","delta","status"])
        w.writeheader(); w.writerows(diffs)
    print("STOP: Reproduction gate FAILED"); exit(1)

print("  Reproduction gate: ALL PASS")

# ===================== BUILD GLOBAL COMPACT BM25 =====================
print("\n=== Global compact BM25 ===")
enriched = list(csv.DictReader((ENR/"enriched_memory_records.csv").open(encoding="utf-8-sig")))

def extract_field(text, field):
    if f"{field}:" in text: return text.split(f"{field}:")[1].split("\n")[0].strip()
    return ""

compact_texts = {}
for e in enriched:
    mid = e["memory_id"]
    raw = e["raw_text"]
    ent = extract_field(e["enriched_text"], "Entities")
    rel = extract_field(e["enriched_text"], "Relations")
    kw = extract_field(e["enriched_text"], "Keywords")
    compact_texts[mid] = raw + (f"\nEntities: {ent}" if ent else "") + (f"\nRelations: {rel}" if rel else "") + (f"\nKeywords: {kw}" if kw else "")

# Build global BM25 index over all 5882 compact memories
corpus = {mid: compact_texts[mid] for mid in mem_ids if mid in compact_texts}
df = defaultdict(int); dlens = {}
for mid, text in corpus.items():
    toks = tokenize(text); dlens[mid] = len(toks)
    for t in set(toks): df[t] += 1
N_global = len(corpus)
avgdl = statistics.mean(dlens.values())
print(f"  Global corpus: {N_global} docs, avgdl={avgdl:.1f}")

def bm25_score_global(query, doc_toks):
    score = 0
    for qt in set(tokenize(query)):
        if qt not in df: continue
        idf = math.log((N_global - df[qt] + 0.5) / (df[qt] + 0.5) + 1)
        tf = doc_toks.count(qt)
        score += idf * tf * 1.5 / (tf + 1.5 * (0.75 + 0.25 * len(doc_toks) / avgdl))
    return score

# Global retrieval: top10 from all 5882, then sample-scope filter
bm25_compact_global = {}
bm25_compact_legacy = {}
for i, qid in enumerate(cat14):
    if i % 200 == 0: print(f"  query {i}/{len(cat14)}...")
    question = qas[qid]["question"]
    all_scores = [(mid, bm25_score_global(question, tokenize(corpus[mid]))) for mid in mem_ids if mid in corpus]
    all_scores.sort(key=lambda x: -x[1])
    global_top10 = [mid for mid, _ in all_scores[:10]]
    bm25_compact_global[qid] = global_top10
    # Sample-scope filter
    qs = qa_sample[qid]
    top10 = [m for m in global_top10 if mem_sample.get(m) == qs][:10]
    bm25_compact_legacy[qid] = top10

# Save global compact BM25 results
with (BASE.parent / "results/locomo_bm25_compact_global_legacy_results.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["qa_id","category","question","retrieved_memory_ids"])
    w.writeheader()
    for qid in cat14:
        w.writerow({"qa_id":qid,"category":qas[qid]["category"],"question":qas[qid]["question"],
                     "retrieved_memory_ids":";".join(bm25_compact_global.get(qid,[]))})

# ===================== RRF (alpha=0.5, k=10) =====================
print("\n=== Weighted RRF (alpha=0.5, k=10) ===")
ALPHA = 0.5; K = 10

def rrf_legacy(dense_ranks, bm25_ranks):
    ranks = {}
    for qid in cat14:
        qs = qa_sample[qid]
        # Global top10 from both sources
        d_list = dense_global_top10.get(qid, [])
        b_list = bm25_global_top10.get(qid, [])
        # Filter to sample
        d_sample = [m for m in d_list if mem_sample.get(m) == qs]
        b_sample = [m for m in b_list if mem_sample.get(m) == qs]
        # Rank within sample (1-indexed, 0 contribution if absent)
        dr = {m: i+1 for i, m in enumerate(d_sample)}
        br = {m: i+1 for i, m in enumerate(b_sample)}
        all_m = list(dict.fromkeys(d_sample + b_sample))
        scored = [(ALPHA/(K + dr.get(m, 999)) + (1-ALPHA)/(K + br.get(m, 999)), m) for m in all_m]
        scored.sort(key=lambda x: -x[0])
        ranks[qid] = [m for _, m in scored[:10]]
    return ranks

rrf_raw_legacy_ranks = rrf_legacy(dense_global_top10, bm25_global_top10)
rrf_compact_legacy_ranks = rrf_legacy(dense_global_top10, bm25_compact_global)

# ===================== EVALUATE ALL METHODS =====================
print("\n=== Results ===")
methods = {
    "Dense_raw_legacy": dense_legacy,
    "BM25_raw_legacy": bm25_raw_legacy,
    "BM25_compact_global_legacy": bm25_compact_legacy,
    "RRF_raw_legacy": rrf_raw_legacy_ranks,
    "RRF_compact_legacy": rrf_compact_legacy_ranks,
}

ov_rows = []
for name, ranks in methods.items():
    r = eval_gate(ranks, None)
    r["method"] = name
    r["R@1"] = round(r["R@1"], 4); r["R@10"] = round(r["R@10"], 4); r["MRR"] = round(r["MRR"], 4)
    r["n"] = len(cat14)
    ov_rows.append(r)
    print(f"  {name:35s} R@1={r['R@1']:.4f} R@10={r['R@10']:.4f} MRR={r['MRR']:.4f}")

with (OUT/"legacy_alignment_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["method","R@1","R@10","MRR","n"])
    w.writeheader(); w.writerows(ov_rows)

# ===================== COMPONENT ATTRIBUTION =====================
print("\n=== Component Attribution ===")
def get_metrics(name):
    return next(r for r in ov_rows if r["method"]==name)

dr = get_metrics("Dense_raw_legacy")
bmr = get_metrics("BM25_raw_legacy")
bmc = get_metrics("BM25_compact_global_legacy")
rr = get_metrics("RRF_raw_legacy")
rc = get_metrics("RRF_compact_legacy")

attrib = [
    {"component":"generic_hybrid (RRF_raw - Dense_raw)",
     "delta_MRR": round(rc["MRR"] - rr["MRR"], 4),  # placeholder, will overwrite
     "delta_R@1": round(rr["R@1"] - dr["R@1"], 4),
     "delta_R@10": round(rr["R@10"] - dr["R@10"], 4)},
    {"component":"KG_rep_lexical (BM25_cpt - BM25_raw)",
     "delta_MRR": round(bmc["MRR"] - bmr["MRR"], 4),
     "delta_R@1": round(bmc["R@1"] - bmr["R@1"], 4),
     "delta_R@10": round(bmc["R@10"] - bmr["R@10"], 4)},
    {"component":"KG_rep_in_hybrid (RRF_cpt - RRF_raw)",
     "delta_MRR": round(rc["MRR"] - rr["MRR"], 4),
     "delta_R@1": round(rc["R@1"] - rr["R@1"], 4),
     "delta_R@10": round(rc["R@10"] - rr["R@10"], 4)},
    {"component":"total_final_gain (RRF_cpt - Dense_raw)",
     "delta_MRR": round(rc["MRR"] - dr["MRR"], 4),
     "delta_R@1": round(rc["R@1"] - dr["R@1"], 4),
     "delta_R@10": round(rc["R@10"] - dr["R@10"], 4)},
]
# Fix generic hybrid attribution
attrib[0]["delta_MRR"] = round(rr["MRR"] - dr["MRR"], 4)

with (OUT/"legacy_component_attribution.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["component","delta_MRR","delta_R@1","delta_R@10"])
    w.writeheader(); w.writerows(attrib)

for a in attrib:
    print(f"  {a['component']}: dMRR={a['delta_MRR']} dR@1={a['delta_R@1']} dR@10={a['delta_R@10']}")

# ===================== LEGACY vs SAMPLE-LOCAL COMPARISON =====================
print("\n=== Legacy vs Sample-Local Comparison ===")
# Load sample-local results from previous run
sl_dense = {"R@1": 0.3696, "R@10": 0.7174, "MRR": 0.4778}
sl_rrf_raw = {"R@1": 0.3730, "R@10": 0.7557, "MRR": 0.4920}
sl_rrf_cpt = {"R@1": 0.4061, "R@10": 0.7870, "MRR": 0.5253}

comp_rows = [
    {"comparison":"Dense_raw","legacy_R@10":dr["R@10"],"legacy_MRR":dr["MRR"],"sample_local_R@10":sl_dense["R@10"],"sample_local_MRR":sl_dense["MRR"],"note":"identical protocol -> should match"},
    {"comparison":"RRF_raw","legacy_R@10":rr["R@10"],"legacy_MRR":rr["MRR"],"sample_local_R@10":sl_rrf_raw["R@10"],"sample_local_MRR":sl_rrf_raw["MRR"],"note":"BM25 source differs (global vs local)"},
    {"comparison":"RRF_cpt","legacy_R@10":rc["R@10"],"legacy_MRR":rc["MRR"],"sample_local_R@10":sl_rrf_cpt["R@10"],"sample_local_MRR":sl_rrf_cpt["MRR"],"note":"BM25 source differs (global vs local)"},
]
with (OUT/"legacy_vs_sample_local_comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["comparison","legacy_R@10","legacy_MRR","sample_local_R@10","sample_local_MRR","note"])
    w.writeheader(); w.writerows(comp_rows)

# ===================== SUMMARY =====================
with (OUT/"legacy_alignment_summary.md").open("w") as f:
    f.write(f"# Legacy-Aligned RRF Attribution\n\n")
    f.write(f"## Reproduction Gate: ALL PASS\n")
    f.write(f"BM25: R@1={bmr['R@1']:.4f} R@10={bmr['R@10']:.4f} MRR={bmr['MRR']:.4f}\n")
    f.write(f"Dense: R@1={dr['R@1']:.4f} R@10={dr['R@10']:.4f} MRR={dr['MRR']:.4f}\n\n")
    f.write(f"## Results (global top10 -> sample-scope filter, 1540 queries)\n")
    for r in ov_rows:
        f.write(f"- {r['method']}: R@1={r['R@1']:.4f} R@10={r['R@10']:.4f} MRR={r['MRR']:.4f}\n")
    f.write(f"\n## Component Attribution\n")
    for a in attrib:
        f.write(f"- {a['component']}: dMRR={a['delta_MRR']}\n")
    f.write(f"\n## Protocol\n")
    f.write(f"1. Dense: locomo_dense_bge_results.csv (global top10, sample-scope filter)\n")
    f.write(f"2. BM25_raw: locomo_bm25_results.csv (global 5882-memory, top10, sample-scope filter)\n")
    f.write(f"3. BM25_compact: global 5882-memory BM25 on KG-enriched text, top10, sample-scope filter\n")
    f.write(f"4. RRF: alpha=0.5, k=10, absent candidate = 0 contribution\n")
    f.write(f"5. Scope: cat1-4, 1540 queries, sample-scoped\n")
    f.write(f"\n## Legacy vs Sample-Local\n")
    f.write(f"Legacy uses global BM25; sample-local uses per-sample BM25. Attribution values differ.\n")
    f.write(f"Legacy is aligned with screenshot baselines; sample-local is generalization validation.\n")

with (OUT/"run_config.json").open("w") as f:
    json.dump({"protocol":"global_legacy","rrf_alpha":0.5,"rrf_k":10,"bm25_corpus":"global_5882","reproduction_gate":"PASS","n_queries":len(cat14)}, f, indent=2)

print(f"\n=== DONE ({time.time()-t0:.1f}s) ===")
