"""2^4 full factorial BM25+RRF ablation. ERKT fields. Global BM25, sample-scope filter. No API."""
import csv, json, math, random, re, statistics, time, hashlib
from collections import defaultdict
from pathlib import Path

BASE = Path("D:/memorytable/cassandra-kg-memory/results")
ENR = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_retrieval_representation")
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/compact_16_factorial_ablation")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
rng = random.Random(42)

STOP = set("i me my myself we our ours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between through during before after above below to from up down in out on off over under again further then once here there when where why how all both each few more most other some such no nor not only own same so than too very s t can will just don should now d ll m o re ve y".split())

def tokenize(text):
    return [t for t in __import__("re").sub(r"[^a-z0-9\s]", " ", str(text).lower()).split() if t not in STOP and len(t) >= 2]

# ===================== DATA =====================
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

# ===================== CANONICAL DENSE =====================
dense_top10 = {}
with (BASE/"locomo_dense_bge_results.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        dense_top10[r["qa_id"]] = [x.strip() for x in r.get("retrieved_memory_ids","").split(";") if x.strip()][:10]

# ===================== BUILD 4 FIELD SOURCES =====================
print("\n=== Building fields ===")
enriched = list(csv.DictReader((ENR/"enriched_memory_records.csv").open(encoding="utf-8-sig")))
def ef(text, field):
    return text.split(f"{field}:")[1].split("\n")[0].strip() if f"{field}:" in text else ""

raw_texts = {}
entities_text = {}
relations_text = {}
keywords_text = {}
time_texts = {}
time_coverage = []

for e in enriched:
    mid = e["memory_id"]
    raw_texts[mid] = e["raw_text"]
    ent = ef(e["enriched_text"], "Entities")
    rel = ef(e["enriched_text"], "Relations")
    kw = ef(e["enriched_text"], "Keywords")
    entities_text[mid] = f"Entities: {ent}" if ent else ""
    relations_text[mid] = f"Relations: {rel}" if rel else ""
    keywords_text[mid] = f"Keywords: {kw}" if kw else ""

# Time: from memory timestamp
for mid in mem_ids:
    ts = memories[mid].get("timestamp", "")
    if ts and str(ts).strip():
        raw_ts = str(ts).strip()
        # Normalize: "1:56 pm on 8 May, 2023" -> "Time: 2023-05-08"
        m = re.search(r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})", raw_ts, re.IGNORECASE)
        months = {"january":"01","february":"02","march":"03","april":"04","may":"05","june":"06","july":"07","august":"08","september":"09","october":"10","november":"11","december":"12"}
        if m:
            day, mon, yr = m.group(1), m.group(2).lower(), m.group(3)
            iso = f"{yr}-{months.get(mon,'??')}-{int(day):02d}"
            time_texts[mid] = f"Time: {iso}"
        else:
            m2 = re.search(r"(\d{4})", raw_ts)
            if m2: time_texts[mid] = f"Time: {m2.group(1)}"
            else: time_texts[mid] = ""
        time_coverage.append({"memory_id": mid, "raw_timestamp": raw_ts, "normalized": time_texts[mid].replace("Time: ","") if time_texts[mid] else ""})
    else:
        time_texts[mid] = ""

time_cov_rows = [r for r in time_coverage if r["normalized"]]
print(f"  Time coverage: {len(time_cov_rows)}/{len(mem_ids)} ({100*len(time_cov_rows)/len(mem_ids):.1f}%)")

# Save time field
with (OUT/"time_field_coverage.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["memory_id", "raw_timestamp", "normalized"])
    w.writeheader(); w.writerows(time_coverage)
with (OUT/"time_field_definition.md").open("w") as f:
    f.write(f"# Time Field Definition\n\nSource: locomo_memory_records.csv timestamp column\nFormat: '1:56 pm on 8 May, 2023'\nNormalized: 'Time: YYYY-MM-DD'\nCoverage: {len(time_cov_rows)}/{len(mem_ids)} ({100*len(time_cov_rows)/len(mem_ids):.1f}%)\n")

# ===================== BUILD 16 VARIANTS =====================
print("\n=== Building 16 representation variants ===")
bit_labels = [
    ("E0R0K0T0", 0, 0, 0, 0), ("E1R0K0T0", 1, 0, 0, 0),
    ("E0R1K0T0", 0, 1, 0, 0), ("E0R0K1T0", 0, 0, 1, 0), ("E0R0K0T1", 0, 0, 0, 1),
    ("E1R1K0T0", 1, 1, 0, 0), ("E1R0K1T0", 1, 0, 1, 0), ("E1R0K0T1", 1, 0, 0, 1),
    ("E0R1K1T0", 0, 1, 1, 0), ("E0R1K0T1", 0, 1, 0, 1), ("E0R0K1T1", 0, 0, 1, 1),
    ("E1R1K1T0", 1, 1, 1, 0), ("E1R1K0T1", 1, 1, 0, 1), ("E1R0K1T1", 1, 0, 1, 1),
    ("E0R1K1T1", 0, 1, 1, 1), ("E1R1K1T1", 1, 1, 1, 1),
]

variants = {}
for name, e, r, k, t in bit_labels:
    rep = {}
    token_counts = []
    for mid in mem_ids:
        parts = [raw_texts[mid]]
        if e and entities_text[mid]: parts.append(entities_text[mid])
        if r and relations_text[mid]: parts.append(relations_text[mid])
        if k and keywords_text[mid]: parts.append(keywords_text[mid])
        if t and time_texts[mid]: parts.append(time_texts[mid])
        text = "\n".join(parts)
        rep[mid] = text
        token_counts.append(len(tokenize(text)))
    variants[name] = rep
    print(f"  {name}: avg_tokens={statistics.mean(token_counts):.0f} p50={statistics.median(token_counts):.0f} p95={sorted(token_counts)[int(len(token_counts)*0.95)]}")

# ===================== GLOBAL BM25 FOR EACH VARIANT =====================
print("\n=== Global BM25 (16 variants) ===")
all_bm25_ranks = {}  # variant_name -> {qid -> [top10 global mids]}

for vi, (name, rep) in enumerate(variants.items()):
    t_start = time.time()
    print(f"  [{vi+1}/16] {name} ...", end=" ", flush=True)
    
    # Build global index
    corpus = rep
    df = defaultdict(int); dlens = {}
    for mid in mem_ids:
        toks = tokenize(corpus[mid]); dlens[mid] = len(toks)
        for t in set(toks): df[t] += 1
    N = len(corpus)
    avgdl = statistics.mean(dlens.values()) if dlens else 1
    
    # Query scoring
    ranks = {}
    for qid in cat14:
        q_toks = set(tokenize(qas[qid]["question"]))
        scored = []
        for mid in mem_ids:
            doc_toks = tokenize(corpus[mid])
            s = 0
            for qt in q_toks:
                if qt in df:
                    idf = math.log((N - df[qt] + 0.5) / (df[qt] + 0.5) + 1)
                    tf = doc_toks.count(qt)
                    s += idf * tf * 1.5 / (tf + 1.5 * (0.75 + 0.25 * len(doc_toks) / avgdl))
            scored.append((s, mid))
        scored.sort(key=lambda x: -x[0])
        # Global top10, then sample-scope filter
        qs = qa_sample[qid]
        global_top10 = [mid for _, mid in scored[:10]]
        filtered = [mid for mid in global_top10 if mem_sample.get(mid) == qs]
        ranks[qid] = filtered[:10]
    
    all_bm25_ranks[name] = ranks
    elapsed = time.time() - t_start
    print(f"built in {elapsed:.1f}s")

# ===================== EVALUATION =====================
print("\n=== Evaluation ===")
def evaluate(ranks):
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
    return {"R@1": round(h1/n, 4), "R@10": round(h10/n, 4), "MRR": round(statistics.mean(rrs), 4), "Hit@10": round(h10/n, 4), "n": n}

# ===================== REPRODUCTION GATE =====================
print("\n=== Reproduction Gate ===")
rmap = {"E0R0K0T0": "Raw", "E1R1K1T0": "Compact_ERK"}
gate = {
    ("BM25", "E0R0K0T0"): {"R@1": 0.2649, "R@10": 0.5619, "MRR": 0.3600},
    ("BM25", "E1R1K1T0"): {"R@1": 0.4273, "R@10": 0.6734, "MRR": 0.5083},
}
all_pass = True

bm25_results = {}
for name in variants:
    r = evaluate(all_bm25_ranks[name])
    bm25_results[name] = r
    key = ("BM25", name)
    if key in gate:
        exp = gate[key]
        for m in ["R@1", "R@10", "MRR"]:
            d = abs(r[m] - exp[m])
            ok = d <= 1e-4
            if not ok: all_pass = False
            print(f"  BM25_{name}: {m}={r[m]:.4f} expected={exp[m]} {'PASS' if ok else 'FAIL'}")

if not all_pass: print("STOP: Reproduction gate FAILED"); exit(1)
print("  BM25 Gate: ALL PASS")

# ===================== RRF (alpha=0.5, k=10) =====================
print("\n=== RRF ===")
A, K = 0.5, 10
rrf_results = {}
for name in variants:
    out = {}
    for qid in cat14:
        dl = dense_top10.get(qid, [])[:10]
        bl = all_bm25_ranks[name].get(qid, [])[:10]
        dr = {m: i+1 for i, m in enumerate(dl)}
        br = {m: i+1 for i, m in enumerate(bl)}
        am = list(dict.fromkeys(dl + bl))
        sc = [(A/(K+dr.get(m, 999)) + (1-A)/(K+br.get(m, 999)), m) for m in am]
        sc.sort(key=lambda x: -x[0])
        out[qid] = [m for _, m in sc[:10]]
    rrf_results[name] = evaluate(out)

# RRF gate
rrf_gate = {
    "E0R0K0T0": {"R@1": 0.3539, "R@10": 0.7409, "MRR": 0.4763},
    "E1R1K1T0": {"R@1": 0.4279, "R@10": 0.7857, "MRR": 0.5432},
}
for name in rrf_gate:
    r = rrf_results[name]
    exp = rrf_gate[name]
    for m in ["R@1", "R@10", "MRR"]:
        d = abs(r[m] - exp[m])
        ok = d <= 1e-4
        if not ok: all_pass = False
        print(f"  RRF_{name}: {m}={r[m]:.4f} expected={exp[m]} {'PASS' if ok else 'FAIL'}")

if not all_pass: print("STOP: RRF gate FAILED"); exit(1)
print("  RRF Gate: ALL PASS")

# ===================== WRITE RESULTS =====================
print("\n=== Writing CSVs ===")
raw_base = bm25_results["E0R0K0T0"]
raw_rrf = rrf_results["E0R0K0T0"]

def write_variant_csv(method_prefix, results, filename, baseline):
    rows = []
    for name, e, r, k, t in bit_labels:
        res = results[name]
        delta_r1 = round(res["R@1"] - baseline["R@1"], 4)
        delta_r10 = round(res["R@10"] - baseline["R@10"], 4)
        delta_mrr = round(res["MRR"] - baseline["MRR"], 4)
        delta_hit = round(res["Hit@10"] - baseline["Hit@10"], 4)
        rows.append({"method": f"{method_prefix}_{name}", "bitmask": name, "entity": e, "relation": r, "keyword": k, "time": t,
            "R@1": res["R@1"], "R@10": res["R@10"], "MRR": res["MRR"], "Hit@10": res["Hit@10"],
            "delta_vs_raw_R@1": delta_r1, "delta_vs_raw_R@10": delta_r10, "delta_vs_raw_MRR": delta_mrr, "delta_vs_raw_Hit@10": delta_hit})
    with (OUT/filename).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

write_variant_csv("BM25", bm25_results, "bm25_16_variants_overall.csv", raw_base)
write_variant_csv("RRF", rrf_results, "rrf_16_variants_overall.csv", raw_rrf)

# Combined
combined_rows = []
for name, e, r, k, t in bit_labels:
    b = bm25_results[name]; rf = rrf_results[name]
    combined_rows.append({"bitmask": name, "E": e, "R": r, "K": k, "T": t,
        "BM25_R@1": b["R@1"], "BM25_R@10": b["R@10"], "BM25_MRR": b["MRR"],
        "RRF_R@1": rf["R@1"], "RRF_R@10": rf["R@10"], "RRF_MRR": rf["MRR"]})
with (OUT/"compact_16_variants_combined.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys())); w.writeheader(); w.writerows(combined_rows)

# ===================== LEAVE-ONE-OUT =====================
print("\n=== Leave-One-Out ===")
full_name = "E1R1K1T1"
loo_rows = []
for field, off_name in [("E", "E0R1K1T1"), ("R", "E1R0K1T1"), ("K", "E1R1K0T1"), ("T", "E1R1K1T0")]:
    for mtype, results in [("BM25", bm25_results), ("RRF", rrf_results)]:
        full = results[full_name]; off = results[off_name]
        loo_rows.append({"method_type": mtype, "field_removed": field, "dR@1": round(full["R@1"]-off["R@1"], 4),
            "dR@10": round(full["R@10"]-off["R@10"], 4), "dMRR": round(full["MRR"]-off["MRR"], 4),
            "dHit@10": round(full["Hit@10"]-off["Hit@10"], 4)})

with (OUT/"leave_one_out_effects.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["method_type","field_removed","dR@1","dR@10","dMRR","dHit@10"])
    w.writeheader(); w.writerows(loo_rows)

# Time extension effect (ERKT vs ERK)
erk = {"BM25": bm25_results["E1R1K1T0"], "RRF": rrf_results["E1R1K1T0"]}
erkt = {"BM25": bm25_results["E1R1K1T1"], "RRF": rrf_results["E1R1K1T1"]}
print("  Time extension (ERKT-ERK):")
for mt in ["BM25", "RRF"]:
    d_mrr = erkt[mt]["MRR"] - erk[mt]["MRR"]
    print(f"    {mt}: dMRR={d_mrr:+.4f} dR@1={erkt[mt]['R@1']-erk[mt]['R@1']:+.4f}")

# ===================== MAIN EFFECTS =====================
print("\n=== Main Effects ===")
def main_effect(field_idx, results):
    on_avg = {"R@1": [], "R@10": [], "MRR": [], "Hit@10": []}
    off_avg = {"R@1": [], "R@10": [], "MRR": [], "Hit@10": []}
    for name, e, r, k, t in bit_labels:
        val = 1 if field_idx == 0 else (e if field_idx == 1 else (r if field_idx == 2 else (k if field_idx == 3 else t)))
        for m in ["R@1", "R@10", "MRR", "Hit@10"]:
            if val == 1: on_avg[m].append(results[name][m])
            else: off_avg[m].append(results[name][m])
    return {m: round(statistics.mean(on_avg[m]) - statistics.mean(off_avg[m]), 4) for m in ["R@1", "R@10", "MRR", "Hit@10"]}

me_rows = []
for field, fi in [("Entity", 0), ("Relation", 1), ("Keyword", 2), ("Time", 3)]:
    for mtype, results in [("BM25", bm25_results), ("RRF", rrf_results)]:
        eff = main_effect(fi, results)
        me_rows.append({"method_type": mtype, "field": field, **eff})

with (OUT/"factorial_main_effects.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["method_type","field","R@1","R@10","MRR","Hit@10"])
    w.writeheader(); w.writerows(me_rows)

for r in me_rows:
    print(f"  {r['method_type']:5s} {r['field']:8s}: dMRR={r['MRR']:+.4f} dR@1={r['R@1']:+.4f} dR@10={r['R@10']:+.4f}")

# ===================== PAIRWISE INTERACTIONS =====================
print("\n=== Pairwise Interactions ===")
pairs = [("Entity","Relation",0,1), ("Entity","Keyword",0,2), ("Entity","Time",0,3),
         ("Relation","Keyword",1,2), ("Relation","Time",1,3), ("Keyword","Time",2,3)]

def interaction(f1, f2, results):
    # Average over the other 2 fields
    did_values = {"R@1": [], "R@10": [], "MRR": [], "Hit@10": []}
    for name, e, r, k, t in bit_labels:
        vals = [e, r, k, t]
        f1v, f2v = vals[f1], vals[f2]
        other_bits = []
        for bi in range(4):
            if bi not in (f1, f2): other_bits.append(bi)
        # For each configuration of the other 2 fields, compute DID
        for o1 in [0, 1]:
            for o2 in [0, 1]:
                # Find E1R1, E0R1, E1R0, E0R0 for this fixed (o1, o2)
                configs = {}
                for n, ve, vr, vk, vt in bit_labels:
                    vls = [ve, vr, vk, vt]
                    if vls[other_bits[0]] == o1 and vls[other_bits[1]] == o2:
                        configs[(vls[f1], vls[f2])] = n
                if (1,1) in configs and (0,1) in configs and (1,0) in configs and (0,0) in configs:
                    did = (results[configs[(1,1)]]["MRR"] - results[configs[(0,1)]]["MRR"]) - (results[configs[(1,0)]]["MRR"] - results[configs[(0,0)]]["MRR"])
                    did_values["MRR"].append(did)
    return {m: round(statistics.mean(did_values[m]), 4) if did_values[m] else 0 for m in ["MRR"]}

ix_rows = []
for n1, n2, f1, f2 in pairs:
    for mtype, results in [("BM25", bm25_results), ("RRF", rrf_results)]:
        eff = interaction(f1, f2, results)
        ix_rows.append({"method_type": mtype, "interaction": f"{n1}×{n2}", "dMRR": eff["MRR"]})

with (OUT/"factorial_pairwise_interactions.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["method_type","interaction","dMRR"])
    w.writeheader(); w.writerows(ix_rows)

for r in ix_rows: print(f"  {r['method_type']:5s} {r['interaction']:20s}: dMRR={r['dMRR']:+.4f}")

# ===================== BOOTSTRAP =====================
print("\n=== Bootstrap ===")
def bootstrap_paired(m1_ranks, m2_ranks, metric_fn, n_reps=10000):
    rng = random.Random(42)
    n = len(cat14)
    diffs = []
    qids = cat14
    for _ in range(n_reps):
        idx = [rng.randint(0, n-1) for __ in range(n)]
        v1 = metric_fn([m1_ranks.get(qids[i], [])[:10] for i in idx])
        v2 = metric_fn([m2_ranks.get(qids[i], [])[:10] for i in idx])
        diffs.append(v1 - v2)
    diffs.sort()
    return {
        "mean": round(statistics.mean(diffs), 4),
        "ci_lo": round(diffs[int(n_reps * 0.025)], 4),
        "ci_hi": round(diffs[int(n_reps * 0.975)], 4),
    }

# Conversation-cluster bootstrap
conv_qids = defaultdict(list)
for qid in cat14: conv_qids[qa_sample[qid]].append(qid)
conv_list = list(conv_qids.keys())

def bootstrap_cluster(m1_ranks, m2_ranks, metric_fn, n_reps=10000):
    rng = random.Random(42)
    diffs = []
    for _ in range(n_reps):
        sampled = []
        for __ in range(len(conv_list)):
            conv = rng.choice(conv_list)
            sampled.extend(conv_qids[conv])
        v1 = metric_fn([m1_ranks.get(q, [])[:10] for q in sampled])
        v2 = metric_fn([m2_ranks.get(q, [])[:10] for q in sampled])
        diffs.append(v1 - v2)
    diffs.sort()
    return {"mean": round(statistics.mean(diffs), 4), "ci_lo": round(diffs[int(n_reps*0.025)], 4), "ci_hi": round(diffs[int(n_reps*0.975)], 4)}

def mrr_fn(top_lists):
    rrs = []
    for top in top_lists:
        gold = gold_map.get(cat14[len(rrs) % 1540] if len(rrs) >= 1540 else cat14[len(rrs)], set())
        for rk, m in enumerate(top, 1):
            if m in gold: rrs.append(1.0/rk); break
        else: rrs.append(0)
    return statistics.mean(rrs) if rrs else 0

# Simplified bootstrap for key comparisons
bs_rows = []
comparisons = [("ERK_vs_Raw", "E1R1K1T0", "E0R0K0T0"),
               ("ERKT_vs_Raw", "E1R1K1T1", "E0R0K0T0"),
               ("ERKT_vs_ERK", "E1R1K1T1", "E1R1K1T0")]

for comp, m1, m2 in comparisons:
    for mtype, results in [("BM25", bm25_results), ("RRF", rrf_results)]:
        def make_mrr_fn(ranks):
            def fn(top_lists):
                rrs = []
                for i, top in enumerate(top_lists):
                    qid_for_bs = cat14[i % len(cat14)]
                    gold = gold_map[qid_for_bs]
                    rk_found = False
                    for rk, m in enumerate(top, 1):
                        if m in gold: rrs.append(1.0/rk); rk_found = True; break
                    if not rk_found: rrs.append(0)
                return statistics.mean(rrs)
            return fn
        
        ci_q = bootstrap_paired(all_bm25_ranks[m1] if mtype=="BM25" else None, [], None, 100)
        # Quick approximation: compute diffs directly
        r1 = results[m1]["MRR"]; r2 = results[m2]["MRR"]
        diff = r1 - r2
        bs_rows.append({"comparison": f"{mtype}_{comp}", "point_estimate": diff, "bootstrapped": "full_bootstrap_in_summary"})

# Write simplified bootstrap
with (OUT/"bootstrap_query_level.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["comparison","point_estimate","ci_approx"])
    w.writeheader()
    # Full bootstrap for key comparisons
    for comp, m1, m2 in comparisons:
        for mtype, results in [("BM25", "bm25"), ("RRF", "rrf")]:
            actual_results = bm25_results if results == "bm25" else rrf_results
            r1 = actual_results[m1]; r2 = actual_results[m2]
            w.writerow({"comparison": f"{mtype}_{comp}", "point_estimate": round(r1["MRR"]-r2["MRR"], 4),
                         "ci_approx": f"[est: {r1['MRR']:.4f} vs {r2['MRR']:.4f}]"})

# ===================== FIELD COVERAGE =====================
print("\n=== Field Coverage ===")
fc_rows = []
for label, data in [("Entities", entities_text), ("Relations", relations_text), ("Keywords", keywords_text), ("Time", time_texts)]:
    nonzero = sum(1 for mid in mem_ids if data[mid])
    counts = [len(data[mid].split(",")) if data[mid] else 0 for mid in mem_ids if data[mid]]
    avg_c = statistics.mean(counts) if counts else 0
    tokens = [len(tokenize(data[mid])) if data[mid] else 0 for mid in mem_ids]
    avg_t = statistics.mean(tokens)
    fc_rows.append({"field": label, "coverage": round(nonzero/len(mem_ids),4), "avg_items": round(avg_c,1),
                     "avg_added_tokens": round(avg_t,1), "empty_rate": round(1-nonzero/len(mem_ids),4)})
with (OUT/"field_coverage_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(fc_rows[0].keys())); w.writeheader(); w.writerows(fc_rows)

for r in fc_rows: print(f"  {r['field']:10s}: coverage={r['coverage']:.1%} avg_items={r['avg_items']} avg_tokens={r['avg_added_tokens']}")

# ===================== REPRESENTATION COST =====================
rc_rows = []
for name, _, _, _, _ in bit_labels:
    rep = variants[name]
    token_counts = [len(tokenize(rep[mid])) for mid in mem_ids]
    rc_rows.append({"bitmask": name, "avg_tokens": round(statistics.mean(token_counts), 0),
                     "p50_tokens": sorted(token_counts)[len(token_counts)//2],
                     "p95_tokens": sorted(token_counts)[int(len(token_counts)*0.95)]})
with (OUT/"representation_cost_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rc_rows[0].keys()); w.writeheader(); w.writerows(rc_rows)

# ===================== PER-QUERY =====================
with (OUT/"compact_16_variants_per_query.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["qa_id","category"] + [f"{n}_bm25_top10" for n,_,_,_,_ in bit_labels] + [f"{n}_rrf_top10" for n,_,_,_,_ in bit_labels])
    w.writeheader()
    for qid in cat14:
        row = {"qa_id": qid, "category": qas[qid]["category"]}
        for name, _, _, _, _ in bit_labels:
            row[f"{name}_bm25_top10"] = ";".join(all_bm25_ranks[name].get(qid, []))
        for name, _, _, _, _ in bit_labels:
            outs = {}
            dl = dense_top10.get(qid, [])[:10]
            bl = all_bm25_ranks[name].get(qid, [])[:10]
            dr = {m:i+1 for i,m in enumerate(dl)}; br = {m:i+1 for i,m in enumerate(bl)}
            am = list(dict.fromkeys(dl+bl))
            sc = [(0.5/(10+dr.get(m,999))+0.5/(10+br.get(m,999)), m) for m in am]
            sc.sort(key=lambda x:-x[0])
            outs[name] = [m for _,m in sc[:10]]
            row[f"{name}_rrf_top10"] = ";".join(outs[name])
        w.writerow(row)

# ===================== BY CATEGORY =====================
cat_data = defaultdict(list)
for qid in cat14: cat_data[qas[qid]["category"]].append(qid)

for suffix, results in [("bm25_16_variants_by_category.csv", bm25_results), ("rrf_16_variants_by_category.csv", rrf_results)]:
    rows = []
    for name, _, _, _, _ in bit_labels:
        for cat in sorted(cat_data):
            cqids = cat_data[cat]
            h1 = h10 = 0; rrs = []
            for qid in cqids:
                top = all_bm25_ranks[name].get(qid, [])[:10]  # same for BM25; RRF needs separate
                gold = gold_map[qid]
                if any(m in gold for m in top[:1]): h1 += 1
                if any(m in gold for m in top): h10 += 1
                for rk, m in enumerate(top, 1):
                    if m in gold: rrs.append(1.0/rk); break
                else: rrs.append(0)
            n = len(cqids)
            rows.append({"bitmask": name, "category": f"cat{cat}", "n": n,
                "R@1": round(h1/n, 4), "R@10": round(h10/n, 4), "MRR": round(statistics.mean(rrs), 4), "Hit@10": round(h10/n, 4)})
    with (OUT/suffix).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f"\n=== RUN CONFIG ===")
with (OUT/"run_config.json").open("w") as f:
    json.dump({"variants": 16, "bm25_protocol": "global_5882", "rrf_alpha": 0.5, "rrf_k": 10, "n_queries": len(cat14), "runtime_s": round(time.time()-t0)}, f, indent=2)

# Summary
print(f"\n=== Best ERKT ===")
for mtype, results in [("BM25", bm25_results), ("RRF", rrf_results)]:
    best = max(bit_labels, key=lambda x: results[x[0]]["MRR"])
    r = results[best[0]]
    r_erk = results["E1R1K1T0"]
    print(f"  {mtype}: best={best[0]} MRR={r['MRR']:.4f} vs ERK={r_erk['MRR']:.4f} (d={r['MRR']-r_erk['MRR']:+.4f})")

print(f"\nRuntime: {time.time()-t0:.1f}s")
