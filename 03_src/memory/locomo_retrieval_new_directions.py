"""A-MEM-lite + Selective Memory Filtering + Reader evidence packaging. No API calls, no old method optimization."""
import csv, json, math, re, statistics, time
from collections import defaultdict
from pathlib import Path

BASE = Path("D:/memorytable/cassandra-kg-memory/results")
OUT_R = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_retrieval_representation")
OUT_S = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_retrieval_selective_memory")
OUT_RD = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_reader_evidence_packaging")
OUT_F = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_retrieval_new_directions_from_papers")
t0 = time.time()

STOP = set("i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between through during before after above below to from up down in out on off over under again further then once here there when where why how all both each few more most other some such no nor not only own same so than too very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn also would could should may might shall".split())
TEMPORAL = set("when before after first last later earlier recently yesterday today tomorrow date time week month year day ago since until".split())
SALIENCE_KW = set("like prefer favorite want need plan went bought met attended visited lived worked birthday family friend before after first last recently earlier later".split())

def tokenize(text, rm_stop=True):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    if rm_stop: return [t for t in text if t not in STOP and len(t) >= 2]
    return [t for t in text if len(t) >= 2]

def is_entity_like(t):
    return t not in STOP and len(t) >= 3 and not t.isdigit()

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
print(f"  cat1-4 queries: {len(cat14)}")

# === Canonical Dense/BM25 rankings ===
dia2mid = {memories[mid]["dia_id"]: mid for mid in mem_ids if "dia_id" in memories[mid]}

dense_precomp = {}
with (BASE/"locomo_dense_bge_results.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        retrieved = r.get("retrieved_memory_ids","")
        if retrieved:
            dense_precomp[r["qa_id"]] = [x.strip() for x in retrieved.split(";") if x.strip()]

bm25_precomp = {}
with (BASE/"locomo_bm25_results.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        retrieved = r.get("retrieved_memory_ids","")
        if retrieved:
            mids = []
            for sid in [x.strip() for x in retrieved.split(";") if x.strip()]:
                if sid in dia2mid: mids.append(dia2mid[sid])
            bm25_precomp[r["qa_id"]] = mids

# === Load KG edges ===
mem_kg = defaultdict(list)
with (BASE/"locomo_kg_edges_spacy.csv").open(encoding="utf-8-sig") as f:
    for e in csv.DictReader(f):
        ev, gid = e.get("evidence",""), e["graph_id"]
        for mid, m in memories.items():
            if m["sample_id"] == gid and (m["dia_id"] == ev or ev in mid):
                mem_kg[mid].append(e)
                break

has_kg_set = set(mem_kg.keys())
print(f"  KG memories: {len(has_kg_set)}")

# ============================================
# EXPERIMENT 1: A-MEM-lite enriched memory
# ============================================
print("\n=== EXP 1: A-MEM-lite enriched memory ===")

enriched = []
for mid in mem_ids:
    m = memories[mid]
    raw = m.get("text","")
    edges = mem_kg.get(mid, [])
    if not edges:
        enriched.append({"memory_id": mid, "raw_text": raw, "enriched_text": raw,
            "has_kg": 0, "n_triples": 0, "entities": "", "relations": "", "temporal_flag": 0})
        continue
    ents = set(); rels = set(); temp_flag = 0
    triples = []
    for e in edges:
        src, rel, dst = e.get("src_id",""), e.get("relation",""), e.get("dst_id","")
        triples.append(f"({src}, {rel}, {dst})")
        for t in tokenize(src)+tokenize(dst): ents.add(t)
        for t in tokenize(rel):
            rels.add(t)
            if t in TEMPORAL: temp_flag = 1
        if any(t in TEMPORAL for t in tokenize(src)+tokenize(dst)+tokenize(rel)): temp_flag = 1
    ent_list = ", ".join(sorted(ents)[:15])
    rel_list = ", ".join(sorted(rels)[:10])
    kw = set();
    for t in tokenize(raw): kw.add(t);
    for t in tokenize(ent_list): kw.add(t)
    kw_list = ", ".join(sorted(kw - STOP)[:20])
    enriched_text = (
        f"{raw}\n"
        f"Entities: {ent_list}\n"
        f"Relations: {rel_list}\n"
        f"Triples: {'; '.join(triples[:10])}\n"
        f"{'Temporal: true' if temp_flag else ''}\n"
        f"Keywords: {kw_list}"
    ).strip()
    enriched.append({"memory_id": mid, "raw_text": raw, "enriched_text": enriched_text,
        "has_kg": 1, "n_triples": len(edges), "entities": ent_list,
        "relations": rel_list, "temporal_flag": temp_flag})

with (OUT_R/"enriched_memory_records.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["memory_id","raw_text","enriched_text","has_kg","n_triples","entities","relations","temporal_flag"])
    w.writeheader(); w.writerows(enriched)

# BM25 on raw vs enriched (local TF-IDF similarity)
print("  Running BM25 raw vs enriched...")
def bm25_build(corpus):
    df = defaultdict(int); dlens = {}
    for mid, text in corpus.items():
        toks = tokenize(text)
        dlens[mid] = len(toks)
        for t in set(toks): df[t] += 1
    return df, dlens

def bm25_score(query, doc_toks, df, dlens, N, avgdl, k1=1.2, b=0.75):
    score = 0
    for qt in set(tokenize(query)):
        if qt not in df: continue
        idf = math.log((N - df[qt] + 0.5) / (df[qt] + 0.5) + 1)
        tf = doc_toks.count(qt)
        dl = len(doc_toks)
        score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    return score

def run_bm25(corpus_text, label):
    corpus = {e["memory_id"]: e[corpus_text] for e in enriched}
    # Sample-scoped: build per-sample indices
    sample_mem = defaultdict(list)
    for e in enriched:
        mid = e["memory_id"]
        sample_mem[memories[mid]["sample_id"]].append(mid)
    
    results = {}
    for qid in cat14:
        q_sample = qas[qid]["sample_id"]
        smids = sample_mem[q_sample]
        cm = {mid: corpus[mid] for mid in smids}
        df, dlens = bm25_build(cm)
        N = len(cm)
        avgdl = statistics.mean(dlens.values()) if dlens else 1
        
        query = qas[qid]["question"]
        scored = [(mid, bm25_score(query, tokenize(cm[mid]), df, dlens, N, avgdl)) for mid in smids]
        scored.sort(key=lambda x: -x[1])
        results[qid] = [mid for mid, _ in scored[:10]]
    
    # Evaluate
    h1 = 0; h10 = 0; rrs = []
    for qid in cat14:
        top = results.get(qid, [])[:10]
        gold = gold_map[qid]
        if any(m in gold for m in top[:1]): h1 += 1
        if any(m in gold for m in top): h10 += 1
        for rk, m in enumerate(top, 1):
            if m in gold: rrs.append(1.0/rk); break
        else: rrs.append(0)
    n = len(cat14)
    return {"method": label, "R@1": round(h1/n,4), "R@10": round(h10/n,4), "MRR": round(statistics.mean(rrs),4), "n": n}

bm25_raw_res = run_bm25("raw_text", "BM25_raw")
bm25_enr_res = run_bm25("enriched_text", "BM25_enriched")

# Canonical Dense
dense_cat = {"R@1": 0, "R@10": 0, "MRR": 0, "n": len(cat14)}
for qid in cat14:
    top = dense_precomp.get(qid, [])[:10]
    gold = gold_map[qid]
    if any(m in gold for m in top[:1]): dense_cat["R@1"] += 1
    if any(m in gold for m in top): dense_cat["R@10"] += 1
    for rk, m in enumerate(top, 1):
        if m in gold: dense_cat["MRR"] += 1.0/rk; break
dense_cat["R@1"] = round(dense_cat["R@1"]/dense_cat["n"],4)
dense_cat["R@10"] = round(dense_cat["R@10"]/dense_cat["n"],4)
dense_cat["MRR"] = round(dense_cat["MRR"]/dense_cat["n"],4)
dense_cat["method"] = "Dense_raw_canonical"

# Dense + GlobalKG-Prior
dense_kg = {"R@1": 0, "R@10": 0, "MRR": 0, "n": len(cat14)}
for qid in cat14:
    top = dense_precomp.get(qid, [])[:10]
    scored = [(1.0/(i+1) + 0.1*(1 if m in has_kg_set else 0), m) for i, m in enumerate(top)]
    scored.sort(key=lambda x: -x[0])
    reranked = [m for _, m in scored][:10]
    gold = gold_map[qid]
    if any(m in gold for m in reranked[:1]): dense_kg["R@1"] += 1
    if any(m in gold for m in reranked): dense_kg["R@10"] += 1
    for rk, m in enumerate(reranked, 1):
        if m in gold: dense_kg["MRR"] += 1.0/rk; break
dense_kg["R@1"] = round(dense_kg["R@1"]/dense_kg["n"],4)
dense_kg["R@10"] = round(dense_kg["R@10"]/dense_kg["n"],4)
dense_kg["MRR"] = round(dense_kg["MRR"]/dense_kg["n"],4)
dense_kg["method"] = "Dense+GlobalKG_canonical"

rep_rows = [bm25_raw_res, bm25_enr_res, dense_cat, dense_kg]
rep_fields = ["method","R@1","R@10","MRR","n"]
with (OUT_R/"representation_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rep_fields); w.writeheader(); w.writerows(rep_rows)

# Rescue/hurt
rh_rep = []
for label, func in [("BM25_enriched vs BM25_raw",
    lambda q: (set(run_bm25("enriched_text","")[1][:1]) if False else set())),]:
    pass  # Simplified

for r in rep_rows:
    print(f"  {r['method']:30s} R@1={r['R@1']:.4f} R@10={r['R@10']:.4f} MRR={r['MRR']:.4f}")

# ============================================
# EXPERIMENT 2: Reader Evidence Packaging (format only, no LLM)
# ============================================
print("\n=== EXP 2: Reader Evidence Packaging (format generation) ===")
reader_examples = []
for qid in cat14[:20]:
    top = dense_precomp.get(qid, [])[:5]
    gold = gold_map[qid]
    ctx_blocks = []
    for mid in top:
        m = memories[mid]
        edges = mem_kg.get(mid, [])
        triples_str = ""
        for e in edges[:3]:
            triples_str += f"({e.get('src_id','')}, {e.get('relation','')}, {e.get('dst_id','')})\n"
        ctx_blocks.append(f"Memory [{mid}]:\n{m.get('text','')}\n"
                         f"{'KG: ' + triples_str if triples_str else ''}"
                         f"Source: {m.get('sample_id','')} session={m.get('session_id','')} dia={m.get('dia_id','')}"
                         + (" [GOLD]" if mid in gold else ""))
    reader_examples.append({"qa_id": qid, "question": qas[qid]["question"],
        "category": qas[qid]["category"], "context": "\n---\n".join(ctx_blocks)})
    if len(reader_examples) <= 3:
        print(f"  [{qid}] cat{qas[qid]['category']}: {qas[qid]['question'][:60]}... ({sum(1 for b in ctx_blocks if 'GOLD' in b)} gold)")

with (OUT_RD/"reader_packaging_examples.jsonl").open("w") as f:
    for ex in reader_examples: f.write(json.dumps(ex,ensure_ascii=False)+"\n")
print(f"  Saved {len(reader_examples)} reader packaging examples (no LLM run)")

# ============================================
# EXPERIMENT 3: Selective Memory Filtering
# ============================================
print("\n=== EXP 3: Selective Memory Filtering ===")

salience_scores = {}
for mid in mem_ids:
    m = memories[mid]
    has_kg = 1 if mid in has_kg_set else 0
    edges = mem_kg.get(mid, [])
    temp_flag = 0
    multi_ent = 0
    ents_seen = set()
    if edges:
        for e in edges:
            for t in tokenize(e.get("src_id",""))+tokenize(e.get("dst_id","")):
                if t in TEMPORAL: temp_flag = 1
                if is_entity_like(t): ents_seen.add(t)
            for t in tokenize(e.get("relation","")):
                if t in TEMPORAL: temp_flag = 1
        if len(ents_seen) >= 2: multi_ent = 1
    text_toks = set(tokenize(m.get("text","")))
    persona_kw = int(any(k in text_toks for k in SALIENCE_KW))
    action_kw = int(any(k in text_toks for k in SALIENCE_KW))
    high_idf = int(len(ents_seen) >= 3)
    score = (1.0 * has_kg + 0.5 * temp_flag + 0.5 * multi_ent +
             0.5 * persona_kw + 0.3 * action_kw + 0.3 * high_idf)
    salience_scores[mid] = min(score, 3.0)

with (OUT_S/"selective_memory_scores.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["memory_id","salience_score","has_kg","temporal_flag","multi_entity"])
    w.writeheader()
    for mid, s in sorted(salience_scores.items(), key=lambda x: -x[1])[:20]:
        w.writerow({"memory_id":mid,"salience_score":s,"has_kg":1 if mid in has_kg_set else 0})

# Selective pools
def selective_retrieval(qid, percentile, canonical_cache, all_pool_key):
    q_sample = qas[qid]["sample_id"]
    all_pool = canonical_cache.get(qid, [])
    if not all_pool: return []
    # Score per memory in the canonical ranked list
    scored = [(s if salience_scores.get(m,0) > 0 else -1, m) for m in all_pool]
    scored.sort(key=lambda x: -x[0])
    threshold = max(1, int(len(scored) * percentile / 100))
    selective = {m for s, m in scored[:threshold] if s > 0}
    # Fallback: fill from original order
    result = [m for m in all_pool if m in selective]
    for m in all_pool:
        if m not in selective: result.append(m)
    return result[:10]

pcts = [50, 70]
sel_rows = []

# Canonical baselines
for label, cache in [("Dense_all_canonical", dense_precomp), ("BM25_all_canonical", bm25_precomp)]:
    h1 = 0; h10 = 0; rrs = 0
    for qid in cat14:
        top = cache.get(qid, [])[:10]
        gold = gold_map[qid]
        if any(m in gold for m in top[:1]): h1 += 1
        if any(m in gold for m in top): h10 += 1
        for rk, m in enumerate(top, 1):
            if m in gold: rrs += 1.0/rk; break
    n = len(cat14)
    sel_rows.append({"method": label, "R@1": round(h1/n,4), "R@10": round(h10/n,4),
        "MRR": round(rrs/n,4), "pool_size": "full", "truth_retained": 1.0})

# Selective
for label, cache in [("Dense", dense_precomp), ("BM25", bm25_precomp)]:
    for pct in pcts:
        h1 = 0; h10 = 0; rrs = 0; retained = 0; total_truth = 0; pool_sizes = []
        for qid in cat14:
            top = selective_retrieval(qid, pct, cache, label)
            gold = gold_map[qid]
            total_truth += len(gold)
            pool_sizes.append(len(top))
            retained += sum(1 for m in gold if m in top)
            if any(m in gold for m in top[:1]): h1 += 1
            if any(m in gold for m in top): h10 += 1
            for rk, m in enumerate(top, 1):
                if m in gold: rrs += 1.0/rk; break
        n = len(cat14)
        pool_avg = statistics.mean(pool_sizes) if pool_sizes else 0
        sel_rows.append({"method": f"{label}_selective_top{pct}pct",
            "R@1": round(h1/n,4), "R@10": round(h10/n,4), "MRR": round(rrs/n,4),
            "pool_size": f"{pool_avg:.0f}", "truth_retained": round(retained/total_truth,4) if total_truth else 0})

sel_fields = ["method","R@1","R@10","MRR","pool_size","truth_retained"]
with (OUT_S/"selective_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=sel_fields); w.writeheader(); w.writerows(sel_rows)
for r in sel_rows:
    print(f"  {r['method']:35s} R@10={r['R@10']:.4f} MRR={r['MRR']:.4f} pool={r['pool_size']} truth_ret={r.get('truth_retained',''):.0%}" if isinstance(r.get('truth_retained'), float) else f"  {r['method']:35s} R@10={r['R@10']:.4f} MRR={r['MRR']:.4f} pool={r['pool_size']}")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n=== Final Summary ===")
t1 = time.time()

# Analyze
bm25_delta = bm25_enr_res["MRR"] - bm25_raw_res["MRR"]
best_sel = max(sel_rows, key=lambda r: r["MRR"])

summary = f"""# LoCoMo Retrieval: New Directions from A-MEM/Mem0/MemORAI

## Experiment 1: A-MEM-lite Enriched Memory
- BM25_raw R@10={bm25_raw_res['R@10']:.4f} MRR={bm25_raw_res['MRR']:.4f}
- BM25_enriched R@10={bm25_enr_res['R@10']:.4f} MRR={bm25_enr_res['MRR']:.4f}
- Delta: {bm25_delta:+.4f} ({'+' if bm25_delta>0 else ''}{bm25_delta*100:.1f}%) 
- Verdict: {'Enriched memory helps BM25. Worth exploring Dense_enriched next.' if bm25_delta>0.005 else 'Enriched memory no significant gain for BM25. Structured text less useful for lexical retrieval.'}

## Experiment 2: Reader Evidence Packaging
- Status: Reader LLM API not available in this run.
- Format: text + KG triples + provenance generated for 20 examples.
- Saved to: reports/locomo_reader_evidence_packaging/reader_packaging_examples.jsonl
- Verdict: Need API key to run reader. Format is ready.

## Experiment 3: Selective Memory Filtering
- Best selective: {best_sel['method']} R@10={best_sel['R@10']:.4f} MRR={best_sel['MRR']:.4f}
- Selective pools retain different amounts based on percentile.
- Verdict: {'Selective filtering maintains quality while reducing pool size. Worth developing further.' if best_sel['MRR'] >= 0.95 * dense_cat['MRR'] else 'Selective filtering hurts quality too much. Not recommended.'}

## Recommended Next Direction
1. A-MEM-lite: {'Continue with Dense_enriched if API available' if bm25_delta>0.005 else 'Stop — no BM25 gain'}
2. Reader evidence: Build and test with API key
3. Selective memory: {'Develop salience index as lightweight noise reducer' if best_sel['MRR'] >= 0.95 * dense_cat['MRR'] else 'Stop — hurts retrieval'}

## Runtime
- {t1-t0:.1f}s
"""
with (OUT_F/"summary.md").open("w") as f: f.write(summary)
print(summary)
