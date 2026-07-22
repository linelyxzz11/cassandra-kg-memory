"""KG-Native LoCoMo retriever. No Dense embeddings. Lexical/entity/relation/temporal anchors + KG structure ranking.""" 
import csv, json, math, re, statistics, time
from collections import defaultdict
from pathlib import Path

BASE = Path("D:/memorytable/cassandra-kg-memory/results")
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_retrieval_kg_native")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

STOP = set("i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between through during before after above below to from up down in out on off over under again further then once here there when where why how all both each few more most other some such no nor not only own same so than too very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn also would could should may might shall".split())
TEMPORAL = set("when before after first last later earlier recently yesterday today tomorrow date time week month year day ago since until".split())
RELATION_KEYWORDS = set("went bought met talked liked visited attended worked lived called gave asked told remembered saw found lost bought sold read wrote learned taught moved traveled started finished".split())

def tokenize(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return [t for t in text if t not in STOP and len(t) >= 2]

def is_entity_like(token):
    return token not in STOP and token not in RELATION_KEYWORDS and token not in TEMPORAL and len(token) >= 3 and not token.isdigit()

def is_relation_like(token):
    return token in RELATION_KEYWORDS

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

cat14_qids = sorted([qid for qid, q in qas.items() if q["category"] != "5"])
print(f"  memories={len(mem_ids)}  cat1-4_queries={len(cat14_qids)}")

# === Build KG index ===
print("\n=== Building KG index ===")
mem_kg_edges = defaultdict(list)
with (BASE/"locomo_kg_edges_spacy.csv").open(encoding="utf-8-sig") as f:
    for e in csv.DictReader(f):
        ev, gid, src, rel, dst = e.get("evidence",""), e["graph_id"], e["src_id"], e["relation"], e["dst_id"]
        for mid, m in memories.items():
            if m["sample_id"] == gid and (m["dia_id"] == ev or ev in mid):
                mem_kg_edges[mid].append({"src": src, "relation": rel, "dst": dst, "evidence": ev,
                    "confidence": float(e.get("confidence", 0) or 0)})
                break

# Per-memory KG features
mem_features = {}
entity2mems = defaultdict(set)
relation2mems = defaultdict(set)
all_token2mems = defaultdict(set)
entity2entities = defaultdict(set)

for mid in mem_ids:
    edges = mem_kg_edges.get(mid, [])
    if not edges:
        mem_features[mid] = {"entity": set(), "relation": set(), "all": set(),
            "temporal": False, "edge_count": 0, "connected_entities": set(), "paths": []}
        continue
    et = set(); rt = set(); temp = False
    connected = set()
    for e in edges:
        for t in tokenize(e["src"]): et.add(t); connected.add(e["src"])
        for t in tokenize(e["dst"]): et.add(t); connected.add(e["dst"])
        for t in tokenize(e["relation"]):
            rt.add(t)
            if t in TEMPORAL: temp = True
        all_toks = set(tokenize(e["src"]) + tokenize(e["dst"]) + tokenize(e["relation"]))
        for t in all_toks:
            if t in TEMPORAL: temp = True
    at = et | rt
    mem_features[mid] = {"entity": et, "relation": rt, "all": at, "temporal": temp,
        "edge_count": len(edges), "connected_entities": connected, "paths": edges}
    for t in et: entity2mems[t].add(mid)
    for t in rt: relation2mems[t].add(mid)
    for t in at: all_token2mems[t].add(mid)

# 1-hop entity-entity adjacency
entity_adj = defaultdict(set)
for mid, edges in mem_kg_edges.items():
    for e in edges:
        entity_adj[e["src"]].add(e["dst"])
        entity_adj[e["dst"]].add(e["src"])

print(f"  KG mems={len(mem_kg_edges)}  entity_index={len(entity2mems)}  relation_index={len(relation2mems)}")

# === Query anchor extraction ===
print("\n=== Extracting query anchors ===")
query_anchors = {}
for qid in cat14_qids:
    question = qas[qid]["question"]
    toks = tokenize(question)
    ents = set(t for t in toks if is_entity_like(t))
    rels = set(t for t in toks if is_relation_like(t))
    temp = any(t in TEMPORAL for t in toks)
    multi_ent = len(ents) >= 2
    query_anchors[qid] = {"tokens": set(toks), "entity": ents, "relation": rels,
        "temporal": temp, "multi_entity": multi_ent}

# === KG-Native candidate generation + structural scoring ===
print("\n=== KG-Native retrieval ===")
sample_map = {mid: memories[mid]["sample_id"] for mid in mem_ids}
qa_sample = {qid: qas[qid]["sample_id"] for qid in cat14_qids}

def kg_native_retrieve(qid, max_candidates=200):
    anchors = query_anchors[qid]
    q_sample = qa_sample[qid]
    q_ents = anchors["entity"]
    q_rels = anchors["relation"]
    q_temp = anchors["temporal"]
    q_multi = anchors["multi_entity"]
    
    candidates = defaultdict(float)
    
    # Seed: direct entity match
    seed_mems = set()
    for ent in q_ents:
        seed_mems |= {m for m in entity2mems.get(ent, set()) if sample_map.get(m) == q_sample}
    
    # 1-hop expansion: from matched entities to neighbor entities
    hop1_mems = set()
    matched_entities = set()
    for ent in q_ents:
        for e2 in entity_adj.get(ent, set()):
            matched_entities.add(e2)
    for ent in matched_entities:
        hop1_mems |= {m for m in entity2mems.get(ent, set()) if sample_map.get(m) == q_sample}
    
    # Relation match
    rel_mems = set()
    for rel in q_rels:
        rel_mems |= {m for m in relation2mems.get(rel, set()) if sample_map.get(m) == q_sample}
    
    # Union all candidate memories
    all_mems = seed_mems | hop1_mems | rel_mems
    
    # Score each memory
    for mid in all_mems:
        feats = mem_features.get(mid)
        if feats is None: continue
        
        # Entity coverage
        if len(q_ents) > 0:
            entity_cov = len(q_ents & feats["entity"]) / len(q_ents)
        else:
            entity_cov = 0.0
        
        # Relation match
        if len(q_rels) > 0:
            rel_match = len(q_rels & feats["relation"]) / len(q_rels)
        else:
            rel_match = 0.0
        
        # Temporal
        temp_match = 1.0 if (q_temp and feats["temporal"]) else 0.0
        
        # Multi-entity
        multi_bonus = 1.0 if (q_multi and len(q_ents & feats["entity"]) >= 2) else 0.0
        
        # Path count & edge count
        path_cnt = feats["edge_count"]
        edge_cnt = feats["edge_count"]
        
        # Path length: 1 if direct entity match, 2 if from hop1
        is_seed = mid in seed_mems
        min_path = 1 if is_seed else 2
        
        score = (2.0 * entity_cov + 1.5 * rel_match + 1.0 * temp_match +
                 1.0 * multi_bonus + 0.5 * math.log1p(path_cnt) +
                 0.5 * math.log1p(edge_cnt) - 0.3 * min_path)
        
        candidates[mid] = score
    
    ranked = sorted(candidates.items(), key=lambda x: -x[1])
    return [mid for mid, _ in ranked[:max_candidates]]

# === Evaluation ===
print("\n=== Evaluation ===")
Ks = [10, 50, 100, 200]
kg_native_ranks = {}
for qid in cat14_qids:
    kg_native_ranks[qid] = kg_native_retrieve(qid, 200)

# Load BM25
bm25_ranks = {}
dia2mid = {memories[mid]["dia_id"]: mid for mid in mem_ids if "dia_id" in memories[mid]}
try:
    with (BASE/"locomo_bm25_results.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            qid = r["qa_id"]
            retrieved = r.get("retrieved_memory_ids","")
            if retrieved:
                mids = []
                for sid in [x.strip() for x in retrieved.split(";") if x.strip()]:
                    if sid in dia2mid: mids.append(dia2mid[sid])
                bm25_ranks[qid] = mids
    print(f"  BM25 loaded: {len(bm25_ranks)} queries")
except Exception as e:
    print(f"  BM25 error: {e}")

# Load Dense-bge (for baseline comparison)
dense_ranks = {}
try:
    with (BASE/"locomo_dense_bge_results.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            retrieved = r.get("retrieved_memory_ids","")
            if retrieved:
                dense_ranks[r["qa_id"]] = [x.strip() for x in retrieved.split(";") if x.strip()]
    print(f"  Dense loaded: {len(dense_ranks)} queries")
except: print("  Dense not available")

# Load GlobalKG-Prior (Dense + w=0.1 has_KG)
globalkg_ranks = {}
kg_set = set(mem_kg_edges.keys())
try:
    with (BASE/"locomo_dense_kg_boost_best_results.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            retrieved = r.get("retrieved_memory_ids","")
            if retrieved:
                globalkg_ranks[r["qa_id"]] = [x.strip() for x in retrieved.split(";") if x.strip()]
    print(f"  GlobalKG-Prior loaded: {len(globalkg_ranks)} queries")
except:
    print("  GlobalKG-Prior not available, computing from Dense+boost")
    for qid in cat14_qids:
        if qid not in dense_ranks: continue
        scored = []
        for mid in dense_ranks[qid]:
            s = 1.0 / (scored.__len__() + 1)  # rank-based
            if mid in kg_set: s += 0.1
            scored.append((s, mid))
        scored.sort(key=lambda x: -x[0])
        globalkg_ranks[qid] = [mid for _, mid in scored]

# BM25 also sample-scoped — filter within eval function, not pre-filter
bm25_src = {}
for qid in cat14_qids:
    if qid in bm25_ranks:
        qs = qa_sample[qid]
        bm25_src[qid] = [m for m in bm25_ranks[qid] if sample_map.get(m) == qs]
bm25_ranks = bm25_src

# Candidate recall (KG-candidate-only: no structural scoring, just entity+relation match)
kg_cand_only = {}
for qid in cat14_qids:
    anchors = query_anchors[qid]
    q_sample = qa_sample[qid]
    seed = set()
    for ent in anchors["entity"]:
        seed |= {m for m in entity2mems.get(ent, set()) if sample_map.get(m) == q_sample}
    for rel in anchors["relation"]:
        seed |= {m for m in relation2mems.get(rel, set()) if sample_map.get(m) == q_sample}
    kg_cand_only[qid] = list(seed)

def eval_retrieval(qids, rank_func, k_vals=Ks):
    results = {}
    for k in k_vals:
        hits = 0; rrs = []
        for qid in qids:
            cands = rank_func(qid)[:k]
            gold = gold_map[qid]
            if any(m in gold for m in cands[:1]): hits += 1
            for rank, mid in enumerate(cands[:10], 1):
                if mid in gold: rrs.append(1.0/rank); break
            else: rrs.append(0)
        results[f"R@1@{k}"] = hits / len(qids)
    # Standard R@10 and MRR (top10)
    hits10 = 0; rrs10 = []
    for qid in qids:
        cands = rank_func(qid)[:10]
        gold = gold_map[qid]
        if any(m in gold for m in cands): hits10 += 1
        for rank, mid in enumerate(cands, 1):
            if mid in gold: rrs10.append(1.0/rank); break
        else: rrs10.append(0)
    results["R@10"] = hits10 / len(qids)
    results["MRR"] = statistics.mean(rrs10)
    
    # Candidate recall
    for k in k_vals:
        hits = 0
        for qid in qids:
            cands = rank_func(qid)[:k]
            if any(m in gold_map[qid] for m in cands): hits += 1
        results[f"cand_recall@{k}"] = hits / len(qids)
    results["n"] = len(qids)
    return results

# Overall
methods = {
    "KG-Native": lambda qid: (kg_native_ranks.get(qid, []) + 
        [m for m in mem_ids if sample_map.get(m)==qa_sample[qid] and m not in kg_native_ranks.get(qid,[])])[:200],
    "KG-candidate-only": lambda qid: (kg_cand_only.get(qid, []) + 
        [m for m in mem_ids if sample_map.get(m)==qa_sample[qid] and m not in kg_cand_only.get(qid,[])])[:200],
}
if bm25_ranks:
    methods["BM25"] = lambda qid: bm25_ranks.get(qid, [])[:200]
if dense_ranks:
    methods["Dense-bge"] = lambda qid: dense_ranks.get(qid, [])[:200]
    methods["GlobalKG-Prior"] = lambda qid: globalkg_ranks.get(qid, [])[:200]

ov_rows = []
for name, func in methods.items():
    r = eval_retrieval(cat14_qids, func)
    r["method"] = name
    ov_rows.append(r)
    print(f"  {name:25s} R@10={r['R@10']:.4f} MRR={r['MRR']:.4f} cand_recall@100={r.get('cand_recall@100',0):.4f}")

of = ["method","n"] + [f"R@1@{k}" for k in Ks] + ["R@10","MRR"] + [f"cand_recall@{k}" for k in Ks]
with (OUT/"kg_native_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=of, extrasaction="ignore")
    w.writeheader(); w.writerows(ov_rows)

# By category
cat_data = defaultdict(list)
for qid in cat14_qids: cat_data[qas[qid]["category"]].append(qid)

cat_rows = []
for cat in sorted(cat_data, key=lambda x: int(x)):
    qids = cat_data[cat]
    for name, func in methods.items():
        r = eval_retrieval(qids, func)
        r["category"] = f"cat{cat}"
        r["method"] = name
        r["n"] = len(qids)
        cat_rows.append(r)
cf = ["category","method","n"] + [f"R@1@{k}" for k in Ks] + ["R@10","MRR"] + [f"cand_recall@{k}" for k in Ks]
with (OUT/"kg_native_by_category.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cf, extrasaction="ignore")
    w.writeheader(); w.writerows(cat_rows)

# Anchor coverage
print("\n=== Anchor coverage ===")
ac_rows = []
for cat in sorted(cat_data, key=lambda x: int(x)):
    qids = cat_data[cat]
    ac = {"category": f"cat{cat}", "n_questions": len(qids)}
    ac["pct_entity_anchor"] = sum(1 for q in qids if len(query_anchors[q]["entity"]) > 0) / len(qids)
    ac["pct_relation_anchor"] = sum(1 for q in qids if len(query_anchors[q]["relation"]) > 0) / len(qids)
    ac["pct_temporal"] = sum(1 for q in qids if query_anchors[q]["temporal"]) / len(qids)
    ac["pct_any_kg_candidate"] = sum(1 for q in qids if len(kg_cand_only.get(q,[])) > 0) / len(qids)
    ac["avg_entity_anchors"] = statistics.mean([len(query_anchors[q]["entity"]) for q in qids])
    ac["avg_relation_anchors"] = statistics.mean([len(query_anchors[q]["relation"]) for q in qids])
    ac["avg_kg_candidates"] = statistics.mean([len(kg_cand_only.get(q,[])) for q in qids])
    ac_rows.append(ac)
    print(f"  cat{cat}: ent_anchor={ac['pct_entity_anchor']:.1%} rel_anchor={ac['pct_relation_anchor']:.1%} temp={ac['pct_temporal']:.1%} any_kg={ac['pct_any_kg_candidate']:.1%} avg_cands={ac['avg_kg_candidates']:.0f}")

acf = ["category","n_questions","pct_entity_anchor","pct_relation_anchor","pct_temporal",
       "pct_any_kg_candidate","avg_entity_anchors","avg_relation_anchors","avg_kg_candidates"]
with (OUT/"kg_native_anchor_coverage.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=acf, extrasaction="ignore")
    w.writeheader(); w.writerows(ac_rows)

# Candidate recall comparison
cr_rows = []
for name, func in methods.items():
    r = eval_retrieval(cat14_qids, func)
    r["method"] = name
    cr_rows.append(r)
crf = ["method"] + [f"cand_recall@{k}" for k in Ks] + ["n"]
with (OUT/"kg_native_candidate_recall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=crf, extrasaction="ignore")
    w.writeheader(); w.writerows(cr_rows)

# Rescue/hurt vs Dense
rh_rows = []
if dense_ranks:
    for name, func in methods.items():
        if name in ("Dense-bge", "KG-candidate-only"): continue
        rescue = hurt = 0
        for qid in cat14_qids:
            d10 = set(dense_ranks.get(qid,[])[:1])
            k10 = set(func(qid)[:1])
            gold = gold_map[qid]
            d_hit = bool(d10 & gold)
            k_hit = bool(k10 & gold)
            if k_hit and not d_hit: rescue += 1
            if d_hit and not k_hit: hurt += 1
        rh_rows.append({"comparison": f"{name} vs Dense-bge", "rescue@1": rescue, "hurt@1": hurt, "net@1": rescue - hurt})
    with (OUT/"kg_native_rescue_hurt.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["comparison","rescue@1","hurt@1","net@1"])
        w.writeheader(); w.writerows(rh_rows)

# Success/failure examples
success_ex = []; fail_ex = []
for qid in cat14_qids:
    d10 = set(dense_ranks.get(qid,[])[:1])
    k10 = set(kg_native_ranks.get(qid,[])[:1])
    gold = gold_map[qid]
    if bool(k10 & gold) and not bool(d10 & gold):
        success_ex.append({"qa_id": qid, "category": int(qas[qid]["category"]),
            "question": qas[qid]["question"], "truth_memory_ids": sorted(gold),
            "kg_native_top5": kg_native_ranks.get(qid, [])[:5],
            "dense_top5": dense_ranks.get(qid, [])[:5],
            "anchors_entity": sorted(query_anchors[qid]["entity"]),
            "anchors_relation": sorted(query_anchors[qid]["relation"])})
    if not bool(d10 & gold) and not bool(k10 & gold) and gold:
        fail_ex.append({"qa_id": qid, "category": int(qas[qid]["category"]),
            "question": qas[qid]["question"], "truth_memory_ids": sorted(gold),
            "kg_native_top5": kg_native_ranks.get(qid, [])[:5],
            "anchors_entity": sorted(query_anchors[qid]["entity"]),
            "anchors_relation": sorted(query_anchors[qid]["relation"]),
            "has_kg_candidate": bool(kg_cand_only.get(qid, {}))})

with (OUT/"kg_native_examples_success.jsonl").open("w") as f:
    for ex in success_ex[:50]: f.write(json.dumps(ex, ensure_ascii=False)+"\n")
with (OUT/"kg_native_examples_failure.jsonl").open("w") as f:
    for ex in fail_ex[:50]: f.write(json.dumps(ex, ensure_ascii=False)+"\n")

# Method comparison summary
kg_r10 = [r for r in ov_rows if r["method"]=="KG-Native"][0]
bm25_r10 = [r for r in ov_rows if r["method"]=="BM25"][0] if bm25_ranks else None
dense_r10 = [r for r in ov_rows if r["method"]=="Dense-bge"][0] if dense_ranks else None
ac_overall = ac_rows[0] if ac_rows else {}

verdict_beats_bm25 = kg_r10["R@10"] >= bm25_r10["R@10"] if bm25_r10 else "?"
verdict_anchor_ok = ac_overall.get("pct_any_kg_candidate", 0) >= 0.70 if ac_overall else "?"

summary = f"""# KG-Native Method Comparison

## Overall (cat1-4, sample-scoped)
- KG-Native: R@10={kg_r10['R@10']:.4f}, MRR={kg_r10['MRR']:.4f}
- BM25: R@10={bm25_r10['R@10']:.4f}, MRR={bm25_r10['MRR']:.4f}
- Dense-bge: R@10={dense_r10['R@10']:.4f}, MRR={dense_r10['MRR']:.4f}

## Anchor Coverage
- Entity anchor: {ac_overall.get('pct_entity_anchor',0):.1%}
- Any KG candidate: {ac_overall.get('pct_any_kg_candidate',0):.1%}
- Avg candidates: {ac_overall.get('avg_kg_candidates',0):.0f}

## Verdict
- Beats BM25: {verdict_beats_bm25}
- Anchor coverage >= 70%: {verdict_anchor_ok}
- Continue to Cassandra serving? {'YES' if (verdict_beats_bm25 or verdict_anchor_ok) else 'NO'}

## Runtime
- {time.time()-t0:.1f}s
"""
with (OUT/"method_comparison_summary.md").open("w") as f: f.write(summary)

with (OUT/"run_config.json").open("w") as f:
    json.dump({"method":"KG-Native","no_dense":True,"sample_scoped":"cat1-4",
        "queries":len(cat14_qids)}, f, indent=2)

print(f"\n=== DONE ({time.time()-t0:.1f}s) ===")
print(f"KG-Native: R@10={kg_r10['R@10']:.4f} MRR={kg_r10['MRR']:.4f}")
print(f"BM25: R@10={bm25_r10['R@10']:.4f} MRR={bm25_r10['MRR']:.4f}")
print(f"Beats BM25? {verdict_beats_bm25} | Anchor>=70%? {verdict_anchor_ok}")
print(f"Success examples: {len(success_ex)} | Failures: {len(fail_ex)}")
for ex in success_ex[:3]:
    print(f"  SUCCESS: {ex['qa_id']} cat{ex['category']}: {ex['question'][:80]}")
for ex in fail_ex[:3]:
    print(f"  FAIL:    {ex['qa_id']} cat{ex['category']}: {ex['question'][:80]}  has_kg={ex['has_kg_candidate']}")
