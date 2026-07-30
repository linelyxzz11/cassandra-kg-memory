#!/usr/bin/env python3
"""P7-A Gate A/B/C — BM25 validation across CSV/Cassandra/Neo4j."""
import csv, hashlib, json, time
from collections import defaultdict
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

BASE = Path("D:/memorytable/cassandra-kg-memory")
QFILE = BASE / "05_reports/evaluator_input_audit/selected_full5_questions.csv"
LEGACY = BASE / "05_reports/official_eval/bm25_raw_ranking_canonical1540.csv"

# ── Load questions ──────────────────────────
questions = {}
with open(QFILE, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        qid = r["qa_id"].strip()
        questions[qid] = {
            "qid": qid,
            "cat": r.get("category","").strip(),
            "question": r.get("question","").strip(),
            "conversation": qid.split("_qa_")[0],
        }
print(f"Questions: {len(questions)}")

# ── Load memory records ────────────────────
mem_raw = {}
with open(BASE / "01_data" / "locomo_memory_records.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        mid = r["memory_id"].strip()
        mem_raw[mid] = {
            "text": r.get("text","").strip(),
            "conv": mid.split("_session_")[0] if "_session_" in mid else "",
        }
print(f"Memories: {len(mem_raw)}")

# ── Load ERK features ──────────────────────
mem_features = {}
with open(BASE / "02_artifacts" / "p3_memory_features.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        mid = r["memory_id"].strip()
        mem_features[mid] = {
            "entities": r.get("entities","").strip(),
            "relations": r.get("relations","").strip(),
            "keywords": r.get("keywords","").strip(),
            "triples": r.get("triples","").strip(),
        }
print(f"Featured: {len(mem_features)}")

# ── BM25 scorer ───────────────────────────
def bm25_rank(query_text, records, top_k=10):
    """records = {memory_id: text}"""
    mids = list(records.keys())
    if len(mids) == 0:
        return {}
    texts = [records[mid] for mid in mids]
    vec = CountVectorizer(min_df=1, stop_words="english")
    dtm = vec.fit_transform(texts)
    q_vec = vec.transform([query_text])
    scores = (dtm @ q_vec.T).toarray().flatten()
    return {mid: float(scores[i]) for i, mid in enumerate(mids)}

def top_k(scores, k=10):
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]

# ── Data sources ──────────────────────────
def csv_raw_records(conv_id):
    return {mid: m["text"] for mid, m in mem_raw.items() if m["conv"] == conv_id}

def cassandra_raw_records(conv_id):
    from cassandra.cluster import Cluster
    cl = Cluster(["127.0.0.1"], port=9042)
    s = cl.connect("kg_memory")
    rows = s.execute(f"SELECT memory_id, raw_text FROM kg_memory_table WHERE conversation_id = '{conv_id}' ALLOW FILTERING")
    result = {r.memory_id: r.raw_text for r in rows if r.raw_text}
    cl.shutdown()
    return result

def neo4j_raw_records(conv_id):
    from neo4j import GraphDatabase
    d = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "REDACTED_NEO4J_PASSWORD"))
    with d.session() as s:
        rows = s.run("MATCH (m:LocoMoMemory {scope_id: $sid}) RETURN m.memory_id AS memory_id, m.raw_text AS raw_text", sid=conv_id)
        result = {r["memory_id"]: r["raw_text"] for r in rows if r["raw_text"]}
    d.close()
    return result

# ── Compare top-10 ────────────────────────
def compare(qid, conv_id, qtext, source_name, source_fn):
    t0 = time.time()
    records = source_fn(conv_id)
    if not records:
        return {"error": "no_records", "qid": qid}
    scores = bm25_rank(qtext, records)
    t10 = top_k(scores)
    elapsed = time.time() - t0
    return {
        "qid": qid, "conv": conv_id,
        "source": source_name,
        "candidates": len(records),
        "top10_ids": [mid for mid, _ in t10],
        "top10_scores": [round(s, 6) for _, s in t10],
        "elapsed_ms": round(elapsed*1000, 1),
    }

# ── Run validation ────────────────────────
def run_gate(label, source_fn, qids_subset=None):
    results = []
    qs = qids_subset or sorted(questions.keys())
    for i, qid in enumerate(qs[:10]):  # smoke 10 first
        q = questions[qid]
        r = compare(qid, q["conversation"], q["question"], label, source_fn)
        results.append(r)
        if "error" not in r:
            print(f"  {qid}: {r['candidates']} candidates, top1={r['top10_ids'][0] if r['top10_ids'] else ''}, {r['elapsed_ms']}ms")
    return results

# ── Execute ────────────────────────────────
print("\n=== Gate A: CSV adapter ===")
res_a = run_gate("CSV", csv_raw_records)

print("\n=== Gate B: Cassandra ===")
res_b = run_gate("Cassandra", cassandra_raw_records)

print("\n=== Gate C: Neo4j ===")
res_c = run_gate("Neo4j", neo4j_raw_records)

# ── Parity check ───────────────────────────
print("\n=== Parity ===")
for i in range(min(len(res_a), len(res_b), len(res_c))):
    match_b = res_a[i].get("top10_ids") == res_b[i].get("top10_ids")
    match_c = res_a[i].get("top10_ids") == res_c[i].get("top10_ids")
    n_cand_a = res_a[i].get("candidates",0)
    n_cand_b = res_b[i].get("candidates",0)
    n_cand_c = res_c[i].get("candidates",0)
    sym = "✅" if (match_b and match_c) else "❌"
    print(f"  {res_a[i]['qid']}: {sym} A={n_cand_a} B={n_cand_b} C={n_cand_c} | B_match={match_b} C_match={match_c}")

print("\nDone. Candidate + top10 parity checked for first 10 queries.")
