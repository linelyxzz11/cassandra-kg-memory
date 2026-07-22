import csv
import re
import sys
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path("D:/memorytable/cassandra-kg-memory")
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "sample_scoped"

MEM_EMB_FILE = RESULTS / "locomo_memory_bge_large.npy"
QA_EMB_FILE = RESULTS / "locomo_qa_bge_large.npy"
MEM_IDS_FILE = RESULTS / "locomo_memory_ids_bge.txt"
QA_IDS_FILE = RESULTS / "locomo_qa_ids_bge.txt"
QA_CSV = RESULTS / "locomo_qa_records.csv"
MEMORY_CSV = RESULTS / "locomo_memory_records.csv"
EVIDENCE_CSV = RESULTS / "locomo_evidence_map.csv"
KG_EDGES_CSV = RESULTS / "locomo_kg_edges_spacy.csv"

STOPWORDS = set(
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their theirs "
    "themselves what which who whom this that these those am is are was were be "
    "been being have has had having do does did doing a an the and but if or "
    "because as until while of at by for with about against between through "
    "during before after above below to from up down in out on off over under "
    "again further then once here there when where why how all both each few "
    "more most other some such no nor not only own same so than too very s t "
    "can will just don should now d ll m o re ve y ain aren couldn didn doesn "
    "hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won "
    "wouldn also would could should may might shall".split()
)

TEMPORAL_WORDS = set(
    "when before after first last later earlier recently yesterday today "
    "tomorrow date time week month year day ago since until".split()
)

FEATURE_CONFIGS = {
    "config_B": {"entity": 0.50, "relation": 0.20, "text": 0.15, "temporal": 0.10, "multi_entity": 0.05},
}

DEFAULT_QUERYKG_TOPN = 50
DEFAULT_QUERYKG_LAM = 0.5
DEFAULT_QUERYKG_CONFIG = "config_B"


def load_csv(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 2]
    return tokens


def is_entity_like(token):
    if token in STOPWORDS:
        return False
    if token.isdigit():
        return False
    if len(token) < 3:
        return False
    return True


def has_temporal_flag(tokens):
    return int(any(t in TEMPORAL_WORDS for t in tokens))


def load_pretrained_embeddings(mem_emb_path, qa_emb_path, mem_ids_path, qa_ids_path):
    mem_embs = np.load(str(mem_emb_path))
    with open(mem_ids_path, "r", encoding="utf-8") as f:
        mem_ids = [line.strip() for line in f if line.strip()]
    qa_embs = np.load(str(qa_emb_path))
    with open(qa_ids_path, "r", encoding="utf-8") as f:
        qa_ids = [line.strip() for line in f if line.strip()]
    mem_embs = mem_embs / np.linalg.norm(mem_embs, axis=1, keepdims=True)
    qa_embs = qa_embs / np.linalg.norm(qa_embs, axis=1, keepdims=True)
    return mem_ids, mem_embs, qa_ids, qa_embs


def load_gold_map():
    gold_mem = defaultdict(set)
    for r in load_csv(EVIDENCE_CSV):
        qid = r["qa_id"].strip()
        mid = r["memory_id"].strip()
        if mid:
            gold_mem[qid].add(mid)
    return dict(gold_mem)


def load_memory_sample_map():
    mem_sample = {}
    sample_memories = defaultdict(list)
    with open(MEMORY_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mid = row["memory_id"].strip()
            sid = row["sample_id"].strip()
            txt = row["text"].strip()
            dia = row["dia_id"].strip()
            mem_sample[mid] = sid
            sample_memories[sid].append({"memory_id": mid, "text": txt, "dia_id": dia})
    return mem_sample, dict(sample_memories)


def load_qa_sample_map():
    qa_sample = {}
    for r in load_csv(QA_CSV):
        qa_sample[r["qa_id"].strip()] = r["sample_id"].strip()
    return qa_sample


def load_kg_edges_mapped():
    edges_by_memory = defaultdict(list)
    mid_by_dia = defaultdict(list)
    with open(MEMORY_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mid = row["memory_id"].strip()
            sid = row["sample_id"].strip()
            dia = row["dia_id"].strip()
            mid_by_dia[(sid, dia)].append(mid)

    with open(KG_EDGES_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gid = row["graph_id"].strip()
            ev = row["evidence"].strip()
            src = row["src_id"].strip()
            rel = row["relation"].strip()
            dst = row["dst_id"].strip()
            for mid in mid_by_dia.get((gid, ev), []):
                edges_by_memory[mid].append({"src": src, "relation": rel, "dst": dst})

    return dict(edges_by_memory)


def build_memory_kg_features(edges_by_memory):
    memory_kg_tokens = {}
    for mid, edges in edges_by_memory.items():
        all_entity_tokens = set()
        all_relation_tokens = set()
        all_tokens = set()
        has_temporal_edge = False

        for e in edges:
            src_toks = set(normalize_text(e["src"]))
            dst_toks = set(normalize_text(e["dst"]))
            rel_toks = set(normalize_text(e["relation"]))
            entity_toks = src_toks | dst_toks
            all_entity_tokens |= entity_toks
            all_relation_tokens |= rel_toks
            all_tokens |= entity_toks | rel_toks
            if has_temporal_flag(list(rel_toks)):
                has_temporal_edge = True
            for t in list(entity_toks) + list(rel_toks):
                if t in TEMPORAL_WORDS:
                    has_temporal_edge = True

        memory_kg_tokens[mid] = {
            "entity_tokens": all_entity_tokens,
            "relation_tokens": all_relation_tokens,
            "all_tokens": all_tokens,
            "has_temporal_edge": has_temporal_edge,
            "n_edges": len(edges),
        }
    return memory_kg_tokens


def compute_query_features(question):
    q_tokens = normalize_text(question)
    q_entity_tokens = set(t for t in q_tokens if is_entity_like(t))
    q_temporal = has_temporal_flag(q_tokens)
    q_token_set = set(q_tokens)
    return {
        "q_tokens": q_token_set,
        "q_entity_tokens": q_entity_tokens,
        "q_temporal": q_temporal,
    }


def compute_kg_score(qf, memory_kg_tok, feat_weights):
    if memory_kg_tok is None:
        return 0.0
    q_toks = qf["q_tokens"]
    if len(q_toks) == 0:
        return 0.0
    entity_overlap = len(q_toks & memory_kg_tok["entity_tokens"]) / max(1, len(q_toks))
    relation_overlap = len(q_toks & memory_kg_tok["relation_tokens"]) / max(1, len(q_toks))
    text_overlap = len(q_toks & memory_kg_tok["all_tokens"]) / max(1, len(q_toks))
    temporal_match = 0.0
    if qf["q_temporal"] and memory_kg_tok["has_temporal_edge"]:
        temporal_match = 1.0
    multi_entity_match = 0.0
    q_ent = qf["q_entity_tokens"]
    if len(q_ent) >= 2:
        kg_ent = memory_kg_tok["entity_tokens"]
        overlap = q_ent & kg_ent
        if len(overlap) >= 2:
            multi_entity_match = 1.0
    score = (
        feat_weights["entity"] * entity_overlap
        + feat_weights["relation"] * relation_overlap
        + feat_weights["text"] * text_overlap
        + feat_weights["temporal"] * temporal_match
        + feat_weights["multi_entity"] * multi_entity_match
    )
    return min(max(score, 0.0), 1.0)


def minmax_normalize(scores):
    mn = min(scores)
    mx = max(scores)
    if mx - mn < 1e-9:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def compute_metrics(retrieved_ids, gold_ids):
    r1 = int(bool(set(retrieved_ids[:1]) & gold_ids))
    r5 = int(bool(set(retrieved_ids[:5]) & gold_ids))
    r10 = int(bool(set(retrieved_ids) & gold_ids))
    rr = 0.0
    for i, mid in enumerate(retrieved_ids[:10], 1):
        if mid in gold_ids:
            rr = 1.0 / i
            break
    return {"R@1": r1, "R@5": r5, "R@10": r10, "MRR": rr}


def parse_qa_sample_id(qa_id):
    m = re.match(r"conv-(\d+)_qa_\d+", qa_id)
    if m:
        return f"conv-{m.group(1)}"
    return None


def parse_memory_sample_id(memory_id):
    if memory_id.startswith("conv-"):
        m = re.match(r"conv-(\d+)_", memory_id)
        if m:
            return f"conv-{m.group(1)}"
    return None


class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = None
        self.doc_tf = None
        self.doc_len = None
        self.avg_dl = None
        self.idf = None
        self.len_norm = None
        self.n_docs = 0

    def fit(self, documents):
        self.vectorizer = CountVectorizer(lowercase=True, stop_words="english",
                                          ngram_range=(1, 2), max_features=50000)
        self.doc_tf = self.vectorizer.fit_transform(documents).tocsc()
        self.n_docs = self.doc_tf.shape[0]
        self.doc_len = np.array(self.doc_tf.sum(axis=1)).flatten()
        self.avg_dl = self.doc_len.mean()
        df = np.array((self.doc_tf > 0).sum(axis=0)).flatten()
        self.idf = np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
        self.len_norm = 1.0 - self.b + self.b * (self.doc_len / self.avg_dl)

    def search(self, query, top_k=10):
        query_vec = self.vectorizer.transform([query]).tocsc()
        q_rows, q_cols = query_vec.nonzero()
        scores = np.zeros(self.n_docs)
        for _, col in zip(q_rows, q_cols):
            col_start = self.doc_tf.indptr[col]
            col_end = self.doc_tf.indptr[col + 1]
            col_rows = self.doc_tf.indices[col_start:col_end]
            col_data = self.doc_tf.data[col_start:col_end]
            tf = np.zeros(self.n_docs)
            tf[col_rows] = col_data
            tf_score = tf * (self.k1 + 1.0) / (tf + self.k1 * self.len_norm + 1e-9)
            scores += tf_score * self.idf[col]
        ranked = np.argsort(-scores)
        top_indices = ranked[:top_k]
        return top_indices.tolist(), [float(scores[i]) for i in top_indices]


def run_bm25_sample_scoped(qa_rows, sample_memories, qa_sample_map, gold_map):
    print("\n[1/4] BM25 sample-scoped...")

    sample_bm25 = {}
    for sid, mem_list in sample_memories.items():
        texts = [m["text"] for m in mem_list]
        bm = BM25Retriever(k1=1.5, b=0.75)
        bm.fit(texts)
        sample_bm25[sid] = {"bm25": bm, "memories": mem_list}

    results = []
    for r in qa_rows:
        qid = r["qa_id"].strip()
        sid = qa_sample_map.get(qid)
        question = r["question"]

        if sid not in sample_bm25:
            continue

        entry = sample_bm25[sid]
        bm = entry["bm25"]
        mems = entry["memories"]
        indices, bm_scores = bm.search(question, top_k=10)
        retrieved = [mems[i]["memory_id"] for i in indices]
        scores = [round(s, 4) for s in bm_scores]

        gold = gold_map.get(qid, set())
        metrics = compute_metrics(retrieved, gold)

        results.append({
            "qa_id": qid,
            "category": r["category"],
            "question": question,
            "gold_memory_ids": ";".join(sorted(gold)),
            "retrieved_memory_ids": ";".join(retrieved),
            "top10_memory_ids": ";".join(retrieved),
            "bm25_top10_memory_ids": ";".join(retrieved),
            "bm25_scores_top10": ";".join(f"{s:.4f}" for s in scores),
            "R@1": metrics["R@1"], "R@5": metrics["R@5"],
            "R@10": metrics["R@10"], "MRR": round(metrics["MRR"], 6),
        })

    out_path = OUT_DIR / "locomo_bm25_sample_scoped_results.csv"
    fields = ["qa_id", "category", "question", "gold_memory_ids",
              "retrieved_memory_ids", "top10_memory_ids", "bm25_top10_memory_ids",
              "bm25_scores_top10", "R@1", "R@5", "R@10", "MRR"]
    write_csv(out_path, results, fields)

    r1 = sum(r["R@1"] for r in results) / len(results)
    r5 = sum(r["R@5"] for r in results) / len(results)
    r10 = sum(r["R@10"] for r in results) / len(results)
    mrr = sum(r["MRR"] for r in results) / len(results)
    print(f"  n={len(results)}  R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}  MRR={mrr:.4f}")
    print(f"  -> {out_path}")
    return results, {"method": "BM25_sample_scoped", "n": len(results),
                     "R@1": r1, "R@5": r5, "R@10": r10, "MRR": mrr}


def run_dense_sample_scoped(qa_rows, mem_ids, mem_embs, qa_ids_list, qa_embs,
                             qa_sample_map, mem_sample_map, gold_map):
    print("\n[2/4] Dense-bge sample-scoped...")

    qid_to_idx = {qid: i for i, qid in enumerate(qa_ids_list)}
    mid_to_idx = {mid: i for i, mid in enumerate(mem_ids)}

    sample_mem_idx = defaultdict(list)
    for mid, sid in mem_sample_map.items():
        if mid in mid_to_idx:
            sample_mem_idx[sid].append(mid_to_idx[mid])

    results = []
    for r in qa_rows:
        qid = r["qa_id"].strip()
        sid = qa_sample_map.get(qid)
        question = r["question"]

        if qid not in qid_to_idx or sid not in sample_mem_idx:
            continue

        qi = qid_to_idx[qid]
        q_emb = qa_embs[qi]

        candidate_indices = sample_mem_idx[sid]
        if not candidate_indices:
            continue

        candidate_embs = mem_embs[candidate_indices]
        scores = np.dot(candidate_embs, q_emb)
        sorted_local = np.argsort(-scores)
        top_k = min(10, len(sorted_local))
        top_local = sorted_local[:top_k]

        retrieved = [mem_ids[candidate_indices[i]] for i in top_local]
        top_scores = [float(scores[i]) for i in top_local]

        gold = gold_map.get(qid, set())
        metrics = compute_metrics(retrieved, gold)

        results.append({
            "qa_id": qid,
            "category": r["category"],
            "question": question,
            "gold_memory_ids": ";".join(sorted(gold)),
            "retrieved_memory_ids": ";".join(retrieved),
            "top10_memory_ids": ";".join(retrieved),
            "dense_bge_top10_memory_ids": ";".join(retrieved),
            "dense_scores_top10": ";".join(f"{s:.4f}" for s in top_scores),
            "R@1": metrics["R@1"], "R@5": metrics["R@5"],
            "R@10": metrics["R@10"], "MRR": round(metrics["MRR"], 6),
        })

    out_path = OUT_DIR / "locomo_dense_bge_sample_scoped_results.csv"
    fields = ["qa_id", "category", "question", "gold_memory_ids",
              "retrieved_memory_ids", "top10_memory_ids", "dense_bge_top10_memory_ids",
              "dense_scores_top10", "R@1", "R@5", "R@10", "MRR"]
    write_csv(out_path, results, fields)

    r1 = sum(r["R@1"] for r in results) / len(results)
    r5 = sum(r["R@5"] for r in results) / len(results)
    r10 = sum(r["R@10"] for r in results) / len(results)
    mrr = sum(r["MRR"] for r in results) / len(results)
    print(f"  n={len(results)}  R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}  MRR={mrr:.4f}")
    print(f"  -> {out_path}")
    return results, {"method": "Dense-bge_sample_scoped", "n": len(results),
                     "R@1": r1, "R@5": r5, "R@10": r10, "MRR": mrr}


def run_dense_global_kg_sample_scoped(qa_rows, mem_ids, mem_embs, qa_ids_list, qa_embs,
                                       qa_sample_map, mem_sample_map, gold_map,
                                       edges_by_memory, kg_weight=0.1):
    print(f"\n[3/4] Dense-bge+GlobalKG sample-scoped (w={kg_weight})...")

    qid_to_idx = {qid: i for i, qid in enumerate(qa_ids_list)}
    mid_to_idx = {mid: i for i, mid in enumerate(mem_ids)}

    sample_mem_idx = defaultdict(list)
    for mid, sid in mem_sample_map.items():
        if mid in mid_to_idx:
            sample_mem_idx[sid].append(mid_to_idx[mid])

    kg_set = set(edges_by_memory.keys())
    kg_mask = {}
    for sid, idxs in sample_mem_idx.items():
        kg_mask[sid] = np.zeros(len(idxs), dtype=np.float32)
        for local_i, global_i in enumerate(idxs):
            if mem_ids[global_i] in kg_set:
                kg_mask[sid][local_i] = 1.0

    results = []
    for r in qa_rows:
        qid = r["qa_id"].strip()
        sid = qa_sample_map.get(qid)
        question = r["question"]

        if qid not in qid_to_idx or sid not in sample_mem_idx:
            continue

        qi = qid_to_idx[qid]
        q_emb = qa_embs[qi]

        candidate_indices = sample_mem_idx[sid]
        if not candidate_indices:
            continue

        candidate_embs = mem_embs[candidate_indices]
        base_scores = np.dot(candidate_embs, q_emb)
        boosted = base_scores + kg_weight * kg_mask[sid]

        sorted_local = np.argsort(-boosted)
        top_k = min(10, len(sorted_local))
        top_local = sorted_local[:top_k]

        retrieved = [mem_ids[candidate_indices[i]] for i in top_local]
        top_dense = [float(base_scores[i]) for i in top_local]
        top_kg = [float(kg_mask[sid][i]) for i in top_local]
        top_final = [float(boosted[i]) for i in top_local]

        gold = gold_map.get(qid, set())
        metrics = compute_metrics(retrieved, gold)

        results.append({
            "qa_id": qid,
            "category": r["category"],
            "question": question,
            "gold_memory_ids": ";".join(sorted(gold)),
            "retrieved_memory_ids": ";".join(retrieved),
            "top10_memory_ids": ";".join(retrieved),
            "dense_scores_top10": ";".join(f"{s:.4f}" for s in top_dense),
            "kg_boost_values_top10": ";".join(f"{s:.4f}" for s in top_kg),
            "final_scores_top10": ";".join(f"{s:.4f}" for s in top_final),
            "R@1": metrics["R@1"], "R@5": metrics["R@5"],
            "R@10": metrics["R@10"], "MRR": round(metrics["MRR"], 6),
        })

    out_path = OUT_DIR / "locomo_dense_global_kg_sample_scoped_results.csv"
    fields = ["qa_id", "category", "question", "gold_memory_ids",
              "retrieved_memory_ids", "top10_memory_ids",
              "dense_scores_top10", "kg_boost_values_top10", "final_scores_top10",
              "R@1", "R@5", "R@10", "MRR"]
    write_csv(out_path, results, fields)

    r1 = sum(r["R@1"] for r in results) / len(results)
    r5 = sum(r["R@5"] for r in results) / len(results)
    r10 = sum(r["R@10"] for r in results) / len(results)
    mrr = sum(r["MRR"] for r in results) / len(results)
    print(f"  n={len(results)}  R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}  MRR={mrr:.4f}")
    print(f"  -> {out_path}")
    return results, {"method": "Dense-bge+GlobalKG_sample_scoped", "n": len(results),
                     "R@1": r1, "R@5": r5, "R@10": r10, "MRR": mrr}


def run_query_kg_sample_scoped(qa_rows, mem_ids, mem_embs, qa_ids_list, qa_embs,
                                qa_sample_map, mem_sample_map, gold_map,
                                edges_by_memory, topn=50, lam=0.5, config_name="config_B"):
    feat_w = FEATURE_CONFIGS[config_name]
    print(f"\n[4/4] Dense-bge+QueryKG sample-scoped (topN={topn}, lam={lam}, {config_name})...")

    qid_to_idx = {qid: i for i, qid in enumerate(qa_ids_list)}
    mid_to_idx = {mid: i for i, mid in enumerate(mem_ids)}

    sample_mem_idx = defaultdict(list)
    for mid, sid in mem_sample_map.items():
        if mid in mid_to_idx:
            sample_mem_idx[sid].append(mid_to_idx[mid])

    memory_kg_tokens = build_memory_kg_features(edges_by_memory)

    results = []
    for r in qa_rows:
        qid = r["qa_id"].strip()
        sid = qa_sample_map.get(qid)
        question = r["question"]

        if qid not in qid_to_idx or sid not in sample_mem_idx:
            continue

        qi = qid_to_idx[qid]
        q_emb = qa_embs[qi]

        candidate_indices = sample_mem_idx[sid]
        if not candidate_indices:
            continue

        candidate_embs = mem_embs[candidate_indices]
        dense_scores = np.dot(candidate_embs, q_emb)

        local_topn = min(topn, len(candidate_indices))
        sorted_local = np.argsort(-dense_scores)
        topn_local = sorted_local[:local_topn]

        topn_ids = [mem_ids[candidate_indices[i]] for i in topn_local]
        topn_dense = [float(dense_scores[i]) for i in topn_local]

        qf = compute_query_features(question)

        kg_scores = []
        for mid in topn_ids:
            mkt = memory_kg_tokens.get(mid)
            ks = compute_kg_score(qf, mkt, feat_w)
            kg_scores.append(ks)

        dense_norm = minmax_normalize(topn_dense)
        final_scores = [dn + lam * ks for dn, ks in zip(dense_norm, kg_scores)]

        paired = list(zip(topn_ids, final_scores, dense_norm, kg_scores))
        paired.sort(key=lambda x: x[1], reverse=True)

        top_k_final = min(10, len(paired))
        retrieved = [p[0] for p in paired[:top_k_final]]
        sorted_final = [p[1] for p in paired[:top_k_final]]
        sorted_dense = [p[2] for p in paired[:top_k_final]]
        sorted_kg = [p[3] for p in paired[:top_k_final]]

        gold = gold_map.get(qid, set())
        metrics = compute_metrics(retrieved, gold)

        results.append({
            "qa_id": qid,
            "category": r["category"],
            "question": question,
            "gold_memory_ids": ";".join(sorted(gold)),
            "retrieved_memory_ids": ";".join(retrieved),
            "top10_memory_ids": ";".join(retrieved),
            "dense_top10_memory_ids": ";".join(topn_ids[:10]),
            "reranked_top10_memory_ids": ";".join(retrieved),
            "dense_scores_top10": ";".join(f"{s:.4f}" for s in sorted_dense),
            "kg_scores_top10": ";".join(f"{s:.4f}" for s in sorted_kg),
            "final_scores_top10": ";".join(f"{s:.4f}" for s in sorted_final),
            "R@1": metrics["R@1"], "R@5": metrics["R@5"],
            "R@10": metrics["R@10"], "MRR": round(metrics["MRR"], 6),
        })

    out_path = OUT_DIR / "locomo_dense_query_kg_sample_scoped_results.csv"
    fields = ["qa_id", "category", "question", "gold_memory_ids",
              "retrieved_memory_ids", "top10_memory_ids",
              "dense_top10_memory_ids", "reranked_top10_memory_ids",
              "dense_scores_top10", "kg_scores_top10", "final_scores_top10",
              "R@1", "R@5", "R@10", "MRR"]
    write_csv(out_path, results, fields)

    r1 = sum(r["R@1"] for r in results) / len(results)
    r5 = sum(r["R@5"] for r in results) / len(results)
    r10 = sum(r["R@10"] for r in results) / len(results)
    mrr = sum(r["MRR"] for r in results) / len(results)
    print(f"  n={len(results)}  R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}  MRR={mrr:.4f}")
    print(f"  -> {out_path}")
    return results, {"method": "Dense-bge+QueryKG_sample_scoped", "n": len(results),
                     "R@1": r1, "R@5": r5, "R@10": r10, "MRR": mrr}


def scope_audit(results_list, method_names, qa_sample_map):
    audit_rows = []
    for method_name, results in zip(method_names, results_list):
        total_checked = 0
        total_cross = 0
        affected = 0
        for r in results:
            qid = r["qa_id"]
            sid = qa_sample_map.get(qid)
            if not sid:
                continue
            top10_str = r.get("retrieved_memory_ids", "")
            top_ids = [x.strip() for x in top10_str.split(";") if x.strip()]
            cross_count = 0
            for mid in top_ids:
                ms = parse_memory_sample_id(mid)
                total_checked += 1
                if ms and ms != sid:
                    cross_count += 1
            total_cross += cross_count
            if cross_count > 0:
                affected += 1
        n = len(results)
        audit_rows.append({
            "method": method_name,
            "n_queries": n,
            "cross_sample_count": total_cross,
            "cross_sample_rate": round(total_cross / max(1, total_checked), 4),
            "affected_query_count": affected,
            "affected_query_rate": round(affected / max(1, n), 4),
        })
    return audit_rows


def summarize(method_results, all_qa_rows, qa_sample_map, sample_memories):
    summaries = []
    sample_counts = {sid: len(mems) for sid, mems in sample_memories.items()}

    for mr in method_results:
        nm = mr["name"]
        results = mr["results"]
        n = len(results)
        r1 = sum(r["R@1"] for r in results) / max(1, n)
        r5 = sum(r["R@5"] for r in results) / max(1, n)
        r10 = sum(r["R@10"] for r in results) / max(1, n)
        mrr = sum(r["MRR"] for r in results) / max(1, n)

        cand_sum = 0
        top10_sum = 0
        for r in results:
            qid = r["qa_id"]
            sid = qa_sample_map.get(qid)
            if sid:
                cand_sum += sample_counts.get(sid, 0)
            top10_str = r.get("retrieved_memory_ids", "")
            top10_ids = [x.strip() for x in top10_str.split(";") if x.strip()]
            top10_sum += len(top10_ids)

        avg_cand = cand_sum / max(1, n)
        avg_top10_len = top10_sum / max(1, n)

        total_cross = 0
        total_checked = 0
        affected = 0
        for r in results:
            qid = r["qa_id"]
            sid = qa_sample_map.get(qid)
            if not sid:
                continue
            top_str = r.get("retrieved_memory_ids", "")
            top_ids = [x.strip() for x in top_str.split(";") if x.strip()]
            cross = 0
            for mid in top_ids:
                ms = parse_memory_sample_id(mid)
                total_checked += 1
                if ms and ms != sid:
                    cross += 1
            total_cross += cross
            if cross > 0:
                affected += 1

        summaries.append({
            "Method": nm,
            "R@1": round(r1, 4),
            "R@5": round(r5, 4),
            "R@10": round(r10, 4),
            "MRR": round(mrr, 4),
            "cross_sample_rate": round(total_cross / max(1, total_checked), 4),
            "affected_query_rate": round(affected / max(1, n), 4),
            "avg_candidate_count": round(avg_cand, 4),
            "avg_top10_len": round(avg_top10_len, 4),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-globalkg", action="store_true")
    parser.add_argument("--skip-querykg", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    qa_rows = load_csv(QA_CSV)
    qa_sample_map = load_qa_sample_map()
    mem_sample_map, sample_memories = load_memory_sample_map()
    gold_map = load_gold_map()
    print(f"  {len(qa_rows)} QA pairs, {len(mem_sample_map)} memories, {len(sample_memories)} samples")

    print("Loading pre-computed bge-large embeddings...")
    mem_ids, mem_embs, qa_ids_list, qa_embs = load_pretrained_embeddings(
        MEM_EMB_FILE, QA_EMB_FILE, MEM_IDS_FILE, QA_IDS_FILE
    )
    print(f"  Memory embs: {mem_embs.shape}, Query embs: {qa_embs.shape}")

    print("Loading KG edges...")
    edges_by_memory = load_kg_edges_mapped()
    print(f"  {len(edges_by_memory)} memories with KG edges")

    method_results = []

    if not args.skip_bm25:
        bm25_res, _ = run_bm25_sample_scoped(qa_rows, sample_memories, qa_sample_map, gold_map)
        method_results.append({"name": "BM25_sample_scoped", "results": bm25_res})

    if not args.skip_dense:
        dense_res, _ = run_dense_sample_scoped(
            qa_rows, mem_ids, mem_embs, qa_ids_list, qa_embs,
            qa_sample_map, mem_sample_map, gold_map
        )
        method_results.append({"name": "Dense-bge_sample_scoped", "results": dense_res})

    if not args.skip_globalkg:
        kg_res, _ = run_dense_global_kg_sample_scoped(
            qa_rows, mem_ids, mem_embs, qa_ids_list, qa_embs,
            qa_sample_map, mem_sample_map, gold_map, edges_by_memory, kg_weight=0.1
        )
        method_results.append({"name": "Dense-bge+GlobalKG_sample_scoped", "results": kg_res})

    if not args.skip_querykg:
        qkg_res, _ = run_query_kg_sample_scoped(
            qa_rows, mem_ids, mem_embs, qa_ids_list, qa_embs,
            qa_sample_map, mem_sample_map, gold_map, edges_by_memory,
            topn=50, lam=0.5, config_name="config_B"
        )
        method_results.append({"name": "Dense-bge+QueryKG_sample_scoped", "results": qkg_res})

    print("\n=== VALIDATION ===")
    for mr in method_results:
        nm = mr["name"]
        res = mr["results"]
        n = len(res)
        if n != len(qa_rows):
            print(f"  {nm}: FAIL - n={n} (expected {len(qa_rows)})")
        else:
            print(f"  {nm}: n={n} OK")

        total_cross = 0
        affected = 0
        for r in res:
            qid = r["qa_id"]
            sid = qa_sample_map.get(qid)
            if not sid:
                affected += 1
                continue
            top_str = r.get("retrieved_memory_ids", "")
            top_ids = [x.strip() for x in top_str.split(";") if x.strip()]
            has_cross = False
            for mid in top_ids:
                ms = parse_memory_sample_id(mid)
                if ms and ms != sid:
                    total_cross += 1
                    has_cross = True
            if has_cross:
                affected += 1
        cr = total_cross / max(1, sum(1 for r in res for _ in r.get("retrieved_memory_ids", "").split(";")))
        ar = affected / max(1, n)
        if total_cross > 0:
            print(f"  {nm}: CROSS-SAMPLE DETECTED! cross={total_cross}, rate={cr:.4e}, affected={affected}")
        else:
            print(f"  {nm}: cross_sample_count=0, affected=0 PASS")

    print("\n=== SUMMARY ===")
    summaries = summarize(method_results, qa_rows, qa_sample_map, sample_memories)
    summary_path = OUT_DIR / "sample_scoped_retrieval_summary.csv"
    summary_fields = ["Method", "R@1", "R@5", "R@10", "MRR",
                      "cross_sample_rate", "affected_query_rate",
                      "avg_candidate_count", "avg_top10_len"]
    write_csv(summary_path, summaries, summary_fields)
    print(f"Summary: {summary_path}")

    for s in summaries:
        print(f"  {s['Method']:<40s} R@1={s['R@1']:<8.4f} R@5={s['R@5']:<8.4f} "
              f"R@10={s['R@10']:<8.4f} MRR={s['MRR']:<8.4f} "
              f"cross_rate={s['cross_sample_rate']} affected={s['affected_query_rate']}")

    print("\nDone.")


if __name__ == "__main__":
    main()