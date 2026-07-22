import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def load_qa(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_gold_evidence(file_path):
    mapping = defaultdict(set)
    with Path(file_path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["qa_id"]].add(row["evidence_id"])
    return mapping


def build_memory_index(memory_file):
    memory_texts = []
    memory_evidence = []
    with Path(memory_file).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mid = row["memory_id"]
            ev_match = re.search(r"D\d+:\d+$", mid)
            if not ev_match:
                continue
            memory_texts.append(row["text"])
            memory_evidence.append(ev_match.group())
    return memory_texts, memory_evidence


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
        return ranked[:top_k].tolist()


def eval_metrics(pred_evidence, gold_evidence):
    if not gold_evidence:
        return {"hit1": 0, "hit5": 0, "hit10": 0, "rr": 0.0}
    hits = [1 if e in gold_evidence else 0 for e in pred_evidence]
    rr = 0.0
    for rank, h in enumerate(hits, start=1):
        if h:
            rr = 1.0 / rank
            break
    return {
        "hit1": 1 if hits[0] == 1 else 0,
        "hit5": 1 if any(hits[:5]) else 0,
        "hit10": 1 if any(hits) else 0,
        "rr": rr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="BM25 retrieval baseline on LoCoMo."
    )
    parser.add_argument("--qa", default="results/locomo_qa_records.csv")
    parser.add_argument("--evidence", default="results/locomo_evidence_map.csv")
    parser.add_argument("--memory", default="results/locomo_memory_records.csv")
    parser.add_argument("--output", default="results/locomo_bm25_results.csv")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    gold_map = load_gold_evidence(args.evidence)
    qa_rows = load_qa(args.qa)
    all_texts, all_evidences = build_memory_index(args.memory)
    print(f"Loaded {len(all_texts)} memory turns, {len(qa_rows)} QA pairs")

    print("Building BM25 index (sparse)...")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.fit(all_texts)
    print(f"  vocab: {len(bm25.vectorizer.vocabulary_)}, avg_dl: {bm25.avg_dl:.1f}")

    results = []
    print(f"Running {len(qa_rows)} queries...")

    for idx, qa in enumerate(qa_rows):
        qa_id = qa["qa_id"]
        question = qa["question"]
        category = qa.get("category", "")
        gold_ids = gold_map.get(qa_id, set())

        top_indices = bm25.search(question, top_k=args.top_k)
        pred_evidence = [all_evidences[i] for i in top_indices]

        metrics = eval_metrics(pred_evidence, gold_ids)
        results.append({
            "qa_id": qa_id, "category": category, "question": question,
            "gold_memory_ids": ";".join(sorted(gold_ids)),
            "retrieved_memory_ids": ";".join(pred_evidence),
            "hit1": metrics["hit1"], "hit5": metrics["hit5"],
            "hit10": metrics["hit10"], "rr": metrics["rr"],
        })

        if (idx + 1) % 200 == 0:
            r1 = sum(r["hit1"] for r in results) / len(results)
            r10 = sum(r["hit10"] for r in results) / len(results)
            print(f"  {idx+1}/{len(qa_rows)}  R@1={r1:.4f}  R@10={r10:.4f}")

    r1 = sum(r["hit1"] for r in results) / len(results)
    r5 = sum(r["hit5"] for r in results) / len(results)
    r10 = sum(r["hit10"] for r in results) / len(results)
    mrr = sum(r["rr"] for r in results) / len(results)

    fieldnames = ["qa_id", "category", "question", "gold_memory_ids",
                  "retrieved_memory_ids", "hit1", "hit5", "hit10", "rr"]
    with Path(args.output).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n=== BM25 Results ===")
    print(f"Method: BM25 (k1=1.5, b=0.75)")
    print(f"QA count: {len(qa_rows)}")
    print(f"Recall@1:  {r1:.4f}")
    print(f"Recall@5:  {r5:.4f}")
    print(f"Recall@10: {r10:.4f}")
    print(f"MRR@10:    {mrr:.4f}")

    by_cat = defaultdict(list)
    for r in results:
        by_cat[str(r["category"])].append(r)

    print("\nCategory-level:")
    for cat in sorted(by_cat.keys()):
        grp = by_cat[cat]
        n = len(grp)
        print(f"  cat {cat}: n={n}  R@1={sum(r['hit1'] for r in grp)/n:.4f}  "
              f"R@10={sum(r['hit10'] for r in grp)/n:.4f}  "
              f"MRR={sum(r['rr'] for r in grp)/n:.4f}")

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
