
import os
import time
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

candidate_paths = [
    {
        "memory": r".\results\locomo_memory_records.csv",
        "qa": r".\results\locomo_qa_records.csv",
        "evidence": r".\results\locomo_evidence_map.csv",
        "out": r".\results\locomo_retrieval_tfidf_results.csv"
    },
    {
        "memory": r".\results\locomo\locomo_memory_records.csv",
        "qa": r".\results\locomo\locomo_qa_records.csv",
        "evidence": r".\results\locomo\locomo_evidence_map.csv",
        "out": r".\results\locomo\locomo_retrieval_tfidf_results.csv"
    }
]

paths = None
for p in candidate_paths:
    if os.path.exists(p["memory"]) and os.path.exists(p["qa"]) and os.path.exists(p["evidence"]):
        paths = p
        break

if paths is None:
    checked = []
    for p in candidate_paths:
        checked.extend([p["memory"], p["qa"], p["evidence"]])
    raise FileNotFoundError("Required CSV files not found. Checked: " + " | ".join(checked))

memory_df = pd.read_csv(paths["memory"]).fillna("")
qa_df = pd.read_csv(paths["qa"]).fillna("")
evidence_df = pd.read_csv(paths["evidence"]).fillna("")

required_memory_cols = {"memory_id", "text"}
required_qa_cols = {"qa_id", "question", "category"}
required_evidence_cols = {"qa_id", "memory_id"}

if not required_memory_cols.issubset(set(memory_df.columns)):
    raise ValueError("memory CSV missing columns: " + str(required_memory_cols - set(memory_df.columns)))
if not required_qa_cols.issubset(set(qa_df.columns)):
    raise ValueError("qa CSV missing columns: " + str(required_qa_cols - set(qa_df.columns)))
if not required_evidence_cols.issubset(set(evidence_df.columns)):
    raise ValueError("evidence CSV missing columns: " + str(required_evidence_cols - set(evidence_df.columns)))

memory_df["memory_id"] = memory_df["memory_id"].astype(str)
memory_df["text"] = memory_df["text"].astype(str)
qa_df["qa_id"] = qa_df["qa_id"].astype(str)
qa_df["question"] = qa_df["question"].astype(str)
qa_df["category"] = qa_df["category"].astype(str)
evidence_df["qa_id"] = evidence_df["qa_id"].astype(str)
evidence_df["memory_id"] = evidence_df["memory_id"].astype(str)

memory_ids = memory_df["memory_id"].tolist()
memory_texts = memory_df["text"].tolist()

evidence_map = defaultdict(set)
for _, row in evidence_df.iterrows():
    evidence_map[row["qa_id"]].add(row["memory_id"])

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50000,
    min_df=1,
    sublinear_tf=True,
    norm="l2"
)

build_start = time.perf_counter()
memory_matrix = vectorizer.fit_transform(memory_texts)
build_ms = (time.perf_counter() - build_start) * 1000

results = []
latencies = []

for _, qa in qa_df.iterrows():
    qa_id = qa["qa_id"]
    question = qa["question"]
    category = qa["category"]
    gold_ids = evidence_map.get(qa_id, set())

    start = time.perf_counter()
    query_vector = vectorizer.transform([question])
    scores = linear_kernel(query_vector, memory_matrix).ravel()
    ranked_idx = scores.argsort()[::-1]
    elapsed_ms = (time.perf_counter() - start) * 1000
    latencies.append(elapsed_ms)

    top1 = [memory_ids[i] for i in ranked_idx[:1]]
    top5 = [memory_ids[i] for i in ranked_idx[:5]]
    top10 = [memory_ids[i] for i in ranked_idx[:10]]

    rr = 0.0
    for rank, mid in enumerate(top10, start=1):
        if mid in gold_ids:
            rr = 1.0 / rank
            break

    results.append({
        "qa_id": qa_id,
        "category": category,
        "question": question,
        "gold_memory_ids": ";".join(sorted(gold_ids)),
        "top1_memory_ids": ";".join(top1),
        "top5_memory_ids": ";".join(top5),
        "top10_memory_ids": ";".join(top10),
        "recall_1": int(any(mid in gold_ids for mid in top1)),
        "recall_5": int(any(mid in gold_ids for mid in top5)),
        "recall_10": int(any(mid in gold_ids for mid in top10)),
        "rr_10": rr,
        "latency_ms": elapsed_ms
    })

results_df = pd.DataFrame(results)

os.makedirs(os.path.dirname(paths["out"]), exist_ok=True)
results_df.to_csv(paths["out"], index=False, encoding="utf-8-sig")

print("LoCoMo TF-IDF retrieval completed.")
print(f"Memory CSV: {paths['memory']}")
print(f"QA CSV: {paths['qa']}")
print(f"Evidence CSV: {paths['evidence']}")
print(f"Memory records: {len(memory_df)}")
print(f"QA records: {len(qa_df)}")
print(f"Evidence mappings: {len(evidence_df)}")
print(f"TF-IDF build latency ms: {build_ms:.3f}")
print(f"Average query latency ms: {sum(latencies) / len(latencies):.3f}")
print(f"Recall@1: {results_df['recall_1'].mean():.4f}")
print(f"Recall@5: {results_df['recall_5'].mean():.4f}")
print(f"Recall@10: {results_df['recall_10'].mean():.4f}")
print(f"MRR@10: {results_df['rr_10'].mean():.4f}")
print("")
print("Category-level metrics:")
for category, group in results_df.groupby("category"):
    print(
        f"category {category}: "
        f"n={len(group)}, "
        f"Recall@1={group['recall_1'].mean():.4f}, "
        f"Recall@5={group['recall_5'].mean():.4f}, "
        f"Recall@10={group['recall_10'].mean():.4f}, "
        f"MRR@10={group['rr_10'].mean():.4f}, "
        f"latency_ms={group['latency_ms'].mean():.3f}"
    )
print("")
print(f"Results saved to: {paths['out']}")