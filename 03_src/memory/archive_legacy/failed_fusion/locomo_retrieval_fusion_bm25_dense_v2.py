import csv
import re
from collections import defaultdict
from pathlib import Path

BASE = Path("D:/memorytable/cassandra-kg-memory/results")

BM25_CSV = BASE / "locomo_bm25_results.csv"
DENSE_CSV = BASE / "locomo_dense_bge_results.csv"
QA_CSV = BASE / "locomo_qa_records.csv"
EVIDENCE_CSV = BASE / "locomo_evidence_map.csv"
OUT_RESULTS = BASE / "locomo_fusion_bm25_dense_bge_results.csv"
OUT_SUMMARY = BASE / "locomo_fusion_bm25_dense_bge_summary.csv"

METHOD = "RRF(BM25, Dense-bge-large)"
RRF_K = 60


def load_csv_rows(file_path):
    with Path(file_path).open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_evidence_list(raw):
    if not raw:
        return []
    return [x for x in raw.split(";") if x]


def compute_rrf(bm25_ranked, dense_ranked, k=60):
    bm25_rank = {}
    dense_rank = {}
    for i, ev in enumerate(bm25_ranked):
        bm25_rank[ev] = i + 1
    for i, ev in enumerate(dense_ranked):
        dense_rank[ev] = i + 1
    all_ev = set(bm25_rank.keys()) | set(dense_rank.keys())
    scores = {}
    for ev in all_ev:
        score = 0.0
        if ev in bm25_rank:
            score += 1.0 / (k + bm25_rank[ev])
        if ev in dense_rank:
            score += 1.0 / (k + dense_rank[ev])
        scores[ev] = score
    ranked = sorted(scores.keys(), key=lambda e: (-scores[e], e))
    return ranked


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
    print("Loading rankings...")
    bm25 = {r["qa_id"]: r for r in load_csv_rows(BM25_CSV)}
    dense = {r["qa_id"]: r for r in load_csv_rows(DENSE_CSV)}
    print(f"  BM25: {len(bm25)} queries")
    print(f"  Dense-bge-large: {len(dense)} queries")

    qa_rows = load_csv_rows(QA_CSV)
    print(f"  QA: {len(qa_rows)} queries")

    gold_map = defaultdict(set)
    for r in load_csv_rows(EVIDENCE_CSV):
        gold_map[r["qa_id"]].add(r["evidence_id"])

    results = []
    for qa in qa_rows:
        qa_id = qa["qa_id"]
        question = qa["question"]
        category = qa.get("category", "")
        gold_ids = gold_map.get(qa_id, set())

        bm25_ranked = parse_evidence_list(bm25.get(qa_id, {}).get("retrieved_memory_ids", ""))
        dense_ranked = parse_evidence_list(dense.get(qa_id, {}).get("retrieved_memory_ids", ""))

        fusion_ranked = compute_rrf(bm25_ranked, dense_ranked, RRF_K)
        fusion_top1 = fusion_ranked[:1]
        fusion_top5 = fusion_ranked[:5]
        fusion_top10 = fusion_ranked[:10]
        retrieved = fusion_ranked[:10]

        metrics = eval_metrics(fusion_top10, gold_ids)

        results.append({
            "qa_id": qa_id,
            "category": category,
            "question": question,
            "gold_memory_ids": ";".join(sorted(gold_ids)),
            "bm25_top10_memory_ids": ";".join(bm25_ranked[:10]),
            "dense_bge_top10_memory_ids": ";".join(dense_ranked[:10]),
            "fusion_top1_memory_ids": ";".join(fusion_top1),
            "fusion_top5_memory_ids": ";".join(fusion_top5),
            "fusion_top10_memory_ids": ";".join(fusion_top10),
            "retrieved_memory_ids": ";".join(retrieved),
            "recall_1": metrics["hit1"],
            "recall_5": metrics["hit5"],
            "recall_10": metrics["hit10"],
            "rr_10": metrics["rr"],
        })

    r_fields = [
        "qa_id", "category", "question", "gold_memory_ids",
        "bm25_top10_memory_ids", "dense_bge_top10_memory_ids",
        "fusion_top1_memory_ids", "fusion_top5_memory_ids",
        "fusion_top10_memory_ids", "retrieved_memory_ids",
        "recall_1", "recall_5", "recall_10", "rr_10",
    ]
    with Path(OUT_RESULTS).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=r_fields)
        w.writeheader()
        w.writerows(results)

    r1 = sum(r["recall_1"] for r in results) / len(results)
    r5 = sum(r["recall_5"] for r in results) / len(results)
    r10 = sum(r["recall_10"] for r in results) / len(results)
    mrr = sum(r["rr_10"] for r in results) / len(results)

    print(f"\n=== BM25 + Dense-bge-large Fusion ===")
    print(f"Method: {METHOD}, k={RRF_K}")
    print(f"QA count: {len(results)}")
    print(f"Recall@1:  {r1:.4f}")
    print(f"Recall@5:  {r5:.4f}")
    print(f"Recall@10: {r10:.4f}")
    print(f"MRR@10:    {mrr:.4f}")

    by_cat = defaultdict(list)
    for r in results:
        by_cat[str(r["category"])].append(r)

    summary_rows = [{
        "method": METHOD,
        "category": "ALL",
        "n": len(results),
        "recall_1": f"{r1:.4f}",
        "recall_5": f"{r5:.4f}",
        "recall_10": f"{r10:.4f}",
        "mrr_10": f"{mrr:.4f}",
    }]

    print("\nCategory-level:")
    for cat in sorted(by_cat.keys()):
        grp = by_cat[cat]
        n = len(grp)
        cr1 = sum(r["recall_1"] for r in grp) / n
        cr5 = sum(r["recall_5"] for r in grp) / n
        cr10 = sum(r["recall_10"] for r in grp) / n
        cmrr = sum(r["rr_10"] for r in grp) / n
        print(f"  cat {cat}: n={n}  R@1={cr1:.4f}  R@10={cr10:.4f}  MRR={cmrr:.4f}")
        summary_rows.append({
            "method": METHOD,
            "category": f"cat_{cat}",
            "n": n,
            "recall_1": f"{cr1:.4f}",
            "recall_5": f"{cr5:.4f}",
            "recall_10": f"{cr10:.4f}",
            "mrr_10": f"{cmrr:.4f}",
        })

    s_fields = ["method", "category", "n", "recall_1", "recall_5", "recall_10", "mrr_10"]
    with Path(OUT_SUMMARY).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=s_fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nOutput: {OUT_RESULTS}")
    print(f"Summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
