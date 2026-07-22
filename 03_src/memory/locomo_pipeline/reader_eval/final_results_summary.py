import csv
from pathlib import Path


def load_csv(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(file_path, fieldnames, rows):
    with Path(file_path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def compute_metrics(rows, hit_col="hit10", rr_col="rr"):
    n = len(rows)
    if n == 0:
        return {"R@1": 0, "R@5": 0, "R@10": 0, "MRR": 0, "n": 0}
    r1 = sum(int(r.get("hit1", 0)) for r in rows) / n
    r5 = sum(int(r.get("hit5", 0)) for r in rows) / n
    r10 = sum(int(r.get(hit_col, 0)) for r in rows) / n
    mrr = sum(float(r.get(rr_col, 0)) for r in rows) / n
    return {"R@1": r1, "R@5": r5, "R@10": r10, "MRR": mrr, "n": n}


def main():
    Path("results").mkdir(parents=True, exist_ok=True)

    tfidf = load_csv("results/locomo_retrieval_tfidf_results.csv") if Path("results/locomo_retrieval_tfidf_results.csv").exists() else None
    if tfidf is None:
        tfidf_ablation = load_csv("results/ablation_boost_00.csv")
        tfidf = tfidf_ablation

    bm25 = load_csv("results/locomo_bm25_results.csv")
    dense = load_csv("results/locomo_dense_onnx_results.csv")
    fusion = load_csv("results/locomo_fusion_bm25_dense_results.csv")
    kg_boost = load_csv("results/locomo_cassandra_kg_results.csv")
    kg_aware = load_csv("results/locomo_kg_aware_fusion_best_results.csv")
    reader = load_csv("results/llm_reader_full_v3_summary.csv") if Path("results/llm_reader_full_v3_summary.csv").exists() else None

    retrieval_table = []

    for name, data in [
        ("TF-IDF", tfidf), ("BM25", bm25), ("Dense-MiniLM", dense),
        ("BM25-Dense-RRF", fusion), ("KG-boost", kg_boost), ("KG-aware-RRF", kg_aware),
    ]:
        m = compute_metrics(data, hit_col="hit10" if name != "BM25-Dense-RRF" else "recall_10", rr_col="rr" if name != "BM25-Dense-RRF" else "rr_10")
        retrieval_table.append({
            "Method": name,
            "R@1": f"{m['R@1']:.4f}",
            "R@5": f"{m['R@5']:.4f}",
            "R@10": f"{m['R@10']:.4f}",
            "MRR@10": f"{m['MRR']:.4f}",
            "n": m["n"],
        })

    write_csv("results/final_retrieval_table.csv", ["Method", "R@1", "R@5", "R@10", "MRR@10", "n"], retrieval_table)

    print("=== Final Retrieval Table ===")
    print(f"{'Method':20s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s} {'MRR@10':>8s}")
    print("-" * 55)
    for r in retrieval_table:
        print(f"{r['Method']:20s} {r['R@1']:>8s} {r['R@5']:>8s} {r['R@10']:>8s} {r['MRR@10']:>8s}")

    if reader:
        reader_table = []
        for r in reader:
            reader_table.append({
                "Method": r["method"],
                "rEM": r.get("relaxed_em", ""),
                "rF1": r.get("relaxed_f1", ""),
                "hit10": r.get("retrieval_hit10", ""),
                "ansInEv": r.get("answer_string_in_evidence_rate", ""),
                "CA%": r.get("cannot_answer_rate", ""),
                "F1_hit": r.get("relaxed_f1_when_hit10", ""),
                "F1_miss": r.get("relaxed_f1_when_miss10", ""),
            })

        write_csv("results/final_reader_table.csv",
                  ["Method", "rEM", "rF1", "hit10", "ansInEv", "CA%", "F1_hit", "F1_miss"],
                  reader_table)

        print(f"\n=== Full v3 Reader Table (ALL) ===")
        print(f"{'Method':20s} {'rEM':>7s} {'rF1':>7s} {'hit10':>7s} {'ansInEv':>7s} {'CA%':>6s} {'F1_hit':>7s} {'F1_miss':>7s}")
        print("-" * 72)
        for r in reader_table:
            print(f"{r['Method']:20s} {r['rEM']:>7s} {r['rF1']:>7s} {r['hit10']:>7s} {r['ansInEv']:>7s} {r['CA%']:>6s} {r['F1_hit']:>7s} {r['F1_miss']:>7s}")

    if Path("results/audit_kg_boost_top10_summary.csv").exists():
        audit = load_csv("results/audit_kg_boost_top10_summary.csv")
        write_csv("results/final_top10_audit_table.csv", list(audit[0].keys()) if audit else [], audit)
        print(f"\nTop10 audit table: results/final_top10_audit_table.csv ({len(audit)} rows)")

    combined = []
    for r in retrieval_table:
        reader_match = None
        if reader:
            for rr in reader:
                if rr["Method"] == r["Method"]:
                    reader_match = rr
                    break
        combined.append({
            "Method": r["Method"],
            "Ret_R@1": r["R@1"],
            "Ret_R@10": r["R@10"],
            "Ret_MRR": r["MRR@10"],
            "Reader_rEM": reader_match["rEM"] if reader_match else "",
            "Reader_rF1": reader_match["rF1"] if reader_match else "",
            "Reader_CA%": reader_match["CA%"] if reader_match else "",
        })

    write_csv("results/final_method_comparison_table.csv",
              ["Method", "Ret_R@1", "Ret_R@10", "Ret_MRR", "Reader_rEM", "Reader_rF1", "Reader_CA%"],
              combined)

    print(f"\n=== Combined Method Comparison ===")
    print(f"{'Method':20s} {'R@1':>8s} {'R@10':>8s} {'MRR':>8s} {'rEM':>8s} {'rF1':>8s} {'CA%':>6s}")
    print("-" * 65)
    for r in combined:
        print(f"{r['Method']:20s} {r['Ret_R@1']:>8s} {r['Ret_R@10']:>8s} {r['Ret_MRR']:>8s} {r['Reader_rEM']:>8s} {r['Reader_rF1']:>8s} {r['Reader_CA%']:>6s}")

    print(f"\nOutput files:")
    print(f"  results/final_retrieval_table.csv")
    print(f"  results/final_reader_table.csv")
    print(f"  results/final_top10_audit_table.csv")
    print(f"  results/final_method_comparison_table.csv")


if __name__ == "__main__":
    main()
