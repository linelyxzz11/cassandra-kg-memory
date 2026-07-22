import csv
import json
from collections import defaultdict
from pathlib import Path


def load_csv(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = load_csv("results/llm_reader_pilot_v2_results.csv")
    ga_map = defaultdict(set)
    for r in load_csv("results/locomo_evidence_map.csv"):
        ga_map[r["qa_id"]].add(r.get("evidence_id", ""))

    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    print("=== LLM Reader v2 Error Audit ===")
    print()

    print(f"{'Method':20s} {'n':>4s} {'EM':>7s} {'F1':>7s} {'hit10':>7s} {'CA%':>6s} {'F1_hit':>7s} {'F1_miss':>7s}")
    print("-" * 70)

    for mname in ["BM25", "Dense-ONNX-MiniLM", "BM25-Dense-RRF", "KG-boost"]:
        grp = by_method[mname]
        n = len(grp)
        em = sum(int(r["exact_match"]) for r in grp) / n if n else 0
        f1 = sum(float(r["token_f1"]) for r in grp) / n if n else 0
        ca = sum(1 for r in grp if r.get("predicted_answer", "").strip().lower() == "cannot answer")
        ca_rate = ca / n if n else 0

        f1_hit_list = []
        f1_miss_list = []
        hit10_count = 0
        for r in grp:
            ids = [x for x in r.get("top10_evidence_ids", "").split(";") if x]
            gold = ga_map.get(r["qa_id"], set())
            if any(g in ids for g in gold):
                hit10_count += 1
                f1_hit_list.append(float(r["token_f1"]))
            else:
                f1_miss_list.append(float(r["token_f1"]))

        hit10_rate = hit10_count / n if n else 0
        f1_hit = sum(f1_hit_list) / len(f1_hit_list) if f1_hit_list else 0
        f1_miss = sum(f1_miss_list) / len(f1_miss_list) if f1_miss_list else 0

        print(f"{mname:20s} {n:4d} {em:7.4f} {f1:7.4f} {hit10_rate:7.4f} {ca_rate:6.4f} {f1_hit:7.4f} {f1_miss:7.4f}")

    print()
    print("=== KG-boost Error Categories ===")

    kg_grp = by_method["KG-boost"]

    cat_a = []
    cat_b = []
    cat_c = []
    cat_d = []

    for r in kg_grp:
        ids = [x for x in r.get("top10_evidence_ids", "").split(";") if x]
        gold = ga_map.get(r["qa_id"], set())
        is_hit = any(g in ids for g in gold)
        f1_val = float(r.get("token_f1", 0))
        is_ca = r.get("predicted_answer", "").strip().lower() == "cannot answer"
        is_cat2 = r.get("category", "") == "2"

        if is_hit and f1_val == 0:
            cat_a.append(r)
        if is_hit and is_ca:
            cat_b.append(r)
        if not is_hit and f1_val > 0:
            cat_c.append(r)
        if is_cat2 and is_ca:
            cat_d.append(r)

    print(f"  A: hit10=1 AND F1=0      : {len(cat_a)} cases")
    print(f"  B: hit10=1 AND CannotAnswer: {len(cat_b)} cases")
    print(f"  C: hit10=0 AND F1>0      : {len(cat_c)} cases")
    print(f"  D: cat2 AND CannotAnswer : {len(cat_d)} cases")

    examples = []

    def add_examples(label, source_list, max_n=10):
        for r in source_list[:max_n]:
            try:
                texts = json.loads(r.get("evidence_texts", "[]"))
                text_preview = " | ".join(t[:200] for t in texts[:5])
            except Exception:
                text_preview = ""
            try:
                prompt_raw = json.loads(r.get("prompt", ""))
                prompt_preview = prompt_raw[:2000]
            except Exception:
                prompt_preview = r.get("prompt", "")[:2000]

            ids = [x for x in r.get("top10_evidence_ids", "").split(";") if x]
            gold = ga_map.get(r["qa_id"], set())
            is_hit = any(g in ids for g in gold)

            examples.append({
                "error_type": label,
                "qa_id": r["qa_id"],
                "category": r["category"],
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "predicted_answer": r.get("predicted_answer", ""),
                "retrieval_hit10": int(is_hit),
                "token_f1": r.get("token_f1", 0),
                "top10_evidence_ids": r.get("top10_evidence_ids", ""),
                "evidence_preview": text_preview,
                "prompt_preview": prompt_preview,
            })

    add_examples("A", cat_a)
    add_examples("B", cat_b)
    add_examples("C", cat_c)
    add_examples("D", cat_d)

    print()
    print("Sample from each category:")

    for label, source, n_show in [("A: hit10=1 & F1=0", cat_a, 2), ("B: hit10=1 & CannotAnswer", cat_b, 2),
                                    ("D: cat2 & CannotAnswer", cat_d, 2)]:
        print(f"\n  --- {label} ---")
        for r in source[:n_show]:
            print(f"    QA: {r['qa_id']} cat={r['category']}")
            print(f"    Q: {r['question'][:120]}")
            print(f"    Gold: {r['gold_answer'][:80]}")
            print(f"    Pred: {r.get('predicted_answer', '')[:80]}")
            ids = r.get("top10_evidence_ids", "")
            print(f"    IDs({len(ids.split(';') if ids else 0)}): {ids[:100]}")

    e_fields = ["error_type", "qa_id", "category", "question", "gold_answer",
                "predicted_answer", "retrieval_hit10", "token_f1",
                "top10_evidence_ids", "evidence_preview", "prompt_preview"]
    with Path("results/llm_reader_v2_error_examples.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=e_fields)
        w.writeheader()
        w.writerows(examples)

    summary = []
    for mname in ["BM25", "Dense-ONNX-MiniLM", "BM25-Dense-RRF", "KG-boost"]:
        grp = by_method[mname]
        n = len(grp)
        em = sum(int(r["exact_match"]) for r in grp) / n if n else 0
        f1_avg = sum(float(r["token_f1"]) for r in grp) / n if n else 0
        ca = sum(1 for r in grp if r.get("predicted_answer", "").strip().lower() == "cannot answer")
        ca_rate = ca / n if n else 0
        hit10_count = sum(1 for r in grp if any(g in r.get("top10_evidence_ids", "").split(";") for g in ga_map.get(r["qa_id"], set())))
        hit10_rate = hit10_count / n if n else 0
        f1_hit_list = [float(r["token_f1"]) for r in grp if any(g in r.get("top10_evidence_ids", "").split(";") for g in ga_map.get(r["qa_id"], set()))]
        f1_miss_list = [float(r["token_f1"]) for r in grp if not any(g in r.get("top10_evidence_ids", "").split(";") for g in ga_map.get(r["qa_id"], set()))]
        summary.append({
            "method": mname, "n": n,
            "EM": f"{em:.4f}", "F1": f"{f1_avg:.4f}",
            "retrieval_hit10": f"{hit10_rate:.4f}",
            "cannot_answer_rate": f"{ca_rate:.4f}",
            "F1_when_hit10": f"{sum(f1_hit_list)/len(f1_hit_list) if f1_hit_list else 0:.4f}",
            "F1_when_miss10": f"{sum(f1_miss_list)/len(f1_miss_list) if f1_miss_list else 0:.4f}",
            "error_A_count": len([r for r in grp if any(g in r.get("top10_evidence_ids", "").split(";") for g in ga_map.get(r["qa_id"], set())) and float(r.get("token_f1", 0)) == 0]),
            "error_B_count": len([r for r in grp if any(g in r.get("top10_evidence_ids", "").split(";") for g in ga_map.get(r["qa_id"], set())) and r.get("predicted_answer", "").strip().lower() == "cannot answer"]),
            "error_D_count": len([r for r in grp if r.get("category", "") == "2" and r.get("predicted_answer", "").strip().lower() == "cannot answer"]),
        })

    s_fields = ["method", "n", "EM", "F1", "retrieval_hit10", "cannot_answer_rate",
                "F1_when_hit10", "F1_when_miss10", "error_A_count", "error_B_count", "error_D_count"]
    with Path("results/llm_reader_v2_error_audit_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=s_fields)
        w.writeheader()
        w.writerows(summary)

    print(f"\nOutput: results/llm_reader_v2_error_audit_summary.csv")
    print(f"Examples: results/llm_reader_v2_error_examples.csv")


if __name__ == "__main__":
    main()
