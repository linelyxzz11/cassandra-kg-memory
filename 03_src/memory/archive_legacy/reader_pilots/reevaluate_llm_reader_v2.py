import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


NUMBER_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

YES_WORDS = {"yes", "yeah", "yep", "correct", "true"}
NO_WORDS = {"no", "nope", "false"}

CA_VARIANTS = {"cannot answer", "cannot determine", "not enough information", "insufficient information"}


def load_csv(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_strict(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_relaxed(text):
    text = normalize_strict(text)
    tokens = text.split()
    result = []
    for tok in tokens:
        if tok in NUMBER_MAP:
            result.append(NUMBER_MAP[tok])
        elif tok in YES_WORDS:
            result.append("yes")
        elif tok in NO_WORDS:
            result.append("no")
        else:
            result.append(tok)
    text = " ".join(result)
    if text in CA_VARIANTS:
        text = "cannot answer"
    return text


def compute_em(pred, gold, normalizer):
    return 1 if normalizer(pred) == normalizer(gold) else 0


def compute_token_f1(pred, gold, normalizer):
    pred_tokens = normalizer(pred).split()
    gold_tokens = normalizer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    rows = load_csv("results/llm_reader_pilot_v2_results.csv")

    ga_map = defaultdict(set)
    for r in load_csv("results/locomo_evidence_map.csv"):
        ga_map[r["qa_id"]].add(r.get("evidence_id", ""))

    reeval = []
    for r in rows:
        pred = r.get("predicted_answer", "")
        gold = r.get("gold_answer", "")

        s_em = compute_em(pred, gold, normalize_strict)
        s_f1 = compute_token_f1(pred, gold, normalize_strict)
        r_em = compute_em(pred, gold, normalize_relaxed)
        r_f1 = compute_token_f1(pred, gold, normalize_relaxed)

        ids = [x for x in r.get("top10_evidence_ids", "").split(";") if x]
        is_hit = any(g in ids for g in ga_map.get(r["qa_id"], set()))
        is_ca = normalize_relaxed(pred) == "cannot answer"

        reeval.append({
            "qa_id": r["qa_id"], "category": r["category"], "method": r["method"],
            "question": r["question"], "gold_answer": gold,
            "predicted_answer": pred,
            "strict_em": s_em, "strict_f1": s_f1,
            "relaxed_em": r_em, "relaxed_f1": r_f1,
            "retrieval_hit10": int(is_hit), "cannot_answer": int(is_ca),
        })

    r_fields = ["qa_id", "category", "method", "question", "gold_answer",
                "predicted_answer", "strict_em", "strict_f1",
                "relaxed_em", "relaxed_f1", "retrieval_hit10", "cannot_answer"]
    with Path("results/llm_reader_pilot_v2_reeval_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=r_fields)
        w.writeheader()
        w.writerows(reeval)

    print("=== LLM Reader v2 Re-evaluation ===")
    print()

    by_method = defaultdict(list)
    for r in reeval:
        by_method[r["method"]].append(r)

    print(f"{'Method':20s} {'sEM':>7s} {'sF1':>7s} {'rEM':>7s} {'rF1':>7s} {'hit10':>7s} {'CA%':>6s}")
    print("-" * 65)

    summary_rows = []

    for mname in ["BM25", "Dense-ONNX-MiniLM", "BM25-Dense-RRF", "KG-boost"]:
        grp = by_method[mname]
        n = len(grp)
        s_em = sum(r["strict_em"] for r in grp) / n
        s_f1 = sum(r["strict_f1"] for r in grp) / n
        r_em = sum(r["relaxed_em"] for r in grp) / n
        r_f1 = sum(r["relaxed_f1"] for r in grp) / n
        hit10_rate = sum(r["retrieval_hit10"] for r in grp) / n
        ca_rate = sum(r["cannot_answer"] for r in grp) / n

        r_f1_hit = [r["relaxed_f1"] for r in grp if r["retrieval_hit10"]]
        r_f1_miss = [r["relaxed_f1"] for r in grp if not r["retrieval_hit10"]]

        print(f"{mname:20s} {s_em:7.4f} {s_f1:7.4f} {r_em:7.4f} {r_f1:7.4f} {hit10_rate:7.4f} {ca_rate:6.4f}")

        summary_rows.append({
            "method": mname, "category": "ALL", "n": n,
            "strict_em": f"{s_em:.4f}", "strict_f1": f"{s_f1:.4f}",
            "relaxed_em": f"{r_em:.4f}", "relaxed_f1": f"{r_f1:.4f}",
            "retrieval_hit10": f"{hit10_rate:.4f}",
            "cannot_answer_rate": f"{ca_rate:.4f}",
            "relaxed_f1_when_hit10": f"{sum(r_f1_hit)/len(r_f1_hit) if r_f1_hit else 0:.4f}",
            "relaxed_f1_when_miss10": f"{sum(r_f1_miss)/len(r_f1_miss) if r_f1_miss else 0:.4f}",
        })

    for mname in ["BM25", "Dense-ONNX-MiniLM", "BM25-Dense-RRF", "KG-boost"]:
        grp = by_method[mname]
        by_cat = defaultdict(list)
        for r in grp:
            by_cat[str(r["category"])].append(r)
        for cat in sorted(by_cat.keys()):
            sub = by_cat[cat]
            n = len(sub)
            s_em = sum(r["strict_em"] for r in sub) / n
            s_f1 = sum(r["strict_f1"] for r in sub) / n
            r_em = sum(r["relaxed_em"] for r in sub) / n
            r_f1 = sum(r["relaxed_f1"] for r in sub) / n
            hit10_rate = sum(r["retrieval_hit10"] for r in sub) / n
            ca_rate = sum(r["cannot_answer"] for r in sub) / n
            r_f1_h = [r["relaxed_f1"] for r in sub if r["retrieval_hit10"]]
            r_f1_m = [r["relaxed_f1"] for r in sub if not r["retrieval_hit10"]]
            summary_rows.append({
                "method": mname, "category": f"cat_{cat}", "n": n,
                "strict_em": f"{s_em:.4f}", "strict_f1": f"{s_f1:.4f}",
                "relaxed_em": f"{r_em:.4f}", "relaxed_f1": f"{r_f1:.4f}",
                "retrieval_hit10": f"{hit10_rate:.4f}",
                "cannot_answer_rate": f"{ca_rate:.4f}",
                "relaxed_f1_when_hit10": f"{sum(r_f1_h)/len(r_f1_h) if r_f1_h else 0:.4f}",
                "relaxed_f1_when_miss10": f"{sum(r_f1_m)/len(r_f1_m) if r_f1_m else 0:.4f}",
            })

    s_fields = ["method", "category", "n", "strict_em", "strict_f1",
                "relaxed_em", "relaxed_f1", "retrieval_hit10",
                "cannot_answer_rate", "relaxed_f1_when_hit10", "relaxed_f1_when_miss10"]
    with Path("results/llm_reader_pilot_v2_reeval_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=s_fields)
        w.writeheader()
        w.writerows(summary_rows)

    changed = [r for r in reeval if r["relaxed_f1"] > r["strict_f1"] or r["relaxed_em"] > r["strict_em"]]
    changed_path = "results/llm_reader_v2_reeval_changed_cases.csv"
    if changed:
        c_fields = list(changed[0].keys())
    else:
        c_fields = ["qa_id", "category", "method", "question", "gold_answer",
                    "predicted_answer", "strict_em", "strict_f1", "relaxed_em",
                    "relaxed_f1", "retrieval_hit10", "cannot_answer"]
    with Path(changed_path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=c_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(changed)

    print(f"\nChanged cases: {len(changed)}")
    if changed:
        print("Examples (relaxed score > strict score):")
        for case in changed[:5]:
            print(f"  [{case['method']}] {case['qa_id']}")
            print(f"    Gold: {case['gold_answer'][:60]}")
            print(f"    Pred: {case['predicted_answer'][:60]}")
            print(f"    sEM={case['strict_em']} sF1={case['strict_f1']}  rEM={case['relaxed_em']} rF1={case['relaxed_f1']}")

    print(f"\nOutput: results/llm_reader_pilot_v2_reeval_results.csv")
    print(f"Summary: results/llm_reader_pilot_v2_reeval_summary.csv")
    print(f"Changed: results/llm_reader_v2_reeval_changed_cases.csv")


if __name__ == "__main__":
    main()
