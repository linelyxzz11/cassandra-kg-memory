import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def load_csv(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_memory_texts(file_path):
    lookup = {}
    with Path(file_path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["memory_id"]] = {"text": row["text"], "timestamp": row.get("timestamp", "")}
    return lookup


def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    rows = load_csv("results/llm_reader_pilot_results.csv")
    mem_lookup = load_memory_texts("results/locomo_memory_records.csv")
    ga_map = defaultdict(set)
    for r in load_csv("results/locomo_evidence_map.csv"):
        ga_map[r["qa_id"]].add(r.get("evidence_id", ""))

    qa_answer = {}
    for r in load_csv("results/locomo_qa_records.csv"):
        qa_answer[r["qa_id"]] = r.get("answer", "")

    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    print("=== LLM Reader Pilot Audit ===")
    print(f"Total rows: {len(rows)}")
    print()

    header_labels = [
        "Method", "rows", "empty_ids", "empty_texts", "avg_ids",
        "avg_text_len", "cannot_answer", "retrieval_hit10",
        "answer_in_evidence", "F1_when_hit10", "F1_when_miss10"
    ]
    print(f"{'Method':20s} {'rows':>5s} {'empty':>5s} {'etext':>5s} {'avgID':>5s} {'avgLen':>6s} {'CA%':>5s} {'hit10%':>7s} {'ansInEv%':>8s} {'F1_hit':>7s} {'F1_miss':>7s}")
    print("-" * 95)

    summary_rows = []
    audit_examples = []

    for mname in ["BM25", "Dense-ONNX-MiniLM", "BM25-Dense-RRF", "KG-boost"]:
        grp = by_method[mname]
        n = len(grp)
        empty_ids = sum(1 for r in grp if not r.get("top10_evidence_ids", "").strip())
        empty_texts = sum(1 for r in grp if not r.get("evidence_texts", "") or r.get("evidence_texts") == "[]")

        avg_ids = 0
        total_text_len = 0
        for r in grp:
            ids_raw = r.get("top10_evidence_ids", "")
            ids_list = [x for x in ids_raw.split(";") if x] if ids_raw else []
            avg_ids += len(ids_list)
            try:
                texts = json.loads(r.get("evidence_texts", "[]"))
                total_text_len += sum(len(t) for t in texts)
            except Exception:
                pass
        avg_ids = avg_ids / n if n else 0
        avg_text_len = total_text_len / n if n else 0

        cannot_answer = sum(1 for r in grp if r.get("predicted_answer", "").strip().lower() == "cannot answer")
        ca_rate = cannot_answer / n if n else 0

        hit10_count = 0
        ans_in_ev_count = 0
        f1_hit = []
        f1_miss = []
        for r in grp:
            ids_raw = r.get("top10_evidence_ids", "")
            ids_list = [x for x in ids_raw.split(";") if x] if ids_raw else []
            gold_ids = ga_map.get(r["qa_id"], set())
            gold_answer = r.get("gold_answer", "")

            is_hit = any(g in ids_list for g in gold_ids)
            if is_hit:
                hit10_count += 1
                f1_hit.append(float(r.get("token_f1", 0)))
            else:
                f1_miss.append(float(r.get("token_f1", 0)))

            gold_norm = normalize_answer(gold_answer)
            try:
                texts = json.loads(r.get("evidence_texts", "[]"))
                text_blob = " ".join(texts).lower()
                if gold_norm and gold_norm in text_blob:
                    ans_in_ev_count += 1
            except Exception:
                pass

        hit10_rate = hit10_count / n if n else 0
        ans_in_ev_rate = ans_in_ev_count / n if n else 0
        avg_f1_hit = sum(f1_hit) / len(f1_hit) if f1_hit else 0
        avg_f1_miss = sum(f1_miss) / len(f1_miss) if f1_miss else 0

        print(f"{mname:20s} {n:5d} {empty_ids:5d} {empty_texts:5d} {avg_ids:5.1f} {avg_text_len:6.0f} {ca_rate:5.2f} {hit10_rate:7.2f} {ans_in_ev_rate:8.2f} {avg_f1_hit:7.4f} {avg_f1_miss:7.4f}")

        summary_rows.append({
            "method": mname, "rows": n, "empty_ids": empty_ids,
            "empty_texts": empty_texts, "avg_ids": avg_ids,
            "avg_evidence_text_len": avg_text_len,
            "cannot_answer_rate": ca_rate,
            "retrieval_hit10": hit10_rate,
            "answer_in_evidence_rate": ans_in_ev_rate,
            "avg_f1_when_hit10": avg_f1_hit,
            "avg_f1_when_miss10": avg_f1_miss,
        })

        examples = [r for r in grp if r.get("predicted_answer", "").strip().lower() != "cannot answer"][:3]
        for r in examples:
            try:
                texts = json.loads(r.get("evidence_texts", "[]"))
                text_preview = " | ".join(t[:120] for t in texts[:3])
            except Exception:
                text_preview = ""
            audit_examples.append({
                "method": mname, "qa_id": r["qa_id"], "category": r["category"],
                "question": r["question"], "gold_answer": r["gold_answer"],
                "top10_ids": r.get("top10_evidence_ids", ""),
                "evidence_preview": text_preview[:400],
                "predicted_answer": r.get("predicted_answer", ""),
                "retrieval_hit10": int(any(g in r.get("top10_evidence_ids", "").split(";") for g in ga_map.get(r["qa_id"], set()))),
                "exact_match": r.get("exact_match", 0),
                "token_f1": r.get("token_f1", 0),
            })

    print()
    print("Suspicious findings:")

    kg_f1_hit = summary_rows[-1]["avg_f1_when_hit10"]
    bm25_f1_hit = summary_rows[0]["avg_f1_when_hit10"]
    print(f"  KG-boost F1 when hit@10: {kg_f1_hit:.4f}")
    print(f"  BM25    F1 when hit@10: {bm25_f1_hit:.4f}")
    if kg_f1_hit < bm25_f1_hit:
        print("  NOTE: KG-boost reader performs worse even when evidence IS found.")
        print("  Possible: evidence text is shorter/less self-contained than BM25 evidence.")

    ca_kg = summary_rows[-1]["cannot_answer_rate"]
    print(f"  KG-boost CannotAnswer rate: {ca_kg:.2f}")
    if ca_kg > 0.3:
        print("  NOTE: >30% CannotAnswer. Evidence may lack temporal metadata or context.")

    ans_in_ev = summary_rows[-1]["answer_in_evidence_rate"]
    print(f"  KG-boost answer literally in evidence: {ans_in_ev:.2f}")
    if ans_in_ev < 0.1:
        print("  NOTE: Gold answers are rarely verbatim in evidence. Reader must infer —")
        print("  and may fail without session timestamps or dia_id context.")

    print()
    print("Category analysis (KG-boost):")
    kg_grp = by_method["KG-boost"]
    by_cat = defaultdict(list)
    for r in kg_grp:
        by_cat[str(r["category"])].append(r)
    for cat in sorted(by_cat.keys()):
        grp = by_cat[cat]
        n = len(grp)
        ca = sum(1 for r in grp if r.get("predicted_answer", "").strip().lower() == "cannot answer")
        f1 = sum(float(r.get("token_f1", 0)) for r in grp) / n if n else 0
        hit10 = 0
        for r in grp:
            ids = [x for x in r.get("top10_evidence_ids", "").split(";") if x]
            if any(g in ids for g in ga_map.get(r["qa_id"], set())):
                hit10 += 1
        print(f"  cat {cat}: n={n}, CannotAnswer={ca}, F1={f1:.4f}, hit10={hit10}/{n} ({100*hit10/n:.0f}%)")

    print()
    print("Evidence metadata check (3 random KG-boost samples):")
    kg_samples = kg_grp[:3]
    for r in kg_samples:
        qa_id = r["qa_id"]
        ids = [x for x in r.get("top10_evidence_ids", "").split(";") if x]
        graph_id = qa_id.split("_qa_")[0]
        print(f"  QA: {qa_id}")
        for ev in ids[:3]:
            parts = ev.split(":")
            if len(parts) == 2 and parts[0].startswith("D"):
                mid = f"{graph_id}_session_{parts[0][1:]}_{ev}"
            else:
                mid = f"{graph_id}_{ev}"
            info = mem_lookup.get(mid, {})
            print(f"    {mid}: ts={info.get('timestamp', 'N/A')}, text={info.get('text', 'N/A')[:80]}")

    s_fields = ["method", "rows", "empty_ids", "empty_texts", "avg_ids",
                "avg_evidence_text_len", "cannot_answer_rate", "retrieval_hit10",
                "answer_in_evidence_rate", "avg_f1_when_hit10", "avg_f1_when_miss10"]
    with Path("results/llm_reader_pilot_audit_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=s_fields)
        w.writeheader()
        w.writerows(summary_rows)

    e_fields = ["method", "qa_id", "category", "question", "gold_answer",
                "top10_ids", "evidence_preview", "predicted_answer",
                "retrieval_hit10", "exact_match", "token_f1"]
    with Path("results/llm_reader_pilot_audit_examples.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=e_fields)
        w.writeheader()
        w.writerows(audit_examples)

    print(f"\nOutput: results/llm_reader_pilot_audit_summary.csv")
    print(f"Examples: results/llm_reader_pilot_audit_examples.csv")


if __name__ == "__main__":
    main()
