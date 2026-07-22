import csv
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


METHODS = {
    "BM25": ("results/locomo_bm25_results.csv", "retrieved_memory_ids"),
    "Dense-ONNX-MiniLM": ("results/locomo_dense_onnx_results.csv", "retrieved_memory_ids"),
    "BM25-Dense-RRF": ("results/locomo_fusion_bm25_dense_results.csv", "fusion_top10_memory_ids"),
    "KG-boost": ("results/locomo_cassandra_kg_results.csv", "retrieved_memory_ids"),
}


def load_csv(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_memory_texts(file_path):
    lookup = {}
    with Path(file_path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["memory_id"]] = row["text"]
    return lookup


def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_em(pred, gold):
    return 1 if normalize_answer(pred) == normalize_answer(gold) else 0


def compute_token_f1(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
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


def build_prompt(question, evidence_texts, evidence_ids=None):
    lines = ["You are answering a memory-based question. Below are relevant conversation turns retrieved from the user's memory, followed by a question."]
    lines.append("")
    lines.append("Use only the provided evidence. If the evidence does not contain enough information, answer \"Cannot answer\".")
    lines.append("")
    lines.append("Return only the answer. Do not explain.")
    lines.append("")
    lines.append("Evidence:")
    for i, text in enumerate(evidence_texts):
        safe_text = text.replace("\n", " ").strip()
        if len(safe_text) > 200:
            safe_text = safe_text[:200]
        lines.append(f"[{i+1}] {safe_text}")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def call_deepseek(client, prompt):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=128,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"


def evidence_to_memory_id(graph_id, evidence_id):
    parts = evidence_id.split(":")
    if len(parts) == 2 and parts[0].startswith("D"):
        session_num = parts[0][1:]
        return f"{graph_id}_session_{session_num}_{evidence_id}"
    return f"{graph_id}_{evidence_id}"


def main():
    random.seed(42)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Please set DEEPSEEK_API_KEY environment variable")
        return

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    qa_rows = load_csv("results/locomo_qa_records.csv")
    memory_lookup = load_memory_texts("results/locomo_memory_records.csv")

    by_cat = defaultdict(list)
    for r in qa_rows:
        by_cat[str(r["category"])].append(r)

    sampled = []
    for cat in ["1", "2", "3", "4", "5"]:
        pool = by_cat[cat]
        chosen = random.sample(pool, min(20, len(pool)))
        sampled.extend(chosen)

    pilot_qa_path = Path("results/llm_reader_pilot_qa.csv")
    if not pilot_qa_path.exists():
        with pilot_qa_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["qa_id", "category", "question", "answer", "evidence_raw"])
            w.writeheader()
            w.writerows([{
                "qa_id": r["qa_id"], "category": r["category"],
                "question": r["question"], "answer": r.get("answer", ""),
                "evidence_raw": r.get("evidence", ""),
            } for r in sampled])
    print(f"Sampled {len(sampled)} QA (20 per category)")

    method_retrieval = {}
    for mname, (mfile, col_name) in METHODS.items():
        rows = load_csv(mfile)
        for r in rows:
            qid = r["qa_id"]
            raw = r.get(col_name, "")
            ev_list = [x for x in raw.split(";") if x] if raw else []
            method_retrieval.setdefault(qid, {})[mname] = ev_list

    result_path = Path("results/llm_reader_pilot_results.csv")
    results_done = set()
    if result_path.exists():
        existing = load_csv(str(result_path))
        for r in existing:
            results_done.add((r["qa_id"], r["method"]))

    r_fields = ["qa_id", "category", "method", "question", "gold_answer",
                "top10_evidence_ids", "evidence_texts", "predicted_answer",
                "exact_match", "token_f1", "prompt"]
    write_header = not result_path.exists()

    total_calls = 0
    all_results = []

    for qa in sampled:
        qa_id = qa["qa_id"]
        graph_id = qa_id.split("_qa_")[0]
        question = qa["question"]
        gold_answer = qa.get("answer", "")

        for mname in ["BM25", "Dense-ONNX-MiniLM", "BM25-Dense-RRF", "KG-boost"]:
            if (qa_id, mname) in results_done:
                continue

            ev_ids = method_retrieval.get(qa_id, {}).get(mname, [])[:10]
            ev_texts = []
            mid_list = []
            for ev in ev_ids:
                mid = evidence_to_memory_id(graph_id, ev)
                mid_list.append(mid)
                text = memory_lookup.get(mid, ev)
                ev_texts.append(text)

            prompt = build_prompt(question, ev_texts, ev_ids)

            predicted = call_deepseek(client, prompt)
            total_calls += 1

            em = compute_em(predicted, gold_answer)
            f1 = compute_token_f1(predicted, gold_answer)

            row = {
                "qa_id": qa_id, "category": qa["category"], "method": mname,
                "question": question, "gold_answer": gold_answer,
                "top10_evidence_ids": ";".join(ev_ids),
                "evidence_texts": json.dumps(ev_texts, ensure_ascii=False),
                "predicted_answer": predicted,
                "exact_match": em, "token_f1": f"{f1:.4f}",
                "prompt": json.dumps(prompt, ensure_ascii=False),
            }
            all_results.append(row)
            results_done.add((qa_id, mname))

            mode = "a" if Path(result_path).exists() else "w"
            with result_path.open(mode, encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=r_fields)
                if mode == "w":
                    w.writeheader()
                w.writerow(row)

            time.sleep(0.3 + random.random() * 0.2)

            if total_calls % 20 == 0:
                print(f"  {total_calls}/400 calls done...")

    final_rows = load_csv(str(result_path))

    print(f"\n=== LLM Reader Pilot ===")
    print(f"QA sample: {len(sampled)}")
    print(f"Methods: BM25, Dense-ONNX-MiniLM, BM25-Dense-RRF, KG-boost")
    print(f"Total calls: {len(final_rows)}")
    print()

    by_method = defaultdict(list)
    for r in final_rows:
        by_method[r["method"]].append(r)

    print(f"{'Method':20s} {'EM':>8s} {'TokenF1':>8s}")
    print("-" * 38)
    summary_rows = []
    for mname in ["BM25", "Dense-ONNX-MiniLM", "BM25-Dense-RRF", "KG-boost"]:
        grp = by_method[mname]
        n = len(grp)
        em = sum(int(r["exact_match"]) for r in grp) / n if n else 0
        f1 = sum(float(r["token_f1"]) for r in grp) / n if n else 0
        print(f"{mname:20s} {em:8.4f} {f1:8.4f}")
        summary_rows.append({"method": mname, "category": "ALL", "n": n,
                            "exact_match_acc": f"{em:.4f}", "avg_token_f1": f"{f1:.4f}"})

    for mname in ["BM25", "Dense-ONNX-MiniLM", "BM25-Dense-RRF", "KG-boost"]:
        grp = by_method[mname]
        by_cat_grp = defaultdict(list)
        for r in grp:
            by_cat_grp[str(r["category"])].append(r)
        for cat in sorted(by_cat_grp.keys()):
            sub = by_cat_grp[cat]
            n = len(sub)
            em = sum(int(r["exact_match"]) for r in sub) / n if n else 0
            f1 = sum(float(r["token_f1"]) for r in sub) / n if n else 0
            summary_rows.append({"method": mname, "category": f"cat_{cat}", "n": n,
                                "exact_match_acc": f"{em:.4f}", "avg_token_f1": f"{f1:.4f}"})

    print("\nCategory-level KG-boost:")
    kg_grp = by_method["KG-boost"]
    kg_by_cat = defaultdict(list)
    for r in kg_grp:
        kg_by_cat[str(r["category"])].append(r)
    for cat in sorted(kg_by_cat.keys()):
        sub = kg_by_cat[cat]
        n = len(sub)
        em = sum(int(r["exact_match"]) for r in sub) / n if n else 0
        f1 = sum(float(r["token_f1"]) for r in sub) / n if n else 0
        print(f"  cat {cat}: EM={em:.4f} F1={f1:.4f}")

    s_fields = ["method", "category", "n", "exact_match_acc", "avg_token_f1"]
    with Path("results/llm_reader_pilot_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=s_fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nOutput: results/llm_reader_pilot_results.csv")
    print(f"Summary: results/llm_reader_pilot_summary.csv")


if __name__ == "__main__":
    main()
