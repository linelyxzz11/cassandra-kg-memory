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


def load_memory_index(file_path):
    mem_by_id = {}
    mem_by_dia = {}
    with Path(file_path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mid = row["memory_id"]
            mem_by_id[mid] = row
            sid = row["sample_id"]
            di = row["dia_id"]
            mem_by_dia[(sid, di)] = row
    return mem_by_id, mem_by_dia


def get_neighbor_turns(memory_id, mem_by_id, mem_by_dia):
    row = mem_by_id.get(memory_id)
    if not row:
        return None, row, None

    sample_id = row["sample_id"]
    dia_id = row["dia_id"]
    parts = dia_id.split(":")
    if len(parts) != 2 or not parts[0].startswith("D"):
        return None, row, None

    session = parts[0]
    turn_num = int(parts[1])

    prev_dia = f"{session}:{turn_num - 1}" if turn_num > 1 else None
    next_dia = f"{session}:{turn_num + 1}" if turn_num < 999 else None

    prev_row = mem_by_dia.get((sample_id, prev_dia)) if prev_dia else None
    next_row = mem_by_dia.get((sample_id, next_dia)) if next_dia else None

    return prev_row, row, next_row


def format_evidence_block(idx, prev_row, cur_row, next_row):
    lines = []
    lines.append(f"[{idx}]")
    lines.append(f"retrieved_memory_id: {cur_row['memory_id']}")
    lines.append(f"timestamp: {cur_row['timestamp']}")
    lines.append(f"session_id: {cur_row['session_id']}")
    lines.append(f"dia_id: {cur_row['dia_id']}")
    lines.append("")

    if prev_row:
        lines.append(f"Previous turn:")
        lines.append(f"{prev_row['dia_id']} | {prev_row['speaker']} | {prev_row['text']}")
        lines.append("")

    lines.append(f"Retrieved turn:")
    lines.append(f"{cur_row['dia_id']} | {cur_row['speaker']} | {cur_row['text']}")
    lines.append("")

    if next_row:
        lines.append(f"Next turn:")
        lines.append(f"{next_row['dia_id']} | {next_row['speaker']} | {next_row['text']}")
        lines.append("")

    return "\n".join(lines)


def build_prompt(question, evidence_blocks_text):
    lines = [
        "You are answering a memory-based question using retrieved conversation evidence.",
        "",
        "Each evidence item includes timestamp metadata and nearby conversation context. Use the timestamp to resolve relative time expressions such as \"yesterday\", \"last week\", \"last Saturday\", \"next month\", and similar phrases.",
        "",
        "Use only the provided evidence and its metadata. If the evidence is clearly insufficient, answer \"Cannot answer\". Otherwise, give the shortest correct answer.",
        "",
        "Return only the answer. Do not explain.",
        "",
        "Evidence:",
        evidence_blocks_text,
        "",
        f"Question: {question}",
        "Answer:",
    ]
    return "\n".join(lines)


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


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Please set DEEPSEEK_API_KEY environment variable")
        return

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    qa_rows = load_csv("results/locomo_qa_records.csv")
    sampled_map = {r["qa_id"]: r for r in load_csv("results/llm_reader_pilot_qa.csv")}
    sampled = [sampled_map[r["qa_id"]] for r in load_csv("results/llm_reader_pilot_qa.csv") if r["qa_id"] in sampled_map]
    print(f"Loaded {len(sampled)} sampled QA pairs")

    mem_by_id, mem_by_dia = load_memory_index("results/locomo_memory_records.csv")

    method_retrieval = {}
    for mname, (mfile, col_name) in METHODS.items():
        for r in load_csv(mfile):
            qid = r["qa_id"]
            raw = r.get(col_name, "")
            ev_list = [x for x in raw.split(";") if x] if raw else []
            method_retrieval.setdefault(qid, {})[mname] = ev_list

    result_path = Path("results/llm_reader_pilot_v2_results.csv")
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
    empty_gold = 0

    for qa in sampled:
        qa_id = qa["qa_id"]
        graph_id = qa_id.split("_qa_")[0]
        question = qa["question"]
        gold_answer = qa.get("answer", "").strip()
        cat5 = qa.get("category", "") == "5"

        if not gold_answer:
            empty_gold += 1
            if cat5:
                continue

        for mname in ["BM25", "Dense-ONNX-MiniLM", "BM25-Dense-RRF", "KG-boost"]:
            if (qa_id, mname) in results_done:
                continue

            ev_ids = method_retrieval.get(qa_id, {}).get(mname, [])[:10]
            evidence_blocks = []
            ev_texts_list = []

            for ev in ev_ids:
                parts = ev.split(":")
                if len(parts) == 2 and parts[0].startswith("D"):
                    session_num = parts[0][1:]
                    mid = f"{graph_id}_session_{session_num}_{ev}"
                else:
                    mid = f"{graph_id}_{ev}"

                prev_row, cur_row, next_row = get_neighbor_turns(mid, mem_by_id, mem_by_dia)
                if cur_row is None:
                    continue

                block = format_evidence_block(len(evidence_blocks) + 1, prev_row, cur_row, next_row)
                evidence_blocks.append(block)
                ev_texts_list.append(cur_row["text"])

            prompt = build_prompt(question, "\n".join(evidence_blocks))

            predicted = call_deepseek(client, prompt)
            total_calls += 1

            em = compute_em(predicted, gold_answer)
            f1 = compute_token_f1(predicted, gold_answer)

            row = {
                "qa_id": qa_id, "category": qa["category"], "method": mname,
                "question": question, "gold_answer": gold_answer,
                "top10_evidence_ids": ";".join(ev_ids),
                "evidence_texts": json.dumps(ev_texts_list, ensure_ascii=False),
                "predicted_answer": predicted,
                "exact_match": em, "token_f1": f"{f1:.4f}",
                "prompt": json.dumps(prompt, ensure_ascii=False),
            }

            mode = "a" if result_path.exists() else "w"
            with result_path.open(mode, encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=r_fields)
                if mode == "w":
                    w.writeheader()
                w.writerow(row)
            results_done.add((qa_id, mname))

            time.sleep(0.3 + random.random() * 0.2)

            if total_calls % 20 == 0:
                print(f"  {total_calls} calls done...")

    final_rows = load_csv(str(result_path))
    print(f"\n=== LLM Reader v2 Pilot ===")
    print(f"QA sample: {len(sampled)}")
    print(f"Empty gold answers: {empty_gold}")
    print(f"Total calls: {len(final_rows)}")

    all_ca = sum(1 for r in final_rows if r.get("predicted_answer", "").strip().lower() == "cannot answer")
    print(f"Overall CannotAnswer: {all_ca}/{len(final_rows)} = {all_ca/len(final_rows)*100:.1f}%")
    print()

    ga_map = defaultdict(set)
    for r in load_csv("results/locomo_evidence_map.csv"):
        ga_map[r["qa_id"]].add(r.get("evidence_id", ""))

    by_method = defaultdict(list)
    for r in final_rows:
        by_method[r["method"]].append(r)

    print(f"{'Method':20s} {'EM':>7s} {'F1':>7s} {'hit10':>7s} {'ansInEv':>7s} {'CA%':>6s} {'F1_hit':>7s} {'F1_miss':>7s}")
    print("-" * 75)

    summary_rows = []

    for mname in ["BM25", "Dense-ONNX-MiniLM", "BM25-Dense-RRF", "KG-boost"]:
        grp = by_method[mname]
        n = len(grp)
        em = sum(int(r["exact_match"]) for r in grp) / n if n else 0
        f1 = sum(float(r["token_f1"]) for r in grp) / n if n else 0

        ca = sum(1 for r in grp if r.get("predicted_answer", "").strip().lower() == "cannot answer")
        ca_rate = ca / n if n else 0

        ans_in_ev = 0
        hit10_count = 0
        f1_hit_list = []
        f1_miss_list = []
        for r in grp:
            ids = [x for x in r.get("top10_evidence_ids", "").split(";") if x]
            gold = ga_map.get(r["qa_id"], set())
            is_hit = any(g in ids for g in gold)
            if is_hit:
                hit10_count += 1
                f1_hit_list.append(float(r["token_f1"]))
            else:
                f1_miss_list.append(float(r["token_f1"]))

            gold_norm = normalize_answer(r.get("gold_answer", ""))
            try:
                texts = json.loads(r.get("evidence_texts", "[]"))
                text_blob = " ".join(texts).lower()
                if gold_norm and gold_norm in text_blob:
                    ans_in_ev += 1
            except Exception:
                pass

        hit10_rate = hit10_count / n if n else 0
        ans_ev_rate = ans_in_ev / n if n else 0
        f1_hit = sum(f1_hit_list) / len(f1_hit_list) if f1_hit_list else 0
        f1_miss = sum(f1_miss_list) / len(f1_miss_list) if f1_miss_list else 0

        print(f"{mname:20s} {em:7.4f} {f1:7.4f} {hit10_rate:7.4f} {ans_ev_rate:7.4f} {ca_rate:6.4f} {f1_hit:7.4f} {f1_miss:7.4f}")

        summary_rows.append({
            "method": mname, "category": "ALL", "n": n,
            "exact_match_acc": f"{em:.4f}", "avg_token_f1": f"{f1:.4f}",
            "retrieval_hit10": f"{hit10_rate:.4f}",
            "answer_string_in_evidence_rate": f"{ans_ev_rate:.4f}",
            "cannot_answer_rate": f"{ca_rate:.4f}",
            "avg_f1_when_hit10": f"{f1_hit:.4f}",
            "avg_f1_when_miss10": f"{f1_miss:.4f}",
        })

    print()
    print("Category-level KG-boost:")
    kg_grp = by_method["KG-boost"]
    kg_by_cat = defaultdict(list)
    for r in kg_grp:
        kg_by_cat[str(r["category"])].append(r)
    for cat in sorted(kg_by_cat.keys()):
        sub = kg_by_cat[cat]
        n = len(sub)
        em = sum(int(r["exact_match"]) for r in sub) / n if n else 0
        f1 = sum(float(r["token_f1"]) for r in sub) / n if n else 0
        ca = sum(1 for r in sub if r.get("predicted_answer", "").strip().lower() == "cannot answer")
        print(f"  cat {cat}: EM={em:.4f} F1={f1:.4f} CannotAnswer={ca}/{n}")

    for mname in ["BM25", "Dense-ONNX-MiniLM", "BM25-Dense-RRF", "KG-boost"]:
        grp = by_method[mname]
        by_cat = defaultdict(list)
        for r in grp:
            by_cat[str(r["category"])].append(r)
        for cat in sorted(by_cat.keys()):
            sub = by_cat[cat]
            n = len(sub)
            em = sum(int(r["exact_match"]) for r in sub) / n if n else 0
            f1 = sum(float(r["token_f1"]) for r in sub) / n if n else 0
            ca = sum(1 for r in sub if r.get("predicted_answer", "").strip().lower() == "cannot answer")
            ca_rate = ca / n if n else 0
            hit10_count = sum(1 for r in sub if any(g in r.get("top10_evidence_ids", "").split(";") for g in ga_map.get(r["qa_id"], set())))
            hit10_rate = hit10_count / n if n else 0
            ans_ev = 0
            for r in sub:
                gnorm = normalize_answer(r.get("gold_answer", ""))
                try:
                    texts = json.loads(r.get("evidence_texts", "[]"))
                    text_blob = " ".join(texts).lower()
                    if gnorm and gnorm in text_blob:
                        ans_ev += 1
                except Exception:
                    pass
            ans_ev_rate = ans_ev / n if n else 0
            f1_hit = sum(float(r["token_f1"]) for r in sub if any(g in r.get("top10_evidence_ids", "").split(";") for g in ga_map.get(r["qa_id"], set())))
            f1_miss = sum(float(r["token_f1"]) for r in sub if not any(g in r.get("top10_evidence_ids", "").split(";") for g in ga_map.get(r["qa_id"], set())))
            hit_n = hit10_count
            miss_n = n - hit_n
            f1_hit_avg = f1_hit / hit_n if hit_n else 0
            f1_miss_avg = f1_miss / miss_n if miss_n else 0
            summary_rows.append({
                "method": mname, "category": f"cat_{cat}", "n": n,
                "exact_match_acc": f"{em:.4f}", "avg_token_f1": f"{f1:.4f}",
                "retrieval_hit10": f"{hit10_rate:.4f}",
                "answer_string_in_evidence_rate": f"{ans_ev_rate:.4f}",
                "cannot_answer_rate": f"{ca_rate:.4f}",
                "avg_f1_when_hit10": f"{f1_hit_avg:.4f}",
                "avg_f1_when_miss10": f"{f1_miss_avg:.4f}",
            })

    s_fields = ["method", "category", "n", "exact_match_acc", "avg_token_f1",
                "retrieval_hit10", "answer_string_in_evidence_rate",
                "cannot_answer_rate", "avg_f1_when_hit10", "avg_f1_when_miss10"]
    with Path("results/llm_reader_pilot_v2_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=s_fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nOutput: results/llm_reader_pilot_v2_results.csv")
    print(f"Summary: results/llm_reader_pilot_v2_summary.csv")


if __name__ == "__main__":
    main()
