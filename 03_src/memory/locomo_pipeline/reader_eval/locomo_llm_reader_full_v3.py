import argparse
import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from locomo_llm_reader_pilot_v3_plus import (
    load_csv,
    write_csv,
    append_csv,
    make_ascii_safe,
    normalize_strict,
    normalize_relaxed,
    compute_em,
    compute_token_f1,
    split_ids,
    unique_keep_order,
    load_memory_index,
    load_qa_answer_map,
    load_gold_map,
    load_locomo_context,
    build_evidence_for_ids,
    build_prompt,
    answer_in_evidence,
    call_deepseek,
)

METHOD_SPECS = {
    "BM25": {
        "file": "results/locomo_bm25_results.csv",
        "columns": ["retrieved_memory_ids", "top10_memory_ids", "top10_evidence_ids"],
    },
    "Dense-ONNX-MiniLM": {
        "file": "results/locomo_dense_onnx_results.csv",
        "columns": ["retrieved_memory_ids", "top10_memory_ids", "top10_evidence_ids"],
    },
    "BM25-Dense-RRF": {
        "file": "results/locomo_fusion_bm25_dense_results.csv",
        "columns": ["fusion_top10_memory_ids", "retrieved_memory_ids", "top10_memory_ids"],
    },
    "KG-boost": {
        "file": "results/locomo_cassandra_kg_results.csv",
        "columns": ["retrieved_memory_ids", "top10_memory_ids", "top10_evidence_ids"],
    },
    "KG-aware-RRF": {
        "file": "results/locomo_kg_aware_fusion_best_results.csv",
        "columns": ["retrieved_memory_ids", "kg_aware_top10_memory_ids"],
    },
}

METHOD_ORDER = [
    "BM25",
    "Dense-ONNX-MiniLM",
    "BM25-Dense-RRF",
    "KG-boost",
    "KG-aware-RRF",
]

CORE_METHODS = [
    "BM25-Dense-RRF",
    "KG-boost",
    "KG-aware-RRF",
]

RESULT_FIELDS = [
    "qa_id",
    "category",
    "method",
    "question",
    "gold_answer",
    "top10_evidence_ids",
    "resolved_memory_ids",
    "evidence_texts",
    "predicted_answer",
    "strict_em",
    "strict_f1",
    "relaxed_em",
    "relaxed_f1",
    "retrieval_hit10",
    "answer_string_in_evidence",
    "prompt",
]

SUMMARY_FIELDS = [
    "method",
    "category",
    "n",
    "strict_em",
    "strict_f1",
    "relaxed_em",
    "relaxed_f1",
    "retrieval_hit10",
    "answer_string_in_evidence_rate",
    "cannot_answer_rate",
    "relaxed_f1_when_hit10",
    "relaxed_f1_when_miss10",
]

REGISTRY_FIELDS = [
    "method",
    "retrieval_file",
    "ranking_column",
    "reader_context",
    "fixed_status",
]

def pick_column(rows, candidates, method, file_path):
    if not rows:
        raise ValueError(f"Empty retrieval file for {method}: {file_path}")
    cols = list(rows[0].keys())
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"No ranking column found for {method} in {file_path}. Available columns: {cols}")

def load_rankings(methods):
    rankings = {}
    used_cols = {}
    for method in methods:
        spec = METHOD_SPECS[method]
        rows = load_csv(spec["file"])
        col = pick_column(rows, spec["columns"], method, spec["file"])
        used_cols[method] = col
        method_map = {}
        for row in rows:
            qid = str(row.get("qa_id", "")).strip()
            if not qid:
                continue
            method_map[qid] = unique_keep_order(split_ids(row.get(col, "")))[:10]
        rankings[method] = method_map
        print(f"{method}: loaded {len(method_map)} rankings from {spec['file']} using column {col}")
    return rankings, used_cols

def load_full_qa(path, answer_map):
    rows = []
    for row in load_csv(path):
        qid = str(row.get("qa_id", "")).strip()
        if not qid:
            continue
        answer = str(row.get("answer", "")).strip()
        adversarial = str(row.get("adversarial_answer", "")).strip()
        if not answer:
            answer = adversarial
        if not answer:
            answer = answer_map.get(qid, "")
        rows.append({
            "qa_id": qid,
            "category": str(row.get("category", "")).strip(),
            "question": str(row.get("question", "")).strip(),
            "answer": answer,
        })
    return rows

def is_retrieval_hit(qa_id, raw_ids, resolved_ids, gold_map):
    gold = gold_map.get(qa_id, set())
    if not gold:
        return 0
    return 1 if (set(raw_ids) | set(resolved_ids)) & gold else 0

def select_methods(value):
    v = str(value).strip()
    if v == "all":
        return METHOD_ORDER
    if v == "core":
        return CORE_METHODS
    methods = [x.strip() for x in v.split(",") if x.strip()]
    unknown = [m for m in methods if m not in METHOD_SPECS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return methods

def summarize_group(method, category, rows):
    n = len(rows)
    if n == 0:
        return {
            "method": method,
            "category": category,
            "n": 0,
            "strict_em": "0.0000",
            "strict_f1": "0.0000",
            "relaxed_em": "0.0000",
            "relaxed_f1": "0.0000",
            "retrieval_hit10": "0.0000",
            "answer_string_in_evidence_rate": "0.0000",
            "cannot_answer_rate": "0.0000",
            "relaxed_f1_when_hit10": "0.0000",
            "relaxed_f1_when_miss10": "0.0000",
        }

    strict_em = sum(int(r.get("strict_em", 0)) for r in rows) / n
    strict_f1 = sum(float(r.get("strict_f1", 0)) for r in rows) / n
    relaxed_em = sum(int(r.get("relaxed_em", 0)) for r in rows) / n
    relaxed_f1 = sum(float(r.get("relaxed_f1", 0)) for r in rows) / n
    hit10 = sum(int(r.get("retrieval_hit10", 0)) for r in rows) / n
    ans_ev = sum(int(r.get("answer_string_in_evidence", 0)) for r in rows) / n
    ca = sum(1 for r in rows if normalize_relaxed(r.get("predicted_answer", "")) == "cannot answer") / n

    f1_hit = [float(r.get("relaxed_f1", 0)) for r in rows if int(r.get("retrieval_hit10", 0)) == 1]
    f1_miss = [float(r.get("relaxed_f1", 0)) for r in rows if int(r.get("retrieval_hit10", 0)) == 0]

    return {
        "method": method,
        "category": category,
        "n": n,
        "strict_em": f"{strict_em:.4f}",
        "strict_f1": f"{strict_f1:.4f}",
        "relaxed_em": f"{relaxed_em:.4f}",
        "relaxed_f1": f"{relaxed_f1:.4f}",
        "retrieval_hit10": f"{hit10:.4f}",
        "answer_string_in_evidence_rate": f"{ans_ev:.4f}",
        "cannot_answer_rate": f"{ca:.4f}",
        "relaxed_f1_when_hit10": f"{sum(f1_hit) / len(f1_hit) if f1_hit else 0:.4f}",
        "relaxed_f1_when_miss10": f"{sum(f1_miss) / len(f1_miss) if f1_miss else 0:.4f}",
    }

def write_summary(rows, methods, out_summary):
    by_method = defaultdict(list)
    for row in rows:
        if row.get("method") in methods:
            by_method[row["method"]].append(row)

    summary = []

    print("")
    print("=== Full LLM Reader v3 Summary ===")
    print(f"Rows used in summary: {sum(len(by_method[m]) for m in methods)}")
    print("")
    print(f"{'Method':20s} {'rEM':>7s} {'rF1':>7s} {'hit10':>7s} {'ansInEv':>7s} {'CA%':>6s} {'F1_hit':>7s} {'F1_miss':>7s}")
    print("-" * 75)

    for method in methods:
        item = summarize_group(method, "ALL", by_method.get(method, []))
        summary.append(item)
        print(
            f"{method:20s} "
            f"{float(item['relaxed_em']):7.4f} "
            f"{float(item['relaxed_f1']):7.4f} "
            f"{float(item['retrieval_hit10']):7.4f} "
            f"{float(item['answer_string_in_evidence_rate']):7.4f} "
            f"{float(item['cannot_answer_rate']):6.4f} "
            f"{float(item['relaxed_f1_when_hit10']):7.4f} "
            f"{float(item['relaxed_f1_when_miss10']):7.4f}"
        )

    for method in methods:
        by_cat = defaultdict(list)
        for row in by_method.get(method, []):
            by_cat[str(row.get("category", ""))].append(row)
        for cat in sorted(by_cat.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
            summary.append(summarize_group(method, f"cat_{cat}", by_cat[cat]))

    write_csv(out_summary, summary, SUMMARY_FIELDS)

def write_registry(methods, used_cols, out_path):
    rows = []
    for method in methods:
        spec = METHOD_SPECS[method]
        rows.append({
            "method": method,
            "retrieval_file": spec["file"],
            "ranking_column": used_cols.get(method, ""),
            "reader_context": "v3: timestamp + neighbor turns + session observations + session summary",
            "fixed_status": "fixed final reader method" if method == "KG-aware-RRF" else "baseline",
        })
    write_csv(out_path, rows, REGISTRY_FIELDS)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="all")
    parser.add_argument("--out-results", default="results/llm_reader_full_v3_results.csv")
    parser.add_argument("--out-summary", default="results/llm_reader_full_v3_summary.csv")
    parser.add_argument("--registry", default="results/locomo_final_method_registry.csv")
    parser.add_argument("--qa-csv", default="results/locomo_qa_records.csv")
    parser.add_argument("--sleep-min", type=float, default=0.3)
    parser.add_argument("--sleep-max", type=float, default=0.5)
    parser.add_argument("--progress-interval", type=int, default=50)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY environment variable")

    methods = select_methods(args.methods)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    answer_map = load_qa_answer_map("results/locomo_qa_records.csv")
    qa_rows = load_full_qa(args.qa_csv, answer_map)
    mem_by_id, mem_by_dia = load_memory_index("results/locomo_memory_records.csv")
    obs_lookup, summary_lookup = load_locomo_context("external_data/locomo10.json")
    gold_map = load_gold_map("results/locomo_evidence_map.csv")
    rankings, used_cols = load_rankings(methods)

    write_registry(methods, used_cols, args.registry)

    result_path = Path(args.out_results)
    done = set()
    if result_path.exists():
        existing = load_csv(args.out_results)
        bad = [r for r in existing if str(r.get("predicted_answer", "")).startswith("ERROR:")]
        if bad:
            raise RuntimeError(f"{args.out_results} contains {len(bad)} ERROR rows")
        for row in existing:
            qid = str(row.get("qa_id", "")).strip()
            method = str(row.get("method", "")).strip()
            if qid and method:
                done.add((qid, method))

    planned = len(qa_rows) * len(methods)
    remaining = planned - sum(1 for qa in qa_rows for m in methods if (qa["qa_id"], m) in done)

    print("")
    print("=== Full LLM Reader v3 Run ===")
    print(f"QA count: {len(qa_rows)}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Planned rows: {planned}")
    print(f"Already done: {planned - remaining}")
    print(f"Remaining calls: {remaining}")
    print(f"Output: {args.out_results}")
    print(f"Summary: {args.out_summary}")
    print("")

    calls = 0

    for qa in qa_rows:
        qa_id = qa["qa_id"]
        category = qa["category"]
        question = qa["question"]
        gold_answer = qa["answer"]

        for method in methods:
            if (qa_id, method) in done:
                continue

            raw_ids = rankings.get(method, {}).get(qa_id, [])[:10]
            resolved_ids, evidence_texts, evidence_blocks = build_evidence_for_ids(
                qa_id,
                raw_ids,
                mem_by_id,
                mem_by_dia,
                obs_lookup,
                summary_lookup,
            )

            prompt = build_prompt(question, evidence_blocks)
            prompt = make_ascii_safe(prompt)

            predicted = call_deepseek(client, prompt).strip()

            strict_em = compute_em(predicted, gold_answer, normalize_strict)
            strict_f1 = compute_token_f1(predicted, gold_answer, normalize_strict)
            relaxed_em = compute_em(predicted, gold_answer, normalize_relaxed)
            relaxed_f1 = compute_token_f1(predicted, gold_answer, normalize_relaxed)

            evidence_blob = "\n".join(evidence_texts)
            retrieval_hit10 = is_retrieval_hit(qa_id, raw_ids, resolved_ids, gold_map)
            ans_in_ev = answer_in_evidence(gold_answer, evidence_blob)

            row = {
                "qa_id": qa_id,
                "category": category,
                "method": method,
                "question": question,
                "gold_answer": gold_answer,
                "top10_evidence_ids": ";".join(raw_ids),
                "resolved_memory_ids": ";".join(resolved_ids),
                "evidence_texts": json.dumps(evidence_texts, ensure_ascii=False),
                "predicted_answer": predicted,
                "strict_em": strict_em,
                "strict_f1": f"{strict_f1:.4f}",
                "relaxed_em": relaxed_em,
                "relaxed_f1": f"{relaxed_f1:.4f}",
                "retrieval_hit10": retrieval_hit10,
                "answer_string_in_evidence": ans_in_ev,
                "prompt": json.dumps(prompt, ensure_ascii=False),
            }

            append_csv(args.out_results, row, RESULT_FIELDS)
            done.add((qa_id, method))
            calls += 1

            if calls % args.progress_interval == 0:
                print(f"  {calls}/{remaining} new calls done...")

            time.sleep(args.sleep_min + random.random() * max(0.0, args.sleep_max - args.sleep_min))

    final_rows = load_csv(args.out_results)
    bad = [r for r in final_rows if str(r.get("predicted_answer", "")).startswith("ERROR:")]
    if bad:
        raise RuntimeError(f"Found {len(bad)} ERROR rows in {args.out_results}")

    write_summary(final_rows, methods, args.out_summary)

    print("")
    print(f"New calls: {calls}")
    print(f"Results: {args.out_results}")
    print(f"Summary: {args.out_summary}")
    print(f"Registry: {args.registry}")

if __name__ == "__main__":
    main()