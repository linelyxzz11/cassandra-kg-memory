import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


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


def load_onnx_model(model_name, cache_dir):
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    model_path = Path(cache_dir) / model_name.replace("/", "_")
    onnx_path = model_path / "model.onnx"

    if onnx_path.exists():
        print(f"  Loading cached ONNX model from {onnx_path}")
        model = ORTModelForFeatureExtraction.from_pretrained(
            str(model_path), file_name="model.onnx"
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    else:
        print(f"  Downloading and exporting {model_name} to ONNX...")
        os.makedirs(str(model_path), exist_ok=True)
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_name, export=True, cache_dir=cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model.save_pretrained(str(model_path))
        tokenizer.save_pretrained(str(model_path))
        print(f"  ONNX model saved to {onnx_path}")

    return model, tokenizer


def encode_batch(model, tokenizer, texts, batch_size):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=128,
            return_tensors="np",
        )
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        pooled = np.sum(token_embeddings * input_mask_expanded, axis=1)
        pooled = pooled / np.clip(np.sum(attention_mask, axis=1, keepdims=True), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, 1e-9, None)
        all_embeddings.append(pooled)
        if (i + len(batch)) % 500 == 0 or i + len(batch) >= len(texts):
            print(f"    {min(i + len(batch), len(texts))}/{len(texts)} encoded")
    return np.vstack(all_embeddings)


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
        description="Dense vector retrieval baseline on LoCoMo (ONNX runtime)."
    )
    parser.add_argument("--qa", default="results/locomo_qa_records.csv")
    parser.add_argument("--evidence", default="results/locomo_evidence_map.csv")
    parser.add_argument("--memory", default="results/locomo_memory_records.csv")
    parser.add_argument("--output", default="results/locomo_dense_results.csv")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-dir", default="models_cache")
    args = parser.parse_args()

    gold_map = load_gold_evidence(args.evidence)
    qa_rows = load_qa(args.qa)
    all_texts, all_evidences = build_memory_index(args.memory)
    questions = [qa["question"] for qa in qa_rows]
    print(f"Loaded {len(all_texts)} memory turns, {len(qa_rows)} QA pairs")

    print(f"Loading ONNX model: {args.model}...")
    model, tokenizer = load_onnx_model(args.model, args.cache_dir)

    print(f"Encoding {len(all_texts)} memory texts...")
    mem_embeddings = encode_batch(model, tokenizer, all_texts, args.batch_size)

    print(f"Encoding {len(questions)} questions...")
    q_embeddings = encode_batch(model, tokenizer, questions, args.batch_size)

    print("Computing cosine similarity...")
    scores = q_embeddings @ mem_embeddings.T
    rankings = np.argsort(-scores, axis=1)

    results = []
    for idx, qa in enumerate(qa_rows):
        qa_id = qa["qa_id"]
        question = qa["question"]
        category = qa.get("category", "")
        gold_ids = gold_map.get(qa_id, set())

        top_indices = rankings[idx][:args.top_k].tolist()
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

    print(f"\n=== Dense Baseline (ONNX) ===")
    print(f"Model: {args.model}")
    print(f"Backend: ONNX Runtime (no torch)")
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
