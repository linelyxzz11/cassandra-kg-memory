import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tokenizers as tk


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


def load_model(model_dir):
    tok_path = Path(model_dir) / "tokenizer.json"
    onnx_path = Path(model_dir) / "onnx" / "model.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tokenizer = tk.Tokenizer.from_file(str(tok_path))
    dim = sess.get_outputs()[0].shape[2]
    return sess, tokenizer, dim


def tokenize(tokenizer, texts, max_seq_len=128):
    encodings = tokenizer.encode_batch(texts)
    batch_size = len(texts)
    input_ids = np.zeros((batch_size, max_seq_len), dtype=np.int64)
    attention_mask = np.zeros((batch_size, max_seq_len), dtype=np.int64)
    token_type_ids = np.zeros((batch_size, max_seq_len), dtype=np.int64)
    for i, e in enumerate(encodings):
        end = min(len(e.ids), max_seq_len)
        input_ids[i, :end] = e.ids[:end]
        attention_mask[i, :end] = e.attention_mask[:end]
        token_type_ids[i, :end] = e.type_ids[:end]
    return input_ids, attention_mask, token_type_ids


def encode_batches(session, tokenizer, texts, batch_size, max_seq_len=128):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        input_ids, attention_mask, token_type_ids = tokenize(tokenizer, batch, max_seq_len)
        out = session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids},
        )
        emb = out[0]
        am3d = attention_mask[:, :emb.shape[1], None]
        pooled = (emb * am3d).sum(axis=1) / am3d.sum(axis=1).clip(1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(1e-9)
        pooled = pooled / norms
        all_vecs.append(pooled)
        done = min(i + batch_size, len(texts))
        if done % 500 == 0 or done == len(texts):
            print(f"    {done}/{len(texts)} encoded")
    return np.vstack(all_vecs)


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
        description="Dense retrieval baseline - pure ONNX Runtime."
    )
    parser.add_argument("--qa", default="results/locomo_qa_records.csv")
    parser.add_argument("--evidence", default="results/locomo_evidence_map.csv")
    parser.add_argument("--memory", default="results/locomo_memory_records.csv")
    parser.add_argument("--output", default="results/locomo_dense_onnx_results.csv")
    parser.add_argument("--model-dir", default="external_data/models/all-MiniLM-L6-v2-onnx")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    gold_map = load_gold_evidence(args.evidence)
    qa_rows = load_qa(args.qa)
    all_texts, all_evidences = build_memory_index(args.memory)
    questions = [qa["question"] for qa in qa_rows]
    print(f"Loaded {len(all_texts)} memory turns, {len(qa_rows)} QA pairs")

    session, tokenizer, dim = load_model(args.model_dir)
    print(f"Model dim: {dim}")

    print(f"Encoding {len(all_texts)} memory texts...")
    mem_emb = encode_batches(session, tokenizer, all_texts, args.batch_size)

    print(f"Encoding {len(questions)} questions...")
    q_emb = encode_batches(session, tokenizer, questions, args.batch_size)

    print("Retrieving...")
    scores = q_emb @ mem_emb.T
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
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n=== Dense Baseline (ONNX) ===")
    print(f"Model: {args.model_dir}")
    print(f"Backend: ONNX Runtime")
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
