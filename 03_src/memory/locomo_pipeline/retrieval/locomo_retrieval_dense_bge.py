import csv
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

MEM_EMB_FILE = "D:/memorytable/cassandra-kg-memory/results/locomo_memory_bge_large.npy"
QA_EMB_FILE = "D:/memorytable/cassandra-kg-memory/results/locomo_qa_bge_large.npy"
MEM_IDS_FILE = "D:/memorytable/cassandra-kg-memory/results/locomo_memory_ids_bge.txt"
QA_IDS_FILE = "D:/memorytable/cassandra-kg-memory/results/locomo_qa_ids_bge.txt"
QA_CSV = "D:/memorytable/cassandra-kg-memory/results/locomo_qa_records.csv"
EVIDENCE_CSV = "D:/memorytable/cassandra-kg-memory/results/locomo_evidence_map.csv"
MEMORY_CSV = "D:/memorytable/cassandra-kg-memory/results/locomo_memory_records.csv"
OUT_RESULTS = "D:/memorytable/cassandra-kg-memory/results/locomo_dense_bge_results.csv"
OUT_SUMMARY = "D:/memorytable/cassandra-kg-memory/results/locomo_dense_bge_summary.csv"


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_embeddings(emb_path, ids_path):
    arr = np.load(emb_path)
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if len(ids) != arr.shape[0]:
        raise ValueError(f"ID count {len(ids)} != embedding count {arr.shape[0]}")
    return ids, arr


def load_gold_map(evidence_csv, memory_csv, qa_csv):
    mem_to_dia = {}
    for r in load_csv(memory_csv):
        mem_to_dia[r["memory_id"].strip()] = r["dia_id"].strip()

    mresp = defaultdict(set)
    ev_resp = defaultdict(set)
    for r in load_csv(evidence_csv):
        qid = r["qa_id"].strip()
        eid = r["evidence_id"].strip()
        mid = r["memory_id"].strip()
        if eid:
            ev_resp[qid].add(eid)
        if mid:
            mresp[qid].add(mid)

    return dict(mresp), dict(ev_resp)


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine_search(query_emb, mem_embs, mem_ids, top_k=10):
    sims = np.dot(mem_embs, query_emb)
    sorted_idx = np.argsort(-sims)[:top_k]
    results = []
    for idx in sorted_idx:
        results.append((mem_ids[idx], float(sims[idx])))
    return results


def hit(top_ids, gold_ids):
    return int(any(mid in gold_ids for mid in top_ids))


def reciprocal_rank(top_ids, gold_ids):
    for i, mid in enumerate(top_ids, start=1):
        if mid in gold_ids:
            return 1.0 / i
    return 0.0


def mean(vals):
    vals = list(vals)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def main():
    print("Loading embeddings...")
    mem_ids, mem_embs = load_embeddings(MEM_EMB_FILE, MEM_IDS_FILE)
    qa_ids_load, qa_embs = load_embeddings(QA_EMB_FILE, QA_IDS_FILE)
    print(f"  Memories: {mem_embs.shape}")
    print(f"  Queries:   {qa_embs.shape}")

    print("Normalizing embeddings...")
    mem_embs = mem_embs / np.linalg.norm(mem_embs, axis=1, keepdims=True)
    qa_embs = qa_embs / np.linalg.norm(qa_embs, axis=1, keepdims=True)

    qa_id_to_idx = {qid: i for i, qid in enumerate(qa_ids_load)}
    mem_id_to_idx = {mid: i for i, mid in enumerate(mem_ids)}

    print("Loading gold evidence map...")
    gold_mem_map, gold_ev_map = load_gold_map(EVIDENCE_CSV, MEMORY_CSV, QA_CSV)

    print("Loading QA records...")
    qa_rows = load_csv(QA_CSV)
    print(f"  {len(qa_rows)} queries")

    print("\nRunning dense retrieval (cosine similarity)...")
    result_rows = []

    for r in qa_rows:
        qid = r["qa_id"].strip()
        if qid not in qa_id_to_idx:
            continue

        q_idx = qa_id_to_idx[qid]
        q_emb = qa_embs[q_idx]

        raw_results = cosine_search(q_emb, mem_embs, mem_ids, top_k=10)
        retrieved_ids = [mid for mid, _ in raw_results]

        gold_ids = gold_mem_map.get(qid, set())

        row = {
            "qa_id": qid,
            "category": r["category"].strip(),
            "question": r["question"].strip(),
            "gold_memory_ids": ";".join(sorted(gold_ids)),
            "retrieved_memory_ids": ";".join(retrieved_ids),
            "dense_bge_top10_memory_ids": ";".join(retrieved_ids),
            "hit1": hit(retrieved_ids[:1], gold_ids),
            "hit5": hit(retrieved_ids[:5], gold_ids),
            "hit10": hit(retrieved_ids, gold_ids),
            "rr": f"{reciprocal_rank(retrieved_ids, gold_ids):.6f}",
        }
        result_rows.append(row)

    result_fields = [
        "qa_id", "category", "question", "gold_memory_ids",
        "retrieved_memory_ids", "dense_bge_top10_memory_ids",
        "hit1", "hit5", "hit10", "rr",
    ]
    write_csv(OUT_RESULTS, result_rows, result_fields)
    print(f"\nResults saved to {OUT_RESULTS}")

    R1 = mean(int(r["hit1"]) for r in result_rows)
    R5 = mean(int(r["hit5"]) for r in result_rows)
    R10 = mean(int(r["hit10"]) for r in result_rows)
    MRR = mean(float(r["rr"]) for r in result_rows)

    summary_rows = [{
        "method": "Dense-bge-large",
        "model": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "n_queries": len(result_rows),
        "n_memories": len(mem_ids),
        "recall_1": f"{R1:.4f}",
        "recall_5": f"{R5:.4f}",
        "recall_10": f"{R10:.4f}",
        "mrr_10": f"{MRR:.4f}",
    }]

    by_cat = defaultdict(list)
    for r in result_rows:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        rows = by_cat[cat]
        summary_rows.append({
            "method": "Dense-bge-large",
            "model": f"cat_{cat}",
            "dim": 1024,
            "n_queries": len(rows),
            "n_memories": len(mem_ids),
            "recall_1": f"{mean(int(r['hit1']) for r in rows):.4f}",
            "recall_5": f"{mean(int(r['hit5']) for r in rows):.4f}",
            "recall_10": f"{mean(int(r['hit10']) for r in rows):.4f}",
            "mrr_10": f"{mean(float(r['rr']) for r in rows):.4f}",
        })

    summary_fields = ["method", "model", "dim", "n_queries", "n_memories",
                      "recall_1", "recall_5", "recall_10", "mrr_10"]
    write_csv(OUT_SUMMARY, summary_rows, summary_fields)
    print(f"Summary saved to {OUT_SUMMARY}")

    print(f"\n{'='*60}")
    print(f"Dense-bge-large (BAAI/bge-large-en-v1.5, 1024-dim)")
    print(f"{'='*60}")
    print(f"  R@1  = {R1:.4f}")
    print(f"  R@5  = {R5:.4f}")
    print(f"  R@10 = {R10:.4f}")
    print(f"  MRR  = {MRR:.4f}")
    print(f"{'='*60}")

    print(f"\nPer-category:")
    for r in summary_rows[1:]:
        print(f"  {r['model']:8s}  R@1={r['recall_1']}  R@5={r['recall_5']}  R@10={r['recall_10']}  MRR={r['mrr_10']}")


if __name__ == "__main__":
    main()
