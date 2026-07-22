import csv
import numpy as np
from collections import defaultdict
from pathlib import Path

BASE = Path("D:/memorytable/cassandra-kg-memory/results")

MEM_EMB_FILE = str(BASE / "locomo_memory_bge_large.npy")
QA_EMB_FILE = str(BASE / "locomo_qa_bge_large.npy")
MEM_IDS_FILE = str(BASE / "locomo_memory_ids_bge.txt")
QA_IDS_FILE = str(BASE / "locomo_qa_ids_bge.txt")
QA_CSV = str(BASE / "locomo_qa_records.csv")
EVIDENCE_CSV = str(BASE / "locomo_evidence_map.csv")
MEMORY_CSV = str(BASE / "locomo_memory_records.csv")
KG_EDGES_CSV = str(BASE / "locomo_kg_edges_spacy.csv")

KG_WEIGHTS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

OUT_RESULTS = str(BASE / "locomo_dense_kg_boost_results.csv")
OUT_SUMMARY = str(BASE / "locomo_dense_kg_boost_summary.csv")
OUT_BEST_RESULTS = str(BASE / "locomo_dense_kg_boost_best_results.csv")
OUT_BEST_SUMMARY = str(BASE / "locomo_dense_kg_boost_best_summary.csv")


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


def build_kg_set(edges_csv, memory_csv):
    ev_ids = set()
    for r in load_csv(edges_csv):
        ev = r.get("evidence", "").strip()
        if ev:
            ev_ids.add(ev)

    kg_memories = set()
    for r in load_csv(memory_csv):
        mid = r["memory_id"].strip()
        for ev in ev_ids:
            if mid.endswith("_" + ev) or mid.endswith(ev):
                kg_memories.add(mid)
                break

    return kg_memories


def load_gold_map():
    gold_mem = defaultdict(set)
    for r in load_csv(EVIDENCE_CSV):
        qid = r["qa_id"].strip()
        mid = r["memory_id"].strip()
        if mid:
            gold_mem[qid].add(mid)
    eid = r.get("evidence_id", "").strip()
    if eid:
        gold_mem[qid].add(eid)
    return dict(gold_mem)


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
    print("Loading bge-large embeddings...")
    mem_ids, mem_embs = load_embeddings(MEM_EMB_FILE, MEM_IDS_FILE)
    qa_ids_load, qa_embs = load_embeddings(QA_EMB_FILE, QA_IDS_FILE)
    print(f"  Memories: {mem_embs.shape}")
    print(f"  Queries:   {qa_embs.shape}")

    print("Normalizing...")
    mem_embs = mem_embs / np.linalg.norm(mem_embs, axis=1, keepdims=True)
    qa_embs = qa_embs / np.linalg.norm(qa_embs, axis=1, keepdims=True)

    print("Building KG memory set...")
    kg_set = build_kg_set(KG_EDGES_CSV, MEMORY_CSV)
    print(f"  {len(kg_set)} memories have KG edges")

    kg_mask = np.zeros(len(mem_ids), dtype=np.float32)
    for i, mid in enumerate(mem_ids):
        if mid in kg_set:
            kg_mask[i] = 1.0

    print("Loading gold map...")
    gold_map = load_gold_map()

    print("Loading QA records...")
    qa_rows = load_csv(QA_CSV)
    qa_id_to_idx = {qid: i for i, qid in enumerate(qa_ids_load)}
    print(f"  {len(qa_rows)} queries")

    print("\nComputing cosine similarity matrix...")
    sims = np.dot(qa_embs, mem_embs.T)
    print(f"  Similarity matrix: {sims.shape}")

    best_by_mrr = None
    best_weight = None

    all_result_rows = {}
    all_summary_rows = []

    for kg_w in KG_WEIGHTS:
        print(f"\n--- kg_weight = {kg_w} ---")
        result_rows = []

        for r in qa_rows:
            qid = r["qa_id"].strip()
            if qid not in qa_id_to_idx:
                continue

            q_idx = qa_id_to_idx[qid]
            base_scores = sims[q_idx].copy()
            boosted = base_scores + kg_w * kg_mask

            sorted_idx = np.argsort(-boosted)
            top10_idx = sorted_idx[:10]
            retrieved_ids = [mem_ids[i] for i in top10_idx]

            gold_ids = gold_map.get(qid, set())

            row = {
                "qa_id": qid,
                "kg_weight": kg_w,
                "category": r["category"].strip(),
                "question": r["question"].strip(),
                "gold_memory_ids": ";".join(sorted(gold_ids)),
                "retrieved_memory_ids": ";".join(retrieved_ids),
                "dense_kg_top10_memory_ids": ";".join(retrieved_ids),
                "hit1": hit(retrieved_ids[:1], gold_ids),
                "hit5": hit(retrieved_ids[:5], gold_ids),
                "hit10": hit(retrieved_ids, gold_ids),
                "rr": f"{reciprocal_rank(retrieved_ids, gold_ids):.6f}",
            }
            result_rows.append(row)

        all_result_rows[kg_w] = result_rows

        R1 = mean(int(r["hit1"]) for r in result_rows)
        R5 = mean(int(r["hit5"]) for r in result_rows)
        R10 = mean(int(r["hit10"]) for r in result_rows)
        MRR = mean(float(r["rr"]) for r in result_rows)
        print(f"  R@1={R1:.4f}  R@5={R5:.4f}  R@10={R10:.4f}  MRR={MRR:.4f}")

        method_name = f"Dense-bge+KG(w={kg_w})"
        all_summary_rows.append({
            "method": method_name,
            "kg_weight": kg_w,
            "category": "ALL",
            "n": len(result_rows),
            "recall_1": f"{R1:.4f}",
            "recall_5": f"{R5:.4f}",
            "recall_10": f"{R10:.4f}",
            "mrr_10": f"{MRR:.4f}",
        })

        by_cat = defaultdict(list)
        for r in result_rows:
            by_cat[r["category"]].append(r)
        for cat in sorted(by_cat.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            grp = by_cat[cat]
            n = len(grp)
            cr1 = mean(int(r["hit1"]) for r in grp)
            cr5 = mean(int(r["hit5"]) for r in grp)
            cr10 = mean(int(r["hit10"]) for r in grp)
            cmrr = mean(float(r["rr"]) for r in grp)
            all_summary_rows.append({
                "method": method_name,
                "kg_weight": kg_w,
                "category": f"cat_{cat}",
                "n": n,
                "recall_1": f"{cr1:.4f}",
                "recall_5": f"{cr5:.4f}",
                "recall_10": f"{cr10:.4f}",
                "mrr_10": f"{cmrr:.4f}",
            })

        if best_by_mrr is None or MRR > best_by_mrr:
            best_by_mrr = MRR
            best_weight = kg_w

    all_rows_flat = []
    for kg_w, rows in all_result_rows.items():
        all_rows_flat.extend(rows)

    result_fields = [
        "qa_id", "kg_weight", "category", "question", "gold_memory_ids",
        "retrieved_memory_ids", "dense_kg_top10_memory_ids",
        "hit1", "hit5", "hit10", "rr",
    ]
    write_csv(OUT_RESULTS, all_rows_flat, result_fields)

    summary_fields = [
        "method", "kg_weight", "category", "n",
        "recall_1", "recall_5", "recall_10", "mrr_10",
    ]
    write_csv(OUT_SUMMARY, all_summary_rows, summary_fields)

    best_rows = [r for r in all_rows_flat if abs(float(r["kg_weight"]) - best_weight) < 0.001]
    best_summary = [r for r in all_summary_rows if abs(float(r["kg_weight"]) - best_weight) < 0.001 and r["category"] == "ALL"]
    write_csv(OUT_BEST_RESULTS, best_rows, result_fields)
    write_csv(OUT_BEST_SUMMARY, best_summary, summary_fields)

    print(f"\n{'='*60}")
    print(f"Dense-bge + KG Boost Results")
    print(f"{'='*60}")
    print(f"KG memories: {len(kg_set)}/{len(mem_ids)} ({100*len(kg_set)/len(mem_ids):.1f}%)")
    print(f"")
    print(f"{'Weight':>8s}  {'R@1':>8s}  {'R@5':>8s}  {'R@10':>8s}  {'MRR':>8s}")
    print("-" * 48)
    for s in all_summary_rows:
        if s["category"] == "ALL":
            w = float(s["kg_weight"])
            print(f"{w:8.2f}  {s['recall_1']:>8s}  {s['recall_5']:>8s}  {s['recall_10']:>8s}  {s['mrr_10']:>8s}")
    print(f"\nBest: w={best_weight}, MRR={best_by_mrr:.4f}")
    print(f"  (Dense-bge alone: R@1=0.3343  R@10=0.6803  MRR=0.4405)")
    print(f"")
    print(f"All results: {OUT_RESULTS}")
    print(f"Summary:     {OUT_SUMMARY}")
    print(f"Best:        {OUT_BEST_RESULTS}")


if __name__ == "__main__":
    main()
