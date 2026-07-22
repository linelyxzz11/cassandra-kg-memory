import csv
import json
import re
from collections import defaultdict
from pathlib import Path

CONFIGS = [
    {
        "config_name": "KG-aware-RRF-A",
        "rrf_k": 60,
        "w_bm25": 1.0,
        "w_dense": 0.5,
        "w_fusion": 1.0,
        "w_kg": 0.25,
    },
    {
        "config_name": "KG-aware-RRF-B",
        "rrf_k": 60,
        "w_bm25": 1.0,
        "w_dense": 0.5,
        "w_fusion": 1.0,
        "w_kg": 0.5,
    },
    {
        "config_name": "KG-aware-RRF-C",
        "rrf_k": 60,
        "w_bm25": 1.0,
        "w_dense": 0.5,
        "w_fusion": 1.0,
        "w_kg": 1.0,
    },
]

INPUTS = {
    "bm25": {
        "path": "results/locomo_bm25_results.csv",
        "columns": ["retrieved_memory_ids", "top10_memory_ids", "top10_evidence_ids", "bm25_top10_memory_ids"],
    },
    "dense": {
        "path": "results/locomo_dense_onnx_results.csv",
        "columns": ["retrieved_memory_ids", "top10_memory_ids", "top10_evidence_ids", "dense_top10_memory_ids"],
    },
    "fusion": {
        "path": "results/locomo_fusion_bm25_dense_results.csv",
        "columns": ["fusion_top10_memory_ids", "retrieved_memory_ids", "top10_memory_ids", "top10_evidence_ids"],
    },
    "kg": {
        "path": "results/locomo_cassandra_kg_results.csv",
        "columns": ["retrieved_memory_ids", "top10_memory_ids", "top10_evidence_ids", "kg_top10_memory_ids"],
    },
}

QA_CSV = "results/locomo_qa_records.csv"
EVIDENCE_CSV = "results/locomo_evidence_map.csv"

OUT_RESULTS = "results/locomo_kg_aware_fusion_results.csv"
OUT_SUMMARY = "results/locomo_kg_aware_fusion_summary.csv"
OUT_BEST_RESULTS = "results/locomo_kg_aware_fusion_best_results.csv"
OUT_BEST_SUMMARY = "results/locomo_kg_aware_fusion_best_summary.csv"

def load_csv(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def split_ids(raw):
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            vals = json.loads(raw)
            return [str(x).strip() for x in vals if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in re.split(r"[;,\|]", raw) if x.strip()]

def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def pick_column(fieldnames, candidates, label, path):
    for c in candidates:
        if c in fieldnames:
            return c
    raise ValueError(f"No top10 column found for {label} in {path}. Available columns: {fieldnames}")

def load_ranking_file(label, spec):
    rows = load_csv(spec["path"])
    if not rows:
        raise ValueError(f"Empty file: {spec['path']}")
    fieldnames = list(rows[0].keys())
    col = pick_column(fieldnames, spec["columns"], label, spec["path"])
    out = {}
    for r in rows:
        qid = str(r.get("qa_id", "")).strip()
        if not qid:
            continue
        out[qid] = unique_keep_order(split_ids(r.get(col, "")))[:10]
    print(f"{label}: loaded {len(out)} QA rankings from {spec['path']} using column {col}")
    return out, col

def load_qa_rows(path):
    rows = load_csv(path)
    out = []
    for r in rows:
        qid = str(r.get("qa_id", "")).strip()
        if not qid:
            continue
        out.append({
            "qa_id": qid,
            "category": str(r.get("category", "")).strip(),
            "question": str(r.get("question", "")).strip(),
        })
    return out

def load_gold_map(path):
    gold = defaultdict(set)
    for r in load_csv(path):
        qid = str(r.get("qa_id", "")).strip()
        if not qid:
            continue
        eid = str(r.get("evidence_id", "")).strip()
        mid = str(r.get("memory_id", "")).strip()
        if eid:
            gold[qid].add(eid)
        if mid:
            gold[qid].add(mid)
    return gold

def rrf_score(rank, k, weight):
    return weight / (k + rank)

def fuse_rankings(bm25_ids, dense_ids, fusion_ids, kg_ids, cfg):
    scores = defaultdict(float)
    sources = [
        (bm25_ids, cfg["w_bm25"]),
        (dense_ids, cfg["w_dense"]),
        (fusion_ids, cfg["w_fusion"]),
        (kg_ids, cfg["w_kg"]),
    ]
    for ids, weight in sources:
        for rank, mid in enumerate(unique_keep_order(ids), start=1):
            scores[mid] += rrf_score(rank, cfg["rrf_k"], weight)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [mid for mid, _ in ranked]

def hit(top_ids, gold_ids):
    return int(any(x in gold_ids for x in top_ids))

def reciprocal_rank(top_ids, gold_ids):
    for i, mid in enumerate(top_ids, start=1):
        if mid in gold_ids:
            return 1.0 / i
    return 0.0

def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)

def summarize_group(method, cfg, category, rows):
    return {
        "method": method,
        "config_name": cfg["config_name"],
        "category": category,
        "n": len(rows),
        "rrf_k": cfg["rrf_k"],
        "w_bm25": cfg["w_bm25"],
        "w_dense": cfg["w_dense"],
        "w_fusion": cfg["w_fusion"],
        "w_kg": cfg["w_kg"],
        "recall_1": f"{mean(float(r['recall_1']) for r in rows):.4f}",
        "recall_5": f"{mean(float(r['recall_5']) for r in rows):.4f}",
        "recall_10": f"{mean(float(r['recall_10']) for r in rows):.4f}",
        "mrr_10": f"{mean(float(r['rr_10']) for r in rows):.4f}",
        "avg_candidate_count": f"{mean(int(r['candidate_count']) for r in rows):.2f}",
    }

def main():
    qa_rows = load_qa_rows(QA_CSV)
    gold_map = load_gold_map(EVIDENCE_CSV)

    rankings = {}
    used_cols = {}
    for label, spec in INPUTS.items():
        rankings[label], used_cols[label] = load_ranking_file(label, spec)

    result_rows = []

    for cfg in CONFIGS:
        cfg_name = cfg["config_name"]
        for qa in qa_rows:
            qid = qa["qa_id"]
            category = qa["category"]
            question = qa["question"]

            bm25_ids = rankings["bm25"].get(qid, [])
            dense_ids = rankings["dense"].get(qid, [])
            fusion_ids = rankings["fusion"].get(qid, [])
            kg_ids = rankings["kg"].get(qid, [])

            fused = fuse_rankings(bm25_ids, dense_ids, fusion_ids, kg_ids, cfg)
            top1 = fused[:1]
            top5 = fused[:5]
            top10 = fused[:10]

            gold_ids = gold_map.get(qid, set())

            row = {
                "method": "KG-aware-RRF",
                "config_name": cfg_name,
                "qa_id": qid,
                "category": category,
                "question": question,
                "gold_memory_ids": ";".join(sorted(gold_ids)),
                "bm25_top10_memory_ids": ";".join(bm25_ids),
                "dense_top10_memory_ids": ";".join(dense_ids),
                "bm25_dense_rrf_top10_memory_ids": ";".join(fusion_ids),
                "kg_boost_top10_memory_ids": ";".join(kg_ids),
                "kg_aware_top1_memory_ids": ";".join(top1),
                "kg_aware_top5_memory_ids": ";".join(top5),
                "kg_aware_top10_memory_ids": ";".join(top10),
                "retrieved_memory_ids": ";".join(top10),
                "candidate_count": len(fused),
                "recall_1": hit(top1, gold_ids),
                "recall_5": hit(top5, gold_ids),
                "recall_10": hit(top10, gold_ids),
                "rr_10": f"{reciprocal_rank(top10, gold_ids):.6f}",
                "rrf_k": cfg["rrf_k"],
                "w_bm25": cfg["w_bm25"],
                "w_dense": cfg["w_dense"],
                "w_fusion": cfg["w_fusion"],
                "w_kg": cfg["w_kg"],
            }
            result_rows.append(row)

    result_fields = [
        "method", "config_name", "qa_id", "category", "question", "gold_memory_ids",
        "bm25_top10_memory_ids", "dense_top10_memory_ids",
        "bm25_dense_rrf_top10_memory_ids", "kg_boost_top10_memory_ids",
        "kg_aware_top1_memory_ids", "kg_aware_top5_memory_ids",
        "kg_aware_top10_memory_ids", "retrieved_memory_ids",
        "candidate_count", "recall_1", "recall_5", "recall_10", "rr_10",
        "rrf_k", "w_bm25", "w_dense", "w_fusion", "w_kg"
    ]

    write_csv(OUT_RESULTS, result_rows, result_fields)

    summary_rows = []
    for cfg in CONFIGS:
        cfg_name = cfg["config_name"]
        rows = [r for r in result_rows if r["config_name"] == cfg_name]
        summary_rows.append(summarize_group("KG-aware-RRF", cfg, "ALL", rows))
        by_cat = defaultdict(list)
        for r in rows:
            by_cat[str(r["category"])].append(r)
        for cat in sorted(by_cat.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            summary_rows.append(summarize_group("KG-aware-RRF", cfg, f"cat_{cat}", by_cat[cat]))

    summary_fields = [
        "method", "config_name", "category", "n", "rrf_k",
        "w_bm25", "w_dense", "w_fusion", "w_kg",
        "recall_1", "recall_5", "recall_10", "mrr_10", "avg_candidate_count"
    ]

    write_csv(OUT_SUMMARY, summary_rows, summary_fields)

    all_summary = [r for r in summary_rows if r["category"] == "ALL"]
    best = sorted(
        all_summary,
        key=lambda r: (
            float(r["mrr_10"]),
            float(r["recall_10"]),
            float(r["recall_5"]),
            float(r["recall_1"]),
        ),
        reverse=True
    )[0]

    best_name = best["config_name"]
    best_result_rows = [r for r in result_rows if r["config_name"] == best_name]
    best_summary_rows = [r for r in summary_rows if r["config_name"] == best_name]

    write_csv(OUT_BEST_RESULTS, best_result_rows, result_fields)
    write_csv(OUT_BEST_SUMMARY, best_summary_rows, summary_fields)

    print("")
    print("=== KG-aware Fusion Retrieval ===")
    print(f"QA count: {len(qa_rows)}")
    print("")
    print(f"{'Config':18s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s} {'MRR@10':>8s} {'Weights':>32s}")
    print("-" * 90)
    for r in all_summary:
        weights = f"bm25={r['w_bm25']},dense={r['w_dense']},fusion={r['w_fusion']},kg={r['w_kg']}"
        print(f"{r['config_name']:18s} {float(r['recall_1']):8.4f} {float(r['recall_5']):8.4f} {float(r['recall_10']):8.4f} {float(r['mrr_10']):8.4f} {weights:>32s}")

    print("")
    print(f"Best config by MRR@10: {best_name}")
    print(f"Best R@1={float(best['recall_1']):.4f}, R@5={float(best['recall_5']):.4f}, R@10={float(best['recall_10']):.4f}, MRR@10={float(best['mrr_10']):.4f}")
    print("")
    print(f"All results: {OUT_RESULTS}")
    print(f"All summary: {OUT_SUMMARY}")
    print(f"Best results: {OUT_BEST_RESULTS}")
    print(f"Best summary: {OUT_BEST_SUMMARY}")
    print("")
    print("Columns used:")
    for k, v in used_cols.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()