"""Recompute held-out quality with the exact online ranking core, no model calls."""
from __future__ import annotations

import argparse
import gc
import math
import pickle
import platform
import time
from collections import defaultdict

from common import ARMS, HERE, ROOT, build_graph_records, index_space, lexical, np, rows, source_manifest, vectors, write_csv, write_json
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex, zscore_fuse


def metrics(gold, ranking):
    ranks = [i for i, mid in enumerate(ranking[:10], 1) if mid in gold]
    dcg = sum(1 / math.log2(i + 1) for i in ranks)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(10, len(gold)) + 1))
    return {"MRR@10": 1 / min(ranks) if ranks else 0.0,
            "Hit@10": float(bool(ranks)), "Recall@10": len(set(ranking[:10]) & gold) / len(gold),
            "nDCG@10": dcg / ideal}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=__import__("pathlib").Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    manifest = {"status": "RUNNING", "python": platform.python_version(), "sources": source_manifest(),
                "arms": ARMS, "alpha_dense": 0.6, "candidate_depth": 50, "top_k": 10,
                "cost_boundary": "Prepared ERK and raw float32 vectors; no extraction or embedding calls."}
    write_json(args.output / "manifest.json", manifest)
    gold_rows = [r for r in rows(ROOT / "data/retrieval_gold/locomo_cat1_4_gold_memory.csv")
                 if r["split"] == "test" and r["gold_status"] == "mapped" and r["gold_memory_ids"]]
    gold = {r["query_id"]: set(r["gold_memory_ids"].split(";")) for r in gold_rows}
    queries = [r for r in rows(ROOT / "data/frozen_retrieval/p3_queries.csv") if r["query_id"] in gold]
    assert len(gold) == len(queries) == 1146
    assert len({q["query_id"] for q in queries}) == 1146
    records = build_graph_records()
    grouped = defaultdict(list)
    for record in records:
        grouped[record.scope_id].append(record)
    mv, qv = vectors("memory"), vectors("qa")
    frozen_dense = defaultdict(list)
    for row in rows(ROOT / "data/frozen_retrieval/frozen_dense_scores_long.csv"):
        if row["query_id"] in gold:
            frozen_dense[row["query_id"]].append((row["memory_id"], float(row["score"])))
    for qid in frozen_dense:
        frozen_dense[qid].sort(key=lambda x: -x[1])
        frozen_dense[qid] = frozen_dense[qid][:50]
    reference_path = ROOT / "results/retrieval/fusion_ablation_aligned_v2/fusion_ablation_per_query_heldout1146.csv"
    ref = {r["query_id"]: r for r in rows(reference_path) if r["method"].lower().startswith("zscore")}
    write_json(args.output / "reference_manifest.json", {"reference_file": str(reference_path), "reference_queries": len(ref)})
    all_rows, spaces, differences = [], [], []
    for arm in ARMS:
        print(f"Building {arm}", flush=True)
        index = ScopedOnlineRetrievalIndex()
        build_start = time.perf_counter()
        for scope, items in grouped.items():
            index.load_scope(scope, {r.memory_id: lexical(r, arm) for r in items}, {r.memory_id: mv[r.memory_id] for r in items})
        space = {"arm": arm, **index_space(index), "initial_build_seconds": time.perf_counter() - build_start}
        space["sparse_snapshot_pickle5_bytes"] = len(pickle.dumps(index._sparse, protocol=5))
        spaces.append(space)
        for ordinal, query in enumerate(queries):
            qid, scope = query["query_id"], query["conversation_id"]
            result = index.search(scope, query["question"], qv[qid])
            for scoring, ranking in (
                ("online_raw_vectors", [x[0] for x in result.top10]),
                ("frozen_dense_scores_audit", [x[0] for x in zscore_fuse(frozen_dense[qid], result.sparse_top50)]),
            ):
                row = {"arm": arm, "scoring": scoring, "query_id": qid, "scope": scope,
                       "category": query["category"], "top10_memory_ids": ";".join(ranking), **metrics(gold[qid], ranking)}
                all_rows.append(row)
                if arm == "RawERK" and qid in ref and row["top10_memory_ids"] != ref[qid]["top10_memory_ids"]:
                    differences.append({**row, "reference_top10": ref[qid]["top10_memory_ids"], "reference_MRR": ref[qid]["MRR@10"]})
            if (ordinal + 1) % 300 == 0:
                print(f"{arm}: {ordinal + 1}/{len(queries)}", flush=True)
        write_csv(args.output / "per_query.csv", all_rows)
        write_csv(args.output / "index_space_original_corpus.csv", spaces)
        del index
        gc.collect()
    summaries = []
    for arm in ARMS:
        for scoring in ("online_raw_vectors", "frozen_dense_scores_audit"):
            selected = [r for r in all_rows if r["arm"] == arm and r["scoring"] == scoring]
            summaries.append({"arm": arm, "scoring": scoring, "n": len(selected),
                              **{m: float(np.mean([r[m] for r in selected])) for m in ("MRR@10", "Hit@10", "Recall@10", "nDCG@10")}})
    write_csv(args.output / "summary.csv", summaries)
    write_csv(args.output / "rawerk_reference_differences.csv", differences)
    manifest.update(status="COMPLETED", elapsed_seconds=time.perf_counter()-began,
                    reference_ranking_mismatches={sc: sum(r["scoring"] == sc for r in differences) for sc in ("online_raw_vectors", "frozen_dense_scores_audit")},
                    reference_gate="PASS" if len(ref) == 1146 and not differences else "REVIEW_REQUIRED")
    write_json(args.output / "manifest.json", manifest)
    print(summaries, flush=True)
    print(f"Reference gate: {manifest['reference_gate']}", flush=True)


if __name__ == "__main__":
    main()
