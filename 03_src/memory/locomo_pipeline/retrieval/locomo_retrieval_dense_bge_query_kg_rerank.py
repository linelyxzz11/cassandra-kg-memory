import csv
import json
import re
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import product

STOPWORDS = set(
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their theirs "
    "themselves what which who whom this that these those am is are was were be "
    "been being have has had having do does did doing a an the and but if or "
    "because as until while of at by for with about against between through "
    "during before after above below to from up down in out on off over under "
    "again further then once here there when where why how all both each few "
    "more most other some such no nor not only own same so than too very s t "
    "can will just don should now d ll m o re ve y ain aren couldn didn doesn "
    "hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won "
    "wouldn also would could should may might shall".split()
)

TEMPORAL_WORDS = set(
    "when before after first last later earlier recently yesterday today "
    "tomorrow date time week month year day ago since until".split()
)

FEATURE_CONFIGS = {
    "config_A": {"entity": 0.40, "relation": 0.25, "text": 0.20, "temporal": 0.10, "multi_entity": 0.05},
    "config_B": {"entity": 0.50, "relation": 0.20, "text": 0.15, "temporal": 0.10, "multi_entity": 0.05},
    "config_C": {"entity": 0.30, "relation": 0.20, "text": 0.20, "temporal": 0.25, "multi_entity": 0.05},
    "config_D": {"entity": 0.35, "relation": 0.30, "text": 0.15, "temporal": 0.10, "multi_entity": 0.10},
}


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 2]
    return tokens


def has_temporal_flag(tokens):
    return int(any(t in TEMPORAL_WORDS for t in tokens))


def is_entity_like(token):
    if token in STOPWORDS:
        return False
    if token.isdigit():
        return False
    if len(token) < 3:
        return False
    return True


def load_data(args):
    base = Path(args.results_dir)

    print("Loading memory records...")
    memories = {}
    with open(base / "locomo_memory_records.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            memories[row["memory_id"]] = row

    print("Loading QA records...")
    qas = {}
    with open(base / "locomo_qa_records.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            qas[row["qa_id"]] = row

    print("Loading evidence map...")
    qa_gold = defaultdict(set)
    with open(base / "locomo_evidence_map.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            qa_gold[row["qa_id"]].add(row["memory_id"])

    print("Loading KG edges...")
    memory_kg_edges = defaultdict(list)
    with open(base / "locomo_kg_edges_spacy.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            evidence = row["evidence"]
            graph_id = row["graph_id"]
            src = row["src_id"]
            rel = row["relation"]
            dst = row["dst_id"]
            mid_candidates = []
            for mid, m in memories.items():
                if m["sample_id"] == graph_id and m["dia_id"] == evidence:
                    mid_candidates.append(mid)
            if not mid_candidates:
                sample_memories = [m for m in memories.values() if m["sample_id"] == graph_id]
                for mid, m in memories.items():
                    if m["sample_id"] == graph_id and evidence in mid:
                        mid_candidates.append(mid)
                        break
            for mid in mid_candidates:
                memory_kg_edges[mid].append({
                    "src": src, "relation": rel, "dst": dst,
                    "graph_id": graph_id, "evidence": evidence
                })

    print("Loading Dense-bge results...")
    dense_results = {}
    with open(base / "locomo_dense_bge_results.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            dense_results[row["qa_id"]] = row

    print("Loading Dense-bge embeddings for scores...")
    memory_ids = []
    with open(base / "locomo_memory_ids_bge.txt", encoding="utf-8") as f:
        for line in f:
            memory_ids.append(line.strip())
    memory_embs = np.load(base / "locomo_memory_bge_large.npy")

    qa_ids_list = []
    with open(base / "locomo_qa_ids_bge.txt", encoding="utf-8") as f:
        for line in f:
            qa_ids_list.append(line.strip())
    qa_embs = np.load(base / "locomo_qa_bge_large.npy")

    mid_to_idx = {mid: i for i, mid in enumerate(memory_ids)}
    qid_to_idx = {qid: i for i, qid in enumerate(qa_ids_list)}

    print(f"  {len(memories)} memories, {len(qas)} QAs, {len(memory_kg_edges)} memories with KG edges")
    kg_covered = len(memory_kg_edges)
    total_mem = len(memories)
    print(f"  KG coverage: {kg_covered}/{total_mem} ({kg_covered/total_mem*100:.1f}%)")

    return memories, qas, qa_gold, memory_kg_edges, dense_results, \
           memory_ids, memory_embs, qa_ids_list, qa_embs, mid_to_idx, qid_to_idx


def build_memory_kg_features(memory_kg_edges):
    memory_kg_tokens = {}
    for mid, edges in memory_kg_edges.items():
        all_entity_tokens = set()
        all_relation_tokens = set()
        all_tokens = set()
        has_temporal_edge = False
        for e in edges:
            src_toks = set(normalize_text(e["src"]))
            dst_toks = set(normalize_text(e["dst"]))
            rel_toks = set(normalize_text(e["relation"]))
            entity_toks = src_toks | dst_toks
            all_entity_tokens |= entity_toks
            all_relation_tokens |= rel_toks
            all_tokens |= entity_toks | rel_toks
            if has_temporal_flag(list(rel_toks)):
                has_temporal_edge = True
            for t in list(entity_toks) + list(rel_toks):
                if t in TEMPORAL_WORDS:
                    has_temporal_edge = True

        memory_kg_tokens[mid] = {
            "entity_tokens": all_entity_tokens,
            "relation_tokens": all_relation_tokens,
            "all_tokens": all_tokens,
            "has_temporal_edge": has_temporal_edge,
            "n_edges": len(edges),
        }
    return memory_kg_tokens


def compute_query_features(question):
    q_tokens = normalize_text(question)
    q_entity_tokens = set(t for t in q_tokens if is_entity_like(t))
    q_temporal = has_temporal_flag(q_tokens)
    q_token_set = set(q_tokens)
    return {
        "q_tokens": q_token_set,
        "q_entity_tokens": q_entity_tokens,
        "q_temporal": q_temporal,
    }


def compute_kg_score(qf, memory_kg_tok, feat_weights):
    if memory_kg_tok is None:
        return 0.0

    q_toks = qf["q_tokens"]
    if len(q_toks) == 0:
        return 0.0

    entity_overlap = len(q_toks & memory_kg_tok["entity_tokens"]) / max(1, len(q_toks))
    relation_overlap = len(q_toks & memory_kg_tok["relation_tokens"]) / max(1, len(q_toks))
    text_overlap = len(q_toks & memory_kg_tok["all_tokens"]) / max(1, len(q_toks))

    temporal_match = 0.0
    if qf["q_temporal"] and memory_kg_tok["has_temporal_edge"]:
        temporal_match = 1.0

    multi_entity_match = 0.0
    q_ent = qf["q_entity_tokens"]
    if len(q_ent) >= 2:
        kg_ent = memory_kg_tok["entity_tokens"]
        overlap = q_ent & kg_ent
        if len(overlap) >= 2:
            multi_entity_match = 1.0

    score = (
        feat_weights["entity"] * entity_overlap
        + feat_weights["relation"] * relation_overlap
        + feat_weights["text"] * text_overlap
        + feat_weights["temporal"] * temporal_match
        + feat_weights["multi_entity"] * multi_entity_match
    )
    return min(max(score, 0.0), 1.0)


def get_dense_topn(qa_id, dense_results, mid_to_idx, memory_embs, qa_embs, qid_to_idx, n):
    row = dense_results[qa_id]
    retrieved = row.get("retrieved_memory_ids", "")
    if not retrieved:
        return [], []
    all_ids = [x.strip() for x in retrieved.split(";") if x.strip()]
    topn_ids = all_ids[:n]

    qi = qid_to_idx.get(qa_id)
    if qi is None:
        return topn_ids, [1.0] * len(topn_ids)

    scores = []
    for mid in topn_ids:
        mi = mid_to_idx.get(mid)
        if mi is not None:
            s = float(np.dot(qa_embs[qi], memory_embs[mi]))
        else:
            s = 0.0
        scores.append(s)
    return topn_ids, scores


def minmax_normalize(scores):
    mn = min(scores)
    mx = max(scores)
    if mx - mn < 1e-9:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def compute_metrics(reranked_ids, gold_ids, k_values=(1, 5, 10)):
    hits = {}
    for k in k_values:
        topk = set(reranked_ids[:k])
        hits[k] = int(bool(topk & gold_ids))
    rr = 0.0
    for i, mid in enumerate(reranked_ids[:10], 1):
        if mid in gold_ids:
            rr = 1.0 / i
            break
    return hits, rr


def stratified_split(qas, qa_gold, dev_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    by_cat = defaultdict(list)
    for qid in sorted(qas.keys()):
        cat = qas[qid]["category"]
        by_cat[cat].append(qid)

    dev_ids = set()
    test_ids = set()
    for cat, ids in by_cat.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_dev = max(1, int(len(ids) * dev_frac))
        dev_ids.update(ids[:n_dev])
        test_ids.update(ids[n_dev:])
    return dev_ids, test_ids


def run_single_config(
    qa_ids_subset, qas, qa_gold, dense_results, memory_kg_tokens,
    mid_to_idx, memory_embs, qa_embs, qid_to_idx,
    topn, lam, feat_weights, gated_info=None
):
    results = []
    for qa_id in qa_ids_subset:
        question = qas[qa_id]["question"]
        gold = qa_gold.get(qa_id, set())

        topn_ids, dense_scores = get_dense_topn(
            qa_id, dense_results, mid_to_idx, memory_embs, qa_embs, qid_to_idx, topn
        )

        if not topn_ids:
            hits, rr = compute_metrics([], gold)
            results.append({
                "qa_id": qa_id, "reranked_ids": [],
                "hits": hits, "rr": rr, "dense_ids": [],
                "kg_scores": [], "final_scores": [], "dense_norm_scores": [],
            })
            continue

        qf = compute_query_features(question)

        use_kg = True
        if gated_info is not None:
            if len(dense_scores) >= 2:
                top1 = dense_scores[0]
                top2 = dense_scores[1] if len(dense_scores) > 1 else 0
                gap = top1 - top2
                if top1 >= gated_info["top1_thresh"] and gap >= gated_info["gap_thresh"]:
                    use_kg = False

        dense_norm = minmax_normalize(dense_scores)

        kg_scores = []
        for mid in topn_ids:
            if use_kg:
                mkt = memory_kg_tokens.get(mid)
                ks = compute_kg_score(qf, mkt, feat_weights)
            else:
                ks = 0.0
            kg_scores.append(ks)

        final_scores = [dn + lam * ks for dn, ks in zip(dense_norm, kg_scores)]

        paired = list(zip(topn_ids, final_scores, dense_norm, kg_scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        reranked = [p[0] for p in paired]
        sorted_final = [p[1] for p in paired]
        sorted_dense_norm = [p[2] for p in paired]
        sorted_kg = [p[3] for p in paired]

        hits, rr = compute_metrics(reranked, gold)

        results.append({
            "qa_id": qa_id,
            "reranked_ids": reranked[:10],
            "dense_ids": topn_ids[:10],
            "hits": hits,
            "rr": rr,
            "kg_scores": sorted_kg[:10],
            "final_scores": sorted_final[:10],
            "dense_norm_scores": sorted_dense_norm[:10],
        })
    return results


def aggregate_metrics(results, qa_gold, dense_results, qas):
    n = len(results)
    if n == 0:
        return {"n": 0, "R@1": 0, "R@5": 0, "R@10": 0, "MRR": 0}

    r1 = sum(r["hits"][1] for r in results) / n
    r5 = sum(r["hits"][5] for r in results) / n
    r10 = sum(r["hits"][10] for r in results) / n
    mrr = sum(r["rr"] for r in results) / n

    rescue_hurt = {"rescue@1": 0, "hurt@1": 0, "rescue@5": 0, "hurt@5": 0, "rescue@10": 0, "hurt@10": 0}
    for r in results:
        qa_id = r["qa_id"]
        gold = qa_gold.get(qa_id, set())
        reranked = r["reranked_ids"]
        dense_row = dense_results.get(qa_id, {})
        dense_top10 = [x.strip() for x in dense_row.get("retrieved_memory_ids", "").split(";") if x.strip()][:10]

        for k in [1, 5, 10]:
            reranked_hit = int(bool(set(reranked[:k]) & gold))
            dense_hit = int(bool(set(dense_top10[:k]) & gold))
            if reranked_hit and not dense_hit:
                rescue_hurt[f"rescue@{k}"] += 1
            elif dense_hit and not reranked_hit:
                rescue_hurt[f"hurt@{k}"] += 1

    return {
        "n": n, "R@1": r1, "R@5": r5, "R@10": r10, "MRR": mrr,
        **rescue_hurt,
        "net@1": rescue_hurt["rescue@1"] - rescue_hurt["hurt@1"],
        "net@5": rescue_hurt["rescue@5"] - rescue_hurt["hurt@5"],
        "net@10": rescue_hurt["rescue@10"] - rescue_hurt["hurt@10"],
    }


def compute_category_metrics(results_by_qa, qas, qa_gold, dense_results):
    by_cat = defaultdict(list)
    for r in results_by_qa:
        cat = qas[r["qa_id"]]["category"]
        by_cat[cat].append(r)

    cat_metrics = {}
    for cat, cat_results in sorted(by_cat.items()):
        cat_metrics[cat] = aggregate_metrics(cat_results, qa_gold, dense_results, qas)
    return cat_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent.parent / "results"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent.parent / "results/query_kg_rerank"))
    parser.add_argument("--dev-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    memories, qas, qa_gold, memory_kg_edges, dense_results, \
        memory_ids, memory_embs, qa_ids_list, qa_embs, mid_to_idx, qid_to_idx = load_data(args)

    memory_kg_tokens = build_memory_kg_features(memory_kg_edges)

    dev_ids, test_ids = stratified_split(qas, qa_gold, dev_frac=args.dev_frac, seed=args.seed)
    all_ids = set(qas.keys())
    print(f"Dev: {len(dev_ids)}, Test: {len(test_ids)}, Total: {len(all_ids)}")

    topn_values = [50, 100, 200]
    lambda_values = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    config_names = list(FEATURE_CONFIGS.keys())

    dense_scores_for_gated = {}
    for qa_id in all_ids:
        row = dense_results.get(qa_id, {})
        retrieved = row.get("retrieved_memory_ids", "")
        if not retrieved:
            continue
        top10 = [x.strip() for x in retrieved.split(";") if x.strip()][:10]
        qi = qid_to_idx.get(qa_id)
        scores = []
        if qi is not None:
            for mid in top10[:2]:
                mi = mid_to_idx.get(mid)
                if mi is not None:
                    scores.append(float(np.dot(qa_embs[qi], memory_embs[mi])))
        if len(scores) >= 2:
            dense_scores_for_gated[qa_id] = {"top1": scores[0], "gap": scores[0] - scores[1]}

    all_top1 = [v["top1"] for v in dense_scores_for_gated.values()]
    all_gaps = [v["gap"] for v in dense_scores_for_gated.values()]

    top1_quantiles = {0.25: np.percentile(all_top1, 25), 0.50: np.percentile(all_top1, 50)}
    gap_quantiles = {0.25: np.percentile(all_gaps, 25), 0.50: np.percentile(all_gaps, 50)}

    print(f"Top1 quantiles: 25th={top1_quantiles[0.25]:.4f}, 50th={top1_quantiles[0.50]:.4f}")
    print(f"Gap quantiles: 25th={gap_quantiles[0.25]:.4f}, 50th={gap_quantiles[0.50]:.4f}")

    grid_results = []
    best_dev_score = (-1, -1, -999, 999)
    best_config = None

    splits = {"dev": dev_ids, "test": test_ids, "full": all_ids}

    total_configs = len(topn_values) * len(lambda_values) * len(config_names)
    gated_configs = 2 * 2  # top1_quantile x gap_quantile
    total_with_gated = total_configs * (1 + gated_configs)
    print(f"Total configs: {total_configs} (non-gated) + {total_configs * gated_configs} (gated) = {total_configs + total_configs * gated_configs}")

    done = 0
    for topn in topn_values:
        for lam in lambda_values:
            for cname in config_names:
                feat_w = FEATURE_CONFIGS[cname]
                done += 1

                for split_name, split_ids in [("dev", dev_ids), ("test", test_ids), ("full", all_ids)]:
                    res = run_single_config(
                        split_ids, qas, qa_gold, dense_results, memory_kg_tokens,
                        mid_to_idx, memory_embs, qa_embs, qid_to_idx,
                        topn, lam, feat_w, gated_info=None
                    )
                    agg = aggregate_metrics(res, qa_gold, dense_results, qas)
                    grid_results.append({
                        "method_name": f"Dense-bge+QueryKG-top{topn}-lam{lam}-{cname}",
                        "topN": topn, "lambda": lam, "feature_config": cname,
                        "gated": False, "top1_quantile": "", "gap_quantile": "",
                        "split": split_name, **agg,
                    })

                    if split_name == "dev":
                        dev_key = (agg["R@10"], agg["MRR"], agg["net@10"], -agg.get("hurt@10", 0))
                        if dev_key > best_dev_score:
                            best_dev_score = dev_key
                            best_config = {"topn": topn, "lambda": lam, "config": cname, "gated": False, "top1_q": "", "gap_q": ""}

                for t1q_label, t1q_val in top1_quantiles.items():
                    for gq_label, gq_val in gap_quantiles.items():
                        gated_info = {"top1_thresh": t1q_val, "gap_thresh": gq_val}
                        for split_name, split_ids in [("dev", dev_ids), ("test", test_ids), ("full", all_ids)]:
                            res = run_single_config(
                                split_ids, qas, qa_gold, dense_results, memory_kg_tokens,
                                mid_to_idx, memory_embs, qa_embs, qid_to_idx,
                                topn, lam, feat_w, gated_info=gated_info
                            )
                            agg = aggregate_metrics(res, qa_gold, dense_results, qas)
                            grid_results.append({
                                "method_name": f"Dense-bge+GatedQueryKG-top{topn}-lam{lam}-{cname}-t1q{t1q_label}-gq{gq_label}",
                                "topN": topn, "lambda": lam, "feature_config": cname,
                                "gated": True, "top1_quantile": t1q_label, "gap_quantile": gq_label,
                                "split": split_name, **agg,
                            })

                            if split_name == "dev":
                                dev_key = (agg["R@10"], agg["MRR"], agg["net@10"], -agg.get("hurt@10", 0))
                                if dev_key > best_dev_score:
                                    best_dev_score = dev_key
                                    best_config = {"topn": topn, "lambda": lam, "config": cname, "gated": True, "top1_q": t1q_label, "gap_q": gq_label}

                if done % 6 == 0:
                    print(f"  Progress: {done}/{total_configs} base configs done...")

    print(f"\nBest dev config: {best_config}")
    print(f"Best dev score (R@10, MRR, net@10, -hurt@10): {best_dev_score}")

    grid_path = os.path.join(args.output_dir, "dense_bge_query_kg_rerank_grid_summary.csv")
    fieldnames = ["method_name", "topN", "lambda", "feature_config", "gated",
                  "top1_quantile", "gap_quantile", "split", "n",
                  "R@1", "R@5", "R@10", "MRR@10",
                  "rescue@1", "hurt@1", "net@1",
                  "rescue@5", "hurt@5", "net@5",
                  "rescue@10", "hurt@10", "net@10"]
    with open(grid_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in grid_results:
            row_out = dict(r)
            row_out["MRR@10"] = row_out.pop("MRR", "")
            w.writerow(row_out)
    print(f"Grid summary saved: {grid_path}")

    bc = best_config
    feat_w = FEATURE_CONFIGS[bc["config"]]
    gated_info = None
    if bc["gated"]:
        gated_info = {
            "top1_thresh": top1_quantiles[bc["top1_q"]],
            "gap_thresh": gap_quantiles[bc["gap_q"]],
        }

    print("\nRunning best config on full dataset for per-query output...")
    full_res = run_single_config(
        all_ids, qas, qa_gold, dense_results, memory_kg_tokens,
        mid_to_idx, memory_embs, qa_embs, qid_to_idx,
        bc["topn"], bc["lambda"], feat_w, gated_info=gated_info
    )

    best_perquery_path = os.path.join(args.output_dir, "dense_bge_query_kg_rerank_best_results.csv")
    with open(best_perquery_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "qa_id", "category", "question", "gold_evidence_ids",
            "dense_top10_memory_ids", "reranked_top10_memory_ids", "retrieved_memory_ids",
            "dense_scores_top10", "kg_scores_top10", "final_scores_top10",
            "retrieval_hit1", "retrieval_hit5", "retrieval_hit10",
        ])
        w.writeheader()
        for r in full_res:
            qa_id = r["qa_id"]
            gold = qa_gold.get(qa_id, set())
            dense_row = dense_results.get(qa_id, {})
            dense_top10 = [x.strip() for x in dense_row.get("retrieved_memory_ids", "").split(";") if x.strip()][:10]

            w.writerow({
                "qa_id": qa_id,
                "category": qas[qa_id]["category"],
                "question": qas[qa_id]["question"],
                "gold_evidence_ids": ";".join(sorted(gold)),
                "dense_top10_memory_ids": ";".join(dense_top10[:10]),
                "reranked_top10_memory_ids": ";".join(r["reranked_ids"][:10]),
                "retrieved_memory_ids": ";".join(r["reranked_ids"][:10]),
                "dense_scores_top10": ";".join(f"{s:.4f}" for s in r["dense_norm_scores"][:10]),
                "kg_scores_top10": ";".join(f"{s:.4f}" for s in r["kg_scores"][:10]),
                "final_scores_top10": ";".join(f"{s:.4f}" for s in r["final_scores"][:10]),
                "retrieval_hit1": r["hits"][1],
                "retrieval_hit5": r["hits"][5],
                "retrieval_hit10": r["hits"][10],
            })
    print(f"Best per-query results saved: {best_perquery_path}")

    best_agg_full = aggregate_metrics(full_res, qa_gold, dense_results, qas)
    best_agg_test = aggregate_metrics(
        [r for r in full_res if r["qa_id"] in test_ids], qa_gold, dense_results, qas
    )
    best_agg_dev = aggregate_metrics(
        [r for r in full_res if r["qa_id"] in dev_ids], qa_gold, dense_results, qas
    )

    cat_metrics = compute_category_metrics(full_res, qas, qa_gold, dense_results)

    dense_only_res = []
    for qa_id in all_ids:
        gold = qa_gold.get(qa_id, set())
        dense_row = dense_results.get(qa_id, {})
        dense_top10 = [x.strip() for x in dense_row.get("retrieved_memory_ids", "").split(";") if x.strip()][:10]
        hits, rr = compute_metrics(dense_top10, gold)
        dense_only_res.append({"qa_id": qa_id, "reranked_ids": dense_top10, "hits": hits, "rr": rr})
    dense_only_agg = aggregate_metrics(dense_only_res, qa_gold, dense_results, qas)

    global_kg_res = []
    for qa_id in all_ids:
        gold = qa_gold.get(qa_id, set())
        dense_row = dense_results.get(qa_id, {})
        retrieved = [x.strip() for x in dense_row.get("retrieved_memory_ids", "").split(";") if x.strip()]
        topn_ids = retrieved[:100]
        dense_scores_list = []
        qi = qid_to_idx.get(qa_id)
        if qi is not None:
            for mid in topn_ids:
                mi = mid_to_idx.get(mid)
                if mi is not None:
                    dense_scores_list.append(float(np.dot(qa_embs[qi], memory_embs[mi])))
                else:
                    dense_scores_list.append(0.0)
        else:
            dense_scores_list = [1.0] * len(topn_ids)

        dense_norm = minmax_normalize(dense_scores_list)
        kg_boost_scores = []
        for mid in topn_ids:
            ks = 0.1 if mid in memory_kg_edges else 0.0
            kg_boost_scores.append(ks)
        final_scores = [dn + ks for dn, ks in zip(dense_norm, kg_boost_scores)]
        paired = sorted(zip(topn_ids, final_scores), key=lambda x: x[1], reverse=True)
        reranked = [p[0] for p in paired]
        hits, rr = compute_metrics(reranked, gold)
        global_kg_res.append({"qa_id": qa_id, "reranked_ids": reranked[:10], "hits": hits, "rr": rr})
    global_kg_agg = aggregate_metrics(global_kg_res, qa_gold, dense_results, qas)

    summary_path = os.path.join(args.output_dir, "dense_bge_query_kg_rerank_best_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "split", "n", "R@1", "R@5", "R@10", "MRR@10",
            "rescue@10", "hurt@10", "net@10",
        ])
        w.writeheader()
        for name, agg, split in [
            ("Dense-bge", dense_only_agg, "full"),
            ("Dense-bge+global-KG", global_kg_agg, "full"),
            (f"Dense-bge+QueryKG ({bc['config']}, top{bc['topn']}, lam={bc['lambda']}, gated={bc['gated']})", best_agg_full, "full"),
            (f"Dense-bge+QueryKG ({bc['config']}, top{bc['topn']}, lam={bc['lambda']}, gated={bc['gated']})", best_agg_dev, "dev"),
            (f"Dense-bge+QueryKG ({bc['config']}, top{bc['topn']}, lam={bc['lambda']}, gated={bc['gated']})", best_agg_test, "test"),
        ]:
            w.writerow({
                "method": name, "split": split, "n": agg["n"],
                "R@1": f"{agg['R@1']:.4f}", "R@5": f"{agg['R@5']:.4f}",
                "R@10": f"{agg['R@10']:.4f}", "MRR@10": f"{agg['MRR']:.4f}",
                "rescue@10": agg.get("rescue@10", 0), "hurt@10": agg.get("hurt@10", 0),
                "net@10": agg.get("net@10", 0),
            })
        for cat, cm in sorted(cat_metrics.items()):
            w.writerow({
                "method": f"Dense-bge+QueryKG cat_{cat}", "split": "full", "n": cm["n"],
                "R@1": f"{cm['R@1']:.4f}", "R@5": f"{cm['R@5']:.4f}",
                "R@10": f"{cm['R@10']:.4f}", "MRR@10": f"{cm['MRR']:.4f}",
                "rescue@10": cm.get("rescue@10", 0), "hurt@10": cm.get("hurt@10", 0),
                "net@10": cm.get("net@10", 0),
            })
    print(f"Best summary saved: {summary_path}")

    rescue_examples = []
    hurt_examples = []
    for r in full_res:
        qa_id = r["qa_id"]
        gold = qa_gold.get(qa_id, set())
        reranked = r["reranked_ids"]
        dense_row = dense_results.get(qa_id, {})
        dense_top10 = [x.strip() for x in dense_row.get("retrieved_memory_ids", "").split(";") if x.strip()][:10]

        reranked_hit10 = int(bool(set(reranked[:10]) & gold))
        dense_hit10 = int(bool(set(dense_top10[:10]) & gold))

        if reranked_hit10 and not dense_hit10:
            rescue_examples.append(r)
        elif dense_hit10 and not reranked_hit10:
            hurt_examples.append(r)

    rescue_examples = rescue_examples[:20] if len(rescue_examples) > 20 else rescue_examples
    hurt_examples = hurt_examples[:20] if len(hurt_examples) > 20 else hurt_examples

    audit_summary_path = os.path.join(args.output_dir, "audit_dense_bge_query_kg_rescue_summary.csv")
    with open(audit_summary_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["type", "count", "pct"])
        w.writeheader()
        n_total = len(full_res)
        n_both = n_total - len(rescue_examples) - len(hurt_examples) - sum(
            1 for r in full_res
            if not (int(bool(set(r["reranked_ids"][:10]) & qa_gold.get(r["qa_id"], set()))))
            and not (int(bool(set(
                [x.strip() for x in dense_results.get(r["qa_id"], {}).get("retrieved_memory_ids", "").split(";") if x.strip()][
                :10]) & qa_gold.get(r["qa_id"], set()))))
        )
        w.writerow({"type": "rescue (Dense miss, QueryKG hit)", "count": len(rescue_examples), "pct": f"{len(rescue_examples)/n_total*100:.2f}%"})
        w.writerow({"type": "hurt (Dense hit, QueryKG miss)", "count": len(hurt_examples), "pct": f"{len(hurt_examples)/n_total*100:.2f}%"})
        w.writerow({"type": "net (rescue - hurt)", "count": len(rescue_examples) - len(hurt_examples), "pct": f"{(len(rescue_examples)-len(hurt_examples))/n_total*100:.2f}%"})
    print(f"Audit summary saved: {audit_summary_path}")

    audit_examples_path = os.path.join(args.output_dir, "audit_dense_bge_query_kg_rescue_examples.csv")
    with open(audit_examples_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "type", "qa_id", "category", "question",
            "gold_ids", "dense_top10", "reranked_top10",
            "kg_scores_top10", "final_scores_top10",
        ])
        w.writeheader()
        for ex in rescue_examples:
            qa_id = ex["qa_id"]
            gold = qa_gold.get(qa_id, set())
            dense_row = dense_results.get(qa_id, {})
            dense_top10 = ";".join([x.strip() for x in dense_row.get("retrieved_memory_ids", "").split(";") if x.strip()][:10])
            w.writerow({
                "type": "rescue", "qa_id": qa_id,
                "category": qas[qa_id]["category"],
                "question": qas[qa_id]["question"],
                "gold_ids": ";".join(sorted(gold)),
                "dense_top10": dense_top10,
                "reranked_top10": ";".join(ex["reranked_ids"][:10]),
                "kg_scores_top10": ";".join(f"{s:.4f}" for s in ex["kg_scores"][:10]),
                "final_scores_top10": ";".join(f"{s:.4f}" for s in ex["final_scores"][:10]),
            })
        for ex in hurt_examples:
            qa_id = ex["qa_id"]
            gold = qa_gold.get(qa_id, set())
            dense_row = dense_results.get(qa_id, {})
            dense_top10 = ";".join([x.strip() for x in dense_row.get("retrieved_memory_ids", "").split(";") if x.strip()][:10])
            w.writerow({
                "type": "hurt", "qa_id": qa_id,
                "category": qas[qa_id]["category"],
                "question": qas[qa_id]["question"],
                "gold_ids": ";".join(sorted(gold)),
                "dense_top10": dense_top10,
                "reranked_top10": ";".join(ex["reranked_ids"][:10]),
                "kg_scores_top10": ";".join(f"{s:.4f}" for s in ex["kg_scores"][:10]),
                "final_scores_top10": ";".join(f"{s:.4f}" for s in ex["final_scores"][:10]),
            })
    print(f"Audit examples saved: {audit_examples_path}")

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"{'Method':<50s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s} {'MRR':>8s} {'net@10':>8s}")
    print("-" * 90)
    print(f"{'Dense-bge':<50s} {dense_only_agg['R@1']:>8.4f} {dense_only_agg['R@5']:>8.4f} {dense_only_agg['R@10']:>8.4f} {dense_only_agg['MRR']:>8.4f} {'N/A':>8s}")
    print(f"{'Dense-bge+global-KG (w=0.1)':<50s} {global_kg_agg['R@1']:>8.4f} {global_kg_agg['R@5']:>8.4f} {global_kg_agg['R@10']:>8.4f} {global_kg_agg['MRR']:>8.4f} {global_kg_agg.get('net@10',0):>+8d}")
    gated_label = f", gated(t1q={bc.get('top1_q','')}, gq={bc.get('gap_q','')})" if bc["gated"] else ""
    best_label = f"QueryKG({bc['config']}, top{bc['topn']}, lam={bc['lambda']}{gated_label})"
    print(f"{'Dense-bge+' + best_label:<50s} {best_agg_full['R@1']:>8.4f} {best_agg_full['R@5']:>8.4f} {best_agg_full['R@10']:>8.4f} {best_agg_full['MRR']:>8.4f} {best_agg_full.get('net@10',0):>+8d}")

    print("\nCategory breakdown (Dense-bge+QueryKG best, full):")
    print(f"{'Category':<15s} {'n':>5s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s} {'MRR':>8s} {'net@10':>8s}")
    print("-" * 60)
    for cat, cm in sorted(cat_metrics.items()):
        print(f"cat_{cat:<10s} {cm['n']:>5d} {cm['R@1']:>8.4f} {cm['R@5']:>8.4f} {cm['R@10']:>8.4f} {cm['MRR']:>8.4f} {cm.get('net@10',0):>+8d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
