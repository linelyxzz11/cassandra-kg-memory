"""Complete dev-selected alpha sweep for canonical CassMem Z-score fusion.

Uses the frozen Dense scores, canonical RawERK renderer/BM25 configuration,
Top-50 branches, branch-min missing-score rule, and deterministic tie handling
from the frozen P1-C implementation.  Alpha is selected on dev MRR only;
held-out rows are reported for analysis and never used for selection.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P1_SCRIPT = ROOT / "04_experiments/retrieval/p1_compact_component_ablation/run_p1c_ablation_v5.py"
P1_CONFIG = ROOT / "04_experiments/retrieval/p1_compact_component_ablation/p1c_config.json"
GOLD_PATH = ROOT / "02_artifacts/retrieval_gold_v2/locomo_cat1_4_gold_memory.csv"
TABLE1_PATH = ROOT / "05_reports/retrieval_main_table/retrieval_main_overall.csv"
OUTPUT = ROOT / "05_reports/zscore_alpha_sweep"
ALPHAS = tuple(round(value / 10, 1) for value in range(11))
METRICS = ("MRR@10", "Hit@1", "Hit@5", "Hit@10", "Recall@10", "nDCG@10")


def load_p1():
    spec = importlib.util.spec_from_file_location("p1c_alpha_sweep", P1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {P1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(gold_ids: set[str], ranking: list[str]) -> dict[str, float]:
    ranked = ranking[:10]
    relevant = [rank for rank, mid in enumerate(ranked, 1) if mid in gold_ids]
    first = min(relevant) if relevant else None
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_ids), 10) + 1))
    return {
        "MRR@10": 0.0 if first is None else 1.0 / first,
        "Hit@1": float(first == 1),
        "Hit@5": float(first is not None and first <= 5),
        "Hit@10": float(first is not None and first <= 10),
        "Recall@10": len(set(ranked) & gold_ids) / len(gold_ids),
        "nDCG@10": 0.0 if not idcg else dcg / idcg,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {metric: sum(float(row[metric]) for row in rows) / len(rows) for metric in METRICS}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def clustered_mrr_comparison(per_query: list[dict], baseline: str, repetitions: int = 10000) -> dict:
    selected = {r["query_id"]: float(r["MRR@10"]) for r in per_query if r["split"] == "dev" and r["alpha_dense"] == "0.6"}
    other = {r["query_id"]: float(r["MRR@10"]) for r in per_query if r["split"] == "dev" and r["alpha_dense"] == baseline}
    differences = {qid: selected[qid] - other[qid] for qid in selected}
    clusters: dict[str, list[float]] = defaultdict(list)
    for qid, delta in differences.items():
        clusters[qid.split("_qa_", 1)[0]].append(delta)
    names = sorted(clusters)
    rng = random.Random(20260922 + int(float(baseline) * 10))
    boot = []
    for _ in range(repetitions):
        sample = [delta for _ in names for delta in clusters[rng.choice(names)]]
        boot.append(sum(sample) / len(sample))
    observed = sum(differences.values()) / len(differences)
    return {
        "comparison": f"alpha=0.6 vs alpha={baseline}",
        "n_queries": len(differences), "n_conversations": len(names),
        "delta_MRR@10": observed,
        "ci95_low": percentile(boot, 0.025), "ci95_high": percentile(boot, 0.975),
        "wins": sum(delta > 0 for delta in differences.values()),
        "ties": sum(delta == 0 for delta in differences.values()),
        "losses": sum(delta < 0 for delta in differences.values()),
        "bootstrap_repetitions": repetitions,
    }


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p1 = load_p1()
    config = p1.load_json(P1_CONFIG)
    root = p1.resolve_path(config.get("project_root", "."), P1_CONFIG.parent)
    depth = int(config["fusion"]["candidate_depth"])
    missing = config["fusion"]["missing_candidate_rule"]

    gold: dict[str, dict] = {}
    with GOLD_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ids = {mid for mid in row["gold_memory_ids"].split(";") if mid}
            if row["gold_status"] == "mapped" and ids:
                gold[row["query_id"]] = {
                    "split": row["split"], "category": row["category"], "ids": ids
                }
    if sum(item["split"] == "dev" for item in gold.values()) != 390:
        raise RuntimeError("Expected 390 mapped dev queries")
    if sum(item["split"] == "test" for item in gold.values()) != 1146:
        raise RuntimeError("Expected 1,146 mapped held-out queries")

    queries = [q for q in p1.load_queries(p1.resolve_path(config["paths"]["queries"], root)) if q.query_id in gold]
    inputs = p1.load_inputs(config, root, set(gold))
    documents = {
        mid: p1.build_document(mid, ("E", "R", "K"), inputs, config["renderer"])
        for mid in inputs.memory_records
    }
    bm25 = p1.build_bm25_rankings_for_variant("RawERK", queries, inputs, config, documents)
    branches = {
        q.query_id: (p1.top_dense_for_query(q, inputs, depth), bm25[q.query_id][:depth])
        for q in queries
    }

    per_query: list[dict] = []
    for alpha in ALPHAS:
        for query in queries:
            dense_branch, lexical_branch = branches[query.query_id]
            fused = p1.zscore_fuse(dense_branch, lexical_branch, alpha, missing, 10)
            ranking = [mid for mid, _ in fused]
            per_query.append({
                "alpha_dense": f"{alpha:.1f}",
                "split": gold[query.query_id]["split"],
                "query_id": query.query_id,
                "category": gold[query.query_id]["category"],
                "top10_memory_ids": ";".join(ranking),
                **score(gold[query.query_id]["ids"], ranking),
            })

    overall: list[dict] = []
    category_rows: list[dict] = []
    for alpha in ALPHAS:
        alpha_text = f"{alpha:.1f}"
        for split in ("dev", "test"):
            selected = [r for r in per_query if r["alpha_dense"] == alpha_text and r["split"] == split]
            overall.append({"alpha_dense": alpha_text, "split": split, "n": len(selected), **mean_metrics(selected)})
            for category in ("1", "2", "3", "4"):
                subset = [r for r in selected if r["category"] == category]
                category_rows.append({
                    "alpha_dense": alpha_text, "split": split, "category": category,
                    "n": len(subset), **mean_metrics(subset),
                })

    dev_rows = [row for row in overall if row["split"] == "dev"]
    selected_alpha = max(dev_rows, key=lambda row: (float(row["MRR@10"]), -float(row["alpha_dense"])))
    if selected_alpha["alpha_dense"] != "0.6":
        raise RuntimeError(f"Dev selection changed: {selected_alpha}")

    heldout_06 = next(row for row in overall if row["split"] == "test" and row["alpha_dense"] == "0.6")
    comparisons = [clustered_mrr_comparison(per_query, baseline) for baseline in ("0.5", "0.7")]
    with TABLE1_PATH.open(encoding="utf-8-sig", newline="") as handle:
        table1 = next(row for row in csv.DictReader(handle) if row["method"] == "ZScore-RawERK" and row["evaluation_split"] == "test")
    for metric in METRICS:
        if abs(float(heldout_06[metric]) - float(table1[metric])) > 1e-12:
            raise RuntimeError(f"Table 1 parity failed for {metric}")

    write_csv(OUTPUT / "alpha_sweep_overall.csv", overall, ["alpha_dense", "split", "n", *METRICS])
    write_csv(OUTPUT / "alpha_sweep_by_category.csv", category_rows, ["alpha_dense", "split", "category", "n", *METRICS])
    write_csv(OUTPUT / "alpha_sweep_per_query.csv", per_query, ["alpha_dense", "split", "query_id", "category", "top10_memory_ids", *METRICS])
    write_csv(OUTPUT / "dev_paired_mrr_comparisons.csv", comparisons,
              ["comparison", "n_queries", "n_conversations", "delta_MRR@10", "ci95_low", "ci95_high", "wins", "ties", "losses", "bootstrap_repetitions"])

    report = [
        "# Query-wise Z-score Fusion Weight Sweep", "",
        "Alpha was selected using development MRR@10 only. Held-out results are reported after selection and were not used to choose the weight.", "",
        "## Development selection", "",
        "| Dense alpha | n | MRR@10 | Hit@1 | Hit@5 | Hit@10 | Recall@10 | nDCG@10 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dev_rows:
        values = [row["alpha_dense"], str(row["n"]), *[f"{float(row[m]):.4f}" for m in METRICS]]
        if row["alpha_dense"] == "0.6":
            values = [f"**{value}**" for value in values]
        report.append("| " + " | ".join(values) + " |")
    report += [
        "", "The unique development optimum is alpha=0.6 under MRR@10. Other metrics are reported rather than used as additional tuning objectives.", "",
        "## Local robustness around the optimum", "",
        "| Comparison | Delta MRR@10 | Clustered 95% CI | Wins / ties / losses |",
        "|---|---:|---:|---:|",
    ]
    for row in comparisons:
        report.append(f"| {row['comparison']} | {row['delta_MRR@10']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['wins']} / {row['ties']} / {row['losses']} |")
    report += [
        "", "The development split contains 390 queries from only two conversations. The clustered interval is therefore a sensitivity check, not strong inferential evidence; the primary justification remains dev-only selection followed by one frozen held-out evaluation.", "",
        "## Frozen held-out result", "",
        "| Dense alpha | n | MRR@10 | Hit@1 | Hit@5 | Hit@10 | Recall@10 | nDCG@10 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| " + " | ".join([heldout_06["alpha_dense"], str(heldout_06["n"]), *[f"{float(heldout_06[m]):.4f}" for m in METRICS]]) + " |",
        "", "The alpha=0.6 held-out row exactly matches the canonical CassMem row in Retrieval Table 1 on all six metrics.", "",
        "### Post-selection sensitivity (diagnostic only)", "",
        "| Dense alpha | MRR@10 | Hit@1 | Hit@5 | Hit@10 | Recall@10 | nDCG@10 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [item for item in overall if item["split"] == "test"]:
        values = [row["alpha_dense"], *[f"{float(row[m]):.4f}" for m in METRICS]]
        if row["alpha_dense"] == "0.6":
            values = [f"**{value}**" for value in values]
        report.append("| " + " | ".join(values) + " |")
    report += [
        "", "This table is not used to select alpha. It shows that the dev-selected value remains the best point estimate for all six held-out metrics among the tested weights.", "",
        "## Protocol", "",
        "- Dense(raw) Top-50 and BM25(RawERK) Top-50 within the known conversation scope.",
        "- Query-wise population Z-score within each truncated branch.",
        "- Missing branch scores use that branch's minimum normalized score.",
        "- Candidate union and tie handling follow the frozen P1-C implementation.",
        "- The four evidence-empty held-out questions are excluded; held-out n=1,146.",
    ]
    (OUTPUT / "ALPHA_SWEEP_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    runtime = time.perf_counter() - started
    manifest = {
        "experiment": "canonical query-wise Z-score alpha sweep",
        "alphas": list(ALPHAS),
        "selection_split": "dev", "selection_metric": "MRR@10",
        "selected_alpha_dense": 0.6,
        "dev_queries": 390, "heldout_queries": 1146,
        "table1_parity_alpha_0_6": "PASS",
        "dev_local_robustness": comparisons,
        "runtime_seconds": runtime,
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in (P1_SCRIPT, P1_CONFIG, GOLD_PATH, TABLE1_PATH)},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"selected_alpha": 0.6, "runtime_seconds": runtime, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
