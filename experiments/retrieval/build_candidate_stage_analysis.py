"""Build the frozen candidate-stage and ranking-loss analysis for CassMem.

This analysis does not invent a graph-expansion stage.  It evaluates the actual
candidate pools consumed by each frozen ranking pipeline: Top-100 for the
single-branch methods and the union of the frozen Top-50 dense/sparse branches
for the fusion methods.  Cat5 is excluded because it has no positive retrieval
target in the LoCoMo protocol.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "retrieval" / "candidate_stage_analysis_v1"
GOLD = ROOT / "data" / "retrieval_gold" / "locomo_cat1_4_gold_memory.csv"
MEMORY = ROOT / "data" / "locomo_memory_records.csv"
POOLS = ROOT / "results" / "backend_equivalence" / "backend_equivalence_v2" / "runs" / "csv" / "diagnostic_pools.csv"
TOP10 = ROOT / "results" / "backend_equivalence" / "backend_equivalence_v2" / "runs" / "csv" / "top10.csv"

METHODS = (
    "BM25",
    "Dense-bge",
    "Dense+GlobalKG",
    "RRF_compact",
    "ZScore-Raw",
    "ZScore-RawERK",
)
CATEGORY_LABELS = {"1": "Multi-Hop", "2": "Temporal", "3": "Open-Domain", "4": "Single-Hop"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_ids(value: str) -> list[str]:
    return [item for item in (value or "").split(";") if item]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gold_rows = read_csv(GOLD)
    memory_rows = read_csv(MEMORY)
    pool_rows = read_csv(POOLS)
    top10_rows = read_csv(TOP10)

    scope_sizes: dict[str, int] = defaultdict(int)
    for row in memory_rows:
        scope = row.get("sample_id") or row.get("conversation_id")
        if not scope:
            raise ValueError("memory corpus lacks sample_id/conversation_id")
        scope_sizes[scope] += 1

    gold_by_qid = {row["query_id"]: row for row in gold_rows}
    evaluable = {qid for qid, row in gold_by_qid.items() if split_ids(row["gold_memory_ids"])}
    if len(gold_by_qid) != 1540 or len(evaluable) != 1536:
        raise ValueError(f"unexpected gold universe: total={len(gold_by_qid)} evaluable={len(evaluable)}")

    pools: dict[tuple[str, str], dict] = {}
    for row in pool_rows:
        key = (row["method"], row["query_id"])
        if key in pools:
            raise ValueError(f"duplicate pool row: {key}")
        ids = split_ids(row["pool_ids"])
        if len(ids) != len(set(ids)) or len(ids) != int(row["pool_size"]):
            raise ValueError(f"invalid pool IDs: {key}")
        pools[key] = {**row, "ids": ids}

    top10: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in sorted(top10_rows, key=lambda item: (item["method"], item["query_id"], int(item["rank"]))):
        top10[(row["method"], row["query_id"])].append(row["memory_id"])

    per_query: list[dict] = []
    for method in METHODS:
        method_qids = {qid for m, qid in pools if m == method}
        if method_qids != set(gold_by_qid):
            raise ValueError(f"pool QID mismatch for {method}: {len(method_qids)}")
        for qid in sorted(evaluable):
            gold_row = gold_by_qid[qid]
            gold = set(split_ids(gold_row["gold_memory_ids"]))
            pool_row = pools[(method, qid)]
            pool_ids = pool_row["ids"]
            pool = set(pool_ids)
            ranked = top10[(method, qid)]
            if len(ranked) != 10 or len(ranked) != len(set(ranked)):
                raise ValueError(f"invalid Top-10: {(method, qid)}")
            if not set(ranked).issubset(pool):
                raise ValueError(f"Top-10 is not a subset of candidate pool: {(method, qid)}")
            candidate_found = len(gold & pool)
            top10_found = len(gold & set(ranked))
            scope = gold_row["conversation_id"]
            per_query.append(
                {
                    "method": method,
                    "query_id": qid,
                    "split": gold_row["split"],
                    "category": gold_row["category"],
                    "category_label": CATEGORY_LABELS[gold_row["category"]],
                    "pool_definition": pool_row["pool_definition"],
                    "conversation_pool_size": scope_sizes[scope],
                    "candidate_pool_size": len(pool_ids),
                    "candidate_reduction": 1.0 - len(pool_ids) / scope_sizes[scope],
                    "gold_count": len(gold),
                    "candidate_gold_count": candidate_found,
                    "candidate_hit": int(candidate_found > 0),
                    "candidate_all_gold": int(candidate_found == len(gold)),
                    "candidate_recall": candidate_found / len(gold),
                    "top10_gold_count": top10_found,
                    "top10_hit": int(top10_found > 0),
                    "top10_recall": top10_found / len(gold),
                    "ranking_loss": (candidate_found - top10_found) / len(gold),
                }
            )

    summary: list[dict] = []
    scopes = [("all", "Overall", None), ("dev", "Overall", None), ("test", "Overall", None)]
    scopes += [("all", CATEGORY_LABELS[category], category) for category in ("4", "1", "2", "3")]
    scopes += [("test", CATEGORY_LABELS[category], category) for category in ("4", "1", "2", "3")]
    for method in METHODS:
        for split, label, category in scopes:
            rows = [row for row in per_query if row["method"] == method]
            if split != "all":
                rows = [row for row in rows if row["split"] == split]
            if category is not None:
                rows = [row for row in rows if row["category"] == category]
            candidate_recall = mean([row["candidate_recall"] for row in rows])
            top10_recall = mean([row["top10_recall"] for row in rows])
            summary.append(
                {
                    "method": method,
                    "split": split,
                    "category": category or "Overall",
                    "category_label": label,
                    "n": len(rows),
                    "pool_definition": rows[0]["pool_definition"] if rows else "",
                    "avg_conversation_pool_size": mean([row["conversation_pool_size"] for row in rows]),
                    "avg_candidate_pool_size": mean([row["candidate_pool_size"] for row in rows]),
                    "candidate_reduction": mean([row["candidate_reduction"] for row in rows]),
                    "candidate_hit": mean([row["candidate_hit"] for row in rows]),
                    "candidate_all_gold": mean([row["candidate_all_gold"] for row in rows]),
                    "candidate_recall": candidate_recall,
                    "top10_hit": mean([row["top10_hit"] for row in rows]),
                    "top10_recall": top10_recall,
                    "ranking_loss": candidate_recall - top10_recall,
                    "ranking_retention": top10_recall / candidate_recall if candidate_recall else 0.0,
                }
            )

    per_query_fields = list(per_query[0])
    summary_fields = list(summary[0])
    write_csv(OUT / "candidate_stage_per_query.csv", per_query, per_query_fields)
    write_csv(OUT / "candidate_stage_summary.csv", summary, summary_fields)

    paper_rows = [row for row in summary if row["split"] == "test" and row["category"] == "Overall"]
    report = [
        "# Candidate-stage recall and ranking-loss decomposition",
        "",
        "## Protocol",
        "",
        "This evaluates the actual frozen candidate stage used by each method. It does not claim a graph-expansion stage. "
        "Single-branch methods use their frozen Top-100 diagnostic pool; fusion methods use the union of their frozen Top-50 branches. "
        "Retrieval metrics exclude the four evidence-empty Cat3 questions.",
        "",
        "## Held-out test results",
        "",
        "| Method | Candidate definition | n | Avg pool | Pool reduction | Candidate Hit | Candidate Recall | Top-10 Hit | Top-10 Recall | Ranking loss | Retention |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paper_rows:
        report.append(
            f"| {row['method']} | {row['pool_definition']} | {row['n']} | {row['avg_candidate_pool_size']:.1f} | "
            f"{row['candidate_reduction']:.3f} | {row['candidate_hit']:.3f} | {row['candidate_recall']:.3f} | "
            f"{row['top10_hit']:.3f} | {row['top10_recall']:.3f} | {row['ranking_loss']:.3f} | {row['ranking_retention']:.3f} |"
        )
    cassmem = next(row for row in paper_rows if row["method"] == "ZScore-RawERK")
    report += [
        "",
        "## Interpretation",
        "",
        f"- CassMem reduces the conversation-scoped pool by {cassmem['candidate_reduction'] * 100:.1f}% on average before final ranking.",
        f"- Its candidate-stage macro Recall is {cassmem['candidate_recall']:.3f}; final Top-10 Recall is {cassmem['top10_recall']:.3f}. "
        f"The {cassmem['ranking_loss']:.3f} absolute gap is attributable to ranking/truncation after candidate generation.",
        "- Candidate Recall and final Top-10 Recall answer different questions and must not be labeled interchangeably.",
        "- This table supports the existing Dense/BM25 fusion pipeline. It does not support a claim that Cassandra performs arbitrary KG expansion.",
        "",
    ]
    (OUT / "CANDIDATE_STAGE_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    outputs = [OUT / "candidate_stage_per_query.csv", OUT / "candidate_stage_summary.csv", OUT / "CANDIDATE_STAGE_REPORT.md"]
    manifest = {
        "status": "PASS",
        "protocol_id": "candidate-stage-analysis-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_calls": 0,
        "query_universe": 1540,
        "evaluable_queries": 1536,
        "heldout_test_queries": sum(1 for row in per_query if row["method"] == "BM25" and row["split"] == "test"),
        "methods": list(METHODS),
        "candidate_definition": "actual frozen Top-100 single-branch pool or union of frozen Top-50 fusion branches",
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in (GOLD, MEMORY, POOLS, TOP10)],
        "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in outputs],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(per_query), "summary_rows": len(summary), "output": str(OUT)}))


if __name__ == "__main__":
    main()
