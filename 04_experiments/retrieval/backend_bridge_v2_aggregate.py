#!/usr/bin/env python3
"""Aggregate the frozen LoCoMo backend-bridge v2 experiment.

This program is deliberately evaluation-only.  It never connects to Cassandra
or Neo4j and it never generates a ranking.  It consumes the frozen runner
contract under::

    05_reports/backend_equivalence_v2/runs/{csv,cassandra,neo4j}/
        top10.csv
        per_query_metrics.csv
        summary.csv
        input_scope_digests.csv
        manifest.json

The long-form ``top10.csv`` schema is:
``backend,method,query_id,category,split,rank,memory_id,score``.  Scope digest
files may use either ``candidate_sha256`` or ``candidate_digest`` and likewise
``projection_sha256``/``projection_digest`` and ``edge_sha256``/``edge_digest``.

The evaluator recomputes every retrieval metric from the rankings and canonical
gold.  Runner-provided metrics are inputs to an audit, not authoritative values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "05_reports" / "backend_equivalence_v2" / "runs"
DEFAULT_OUTPUT = ROOT / "05_reports" / "backend_equivalence_v2"
DEFAULT_GOLD = (
    ROOT
    / "02_artifacts"
    / "retrieval_gold_v2"
    / "locomo_cat1_4_gold_memory.csv"
)

BACKENDS = ("csv", "cassandra", "neo4j")
METHODS = (
    "BM25",
    "Dense-bge",
    "Dense+GlobalKG",
    "RRF_compact",
    "ZScore-Raw",
    "ZScore-RawERK",
)
METRICS = ("MRR@10", "Hit@1", "Hit@5", "Hit@10", "Recall@10", "nDCG@10")
REQUIRED_RUN_FILES = (
    "top10.csv",
    "per_query_metrics.csv",
    "summary.csv",
    "input_scope_digests.csv",
    "manifest.json",
)

FROZEN_REFERENCES = {
    "BM25": ROOT / "05_reports" / "official_eval" / "bm25_raw_ranking_canonical1540.csv",
    "Dense-bge": ROOT / "05_reports" / "official_eval" / "dense_bge_ranking_canonical1540.csv",
    "Dense+GlobalKG": (
        ROOT / "05_reports" / "dense_global_kg_rerun" / "dense_global_kg_top10.csv"
    ),
    "RRF_compact": (
        ROOT
        / "05_reports"
        / "official_eval"
        / "rrf_compact_canonical1540"
        / "rrf_compact_top10.csv"
    ),
    "ZScore-Raw": ROOT / "05_reports" / "official_eval" / "zscore_raw_ranking_canonical1540.csv",
    "ZScore-RawERK": (
        ROOT / "05_reports" / "official_eval" / "zscore_rawerk_ranking_canonical1540.csv"
    ),
}

METHOD_ALIASES = {
    "bm25": "BM25",
    "densebge": "Dense-bge",
    "denseglobalkg": "Dense+GlobalKG",
    "rrfcompact": "RRF_compact",
    "zscoreraw": "ZScore-Raw",
    "zscorerawerk": "ZScore-RawERK",
    "cassmem": "ZScore-RawERK",
}


def normalized_token(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def canonical_method(value: str) -> str:
    token = normalized_token(value)
    if token not in METHOD_ALIASES:
        raise ValueError(f"Unknown retrieval method {value!r}")
    return METHOD_ALIASES[token]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def first(row: Mapping[str, Any], names: Iterable[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def parse_gold(value: str) -> list[str]:
    return list(dict.fromkeys(part.strip() for part in value.split(";") if part.strip()))


def require_inputs(input_dir: Path, gold_path: Path) -> list[Path]:
    paths = [gold_path, *FROZEN_REFERENCES.values()]
    paths.extend(input_dir / backend / name for backend in BACKENDS for name in REQUIRED_RUN_FILES)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Backend bridge aggregation was not run because required frozen inputs "
            f"are missing:\n{rendered}"
        )
    return paths


def load_gold(path: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    questions: dict[str, dict[str, Any]] = {}
    empty_gold: set[str] = set()
    for row in read_csv(path):
        query_id = first(row, ("query_id", "qa_id"))
        if not query_id or query_id in questions:
            raise ValueError(f"Missing or duplicate query_id in {path}: {query_id!r}")
        gold_ids = parse_gold(first(row, ("gold_memory_ids", "gold_ids")))
        questions[query_id] = {
            "category": first(row, ("category",)),
            "split": first(row, ("split",)),
            "conversation_id": first(row, ("conversation_id", "sample_id", "scope_id")),
            "gold_ids": gold_ids,
        }
        if not gold_ids:
            empty_gold.add(query_id)
    return questions, empty_gold


def load_rankings(path: Path, expected_backend: str | None = None) -> dict[str, dict[str, list[dict[str, Any]]]]:
    rows = read_csv(path)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        backend = first(row, ("backend",), expected_backend or "")
        if expected_backend and backend.lower() != expected_backend:
            raise ValueError(f"{path}: row backend {backend!r} != directory {expected_backend!r}")
        method = canonical_method(first(row, ("method",)))
        query_id = first(row, ("query_id", "qa_id", "question_id"))
        memory_id = first(row, ("memory_id",))
        try:
            rank = int(float(first(row, ("rank",))))
        except ValueError as exc:
            raise ValueError(f"{path}: invalid rank in row {row}") from exc
        key = (method, query_id, rank)
        if not query_id or not memory_id or key in seen:
            raise ValueError(f"{path}: blank ID or duplicate method/query/rank {key}")
        if not 1 <= rank <= 10:
            raise ValueError(f"{path}: rank outside 1..10 for {key}")
        seen.add(key)
        grouped[method][query_id].append(
            {
                "rank": rank,
                "memory_id": memory_id,
                "score": first(row, ("score", "final_score")),
                "category": first(row, ("category",)),
                "split": first(row, ("split",)),
            }
        )
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for method, query_rows in grouped.items():
        output[method] = {}
        for query_id, items in query_rows.items():
            items.sort(key=lambda item: item["rank"])
            ranks = [item["rank"] for item in items]
            ids = [item["memory_id"] for item in items]
            if ranks != list(range(1, 11)) or len(ids) != len(set(ids)):
                raise ValueError(f"{path}: {method}/{query_id} is not a unique complete Top-10")
            output[method][query_id] = items
    return output


def load_frozen_reference(path: Path, method: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(path):
        query_id = first(row, ("query_id", "qa_id"))
        rank = int(float(first(row, ("rank",))))
        if rank > 10:
            continue
        grouped[query_id].append(
            {
                "rank": rank,
                "memory_id": first(row, ("memory_id",)),
                "score": first(row, ("score", "final_score")),
            }
        )
    result = {}
    for query_id, items in grouped.items():
        items.sort(key=lambda item: item["rank"])
        if [item["rank"] for item in items] != list(range(1, 11)):
            raise ValueError(f"Frozen reference {method}/{query_id} is not a complete Top-10")
        result[query_id] = items
    return result


def metric_row(gold_ids: list[str], items: list[dict[str, Any]]) -> dict[str, float]:
    gold = set(gold_ids)
    ranked = [item["memory_id"] for item in items[:10]]
    relevant_ranks = [rank for rank, memory_id in enumerate(ranked, 1) if memory_id in gold]
    first_rank = min(relevant_ranks) if relevant_ranks else None
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), 10) + 1))
    return {
        "MRR@10": 0.0 if first_rank is None else 1.0 / first_rank,
        "Hit@1": float(first_rank is not None and first_rank <= 1),
        "Hit@5": float(first_rank is not None and first_rank <= 5),
        "Hit@10": float(first_rank is not None and first_rank <= 10),
        "Recall@10": len(set(ranked) & gold) / len(gold),
        "nDCG@10": 0.0 if not idcg else dcg / idcg,
    }


def summarize_metrics(
    rankings: Mapping[str, list[dict[str, Any]]],
    questions: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    per_query = {
        query_id: metric_row(questions[query_id]["gold_ids"], items)
        for query_id, items in rankings.items()
        if query_id in questions and questions[query_id]["gold_ids"]
    }
    if not per_query:
        raise ValueError("No evaluable queries in ranking")
    summary = {
        metric: sum(row[metric] for row in per_query.values()) / len(per_query)
        for metric in METRICS
    }
    return summary, per_query


def load_scope_digests(path: Path, backend: str) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(path):
        row_backend = first(row, ("backend",), backend).lower()
        if row_backend != backend:
            raise ValueError(f"{path}: backend mismatch {row_backend!r}")
        method_value = first(row, ("method",), "__shared__")
        method = "__shared__" if method_value == "__shared__" else canonical_method(method_value)
        scope_id = first(row, ("scope_id", "conversation_id", "sample_id"))
        key = (method, scope_id)
        if not scope_id or key in result:
            raise ValueError(f"{path}: blank or duplicate scope digest {key}")
        normalized = {
            "candidate_count": first(row, ("candidate_count",)),
            "candidate_digest": first(row, ("candidate_sha256", "candidate_digest")),
            "projection_digest": first(row, ("projection_sha256", "projection_digest")),
            "corpus_order_digest": first(row, ("corpus_order_sha256", "corpus_order_digest")),
            "edge_row_count": first(row, ("edge_row_count",)),
            "edge_assignment_count": first(row, ("edge_assignment_count",)),
            "edge_digest": first(row, ("edge_sha256", "edge_digest")),
        }
        required_digests = (
            "candidate_digest",
            "projection_digest",
            "corpus_order_digest",
            "edge_digest",
        )
        missing_digests = [field for field in required_digests if not normalized[field]]
        if missing_digests:
            raise ValueError(
                f"{path}: {method}/{scope_id} is missing required digest fields "
                f"{missing_digests}; hash an empty edge set instead of writing a blank digest"
            )
        result[key] = normalized
    return result


def compare_backend_rankings(
    all_rankings: Mapping[str, Mapping[str, Mapping[str, list[dict[str, Any]]]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    reference_backend = "csv"
    for method in METHODS:
        expected = all_rankings[reference_backend].get(method, {})
        for backend in ("cassandra", "neo4j"):
            actual = all_rankings[backend].get(method, {})
            shared = sorted(set(expected) & set(actual))
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            exact_queries = 0
            rank_mismatch_count = 0
            for query_id in shared:
                expected_ids = [row["memory_id"] for row in expected[query_id]]
                actual_ids = [row["memory_id"] for row in actual[query_id]]
                if expected_ids == actual_ids:
                    exact_queries += 1
                    continue
                for rank, (expected_id, actual_id) in enumerate(zip(expected_ids, actual_ids), 1):
                    if expected_id != actual_id:
                        rank_mismatch_count += 1
                        mismatches.append(
                            {
                                "comparison_type": "backend_top10",
                                "method": method,
                                "query_id": query_id,
                                "scope_id": "",
                                "backend": backend,
                                "reference_backend": reference_backend,
                                "field": "memory_id",
                                "rank": rank,
                                "expected": expected_id,
                                "actual": actual_id,
                                "status": "MISMATCH",
                                "detail": "Top-10 rank differs from CSV",
                            }
                        )
            for query_id, status in [(qid, "MISSING") for qid in missing] + [(qid, "EXTRA") for qid in extra]:
                mismatches.append(
                    {
                        "comparison_type": "backend_query_coverage",
                        "method": method,
                        "query_id": query_id,
                        "scope_id": "",
                        "backend": backend,
                        "reference_backend": reference_backend,
                        "field": "query_id",
                        "rank": "",
                        "expected": "present" if status == "MISSING" else "absent",
                        "actual": "absent" if status == "MISSING" else "present",
                        "status": status,
                        "detail": "Backend and CSV query universes differ",
                    }
                )
            summaries.append(
                {
                    "method": method,
                    "reference_backend": reference_backend,
                    "backend": backend,
                    "csv_queries": len(expected),
                    "backend_queries": len(actual),
                    "shared_queries": len(shared),
                    "exact_top10_queries": exact_queries,
                    "exact_top10_rate": exact_queries / len(shared) if shared else 0.0,
                    "rank_mismatches": rank_mismatch_count,
                    "missing_queries": len(missing),
                    "extra_queries": len(extra),
                    "status": "PASS" if len(expected) == len(actual) == exact_queries else "FAIL",
                }
            )
    return summaries, mismatches


def compare_scope_digests(
    digests: Mapping[str, Mapping[tuple[str, str], Mapping[str, str]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = (
        "candidate_count",
        "candidate_digest",
        "projection_digest",
        "corpus_order_digest",
        "edge_row_count",
        "edge_assignment_count",
        "edge_digest",
    )
    summaries: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for backend in ("cassandra", "neo4j"):
        expected = digests["csv"]
        actual = digests[backend]
        for field in fields:
            comparable = matched = missing_value = 0
            for key in sorted(set(expected) | set(actual)):
                expected_value = expected.get(key, {}).get(field, "")
                actual_value = actual.get(key, {}).get(field, "")
                if not expected_value and not actual_value:
                    continue
                comparable += 1
                if expected_value and actual_value and expected_value == actual_value:
                    matched += 1
                    continue
                if not expected_value or not actual_value:
                    missing_value += 1
                method, scope_id = key
                mismatches.append(
                    {
                        "comparison_type": "backend_input_digest",
                        "method": method,
                        "query_id": "",
                        "scope_id": scope_id,
                        "backend": backend,
                        "reference_backend": "csv",
                        "field": field,
                        "rank": "",
                        "expected": expected_value,
                        "actual": actual_value,
                        "status": "MISSING" if not expected_value or not actual_value else "MISMATCH",
                        "detail": "Frozen backend input scope differs",
                    }
                )
            summaries.append(
                {
                    "backend": backend,
                    "field": field,
                    "comparable_scopes": comparable,
                    "matched_scopes": matched,
                    "match_rate": matched / comparable if comparable else "",
                    "missing_values": missing_value,
                    "status": "N/A" if not comparable else ("PASS" if matched == comparable else "FAIL"),
                }
            )
    return summaries, mismatches


def compare_frozen_references(
    csv_rankings: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    references: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    questions: Mapping[str, Mapping[str, Any]],
    empty_gold: set[str],
    all_rankings: Mapping[str, Mapping[str, Mapping[str, list[dict[str, Any]]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    empty_rows: list[dict[str, Any]] = []
    for method in METHODS:
        actual = csv_rankings.get(method, {})
        expected = references[method]
        shared = sorted(set(actual) & set(expected))
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        expected_empty_extras = sorted(set(extra) & empty_gold) if method == "Dense+GlobalKG" else []
        unexpected_extra = sorted(set(extra) - set(expected_empty_extras))
        exact = 0
        rank_mismatch_count = 0
        for query_id in shared:
            expected_ids = [row["memory_id"] for row in expected[query_id]]
            actual_ids = [row["memory_id"] for row in actual[query_id]]
            if expected_ids == actual_ids:
                exact += 1
            else:
                for rank, (expected_id, actual_id) in enumerate(zip(expected_ids, actual_ids), 1):
                    if expected_id != actual_id:
                        rank_mismatch_count += 1
                        mismatches.append(
                            {
                                "comparison_type": "csv_frozen_reference",
                                "method": method,
                                "query_id": query_id,
                                "scope_id": "",
                                "backend": "csv",
                                "reference_backend": "frozen_reference",
                                "field": "memory_id",
                                "rank": rank,
                                "expected": expected_id,
                                "actual": actual_id,
                                "status": "MISMATCH",
                                "detail": "CSV adapter differs from frozen Top-10",
                            }
                        )
        for query_id in missing + unexpected_extra:
            status = "MISSING" if query_id in missing else "EXTRA"
            mismatches.append(
                {
                    "comparison_type": "csv_frozen_query_coverage",
                    "method": method,
                    "query_id": query_id,
                    "scope_id": "",
                    "backend": "csv",
                    "reference_backend": "frozen_reference",
                    "field": "query_id",
                    "rank": "",
                    "expected": "present" if status == "MISSING" else "absent",
                    "actual": "absent" if status == "MISSING" else "present",
                    "status": status,
                    "detail": "Unexpected CSV/frozen query-universe difference",
                }
            )
        for query_id in expected_empty_extras:
            csv_ids = [row["memory_id"] for row in all_rankings["csv"][method][query_id]]
            cassandra_ids = [
                row["memory_id"]
                for row in all_rankings["cassandra"].get(method, {}).get(query_id, [])
            ]
            neo4j_ids = [
                row["memory_id"]
                for row in all_rankings["neo4j"].get(method, {}).get(query_id, [])
            ]
            empty_rows.append(
                {
                    "method": method,
                    "query_id": query_id,
                    "category": questions[query_id]["category"],
                    "csv_top10_present": True,
                    "frozen_reference_present": False,
                    "gold_status": "evidence-empty",
                    "cassandra_top10_exact_csv": cassandra_ids == csv_ids,
                    "neo4j_top10_exact_csv": neo4j_ids == csv_ids,
                    "backend_parity_status": (
                        "PASS"
                        if cassandra_ids == csv_ids and neo4j_ids == csv_ids
                        else "FAIL"
                    ),
                    "interpretation": "Expected: corrected frozen Dense+GlobalKG artifact excludes empty-gold queries",
                }
            )
        status = (
            "PASS_WITH_EXPECTED_EMPTY_GOLD_CAVEAT"
            if not missing and not unexpected_extra and exact == len(shared) and expected_empty_extras
            else "PASS"
            if not missing and not unexpected_extra and exact == len(shared)
            else "FAIL"
        )
        summaries.append(
            {
                "method": method,
                "frozen_reference_queries": len(expected),
                "csv_queries": len(actual),
                "shared_queries": len(shared),
                "exact_top10_queries": exact,
                "exact_top10_rate": exact / len(shared) if shared else 0.0,
                "rank_mismatches": rank_mismatch_count,
                "missing_csv_queries": len(missing),
                "unexpected_extra_csv_queries": len(unexpected_extra),
                "expected_empty_gold_extra_queries": len(expected_empty_extras),
                "status": status,
                "reference_path": str(FROZEN_REFERENCES[method]),
                "reference_sha256": sha256(FROZEN_REFERENCES[method]),
            }
        )
    return summaries, mismatches, empty_rows


def build_metric_tables(
    all_rankings: Mapping[str, Mapping[str, Mapping[str, list[dict[str, Any]]]]],
    questions: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    summary_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for backend in BACKENDS:
        for method in METHODS:
            rankings = all_rankings[backend].get(method, {})
            summary, per_query = summarize_metrics(rankings, questions)
            summaries[backend][method] = summary
            summary_rows.append(
                {
                    "backend": backend,
                    "method": method,
                    "n_queries": len(rankings),
                    "n_evaluable": len(per_query),
                    **summary,
                }
            )
    for backend in ("cassandra", "neo4j"):
        for method in METHODS:
            for metric in METRICS:
                expected = summaries["csv"][method][metric]
                actual = summaries[backend][method][metric]
                difference_rows.append(
                    {
                        "backend": backend,
                        "reference_backend": "csv",
                        "method": method,
                        "metric": metric,
                        "csv_value": expected,
                        "backend_value": actual,
                        "absolute_difference": actual - expected,
                        "status": "PASS" if abs(actual - expected) <= 1e-12 else "FAIL",
                    }
                )
    return summary_rows, difference_rows, summaries


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(
    backend_summary: list[dict[str, Any]],
    digest_summary: list[dict[str, Any]],
    metric_summary: list[dict[str, Any]],
    metric_differences: list[dict[str, Any]],
    reference_summary: list[dict[str, Any]],
    empty_rows: list[dict[str, Any]],
    mismatch_count: int,
) -> str:
    backend_pass = all(row["status"] == "PASS" for row in backend_summary)
    digest_pass = all(row["status"] in {"PASS", "N/A"} for row in digest_summary)
    metric_pass = all(row["status"] == "PASS" for row in metric_differences)
    reference_pass = all(str(row["status"]).startswith("PASS") for row in reference_summary)
    overall = "PASS" if backend_pass and digest_pass and metric_pass and reference_pass else "FAIL"
    lines = [
        "# Backend Bridge v2 Report",
        "",
        f"**Overall gate: {overall}.** This report separates storage-input parity, exact ranking parity, retrieval-metric parity, and agreement with frozen CSV references.",
        "",
        "## Protocol",
        "",
        "All six methods use the canonical Cat1-4 query universe (1,540 questions). Retrieval effectiveness excludes the four official evidence-empty questions and therefore uses 1,536 questions. MRR is truncated at 10; Recall@10 is macro-averaged relevant-memory recall, not the historical Hit@10 label; nDCG@10 uses binary relevance.",
        "",
        "## Exact backend Top-10 parity",
        "",
        md_table(
            ["Method", "Backend", "Shared", "Exact", "Rate", "Rank mismatches", "Status"],
            [[row["method"], row["backend"], row["shared_queries"], row["exact_top10_queries"], row["exact_top10_rate"], row["rank_mismatches"], row["status"]] for row in backend_summary],
        ),
        "",
        "## Retrieval metrics",
        "",
        md_table(
            ["Backend", "Method", "n", "MRR@10", "Hit@10", "Recall@10", "nDCG@10"],
            [[row["backend"], row["method"], row["n_evaluable"], row["MRR@10"], row["Hit@10"], row["Recall@10"], row["nDCG@10"]] for row in metric_summary],
        ),
        "",
        "## CSV adapter versus frozen references",
        "",
        md_table(
            ["Method", "Frozen q", "CSV q", "Exact shared", "Expected empty-gold extras", "Status"],
            [[row["method"], row["frozen_reference_queries"], row["csv_queries"], row["exact_top10_queries"], row["expected_empty_gold_extra_queries"], row["status"]] for row in reference_summary],
        ),
        "",
        "Dense+GlobalKG uses the corrected degree-centrality ranking in `05_reports/dense_global_kg_rerun/dense_global_kg_top10.csv`. That frozen artifact intentionally contains only the 1,536 evidence-bearing questions. The four evidence-empty rankings are tested for CSV/Cassandra/Neo4j parity but have no retrieval-effectiveness metric.",
        "",
        "## Input scope digests",
        "",
        md_table(
            ["Backend", "Field", "Comparable scopes", "Matched", "Status"],
            [[row["backend"], row["field"], row["comparable_scopes"], row["matched_scopes"], row["status"]] for row in digest_summary],
        ),
        "",
        "## Gate interpretation",
        "",
        f"The detailed mismatch ledger contains **{mismatch_count}** rows. A publication claim of lossless backend migration requires every non-N/A digest gate, every exact Top-10 gate, and every metric-difference gate to pass. Missing digest values are not silently treated as matches.",
        "",
        "Machine-readable evidence is in `backend_parity_summary.csv`, `input_digest_parity_summary.csv`, `retrieval_metrics_by_backend.csv`, `retrieval_metric_differences.csv`, `csv_frozen_reference_summary.csv`, `densekg_empty_gold_ranking_parity.csv`, `mismatches.csv`, and `manifest.json`.",
        "",
    ]
    if empty_rows:
        lines.extend([f"Dense+GlobalKG expected empty-gold reference exclusions recorded: {len(empty_rows)}.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    args = parser.parse_args()

    input_paths = require_inputs(args.input_dir, args.gold)
    questions, empty_gold = load_gold(args.gold)
    if len(questions) != 1540 or len(empty_gold) != 4:
        raise ValueError(
            f"Canonical gold gate failed: expected 1540 queries/4 empty gold, got "
            f"{len(questions)}/{len(empty_gold)}"
        )

    all_rankings = {
        backend: load_rankings(args.input_dir / backend / "top10.csv", backend)
        for backend in BACKENDS
    }
    for backend in BACKENDS:
        missing_methods = sorted(set(METHODS) - set(all_rankings[backend]))
        if missing_methods:
            raise ValueError(f"{backend} top10.csv is missing methods: {missing_methods}")
        for method in METHODS:
            query_ids = set(all_rankings[backend][method])
            if query_ids != set(questions):
                raise ValueError(
                    f"{backend}/{method} query universe differs from canonical 1540: "
                    f"missing={len(set(questions)-query_ids)}, extra={len(query_ids-set(questions))}"
                )

    scope_digests = {
        backend: load_scope_digests(args.input_dir / backend / "input_scope_digests.csv", backend)
        for backend in BACKENDS
    }
    references = {
        method: load_frozen_reference(path, method)
        for method, path in FROZEN_REFERENCES.items()
    }

    backend_summary, backend_mismatches = compare_backend_rankings(all_rankings)
    digest_summary, digest_mismatches = compare_scope_digests(scope_digests)
    reference_summary, reference_mismatches, empty_rows = compare_frozen_references(
        all_rankings["csv"], references, questions, empty_gold, all_rankings
    )
    metric_summary, metric_differences, _ = build_metric_tables(all_rankings, questions)
    mismatches = backend_mismatches + digest_mismatches + reference_mismatches

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    output_specs = [
        ("backend_parity_summary.csv", backend_summary, list(backend_summary[0])),
        ("input_digest_parity_summary.csv", digest_summary, list(digest_summary[0])),
        ("retrieval_metrics_by_backend.csv", metric_summary, list(metric_summary[0])),
        ("retrieval_metric_differences.csv", metric_differences, list(metric_differences[0])),
        ("csv_frozen_reference_summary.csv", reference_summary, list(reference_summary[0])),
        (
            "densekg_empty_gold_ranking_parity.csv",
            empty_rows,
            ["method", "query_id", "category", "csv_top10_present", "frozen_reference_present", "gold_status", "cassandra_top10_exact_csv", "neo4j_top10_exact_csv", "backend_parity_status", "interpretation"],
        ),
        (
            "mismatches.csv",
            mismatches,
            ["comparison_type", "method", "query_id", "scope_id", "backend", "reference_backend", "field", "rank", "expected", "actual", "status", "detail"],
        ),
    ]
    for name, rows, fields in output_specs:
        path = args.output_dir / name
        write_csv(path, rows, fields)
        outputs.append(path)

    report_path = args.output_dir / "BACKEND_BRIDGE_V2_REPORT.md"
    report_path.write_text(
        render_report(
            backend_summary,
            digest_summary,
            metric_summary,
            metric_differences,
            reference_summary,
            empty_rows,
            len(mismatches),
        ),
        encoding="utf-8",
    )
    outputs.append(report_path)

    overall_status = "PASS" if (
        all(row["status"] == "PASS" for row in backend_summary)
        and all(row["status"] in {"PASS", "N/A"} for row in digest_summary)
        and all(row["status"] == "PASS" for row in metric_differences)
        and all(str(row["status"]).startswith("PASS") for row in reference_summary)
    ) else "FAIL"
    manifest = {
        "experiment": "LoCoMo Backend Bridge v2 aggregation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status,
        "protocol": {
            "backends": list(BACKENDS),
            "methods": list(METHODS),
            "ranking_cutoff": 10,
            "query_universe": 1540,
            "evaluable_queries": 1536,
            "evidence_empty_queries": 4,
            "reference_backend": "csv",
            "metric_definitions": {
                "MRR@10": "reciprocal rank of first relevant memory in Top-10, else zero",
                "Hit@K": "at least one relevant memory in Top-K",
                "Recall@10": "macro mean of unique relevant memories retrieved divided by gold count",
                "nDCG@10": "binary-relevance DCG normalized by min(gold count, 10) ideal ranks",
            },
            "dense_global_kg_caveat": (
                "Corrected frozen reference covers 1536 evidence-bearing queries; the four "
                "empty-gold rankings are evaluated only for cross-backend ranking parity."
            ),
        },
        "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in input_paths],
        "outputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in outputs],
        "counts": {
            "mismatch_rows": len(mismatches),
            "backend_parity_gates": len(backend_summary),
            "digest_gates": len(digest_summary),
            "metric_difference_gates": len(metric_differences),
            "reference_gates": len(reference_summary),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": overall_status, "output_dir": str(args.output_dir), "mismatches": len(mismatches)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
