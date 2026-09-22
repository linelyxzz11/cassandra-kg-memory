"""Freeze the empirical profile used by the LoCoMo-shaped trace replay.

This script does not synthesize observations.  It audits the canonical 5,882
memory / 1,986 QA inputs and writes a machine-readable profile plus a compact
human-readable report.  The scaled workload generator consumes the profile.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "serving_100k" / "locomo_workload_profile"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_json_list(value: str) -> list[str]:
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got {type(parsed).__name__}")
    return [str(item) for item in parsed]


def split_semicolon(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(";") if part.strip()]


EVIDENCE_RE = re.compile(r"D:?(\d+):0*(\d+)", re.IGNORECASE)


def normalize_evidence_items(items: list[str]) -> tuple[list[str], list[str]]:
    """Expand compound/typo-tolerant evidence strings without inventing IDs.

    Accepted normalizations are mechanical only: split embedded Dn:m tokens,
    remove the stray colon in D:n:m, and remove leading zeros in the turn.
    Any non-empty item yielding no token is retained as an unresolved fragment.
    """

    normalized: list[str] = []
    unresolved_fragments: list[str] = []
    for item in items:
        matches = EVIDENCE_RE.findall(item)
        if not matches and item.strip():
            unresolved_fragments.append(item.strip())
        normalized.extend(f"D{int(session)}:{int(turn)}" for session, turn in matches)
    return list(dict.fromkeys(normalized)), unresolved_fragments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    memory_path = ROOT / "data" / "locomo_memory_records.csv"
    qa_path = ROOT / "data" / "locomo_qa_records.csv"
    feature_path = ROOT / "data" / "frozen_retrieval" / "p3_memory_features.csv"
    memory_ids_path = ROOT / "data" / "locomo_memory_ids_bge.txt"
    qa_ids_path = ROOT / "data" / "locomo_qa_ids_bge.txt"

    memories = read_csv(memory_path)
    questions = read_csv(qa_path)
    features = read_csv(feature_path)
    if len(memories) != len(features):
        raise RuntimeError("Memory/features row counts differ")

    memory_by_key = {(row["sample_id"], row["dia_id"]): row for row in memories}
    feature_ids = {row["memory_id"] for row in features}
    memory_ids = {row["memory_id"] for row in memories}
    if feature_ids != memory_ids:
        raise RuntimeError("Memory/features ID sets differ")

    scope_lengths = Counter(row["sample_id"] for row in memories)
    scope_order = list(dict.fromkeys(row["sample_id"] for row in memories))
    session_lengths: dict[str, Counter[str]] = defaultdict(Counter)
    for row in memories:
        session_lengths[row["sample_id"]][row["session_id"]] += 1

    evidence_counts: list[int] = []
    unresolved_evidence: list[dict[str, str]] = []
    evidence_by_category: dict[str, list[int]] = defaultdict(list)
    eligible_freshness: list[dict[str, str]] = []
    for row in questions:
        raw_evidence = parse_json_list(row["evidence"])
        evidence, unresolved_fragments = normalize_evidence_items(raw_evidence)
        evidence_counts.append(len(evidence))
        evidence_by_category[row["category"]].append(len(evidence))
        resolved_memory_ids: list[str] = []
        unresolved_for_qa = list(unresolved_fragments)
        for dia_id in evidence:
            memory = memory_by_key.get((row["sample_id"], dia_id))
            if memory is None:
                unresolved_for_qa.append(dia_id)
            else:
                resolved_memory_ids.append(memory["memory_id"])
        for fragment in unresolved_for_qa:
            unresolved_evidence.append({"qa_id": row["qa_id"], "evidence": fragment})
        if row["category"] in {"1", "2", "3", "4"} and resolved_memory_ids and not unresolved_for_qa:
            eligible_freshness.append({
                "qa_id": row["qa_id"],
                "sample_id": row["sample_id"],
                "category": row["category"],
                "question": row["question"],
                "gold_memory_ids": json.dumps(resolved_memory_ids, ensure_ascii=False),
                "gold_count": str(len(resolved_memory_ids)),
            })

    triple_counts = [len(split_semicolon(row["triples"])) for row in features]
    kg_triple_counts = [count for count in triple_counts if count > 0]
    relation_frequency: Counter[str] = Counter()
    for row in features:
        relation_frequency.update(split_semicolon(row["relations"].replace(",", ";")))

    profile = {
        "profile_id": "locomo-shaped-workload-profile-v1",
        "terminology_guardrail": (
            "The scaled run is LoCoMo-shaped trace replay, not original 100K LoCoMo."
        ),
        "source": {
            "memory_csv": str(memory_path.relative_to(ROOT)).replace("\\", "/"),
            "qa_csv": str(qa_path.relative_to(ROOT)).replace("\\", "/"),
            "feature_csv": str(feature_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": {
                "memory_csv": sha256(memory_path),
                "qa_csv": sha256(qa_path),
                "feature_csv": sha256(feature_path),
                "memory_ids": sha256(memory_ids_path),
                "qa_ids": sha256(qa_ids_path),
            },
        },
        "counts": {
            "conversation_scopes": len(scope_lengths),
            "memories": len(memories),
            "questions": len(questions),
            "evidence_bearing_questions_after_normalization": sum(value > 0 for value in evidence_counts),
            "cat1_4_unambiguous_freshness_questions": len(eligible_freshness),
        },
        "conversation_scope_memory_count": {
            "by_scope": dict(sorted(scope_lengths.items())),
            "source_row_order": scope_order,
            "min": min(scope_lengths.values()),
            "median": statistics.median(scope_lengths.values()),
            "max": max(scope_lengths.values()),
        },
        "sessions_per_scope": {
            scope: len(counts) for scope, counts in sorted(session_lengths.items())
        },
        "qa_category_count": dict(sorted(Counter(row["category"] for row in questions).items())),
        "evidence_count": {
            "histogram": {str(k): v for k, v in sorted(Counter(evidence_counts).items())},
            "median": statistics.median(evidence_counts),
            "p95": percentile(evidence_counts, 0.95),
            "max": max(evidence_counts),
            "by_category_median": {
                category: statistics.median(values)
                for category, values in sorted(evidence_by_category.items())
            },
            "unresolved_reference_count": len(unresolved_evidence),
            "unresolved_references": unresolved_evidence,
        },
        "kg": {
            "covered_memories": len(kg_triple_counts),
            "coverage_fraction": len(kg_triple_counts) / len(features),
            "triples_per_all_memory_histogram": {
                str(k): v for k, v in sorted(Counter(triple_counts).items())
            },
            "triples_per_covered_memory_median": statistics.median(kg_triple_counts),
            "triples_per_covered_memory_p95": percentile(kg_triple_counts, 0.95),
            "triples_per_covered_memory_max": max(kg_triple_counts),
            "relation_frequency_top20": dict(relation_frequency.most_common(20)),
        },
        "embedding": {
            "memory_rows": len(memory_ids_path.read_text(encoding="utf-8-sig").splitlines()),
            "qa_rows": len(qa_ids_path.read_text(encoding="utf-8-sig").splitlines()),
            "dimension": 1024,
            "frozen": True,
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "locomo_workload_profile.json"
    json_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = [
        ("conversation_scopes", profile["counts"]["conversation_scopes"]),
        ("memories", profile["counts"]["memories"]),
        ("questions", profile["counts"]["questions"]),
        ("cat1_4_unambiguous_freshness_questions", profile["counts"]["cat1_4_unambiguous_freshness_questions"]),
        ("scope_memories_min", profile["conversation_scope_memory_count"]["min"]),
        ("scope_memories_median", profile["conversation_scope_memory_count"]["median"]),
        ("scope_memories_max", profile["conversation_scope_memory_count"]["max"]),
        ("evidence_median", profile["evidence_count"]["median"]),
        ("evidence_p95", profile["evidence_count"]["p95"]),
        ("evidence_max", profile["evidence_count"]["max"]),
        ("kg_coverage_fraction", profile["kg"]["coverage_fraction"]),
        ("triples_covered_median", profile["kg"]["triples_per_covered_memory_median"]),
        ("triples_covered_p95", profile["kg"]["triples_per_covered_memory_p95"]),
        ("triples_covered_max", profile["kg"]["triples_per_covered_memory_max"]),
        ("unresolved_evidence", profile["evidence_count"]["unresolved_reference_count"]),
    ]
    with (args.output / "locomo_workload_profile.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    with (args.output / "freshness_eligible_cat1_4.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["qa_id", "sample_id", "category", "question", "gold_memory_ids", "gold_count"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eligible_freshness)

    with (args.output / "evidence_exclusions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qa_id", "evidence"])
        writer.writeheader()
        writer.writerows(unresolved_evidence)

    report = f"""# LoCoMo-shaped workload profile

This profile is computed from the canonical local inputs. The scaled experiment must be described as **LoCoMo-shaped trace replay**, not as original 100K LoCoMo.

- Conversation scopes: {profile['counts']['conversation_scopes']}
- Memories: {profile['counts']['memories']}
- QA: {profile['counts']['questions']}
- Cat1-4 unambiguous QA eligible for freshness events: {profile['counts']['cat1_4_unambiguous_freshness_questions']}
- Memories/scope: min {profile['conversation_scope_memory_count']['min']}, median {profile['conversation_scope_memory_count']['median']}, max {profile['conversation_scope_memory_count']['max']}
- Evidence/question: median {profile['evidence_count']['median']}, P95 {profile['evidence_count']['p95']:.1f}, max {profile['evidence_count']['max']}
- KG coverage: {profile['kg']['coverage_fraction']:.4%} ({profile['kg']['covered_memories']}/{profile['counts']['memories']})
- Triples/covered memory: median {profile['kg']['triples_per_covered_memory_median']}, P95 {profile['kg']['triples_per_covered_memory_p95']:.1f}, max {profile['kg']['triples_per_covered_memory_max']}
- Unresolved evidence references: {profile['evidence_count']['unresolved_reference_count']}

The JSON file is the normative machine-readable artifact; this Markdown file is only a summary.
"""
    (args.output / "WORKLOAD_PROFILE.md").write_text(report, encoding="utf-8")
    print(json_path)


if __name__ == "__main__":
    main()
