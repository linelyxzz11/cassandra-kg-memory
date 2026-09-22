#!/usr/bin/env python3
"""Rebuild canonical LoCoMo Cat1-4 gold-memory IDs from original turn evidence."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_JSON_PATH = ROOT / "external_data" / "locomo10.json"
MEMORY_PATH = ROOT / "data" / "locomo_memory_records.csv"
LEGACY_QUERY_PATH = (
    ROOT / "scripts" / "experiments" / "artifacts" / "p3_queries.csv"
)
OUT_DIR = ROOT / "data" / "retrieval_gold"

# Four annotation typos in the released source can be resolved deterministically
# from the question, answer, and the cited conversation.  They are preserved in
# a separate correction audit rather than silently changing the source.
EVIDENCE_CORRECTIONS = {
    ("conv-42_qa_58", "D10:19"): (
        "D20:15",
        "The cited turn does not exist; D20:15 contains the dairy-free margarine "
        "and coconut-oil recommendation named in the answer.",
    ),
    ("conv-42_qa_88", "D"): (
        "D1:16",
        "The cited token is truncated; D1:16 introduces the movie via its image "
        "query and is the missing evidence between the other cited turns.",
    ),
    ("conv-43_qa_18", "D:11:26"): (
        "D11:26",
        "Malformed session prefix; D11:26 explicitly mentions The Alchemist.",
    ),
    ("conv-47_qa_38", "D4:36"): (
        "D13:3",
        "The cited turn does not exist; D13:3 states that John secured his dream "
        "job, matching the missing clause of the annotated answer.",
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_qas() -> list[dict]:
    source = json.loads(SOURCE_JSON_PATH.read_text(encoding="utf-8"))
    rows = []
    for sample in source:
        sample_id = sample["sample_id"]
        for index, qa in enumerate(sample.get("qa", [])):
            rows.append(
                {
                    "qa_id": f"{sample_id}_qa_{index}",
                    "sample_id": sample_id,
                    "question": qa["question"],
                    "answer": qa.get("answer", ""),
                    "category": str(qa["category"]),
                    "evidence": qa.get("evidence", []),
                }
            )
    return rows


def parse_evidence(value) -> list[str]:
    if isinstance(value, list):
        parsed = value
    else:
        value = value.strip()
        if not value or value == "[]":
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
    if not parsed:
        return []
    if isinstance(parsed, str):
        parsed = [parsed]
    flattened = []
    for item in parsed:
        item = str(item).strip()
        if ";" in item:
            parts = [part.strip() for part in item.split(";") if part.strip()]
        elif len(re.findall(r"D\d+:\d+", item)) > 1:
            parts = re.findall(r"D\d+:\d+", item)
        else:
            parts = [item]
        flattened.extend(parts)
    return list(dict.fromkeys(flattened))


def normalize_turn_id(turn_id: str) -> str:
    match = re.fullmatch(r"D(\d+):0*(\d+)", turn_id)
    if not match:
        return turn_id
    return f"D{int(match.group(1))}:{int(match.group(2))}"


def parse_ids(value: str) -> list[str]:
    return list(dict.fromkeys(part.strip() for part in value.split(";") if part.strip()))


def main() -> None:
    qa_rows = load_source_qas()
    memory_rows = read_csv(MEMORY_PATH)
    legacy_rows = read_csv(LEGACY_QUERY_PATH)
    legacy_by_query = {row["query_id"].strip(): row for row in legacy_rows}

    memory_by_turn: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in memory_rows:
        memory_by_turn[
            (row["sample_id"].strip(), row["dia_id"].strip())
        ].append(row["memory_id"].strip())

    canonical_rows = []
    audit_rows = []
    unmapped_rows = []
    correction_rows = []
    for qa in qa_rows:
        category = qa["category"].strip()
        if category not in {"1", "2", "3", "4"}:
            continue
        query_id = qa["qa_id"].strip()
        sample_id = qa["sample_id"].strip()
        source_evidence_turns = parse_evidence(qa.get("evidence", []))
        evidence_turns = []
        for source_turn_id in source_evidence_turns:
            corrected_turn_id, rationale = EVIDENCE_CORRECTIONS.get(
                (query_id, source_turn_id),
                (source_turn_id, ""),
            )
            corrected_turn_id = normalize_turn_id(corrected_turn_id)
            evidence_turns.append(corrected_turn_id)
            if corrected_turn_id != source_turn_id:
                correction_rows.append(
                    {
                        "query_id": query_id,
                        "category": category,
                        "conversation_id": sample_id,
                        "source_evidence_turn_id": source_turn_id,
                        "corrected_evidence_turn_id": corrected_turn_id,
                        "rationale": rationale
                        or "Normalized leading zero in released evidence ID.",
                    }
                )
        evidence_turns = list(dict.fromkeys(evidence_turns))
        gold_ids = []
        for turn_id in evidence_turns:
            mapped = memory_by_turn.get((sample_id, turn_id), [])
            if not mapped:
                unmapped_rows.append(
                    {
                        "query_id": query_id,
                        "category": category,
                        "conversation_id": sample_id,
                        "evidence_turn_id": turn_id,
                        "reason": "no memory row with matching sample_id and dia_id",
                    }
                )
            gold_ids.extend(mapped)
        gold_ids = list(dict.fromkeys(gold_ids))

        legacy = legacy_by_query.get(query_id, {})
        legacy_ids = parse_ids(legacy.get("gold_memory_ids", ""))
        split = legacy.get("split", "")
        status = (
            "evidence_empty"
            if not evidence_turns
            else "mapped"
            if gold_ids and len(gold_ids) == len(evidence_turns)
            else "partial_or_unmapped"
        )
        canonical_rows.append(
            {
                "query_id": query_id,
                "question": qa["question"].strip(),
                "category": category,
                "conversation_id": sample_id,
                "split": split,
                "source_evidence_turn_ids": ";".join(source_evidence_turns),
                "evidence_turn_ids": ";".join(evidence_turns),
                "gold_memory_ids": ";".join(gold_ids),
                "gold_count": len(gold_ids),
                "gold_status": status,
            }
        )
        audit_rows.append(
            {
                "query_id": query_id,
                "category": category,
                "split": split,
                "legacy_gold_memory_ids": ";".join(legacy_ids),
                "canonical_gold_memory_ids": ";".join(gold_ids),
                "legacy_gold_count": len(legacy_ids),
                "canonical_gold_count": len(gold_ids),
                "changed": legacy_ids != gold_ids,
                "status": status,
            }
        )

    if len(canonical_rows) != 1540:
        raise RuntimeError(f"Expected 1540 Cat1-4 queries, found {len(canonical_rows)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUT_DIR / "locomo_cat1_4_gold_memory.csv",
        canonical_rows,
        [
            "query_id",
            "question",
            "category",
            "conversation_id",
            "split",
            "source_evidence_turn_ids",
            "evidence_turn_ids",
            "gold_memory_ids",
            "gold_count",
            "gold_status",
        ],
    )
    write_csv(
        OUT_DIR / "legacy_gold_mapping_audit.csv",
        audit_rows,
        [
            "query_id",
            "category",
            "split",
            "legacy_gold_memory_ids",
            "canonical_gold_memory_ids",
            "legacy_gold_count",
            "canonical_gold_count",
            "changed",
            "status",
        ],
    )
    write_csv(
        OUT_DIR / "unmapped_evidence_turns.csv",
        unmapped_rows,
        [
            "query_id",
            "category",
            "conversation_id",
            "evidence_turn_id",
            "reason",
        ],
    )
    write_csv(
        OUT_DIR / "evidence_corrections.csv",
        correction_rows,
        [
            "query_id",
            "category",
            "conversation_id",
            "source_evidence_turn_id",
            "corrected_evidence_turn_id",
            "rationale",
        ],
    )

    status_counts = Counter(row["gold_status"] for row in canonical_rows)
    category_counts = Counter(
        (row["category"], row["gold_status"]) for row in canonical_rows
    )
    manifest = {
        "artifact": "LoCoMo Cat1-4 canonical gold-memory mapping v2",
        "method": (
            "Parse the original QA evidence field, split compound semicolon-delimited "
            "turn IDs, and map each (sample_id, dia_id) exactly to memory_id."
        ),
        "counts": {
            "queries": len(canonical_rows),
            "status": dict(status_counts),
            "changed_vs_legacy": sum(row["changed"] for row in audit_rows),
            "unmapped_evidence_turns": len(unmapped_rows),
            "documented_evidence_corrections": len(correction_rows),
            "by_category_and_status": {
                f"cat{category}:{status}": count
                for (category, status), count in sorted(category_counts.items())
            },
        },
        "policy": {
            "evidence_empty": (
                "Retained as official evidence-empty; excluded from positive-evidence "
                "retrieval metrics and reported separately."
            ),
            "manual_supplement": (
                "Any future human-authored evidence must be versioned as an extended "
                "annotation and must not overwrite this official mapping."
            ),
        },
        "inputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (SOURCE_JSON_PATH, MEMORY_PATH, LEGACY_QUERY_PATH)
        ],
        "outputs": [
            "locomo_cat1_4_gold_memory.csv",
            "legacy_gold_mapping_audit.csv",
            "unmapped_evidence_turns.csv",
            "evidence_corrections.csv",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
