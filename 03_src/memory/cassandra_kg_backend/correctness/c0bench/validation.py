from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .executors.base import BackendExecutor
from .models import QuerySpec, TraversalResult
from .semantics import ReferenceGraph


def _diff(expected: TraversalResult, actual: TraversalResult) -> dict[str, list[list[str]]]:
    expected_paths, actual_paths = set(expected.normalized_paths()), set(actual.normalized_paths())
    return {
        "missing_paths": [list(path) for path in sorted(expected_paths - actual_paths)],
        "unexpected_paths": [list(path) for path in sorted(actual_paths - expected_paths)],
    }


def validate_static_reads(reference: ReferenceGraph, executors: dict[str, BackendExecutor], manifest: Iterable[QuerySpec], report_dir: str | Path) -> dict:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    records = [event for event in manifest if event.op_type == "read"]
    summary = {"read_events": len(records), "systems": {name: {"checked": 0, "disagreements": 0, "errors": 0} for name in executors}}
    with (report_dir / "mismatches.jsonl").open("w", encoding="utf-8") as output:
        for query in records:
            expected = reference.execute(query)
            for name, executor in executors.items():
                stats = summary["systems"][name]
                stats["checked"] += 1
                try:
                    actual = executor.execute(query)
                    difference = _diff(expected, actual)
                    if difference["missing_paths"] or difference["unexpected_paths"]:
                        stats["disagreements"] += 1
                        output.write(json.dumps({"system": name, "query": query.to_dict(), "expected": expected.to_dict(), "actual": actual.to_dict(), "difference": difference}, ensure_ascii=False) + "\n")
                except Exception as exc:  # noqa: BLE001
                    stats["errors"] += 1
                    output.write(json.dumps({"system": name, "query": query.to_dict(), "error": repr(exc)}, ensure_ascii=False) + "\n")
    summary["all_pass"] = all(v["disagreements"] == 0 and v["errors"] == 0 for v in summary["systems"].values())
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
