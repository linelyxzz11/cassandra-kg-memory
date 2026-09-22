"""Build a deterministic manifest for the paper-facing result artifacts."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "ARTIFACT_MANIFEST.csv"


def included(path: Path) -> bool:
    relative = path.relative_to(RESULTS)
    if path == OUTPUT or "runs" in relative.parts or "traces" in relative.parts:
        return False
    name = path.name.lower()
    return not (
        name.endswith("_per_event.csv")
        or name in {"scored_predictions.jsonl", "judgments_unique.jsonl"}
        or (name == "reader_metrics_per_query.jsonl" and "reader" in relative.parts)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file() and included(path):
            rows.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} entries to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
