from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .models import Edge, QuerySpec

REQUIRED_COLUMNS = ("graph_id", "src_id", "relation", "dst_id")
CANONICAL_COLUMNS = (*REQUIRED_COLUMNS, "source", "logical_edge_id")


def canonicalize_csv(input_path: str | Path, output_path: str | Path) -> tuple[list[Edge], int]:
    """Create canonical triples without relying on Cassandra's physical timeuuid.

    Exact duplicates on (graph_id, src_id, relation, dst_id, source) are collapsed. This
    is intentional: physical timeuuid values are backend-specific and do not constitute
    a portable semantic distinction for C0's traversal-equivalence gate.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header")
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV misses required columns: {missing}")
        edges: list[Edge] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        duplicates = 0
        for row_index, raw in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in raw.items()}
            edge = Edge.from_mapping(row)
            if not all((edge.graph_id, edge.src_id, edge.relation, edge.dst_id)):
                raise ValueError(f"Empty required field at CSV line {row_index}")
            key = (edge.graph_id, edge.src_id, edge.relation, edge.dst_id, edge.source)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            supplied = row.get("logical_edge_id", "")
            if supplied and supplied != edge.logical_id:
                raise ValueError(
                    f"CSV line {row_index} has inconsistent logical_edge_id; "
                    "remove it or regenerate the canonical file."
                )
            edges.append(edge)
    write_canonical_edges(output_path, edges)
    return edges, duplicates


def write_canonical_edges(path: str | Path, edges: Iterable[Edge]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        for edge in edges:
            writer.writerow(edge.to_dict())


def load_canonical_edges(path: str | Path) -> list[Edge]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = REQUIRED_COLUMNS + ("source", "logical_edge_id")
        missing = [col for col in required if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Canonical CSV missing {missing}. Run canonicalize first.")
        edges: list[Edge] = []
        seen: set[str] = set()
        for index, row in enumerate(reader, start=2):
            edge = Edge.from_mapping(row)
            if row["logical_edge_id"] != edge.logical_id:
                raise ValueError(f"Canonical CSV line {index} has invalid logical_edge_id")
            if edge.logical_id in seen:
                raise ValueError(f"Canonical CSV line {index} duplicates a logical edge")
            seen.add(edge.logical_id)
            edges.append(edge)
    return edges


def read_manifest(path: str | Path, default_graph_id: str | None,
                  default_cycle_policy: str = "path") -> list[QuerySpec]:
    records: list[QuerySpec] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(QuerySpec.from_mapping(json.loads(line), default_graph_id, default_cycle_policy))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    ids = [item.query_id for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("workload manifest must have unique query_id values")
    return records


def write_manifest(path: str | Path, records: Iterable[QuerySpec]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
