#!/usr/bin/env python3
"""Build the mixed HingeMem-style Reader main-table source CSV.

All six local methods take F1/B1 from the single frozen v4 offline evaluator.
No Reader API is called by this script.  J is read from the already-frozen
judge summaries; corrected Dense+GlobalKG has its own judge cache.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OFFLINE_METRICS_DIR = ROOT / "results" / "reader" / "reader_offline_metrics_v4"
OUT_DIR = ROOT / "results" / "reader" / "reader_main_hingemem_style"
JUDGE_DIR = ROOT / "results" / "reader" / "llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected"
CORRECTED_DENSEKG_JUDGE = ROOT / "results" / "reader" / "llm_judge_gpt4o_corrected_densekg_v2"

FIELDS = (
    "single_hop_f1", "single_hop_j",
    "multi_hop_f1", "multi_hop_j",
    "temporal_f1", "temporal_j",
    "open_domain_f1", "open_domain_j",
    "adversarial_f1", "adversarial_j",
    "overall_f1", "overall_j", "overall_b1",
)


def paper_row(
    order: int,
    block: int,
    method: str,
    cat_format: str,
    values: tuple,
    *,
    note: str = "Quoted from HingeMem Table 1; not locally rerun",
) -> dict:
    return {
        "order": order,
        "block": block,
        "method": method,
        "cat_format": cat_format,
        **dict(zip(FIELDS, values)),
        "source_type": "quoted_hingemem_table1",
        "source": "High-mem.pdf Table 1 / HingeMem WWW 2026",
        "note": note,
    }


PAPER_ROWS = [
    paper_row(10, 1, "LOCOMO", "Cat.✓", (12.7,16.5,19.7,20.9,10.4,11.5,20.1,30.2,66.8,90.1,25.8,33.5,0.132)),
    paper_row(20, 1, "RAG (Top-10)", "Cat.✓", (36.4,50.7,29.6,38.6,27.7,22.7,21.2,37.5,86.8,86.5,44.6,51.9,0.306)),
    paper_row(80, 2, "Mem0", "Cat.✗", (45.1,56.7,42.7,48.2,49.7,50.1,27.7,47.2,6.5,56.6,36.0,53.7,0.254)),
    paper_row(81, 2, "Mem0", "Cat.✓", (44.0,59.0,38.6,47.8,45.6,47.6,20.8,42.0,84.3,72.7,51.4,59.6,0.351)),
    paper_row(110, 3, "Mem0g", "Cat.✗", (45.3,55.1,40.4,48.5,47.9,52.0,28.4,41.0,6.7,54.1,35.5,51.7,0.258)),
    paper_row(111, 3, "Mem0g", "Cat.✓", (43.4,62.0,38.3,45.7,44.4,48.5,23.4,44.1,82.7,68.5,50.7,59.7,0.348)),
    paper_row(120, 3, "HippoRAG2", "Cat.✗", (54.4,78.5,35.4,52.8,55.5,61.0,23.4,35.2,4.3,69.5,39.1,68.5,0.289)),
    paper_row(121, 3, "HippoRAG2", "Cat.✓", (59.2,75.5,38.0,46.4,44.9,66.6,21.2,35.4,87.7,87.2,58.4,70.6,0.396)),
    paper_row(130, 3, "HingeMem", "Cat.✗", (61.1,78.8,53.6,62.8,57.4,66.9,30.7,46.4,87.4,87.8,63.9,75.1,0.404)),
]

LOCAL_ORDER = {
    "BM25": (30, 1),
    "Dense-bge": (40, 1),
    "RRF_compact": (50, 1),
    "Dense+GlobalKG": (100, 3),
    "ZScore-Raw": (140, 3),
    "ZScore-RawERK": (170, 4),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    summaries = read_csv(OFFLINE_METRICS_DIR / "reader_metrics_summary.csv")
    category_metrics = {
        (row["method"], row["setting"], row["category"]): row
        for row in read_csv(OFFLINE_METRICS_DIR / "reader_metrics_by_category.csv")
    }
    common_judge = {
        (row["method"], row["setting"]): row
        for row in read_csv(JUDGE_DIR / "j_scores_reader_main_wide.csv")
    }
    densekg_judge = {
        (row["method"], row["setting"]): row
        for row in read_csv(CORRECTED_DENSEKG_JUDGE / "j_scores_reader_main_wide.csv")
    }

    rows: list[dict] = []
    method_offset: dict[str, int] = {}
    for summary in summaries:
        method = summary["method"]
        setting = summary["setting"]
        base_order, block = LOCAL_ORDER[method]
        offset = method_offset.get(method, 0)
        method_offset[method] = offset + 1
        judge_map = densekg_judge if method == "Dense+GlobalKG" else common_judge
        judge = judge_map[(method, setting)]

        def category_f1(category: int) -> float:
            key = (method, setting, str(category))
            return 100 * float(category_metrics[key]["f1"])

        rows.append(
            {
                "order": base_order + offset,
                "block": block,
                "method": method,
                "cat_format": "Cat.✗" if setting == "a_unified" else "Cat.✓",
                "single_hop_f1": category_f1(4),
                "single_hop_j": float(judge["single_hop_j"]),
                "multi_hop_f1": category_f1(1),
                "multi_hop_j": float(judge["multi_hop_j"]),
                "temporal_f1": category_f1(2),
                "temporal_j": float(judge["temporal_j"]),
                "open_domain_f1": category_f1(3),
                "open_domain_j": float(judge["open_domain_j"]),
                "adversarial_f1": category_f1(5),
                "adversarial_j": float(judge["adversarial_j"]),
                "overall_f1": 100 * float(summary["overall_f1"]),
                "overall_j": float(judge["overall_j"]),
                "overall_b1": float(summary["overall_b1"]),
                "source_type": "local_offline_metrics_v4",
                "source": (
                    "reader_offline_metrics_v4 + llm_judge_gpt4o_corrected_densekg_v2"
                    if method == "Dense+GlobalKG"
                    else "reader_offline_metrics_v4 + llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected"
                ),
                "note": "Cat5 option text restored before F1/B1 evaluation; no new Reader API calls",
            }
        )

    if len(rows) != 12 or {(row["method"], row["cat_format"]) for row in rows} != {
        (method, cat_format)
        for method in LOCAL_ORDER
        for cat_format in ("Cat.✗", "Cat.✓")
    }:
        raise RuntimeError("Expected exactly two v4 rows for each of the six local methods")

    rows.extend(PAPER_ROWS)
    rows.sort(key=lambda row: int(row["order"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / "reader_main_data.csv"
    fieldnames = [
        "order", "block", "method", "cat_format", *FIELDS,
        "source_type", "source", "note",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
