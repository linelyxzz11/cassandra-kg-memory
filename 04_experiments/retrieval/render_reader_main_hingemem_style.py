#!/usr/bin/env python3
"""Render the frozen Reader main table in the restrained HingeMem style."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "05_reports" / "reader_main_hingemem_style"
CSV_PATH = OUT_DIR / "reader_main_data.csv"
PNG_PATH = OUT_DIR / "reader_main_hingemem_style_with_j.png"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: str, metric: str) -> str:
    if value == "" or value is None:
        return "-"
    return f"{float(value):.3f}" if metric == "overall_b1" else f"{float(value):.1f}"


def main() -> None:
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    fig, ax = plt.subplots(figsize=(19.2, 12.4), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left, right = 0.025, 0.985
    method_x, cat_x = 0.038, 0.165
    groups = [
        ("Single-Hop", ("single_hop_f1", "single_hop_j")),
        ("Multi-Hop", ("multi_hop_f1", "multi_hop_j")),
        ("Temporal", ("temporal_f1", "temporal_j")),
        ("Open-Domain", ("open_domain_f1", "open_domain_j")),
        ("Adversarial", ("adversarial_f1", "adversarial_j")),
        ("Overall", ("overall_f1", "overall_j", "overall_b1")),
    ]
    metric_start, metric_end = 0.225, 0.972
    total_subcolumns = sum(len(fields) for _, fields in groups)
    sub_w = (metric_end - metric_start) / total_subcolumns
    field_centers: dict[str, float] = {}
    cursor = metric_start
    for label, fields in groups:
        group_left = cursor
        for field in fields:
            field_centers[field] = cursor + sub_w / 2
            cursor += sub_w
        center = (group_left + cursor) / 2
        ax.text(
            center, 0.962, label, ha="center", va="center",
            fontsize=12.2, fontweight="bold", family="serif",
        )
        ax.plot(
            [group_left + 0.006, cursor - 0.006], [0.944, 0.944],
            color="black", lw=0.75,
        )

    ax.text(
        method_x, 0.942, "Method", ha="left", va="center",
        fontsize=12.2, fontweight="bold", family="serif",
    )
    ax.text(
        cat_x, 0.942, "Cat.", ha="center", va="center",
        fontsize=12.2, fontweight="bold", family="serif",
    )
    for field, x in field_centers.items():
        symbol = r"$B_1$" if field == "overall_b1" else (
            r"$J$" if field.endswith("_j") else r"$F_1$"
        )
        ax.text(x, 0.921, symbol, ha="center", va="center", fontsize=11.4, family="serif")
    ax.plot([left, right], [0.982, 0.982], color="black", lw=1.05)
    ax.plot([left, right], [0.902, 0.902], color="black", lw=0.85)

    top_y, bottom_y = 0.882, 0.035
    row_h = (top_y - bottom_y) / len(rows)
    seen_methods: set[str] = set()
    previous_block = None
    metric_fields = [field for _, group_fields in groups for field in group_fields]
    for index, row in enumerate(rows):
        y_top = top_y - index * row_h
        y = y_top - row_h / 2
        block = row["block"]
        if previous_block is not None and block != previous_block:
            ax.plot([left, right], [y_top, y_top], color="black", lw=0.75)
        previous_block = block

        method = row["method"]
        is_second = method in seen_methods
        seen_methods.add(method)
        is_cassmem = method == "ZScore-RawERK"
        if is_second:
            method_label = "+ Cat. format"
            method_color = "#777777"
            method_style = "italic"
        else:
            method_label = "CassMem (Ours)" if is_cassmem else method
            method_color = "black"
            method_style = "normal"
        weight = "bold" if is_cassmem else "normal"
        ax.text(
            method_x, y, method_label, ha="left", va="center", fontsize=10.5,
            family="serif", color=method_color, style=method_style, fontweight=weight,
        )
        cat_label = "✓" if "✓" in row["cat_format"] else "✗"
        ax.text(
            cat_x, y, cat_label, ha="center", va="center", fontsize=11.2,
            family="DejaVu Sans", fontweight=weight,
        )
        for field in metric_fields:
            ax.text(
                field_centers[field], y, number(row[field], field),
                ha="center", va="center", fontsize=10.15,
                family="serif", fontweight=weight,
            )

    ax.plot([left, right], [bottom_y, bottom_y], color="black", lw=1.05)
    fig.savefig(
        PNG_PATH, dpi=180, bbox_inches="tight", pad_inches=0.08, facecolor="white",
    )
    plt.close(fig)

    manifest_path = OUT_DIR / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    manifest.update(
        {
            "artifact": "HingeMem-style Reader main table",
            "local_sources": [
                "05_reports/reader_offline_metrics_v4/reader_metrics_summary.csv",
                "05_reports/reader_offline_metrics_v4/reader_metrics_by_category.csv",
                "05_reports/reader_offline_metrics_v4/manifest.json",
                "05_reports/llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected/j_scores_reader_main_wide.csv",
                "05_reports/llm_judge_gpt4o_corrected_densekg_v2/j_scores_reader_main_wide.csv",
            ],
            "offline_metric_protocol": {
                "version": "reader_offline_metrics_v4",
                "api_calls": 0,
                "cat5_policy": "restore deterministic option text before F1 and BLEU-1",
                "overall_scope": "Full5 micro over 1986 questions",
            },
            "dense_global_kg_reader_status": "corrected_v4_f1_b1_and_corrected_judge_complete",
            "outputs": {
                "csv": {"path": CSV_PATH.name, "sha256": sha256(CSV_PATH)},
                "png": {"path": PNG_PATH.name, "sha256": sha256(PNG_PATH)},
            },
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
