import csv
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_FILE = Path("results/parallel_path_relation_index_summary.csv")
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_rows():
    rows = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append({
                "graph_id": row["graph_id"],
                "total_edges": int(row["total_edges"]),
                "baseline_raw": int(row["baseline_raw"]),
                "index_raw": int(row["index_raw"]),
                "baseline_latency_ms": float(row["baseline_latency_ms"]),
                "index_latency_ms": float(row["index_latency_ms"]),
                "index_speedup": float(row["index_speedup"]),
            })

    return rows


def plot_latency(rows):
    x = [row["total_edges"] for row in rows]
    baseline = [row["baseline_latency_ms"] for row in rows]
    index = [row["index_latency_ms"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, baseline, marker="o", label="Baseline traversal")
    plt.plot(x, index, marker="o", label="Relation-aware index")

    plt.xlabel("Total edges in noisy path graph")
    plt.ylabel("Average total latency (ms)")
    plt.title("Relation-aware Index under Increasing Relation Noise")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = FIGURE_DIR / "figure_relation_index_latency_vs_noise.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved {output}")


def plot_raw_count(rows):
    x = [row["total_edges"] for row in rows]
    baseline_raw = [row["baseline_raw"] for row in rows]
    index_raw = [row["index_raw"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, baseline_raw, marker="o", label="Baseline raw edges")
    plt.plot(x, index_raw, marker="o", label="Index raw edges")

    plt.xlabel("Total edges in noisy path graph")
    plt.ylabel("Raw edges scanned")
    plt.title("Raw Edge Scan Reduction by Relation-aware Index")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = FIGURE_DIR / "figure_relation_index_raw_scan_vs_noise.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved {output}")


def plot_speedup(rows):
    x = [row["total_edges"] for row in rows]
    speedup = [row["index_speedup"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, speedup, marker="o")

    plt.xlabel("Total edges in noisy path graph")
    plt.ylabel("Speedup over baseline")
    plt.title("Index Speedup under Increasing Relation Noise")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = FIGURE_DIR / "figure_relation_index_speedup_vs_noise.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved {output}")


def main():
    rows = load_rows()

    plot_latency(rows)
    plot_raw_count(rows)
    plot_speedup(rows)


if __name__ == "__main__":
    main()