import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_FILE = Path("results/parallel_frontier_summary.csv")
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_rows():
    rows = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append({
                "graph_id": row["graph_id"],
                "workers": int(row["workers"]),
                "parallel_latency_ms": float(row["parallel_latency_ms"]),
                "speedup": float(row["speedup"]),
            })

    return rows


def group_by_graph(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["graph_id"]].append(row)

    for graph_id in grouped:
        grouped[graph_id].sort(key=lambda x: x["workers"])

    return grouped


def plot_latency(grouped):
    plt.figure(figsize=(8, 5))

    for graph_id, rows in grouped.items():
        workers = [row["workers"] for row in rows]
        latency = [row["parallel_latency_ms"] for row in rows]
        plt.plot(workers, latency, marker="o", label=graph_id)

    plt.xlabel("Number of workers")
    plt.ylabel("Average total wall latency (ms)")
    plt.title("Parallel Frontier Traversal: Latency vs Workers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = FIGURE_DIR / "figure_parallel_frontier_latency.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved {output}")


def plot_speedup(grouped):
    plt.figure(figsize=(8, 5))

    for graph_id, rows in grouped.items():
        workers = [row["workers"] for row in rows]
        speedup = [row["speedup"] for row in rows]
        plt.plot(workers, speedup, marker="o", label=graph_id)

    plt.xlabel("Number of workers")
    plt.ylabel("Speedup over serial traversal")
    plt.title("Parallel Frontier Traversal: Speedup vs Workers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = FIGURE_DIR / "figure_parallel_frontier_speedup.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved {output}")


def main():
    rows = load_rows()
    grouped = group_by_graph(rows)

    plot_latency(grouped)
    plot_speedup(grouped)


if __name__ == "__main__":
    main()