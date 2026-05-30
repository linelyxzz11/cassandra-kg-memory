import argparse
import csv
import time
from pathlib import Path
from statistics import mean

from cassandra.cluster import Cluster


BASELINE_QUERY = """
SELECT src_id, relation, dst_id, confidence
FROM kg_edges_by_src
WHERE graph_id=%s AND src_id=%s
"""


INDEX_QUERY = """
SELECT src_id, relation, dst_id, confidence
FROM kg_edges_by_src_relation
WHERE graph_id=%s AND src_id=%s AND relation=%s
"""


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)
    index = int(round((p / 100) * (len(values) - 1)))
    return values[index]


def run_baseline(session, graph_id, src_id, relation):
    start = time.perf_counter()

    rows = list(session.execute(BASELINE_QUERY, (graph_id, src_id)))

    # Baseline reads all outgoing edges first, then filters relation in Python.
    matched_rows = [
        row for row in rows
        if row.relation == relation
    ]

    end = time.perf_counter()

    return {
        "method": "baseline_src_scan",
        "raw_count": len(rows),
        "matched_count": len(matched_rows),
        "latency_ms": (end - start) * 1000,
    }


def run_index(session, graph_id, src_id, relation):
    start = time.perf_counter()

    # Index query directly locates edges by graph_id + src_id + relation.
    rows = list(session.execute(INDEX_QUERY, (graph_id, src_id, relation)))

    end = time.perf_counter()

    return {
        "method": "src_relation_index",
        "raw_count": len(rows),
        "matched_count": len(rows),
        "latency_ms": (end - start) * 1000,
    }


def benchmark_method(fn, session, graph_id, src_id, relation, repeat, warmup):
    for _ in range(warmup):
        fn(session, graph_id, src_id, relation)

    results = []

    for _ in range(repeat):
        results.append(
            fn(session, graph_id, src_id, relation)
        )

    latencies = [row["latency_ms"] for row in results]
    raw_counts = [row["raw_count"] for row in results]
    matched_counts = [row["matched_count"] for row in results]

    return {
        "method": results[0]["method"],
        "avg_latency_ms": mean(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "avg_raw_count": mean(raw_counts),
        "avg_matched_count": mean(matched_counts),
    }


def print_result(args, baseline, index):
    print("Source Relation Index Benchmark")
    print("-------------------------------")
    print(f"graph_id : {args.graph_id}")
    print(f"src_id   : {args.src}")
    print(f"relation : {args.relation}")
    print(f"repeat   : {args.repeat}")
    print(f"warmup   : {args.warmup}")
    print()

    print(
        f"{'method':<24}"
        f"{'avg_ms':<12}"
        f"{'p95_ms':<12}"
        f"{'raw_count':<12}"
        f"{'matched':<12}"
    )
    print("-" * 72)

    for row in [baseline, index]:
        print(
            f"{row['method']:<24}"
            f"{row['avg_latency_ms']:<12.3f}"
            f"{row['p95_latency_ms']:<12.3f}"
            f"{row['avg_raw_count']:<12.1f}"
            f"{row['avg_matched_count']:<12.1f}"
        )

    if index["avg_latency_ms"] > 0:
        speedup = baseline["avg_latency_ms"] / index["avg_latency_ms"]
        print()
        print(f"Index speedup: {speedup:.2f}x")

    raw_reduction = baseline["avg_raw_count"] - index["avg_raw_count"]

    print(f"Raw edge reduction: {raw_reduction:.1f}")


def write_csv(args, baseline, index):
    if not args.output:
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "graph_id",
        "src_id",
        "relation",
        "repeat",
        "warmup",
        "method",
        "avg_latency_ms",
        "p95_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "avg_raw_count",
        "avg_matched_count",
    ]

    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in [baseline, index]:
            writer.writerow({
                "graph_id": args.graph_id,
                "src_id": args.src,
                "relation": args.relation,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "method": row["method"],
                "avg_latency_ms": round(row["avg_latency_ms"], 3),
                "p95_latency_ms": round(row["p95_latency_ms"], 3),
                "min_latency_ms": round(row["min_latency_ms"], 3),
                "max_latency_ms": round(row["max_latency_ms"], 3),
                "avg_raw_count": round(row["avg_raw_count"], 3),
                "avg_matched_count": round(row["avg_matched_count"], 3),
            })


def main():
    parser = argparse.ArgumentParser(
        description="Compare src scan vs src+relation index query."
    )

    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--keyspace", default="ai_memory")
    parser.add_argument(
        "--output",
        default="results/benchmark_src_relation_index_results.csv",
    )

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    baseline = benchmark_method(
        fn=run_baseline,
        session=session,
        graph_id=args.graph_id,
        src_id=args.src,
        relation=args.relation,
        repeat=args.repeat,
        warmup=args.warmup,
    )

    index = benchmark_method(
        fn=run_index,
        session=session,
        graph_id=args.graph_id,
        src_id=args.src,
        relation=args.relation,
        repeat=args.repeat,
        warmup=args.warmup,
    )

    print_result(args, baseline, index)
    write_csv(args, baseline, index)

    cluster.shutdown()


if __name__ == "__main__":
    main()