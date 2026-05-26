
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean

from cassandra.cluster import Cluster


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)
    index = int(round((p / 100) * (len(values) - 1)))
    return values[index]


def dedupe_edges(edges):
    seen = set()
    result = []

    for edge in edges:
        key = (edge.src_id, edge.relation, edge.dst_id)

        if key in seen:
            continue

        seen.add(key)
        result.append(edge)

    return result


def query_relation_bucket(session, graph_id, relation, bucket):
    query = """
    SELECT src_id, relation, dst_id
    FROM kg_edges_by_relation_bucket
    WHERE graph_id=%s AND relation=%s AND bucket=%s
    """

    rows = session.execute(query, (graph_id, relation, bucket))
    return list(rows)


def query_relation_serial(session, graph_id, relation, bucket_count):
    all_edges = []

    for bucket in range(bucket_count):
        rows = query_relation_bucket(
            session=session,
            graph_id=graph_id,
            relation=relation,
            bucket=bucket,
        )
        all_edges.extend(rows)

    return dedupe_edges(all_edges)


def query_relation_parallel(session, graph_id, relation, bucket_count, workers):
    all_edges = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_bucket = {
            executor.submit(
                query_relation_bucket,
                session,
                graph_id,
                relation,
                bucket,
            ): bucket
            for bucket in range(bucket_count)
        }

        for future in as_completed(future_to_bucket):
            bucket = future_to_bucket[future]

            try:
                rows = future.result()
                all_edges.extend(rows)
            except Exception as exc:
                print(f"Bucket {bucket} query failed: {exc}")

    return dedupe_edges(all_edges)


def benchmark_one(name, func, repeat=20, warmup=3):
    for _ in range(warmup):
        func()

    times_ms = []
    last_result = None

    for _ in range(repeat):
        start = time.perf_counter()
        last_result = func()
        end = time.perf_counter()

        times_ms.append((end - start) * 1000)

    result_count = len(last_result) if last_result is not None else 0

    return {
        "name": name,
        "avg_ms": mean(times_ms),
        "p95_ms": percentile(times_ms, 95),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "result_count": result_count,
    }


def print_result(result):
    print(
        f"{result['name']:<38} "
        f"avg={result['avg_ms']:.3f} ms | "
        f"p95={result['p95_ms']:.3f} ms | "
        f"min={result['min_ms']:.3f} ms | "
        f"max={result['max_ms']:.3f} ms | "
        f"count={result['result_count']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark serial and parallel relation queries on Cassandra KG."
    )

    parser.add_argument(
        "--graph-id",
        default="synthetic_10k",
        help="Graph id to benchmark."
    )

    parser.add_argument(
        "--relation",
        default="suitable_for",
        help="Relation to query."
    )

    parser.add_argument(
        "--bucket-count",
        type=int,
        default=64,
        help="Bucket count used during insertion."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel worker threads."
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Number of timed repetitions."
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup repetitions."
    )

    parser.add_argument(
        "--keyspace",
        default="ai_memory",
        help="Cassandra keyspace."
    )

    args = parser.parse_args()

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect(args.keyspace)

    print("Relation Query Benchmark: Serial vs Parallel")
    print("--------------------------------------------")
    print(f"graph_id     : {args.graph_id}")
    print(f"relation     : {args.relation}")
    print(f"bucket_count : {args.bucket_count}")
    print(f"workers      : {args.workers}")
    print(f"repeat       : {args.repeat}")
    print(f"warmup       : {args.warmup}")
    print()

    serial_result = benchmark_one(
        name="Serial bucket relation query",
        func=lambda: query_relation_serial(
            session=session,
            graph_id=args.graph_id,
            relation=args.relation,
            bucket_count=args.bucket_count,
        ),
        repeat=args.repeat,
        warmup=args.warmup,
    )

    parallel_result = benchmark_one(
        name="Parallel bucket relation query",
        func=lambda: query_relation_parallel(
            session=session,
            graph_id=args.graph_id,
            relation=args.relation,
            bucket_count=args.bucket_count,
            workers=args.workers,
        ),
        repeat=args.repeat,
        warmup=args.warmup,
    )

    print_result(serial_result)
    print_result(parallel_result)

    if parallel_result["avg_ms"] > 0:
        speedup = serial_result["avg_ms"] / parallel_result["avg_ms"]
        print()
        print(f"Speedup: {speedup:.2f}x")

    cluster.shutdown()


if __name__ == "__main__":
    main()