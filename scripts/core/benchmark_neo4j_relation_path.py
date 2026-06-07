import argparse
import csv
import time
from pathlib import Path
from statistics import mean

from neo4j import GraphDatabase


RELATION_PATH_QUERY = """
MATCH path =
    (s:KGNode {node_id: $start})-[r1:KG_EDGE]->(n1:KGNode)
    -[r2:KG_EDGE]->(n2:KGNode)
    -[r3:KG_EDGE]->(n3:KGNode)
    -[r4:KG_EDGE]->(n4:KGNode)
WHERE r1.graph_id = $graph_id
  AND r2.graph_id = $graph_id
  AND r3.graph_id = $graph_id
  AND r4.graph_id = $graph_id
  AND r1.relation = $rel1
  AND r2.relation = $rel2
  AND r3.relation = $rel3
  AND r4.relation = $rel4
RETURN path
LIMIT $limit
"""


ONE_HOP_RELATION_QUERY = """
MATCH (s:KGNode {node_id: $start})-[r:KG_EDGE]->(d:KGNode)
WHERE r.graph_id = $graph_id
  AND r.relation = $relation
RETURN s.node_id AS src_id, r.relation AS relation, d.node_id AS dst_id
LIMIT $limit
"""


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)
    index = int(round((p / 100) * (len(values) - 1)))
    return values[index]


def run_query(session, query, params):
    start = time.perf_counter()
    rows = list(session.run(query, **params))
    end = time.perf_counter()

    return {
        "latency_ms": (end - start) * 1000,
        "count": len(rows),
    }


def benchmark(session, name, query, params, repeat, warmup):
    for _ in range(warmup):
        run_query(session, query, params)

    results = []

    for _ in range(repeat):
        results.append(run_query(session, query, params))

    latencies = [row["latency_ms"] for row in results]
    counts = [row["count"] for row in results]

    return {
        "name": name,
        "avg_ms": mean(latencies),
        "p95_ms": percentile(latencies, 95),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "count": mean(counts),
    }


def write_csv(args, rows):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()

    fieldnames = [
        "graph_id",
        "start",
        "relations",
        "repeat",
        "warmup",
        "query_name",
        "avg_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "count",
    ]

    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in rows:
            writer.writerow({
                "graph_id": args.graph_id,
                "start": args.start,
                "relations": args.relations,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "query_name": row["name"],
                "avg_ms": round(row["avg_ms"], 3),
                "p95_ms": round(row["p95_ms"], 3),
                "min_ms": round(row["min_ms"], 3),
                "max_ms": round(row["max_ms"], 3),
                "count": round(row["count"], 3),
            })


def main():
    parser = argparse.ArgumentParser(
        description="Neo4j baseline for relation-specific 4-hop path query."
    )

    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument(
        "--relations",
        default="likes,suitable_for,related_to,suggests",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password123")
    parser.add_argument(
        "--output",
        default="results/benchmark_neo4j_relation_path_results.csv",
    )

    args = parser.parse_args()

    relations = [
        item.strip()
        for item in args.relations.split(",")
        if item.strip()
    ]

    if len(relations) != 4:
        raise ValueError("This benchmark requires exactly 4 relations.")

    driver = GraphDatabase.driver(
        args.uri,
        auth=(args.user, args.password),
    )

    rows = []

    with driver.session() as session:
        rows.append(
            benchmark(
                session=session,
                name="neo4j_relation_one_hop",
                query=ONE_HOP_RELATION_QUERY,
                params={
                    "graph_id": args.graph_id,
                    "start": args.start,
                    "relation": relations[0],
                    "limit": args.limit,
                },
                repeat=args.repeat,
                warmup=args.warmup,
            )
        )

        rows.append(
            benchmark(
                session=session,
                name="neo4j_relation_sequence_4hop",
                query=RELATION_PATH_QUERY,
                params={
                    "graph_id": args.graph_id,
                    "start": args.start,
                    "rel1": relations[0],
                    "rel2": relations[1],
                    "rel3": relations[2],
                    "rel4": relations[3],
                    "limit": args.limit,
                },
                repeat=args.repeat,
                warmup=args.warmup,
            )
        )

    driver.close()

    print("Neo4j Relation Path Benchmark")
    print("-----------------------------")
    print(f"graph_id  : {args.graph_id}")
    print(f"start     : {args.start}")
    print(f"relations : {args.relations}")
    print()

    print(
        f"{'query':<34}"
        f"{'avg_ms':<12}"
        f"{'p95_ms':<12}"
        f"{'count':<10}"
    )
    print("-" * 70)

    for row in rows:
        print(
            f"{row['name']:<34}"
            f"{row['avg_ms']:<12.3f}"
            f"{row['p95_ms']:<12.3f}"
            f"{row['count']:<10.1f}"
        )

    write_csv(args, rows)


if __name__ == "__main__":
    main()