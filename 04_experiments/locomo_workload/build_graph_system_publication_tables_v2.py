"""Build rounded, publication-facing tables for graph-aware system experiments."""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "05_reports" / "locomo_workload_graph_v2_100k"
MAIN_DIR = BASE / "main_95_5_v2"
QUERY_DIR = BASE / "query_types_v2"
REPORT = BASE / "GRAPH_AWARE_SYSTEM_RESULTS_V2.md"


CELL_NAMES = {
    "cassandra-base": "Cassandra-base",
    "cassandra-materialized": "Cassandra-materialized",
    "neo4j-native": "Neo4j-native",
    "neo4j-materialized": "Neo4j-materialized",
}

QUERY_NAMES = {
    "scope_fetch": "Scope fetch",
    "relation_filter": "Relation filter",
    "candidate_projection": "Candidate projection",
    "top10_retrieval": "End-to-end Top-10",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(value: str, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def markdown_table(fields: list[str], rows: list[dict[str, object]]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row[field]) for field in fields) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    raw_main = read_csv(MAIN_DIR / "main_95_5_summary.csv")
    raw_query = read_csv(QUERY_DIR / "query_type_summary.csv")

    main_rows = []
    for row in raw_main:
        main_rows.append({
            "Backend design": CELL_NAMES[row["cell"]],
            "Concurrency": int(row["concurrency"]),
            "p50 (ms)": f(row["latency_p50_ms_median"]),
            "p95 (ms)": f(row["latency_p95_ms_median"]),
            "p99 (ms)": f(row["latency_p99_ms_median"]),
            "Throughput (ops/s)": f(row["throughput_ops_s_median"]),
            "FreshHit@10": f(row["fresh_hit_at_10_median"], 3),
            "Error rate": f(row["error_rate_median"], 3),
        })

    query_rows = []
    for row in raw_query:
        query_rows.append({
            "Backend design": CELL_NAMES[row["cell"]],
            "Query type": QUERY_NAMES[row["query_type"]],
            "p50 (ms)": f(row["p50_ms_median"]),
            "p95 (ms)": f(row["p95_ms_median"]),
            "p99 (ms)": f(row["p99_ms_median"]),
            "Throughput (QPS)": f(row["qps_median"]),
            "Error rate": f(row["error_rate_median"], 3),
        })

    main_fields = list(main_rows[0])
    query_fields = list(query_rows[0])
    write_csv(MAIN_DIR / "main_95_5_publication_table.csv", main_fields, main_rows)
    write_csv(QUERY_DIR / "query_type_publication_table.csv", query_fields, query_rows)

    report = f"""# Graph-aware 100K system experiments (v2)

## Protocol

- Workload: deterministic LoCoMo-shaped trace replay expanded to exactly 100,000 memories across 171 namespaces.
- Logical event: `graph-aware-logical-event-v2`; every update writes the same semantic memory, feature, entity, edge, and retrieval-projection content in all four cells.
- Four cells: Cassandra-base, Cassandra-materialized, Neo4j-native, and Neo4j-materialized.
- Main workload: 95% reads / 5% updates; concurrency 1, 8, 16, 32, and 64; three repetitions; 500 warm-up plus 5,000 measured operations per cell/concurrency/repetition.
- Query-type experiment: concurrency 32; three repetitions; 500 warm-up plus 5,000 measured queries per query type/repetition.
- Reported values: median of three run-level measurements. Raw tables retain min/max values.
- Correctness: all formal runs passed, with zero observed operation errors. After the runs, all four cells still contained exactly 100,000 memory records.

## Table 1. Graph-aware 100K mixed workload (95:5)

{markdown_table(main_fields, main_rows)}

## Table 2. Graph-aware query-type performance (concurrency 32)

{markdown_table(query_fields, query_rows)}

## Main observations

1. Cassandra scales markedly better after concurrency 1. At concurrency 32, Cassandra-base sustains 222.55 ops/s and Cassandra-materialized 220.21 ops/s, versus 28.83 and 28.17 ops/s for the two Neo4j cells.
2. Materialization has its clearest benefit on structured primitives. At concurrency 32, Cassandra-materialized reaches 3,587.46 QPS for relation filtering and 3,412.06 QPS for candidate projection, versus 1,356.83 and 1,654.73 QPS for Cassandra-base.
3. End-to-end Top-10 remains dominated by shared retrieval work: Cassandra-base and Cassandra-materialized achieve 367.29 and 355.64 QPS. Materialization therefore improves structured-view access but does not automatically improve the full ranking pipeline.
4. FreshHit@10 is identical (0.764 median) across all cells because the four backends preserve the same candidate and ranking semantics after a completed logical event. It is an effectiveness/correctness measure here, not a deadline-based freshness curve.
5. These measurements characterize this fixed local hardware, driver, schema, and LoCoMo-shaped workload. They should not be generalized as universal Cassandra-versus-Neo4j performance claims.

## Artifact map

- Main raw run table: `main_95_5_v2/main_95_5_runs.csv`
- Main full aggregate: `main_95_5_v2/main_95_5_summary.csv`
- Main publication table: `main_95_5_v2/main_95_5_publication_table.csv`
- Query raw run table: `query_types_v2/query_type_runs.csv`
- Query full aggregate: `query_types_v2/query_type_summary.csv`
- Query publication table: `query_types_v2/query_type_publication_table.csv`
"""
    REPORT.write_text(report, encoding="utf-8")
    print(MAIN_DIR / "main_95_5_publication_table.csv")
    print(QUERY_DIR / "query_type_publication_table.csv")
    print(REPORT)


if __name__ == "__main__":
    main()
