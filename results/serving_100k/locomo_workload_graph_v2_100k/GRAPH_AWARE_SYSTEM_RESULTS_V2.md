# Graph-aware 100K system experiments (v2)

## Protocol

- Workload: deterministic LoCoMo-shaped trace replay expanded to exactly 100,000 memories across 171 namespaces.
- Logical event: `graph-aware-logical-event-v2`; every update writes the same semantic memory, feature, entity, edge, and retrieval-projection content in all four cells.
- Four cells: Cassandra-base, Cassandra-materialized, Neo4j-native, and Neo4j-materialized.
- Main workload: 95% reads / 5% updates; concurrency 1, 8, 16, 32, and 64; three repetitions; 500 warm-up plus 5,000 measured operations per cell/concurrency/repetition.
- Query-type experiment: concurrency 32; three repetitions; 500 warm-up plus 5,000 measured queries per query type/repetition.
- Reported values: median of three run-level measurements. Raw tables retain min/max values.
- Correctness: all formal runs passed, with zero observed operation errors. After the runs, all four cells still contained exactly 100,000 memory records.

## Table 1. Graph-aware 100K mixed workload (95:5)

| Backend design | Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (ops/s) | FreshHit@10 | Error rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cassandra-base | 1 | 16.10 | 91.51 | 134.97 | 46.28 | 0.764 | 0.000 |
| Cassandra-base | 8 | 20.95 | 118.52 | 255.02 | 222.85 | 0.764 | 0.000 |
| Cassandra-base | 16 | 37.99 | 317.38 | 628.45 | 223.87 | 0.764 | 0.000 |
| Cassandra-base | 32 | 74.10 | 712.84 | 1204.11 | 222.55 | 0.764 | 0.000 |
| Cassandra-base | 64 | 159.64 | 1063.39 | 2099.98 | 234.08 | 0.764 | 0.000 |
| Cassandra-materialized | 1 | 15.89 | 83.84 | 118.75 | 48.70 | 0.764 | 0.000 |
| Cassandra-materialized | 8 | 20.73 | 119.21 | 261.40 | 223.98 | 0.764 | 0.000 |
| Cassandra-materialized | 16 | 38.11 | 342.85 | 608.61 | 221.97 | 0.764 | 0.000 |
| Cassandra-materialized | 32 | 75.30 | 720.12 | 1172.27 | 220.21 | 0.764 | 0.000 |
| Cassandra-materialized | 64 | 166.28 | 1042.13 | 2012.92 | 232.23 | 0.764 | 0.000 |
| Neo4j-native | 1 | 13.11 | 48.52 | 78.34 | 64.10 | 0.764 | 0.000 |
| Neo4j-native | 8 | 251.29 | 330.65 | 409.36 | 32.52 | 0.764 | 0.000 |
| Neo4j-native | 16 | 522.70 | 650.60 | 794.69 | 31.54 | 0.764 | 0.000 |
| Neo4j-native | 32 | 1146.14 | 1416.08 | 1776.94 | 28.83 | 0.764 | 0.000 |
| Neo4j-native | 64 | 2374.95 | 2858.48 | 3497.56 | 28.09 | 0.764 | 0.000 |
| Neo4j-materialized | 1 | 13.39 | 47.34 | 76.95 | 61.61 | 0.764 | 0.000 |
| Neo4j-materialized | 8 | 266.05 | 371.17 | 468.31 | 30.28 | 0.764 | 0.000 |
| Neo4j-materialized | 16 | 559.09 | 711.03 | 925.68 | 29.34 | 0.764 | 0.000 |
| Neo4j-materialized | 32 | 1150.83 | 1538.14 | 1819.89 | 28.17 | 0.764 | 0.000 |
| Neo4j-materialized | 64 | 2191.12 | 2563.40 | 3048.63 | 30.56 | 0.764 | 0.000 |

## Table 2. Graph-aware query-type performance (concurrency 32)

| Backend design | Query type | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (QPS) | Error rate |
| --- | --- | --- | --- | --- | --- | --- |
| Cassandra-base | Scope fetch | 24.75 | 116.89 | 191.29 | 871.14 | 0.000 |
| Cassandra-base | Relation filter | 15.67 | 98.10 | 148.16 | 1356.83 | 0.000 |
| Cassandra-base | Candidate projection | 15.40 | 30.24 | 109.42 | 1654.73 | 0.000 |
| Cassandra-base | End-to-end Top-10 | 74.12 | 189.09 | 274.05 | 367.29 | 0.000 |
| Cassandra-materialized | Scope fetch | 27.71 | 116.41 | 182.27 | 822.92 | 0.000 |
| Cassandra-materialized | Relation filter | 6.59 | 12.87 | 73.33 | 3587.46 | 0.000 |
| Cassandra-materialized | Candidate projection | 6.98 | 11.98 | 98.36 | 3412.06 | 0.000 |
| Cassandra-materialized | End-to-end Top-10 | 77.20 | 192.89 | 271.36 | 355.64 | 0.000 |
| Neo4j-native | Scope fetch | 877.11 | 1010.60 | 1070.06 | 39.37 | 0.000 |
| Neo4j-native | Relation filter | 38.32 | 52.80 | 130.30 | 777.22 | 0.000 |
| Neo4j-native | Candidate projection | 9.95 | 21.28 | 64.00 | 1942.77 | 0.000 |
| Neo4j-native | End-to-end Top-10 | 943.66 | 1080.10 | 1146.22 | 36.35 | 0.000 |
| Neo4j-materialized | Scope fetch | 876.26 | 1015.16 | 1081.39 | 39.24 | 0.000 |
| Neo4j-materialized | Relation filter | 33.44 | 62.59 | 108.77 | 842.35 | 0.000 |
| Neo4j-materialized | Candidate projection | 13.81 | 20.94 | 62.67 | 2027.89 | 0.000 |
| Neo4j-materialized | End-to-end Top-10 | 948.17 | 1097.68 | 1168.94 | 36.24 | 0.000 |

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
