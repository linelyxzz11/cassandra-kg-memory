# Table 7: Backend Serving Performance

## Table 7A. Actual storage work for the 100K graph-aware workload

| Backend design | Memory rows / nodes | Entity records | Mention records | Semantic edges | Materialized candidates | Physical records | Logical writes / event |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cassandra-base | 100,000 | 76,475 | 129,156 | 42,944 | 0 | 534,463 | 6.1 |
| Cassandra-materialized | 100,000 | 76,475 | 129,156 | 42,944 | 42,944 | 391,519 | 4.7 |
| Neo4j-native | 100,000 | 76,475 | 129,156 | 42,944 | 0 | 548,575 | 6.3 |
| Neo4j-materialized | 100,000 | 76,475 | 129,156 | 42,944 | 42,723 | 391,298 | 4.7 |

All four cells contain the same 100,000 logical memories, 76,475 entities, 129,156 mentions, and 42,944 semantic edges. `Physical records` counts rows, nodes, and relationships rather than bytes on disk.

## Table 7B. Serving scalability under the 95% read / 5% update workload

| Backend design | Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (ops/s) | FreshHit@10 | Error rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cassandra-base | 1 | 16.1 | 91.5 | 135.0 | 46.3 | 76.4% | 0.0% |
| Cassandra-materialized | 1 | 15.9 | 83.8 | 118.8 | 48.7 | 76.4% | 0.0% |
| Neo4j-native | 1 | 13.1 | 48.5 | 78.3 | 64.1 | 76.4% | 0.0% |
| Neo4j-materialized | 1 | 13.4 | 47.3 | 77.0 | 61.6 | 76.4% | 0.0% |
| Cassandra-base | 8 | 21.0 | 118.5 | 255.0 | 222.9 | 76.4% | 0.0% |
| Cassandra-materialized | 8 | 20.7 | 119.2 | 261.4 | 224.0 | 76.4% | 0.0% |
| Neo4j-native | 8 | 251.3 | 330.6 | 409.4 | 32.5 | 76.4% | 0.0% |
| Neo4j-materialized | 8 | 266.1 | 371.2 | 468.3 | 30.3 | 76.4% | 0.0% |
| Cassandra-base | 16 | 38.0 | 317.4 | 628.5 | 223.9 | 76.4% | 0.0% |
| Cassandra-materialized | 16 | 38.1 | 342.8 | 608.6 | 222.0 | 76.4% | 0.0% |
| Neo4j-native | 16 | 522.7 | 650.6 | 794.7 | 31.5 | 76.4% | 0.0% |
| Neo4j-materialized | 16 | 559.1 | 711.0 | 925.7 | 29.3 | 76.4% | 0.0% |
| Cassandra-base | 32 | 74.1 | 712.8 | 1,204.1 | 222.6 | 76.4% | 0.0% |
| Cassandra-materialized | 32 | 75.3 | 720.1 | 1,172.3 | 220.2 | 76.4% | 0.0% |
| Neo4j-native | 32 | 1,146.1 | 1,416.1 | 1,776.9 | 28.8 | 76.4% | 0.0% |
| Neo4j-materialized | 32 | 1,150.8 | 1,538.1 | 1,819.9 | 28.2 | 76.4% | 0.0% |
| Cassandra-base | 64 | 159.6 | 1,063.4 | 2,100.0 | 234.1 | 76.4% | 0.0% |
| Cassandra-materialized | 64 | 166.3 | 1,042.1 | 2,012.9 | 232.2 | 76.4% | 0.0% |
| Neo4j-native | 64 | 2,374.9 | 2,858.5 | 3,497.6 | 28.1 | 76.4% | 0.0% |
| Neo4j-materialized | 64 | 2,191.1 | 2,563.4 | 3,048.6 | 30.6 | 76.4% | 0.0% |

## Table 7C. Query-type performance at concurrency 32

| Backend design | Query type | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (QPS) | Error rate |
|---|---|---:|---:|---:|---:|---:|
| Cassandra-base | Scope fetch | 24.8 | 116.9 | 191.3 | 871.1 | 0.0% |
| Cassandra-materialized | Scope fetch | 27.7 | 116.4 | 182.3 | 822.9 | 0.0% |
| Neo4j-native | Scope fetch | 877.1 | 1,010.6 | 1,070.1 | 39.4 | 0.0% |
| Neo4j-materialized | Scope fetch | 876.3 | 1,015.2 | 1,081.4 | 39.2 | 0.0% |
| Cassandra-base | Relation filter | 15.7 | 98.1 | 148.2 | 1,356.8 | 0.0% |
| **Cassandra-materialized** | **Relation filter** | **6.6** | **12.9** | **73.3** | **3,587.5** | **0.0%** |
| Neo4j-native | Relation filter | 38.3 | 52.8 | 130.3 | 777.2 | 0.0% |
| Neo4j-materialized | Relation filter | 33.4 | 62.6 | 108.8 | 842.4 | 0.0% |
| Cassandra-base | Candidate projection | 15.4 | 30.2 | 109.4 | 1,654.7 | 0.0% |
| **Cassandra-materialized** | **Candidate projection** | **7.0** | **12.0** | **98.4** | **3,412.1** | **0.0%** |
| Neo4j-native | Candidate projection | 9.9 | 21.3 | 64.0 | 1,942.8 | 0.0% |
| Neo4j-materialized | Candidate projection | 13.8 | 20.9 | 62.7 | 2,027.9 | 0.0% |
| Cassandra-base | End-to-end Top-10 | 74.1 | 189.1 | 274.0 | 367.3 | 0.0% |
| Cassandra-materialized | End-to-end Top-10 | 77.2 | 192.9 | 271.4 | 355.6 | 0.0% |
| Neo4j-native | End-to-end Top-10 | 943.7 | 1,080.1 | 1,146.2 | 36.4 | 0.0% |
| Neo4j-materialized | End-to-end Top-10 | 948.2 | 1,097.7 | 1,168.9 | 36.2 | 0.0% |

## Protocol

- Workload: LoCoMo-shaped graph-aware trace replay with 100,000 logical memories across 171 namespaces.
- Logical-event parity gate: 4,000 sampled graph-digest comparisons and 4,000 relation-candidate comparisons, with zero mismatches.
- Table 7B: 95% reads and 5% updates; concurrency 1, 8, 16, 32, and 64; three repetitions; 5,000 measured operations per run; each cell reports the median of the three run-level statistics.
- Table 7C: concurrency 32; three repetitions; 5,000 operations per query type per repetition; each cell reports the median of the three run-level statistics.
- Raw text, embeddings, triples, and ERK features are prepared before the timed database operation.
- FreshHit@10 is invariant across the four cells, confirming that throughput differences are not caused by retrieval-result degradation.

## Interpretation

1. Neo4j is faster at concurrency 1, but its throughput drops to roughly 28-33 ops/s from concurrency 8 onward. Cassandra sustains roughly 220-234 ops/s from concurrency 8 to 64.
2. At concurrency 32, Cassandra-materialized achieves 220.2 ops/s versus 28.2 ops/s for Neo4j-materialized, a 7.8x throughput advantage, while preserving the same FreshHit@10 and zero error rate.
3. Cassandra materialization strongly accelerates the targeted structured stages: relation-filter p95 falls from 98.1 ms to 12.9 ms, and candidate-projection p95 falls from 30.2 ms to 12.0 ms.
4. Materialization does not improve the static end-to-end Top-10 row: Cassandra-base and Cassandra-materialized are effectively similar there. Therefore, the paper should claim a stage-level materialization benefit and a Cassandra high-concurrency serving benefit, not a universal end-to-end materialization speedup.
5. Neo4j-materialized does not materially improve scope fetch in this design. This should be reported transparently rather than interpreted as evidence that materialization is universally ineffective.

## Supported claim

> Under the graph-aware 100K LoCoMo-shaped 95:5 online workload, Cassandra preserves retrieval freshness while sustaining substantially higher throughput and lower high-concurrency latency than Neo4j; query-oriented materialization specifically accelerates relation filtering and candidate projection.
