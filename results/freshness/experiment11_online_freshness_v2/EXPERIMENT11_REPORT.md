# Experiment 11: Online Memory Freshness / Time-to-Top10

## Frozen protocol

- Corpus: deterministic 100K LoCoMo-shaped trace replay, 171 namespaces.
- Cells: Cassandra-base, Cassandra-materialized, Neo4j-native, Neo4j-materialized.
- Workload: open-loop 95% reads / 5% updates, concurrency 32.
- Update rates: 1, 2, 5, and 10 updates/s (20, 40, 100, and 200 total operations/s).
- Sampling: at least 40 updates per rate/repetition; three repetitions; 2,400 measured staged updates and 48,000 total operations.
- Inputs prepared before t0: raw text, ERK/triples, frozen BGE embedding, and query embedding. LLM extraction and embedding generation are excluded.
- Aggregation: median of three repetition-level statistics. No failed, missed, or duplicate updates were removed.

## Timing boundaries

- `t_commit`: arrival to acknowledged raw-memory row/node commit.
- `t_structured_view_visible`: arrival to full graph digest, RawERK candidate projection, and relation-candidate visibility.
- `t_index_visible`: arrival to both sparse and dense index version visibility.
- `time_to_top10`: arrival to first inclusion in the real Dense + BM25-RawERK + query-wise ZScore Top-10; non-hits remain in the FreshHit denominator.
- `Pipeline complete`: arrival to completed Top-10 computation, including final non-hits.

## Table 11A. Stage latency at the main 100 ops/s point

| Cell | Stage (cumulative from arrival) | Update rate (/s) | Read rate (/s) | p50 ms | p95 ms | p99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| cassandra-base | Raw backend commit | 5.0 | 95.0 | 7.716 | 12.862 | 14.211 |
| cassandra-base | Structured view visible | 5.0 | 95.0 | 62.696 | 124.351 | 134.264 |
| cassandra-base | Sparse+dense index visible | 5.0 | 95.0 | 93.48 | 164.083 | 189.434 |
| cassandra-base | Pipeline complete | 5.0 | 95.0 | 109.196 | 178.977 | 203.849 |
| cassandra-base | First inclusion in Top-10 (final hits) | 5.0 | 95.0 | 104.588 | 187.297 | 200.296 |
| cassandra-materialized | Raw backend commit | 5.0 | 95.0 | 6.964 | 12.624 | 12.894 |
| cassandra-materialized | Structured view visible | 5.0 | 95.0 | 47.693 | 69.815 | 107.464 |
| cassandra-materialized | Sparse+dense index visible | 5.0 | 95.0 | 77.472 | 153.509 | 188.862 |
| cassandra-materialized | Pipeline complete | 5.0 | 95.0 | 86.682 | 163.278 | 205.587 |
| cassandra-materialized | First inclusion in Top-10 (final hits) | 5.0 | 95.0 | 88.21 | 161.296 | 204.351 |
| neo4j-native | Raw backend commit | 5.0 | 95.0 | 8134.968 | 15695.011 | 16439.624 |
| neo4j-native | Structured view visible | 5.0 | 95.0 | 8203.85 | 15789.028 | 16517.064 |
| neo4j-native | Sparse+dense index visible | 5.0 | 95.0 | 8498.87 | 15983.783 | 16751.792 |
| neo4j-native | Pipeline complete | 5.0 | 95.0 | 9791.851 | 16574.197 | 17209.074 |
| neo4j-native | First inclusion in Top-10 (final hits) | 5.0 | 95.0 | 9991.408 | 16543.619 | 16856.62 |
| neo4j-materialized | Raw backend commit | 5.0 | 95.0 | 8352.626 | 16642.193 | 17243.42 |
| neo4j-materialized | Structured view visible | 5.0 | 95.0 | 8418.735 | 16716.849 | 17302.214 |
| neo4j-materialized | Sparse+dense index visible | 5.0 | 95.0 | 8719.219 | 16928.515 | 17555.241 |
| neo4j-materialized | Pipeline complete | 5.0 | 95.0 | 9829.912 | 17605.288 | 18071.973 |
| neo4j-materialized | First inclusion in Top-10 (final hits) | 5.0 | 95.0 | 11583.611 | 17827.498 | 18094.567 |

## Table 11B. Deadline freshness and overload behavior

| Cell | Update rate (/s) | Total arrival rate (/s) | Pipeline p95 ms | FreshHit@10 50ms | FreshHit@10 100ms | FreshHit@10 250ms | FreshHit@10 500ms | FreshHit@10 1s | Pipeline visible @1s | Final Hit@10 | Update backlog peak | Backlog drain ms | Timeout rate (>5s) | Error rate | Missed update rate | Duplicate update rate | Stale-result rate @1s | SLO pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cassandra-base | 1.0 | 20.0 | 168.176 | 0.0 | 0.0 | 0.7 | 0.7 | 0.7 | 1.0 | 0.7 | 1 | 172.124 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| cassandra-base | 2.0 | 40.0 | 183.849 | 0.0 | 0.0 | 0.725 | 0.725 | 0.725 | 1.0 | 0.725 | 1 | 174.025 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| cassandra-base | 5.0 | 100.0 | 178.977 | 0.0 | 0.275 | 0.725 | 0.725 | 0.725 | 1.0 | 0.725 | 2 | 159.92 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| cassandra-base | 10.0 | 200.0 | 366.024 | 0.0 | 0.1625 | 0.625 | 0.7625 | 0.7625 | 1.0 | 0.7625 | 5 | 189.585 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| cassandra-materialized | 1.0 | 20.0 | 170.365 | 0.0 | 0.025 | 0.7 | 0.7 | 0.7 | 1.0 | 0.7 | 1 | 142.031 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| cassandra-materialized | 2.0 | 40.0 | 137.164 | 0.0 | 0.25 | 0.725 | 0.725 | 0.725 | 1.0 | 0.725 | 1 | 136.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| cassandra-materialized | 5.0 | 100.0 | 163.278 | 0.0 | 0.525 | 0.725 | 0.725 | 0.725 | 1.0 | 0.725 | 2 | 132.264 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| cassandra-materialized | 10.0 | 200.0 | 376.475 | 0.0 | 0.225 | 0.6 | 0.75 | 0.7625 | 1.0 | 0.7625 | 6 | 141.429 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| neo4j-native | 1.0 | 20.0 | 137.557 | 0.0 | 0.4 | 0.7 | 0.7 | 0.7 | 1.0 | 0.7 | 1 | 85.805 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| neo4j-native | 2.0 | 40.0 | 4480.438 | 0.0 | 0.025 | 0.075 | 0.1 | 0.15 | 0.175 | 0.725 | 8 | 4034.446 | 0.0 | 0.0 | 0.0 | 0.0 | 0.825 | 0 |
| neo4j-native | 5.0 | 100.0 | 16574.197 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.725 | 29 | 17106.488 | 0.775 | 0.0 | 0.0 | 0.0 | 1.0 | 0 |
| neo4j-native | 10.0 | 200.0 | 41235.23 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7625 | 69 | 42459.856 | 0.9 | 0.0 | 0.0 | 0.0 | 1.0 | 0 |
| neo4j-materialized | 1.0 | 20.0 | 141.443 | 0.0 | 0.4 | 0.7 | 0.7 | 0.7 | 1.0 | 0.7 | 1 | 81.459 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1 |
| neo4j-materialized | 2.0 | 40.0 | 5313.437 | 0.0 | 0.025 | 0.075 | 0.1 | 0.125 | 0.175 | 0.725 | 9 | 4920.323 | 0.1 | 0.0 | 0.0 | 0.0 | 0.825 | 0 |
| neo4j-materialized | 5.0 | 100.0 | 17605.288 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.725 | 29 | 17958.213 | 0.775 | 0.0 | 0.0 | 0.0 | 1.0 | 0 |
| neo4j-materialized | 10.0 | 200.0 | 44583.025 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7625 | 70 | 46006.43 | 0.9125 | 0.0 | 0.0 | 0.0 | 1.0 | 0 |

## Sustainable update rate under the frozen SLO

SLO: pipeline p95 <= 2,000 ms, backlog drain <= 2,000 ms, and zero timeout/error/missed/duplicate updates.

| Cell | Max tested sustainable update rate (/s) | Corresponding total arrival rate (/s) | SLO |
| --- | --- | --- | --- |
| cassandra-base | 10.0 | 200.0 | p95 pipeline <= 2000 ms; drain <= 2000 ms; zero timeout/error/missed/duplicate |
| cassandra-materialized | 10.0 | 200.0 | p95 pipeline <= 2000 ms; drain <= 2000 ms; zero timeout/error/missed/duplicate |
| neo4j-native | 1.0 | 20.0 | p95 pipeline <= 2000 ms; drain <= 2000 ms; zero timeout/error/missed/duplicate |
| neo4j-materialized | 1.0 | 20.0 | p95 pipeline <= 2000 ms; drain <= 2000 ms; zero timeout/error/missed/duplicate |

## Conclusions and claim boundary

1. At light load (20 total ops/s), Neo4j is competitive and has lower p95 than Cassandra in this local setup. The result is not a universal claim that Cassandra has lower single-event latency.
2. At the main 100 ops/s point, Cassandra-materialized reaches structured-view visibility at p95 69.815 ms and complete Top-10 at p95 163.278 ms. Neo4j-materialized is queue-saturated: the corresponding p95 values are 16,716.849 ms and 17,605.288 ms.
3. Cassandra-materialized sustains the highest tested 10 updates/s (200 total ops/s) under the frozen SLO. Both Neo4j cells sustain 1 update/s (20 total ops/s); 2 updates/s already exceeds the drain and p95 limits.
4. Materialization reduces Cassandra structured-view p95 at the 100 ops/s point (124.351 -> 69.815 ms), but it does not remove the shared sparse/dense reranking cost. Neo4j materialization does not prevent queue saturation at 40+ total ops/s in this implementation.
5. Supported claim: under the tested 100K LoCoMo-shaped open-loop workload, Cassandra-materialized keeps newly committed structured evidence entering the real CassMem Top-10 within the serving SLO at substantially higher arrival rates. Do not claim that Cassandra is always faster at low load.
