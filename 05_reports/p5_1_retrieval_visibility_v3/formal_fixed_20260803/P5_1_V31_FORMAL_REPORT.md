# P5-1 v3.1 Formal Report

## Verdict

The fixed-candidate-cohort benchmark is citation-ready under its stated scope. All 36,000 formal events reached the final RawERK BM25 Top-10, every target memory ranked first, every timed retrieval ranked exactly 33 candidate documents, and cross-backend logical-state/candidate/Top-10 parity passed with zero mismatches.

## Primary results

| Backend | Concurrency | Events | p50 update→TopK ms | p95 | p99 | Mean throughput ± SD events/s |
|---|---:|---:|---:|---:|---:|---:|
| cassandra | 8 | 6000 | 29.26 | 55.61 | 124.38 | 237.02 ± 14.84 |
| cassandra | 32 | 6000 | 85.62 | 201.38 | 252.94 | 316.55 ± 3.60 |
| cassandra | 64 | 6000 | 173.04 | 355.53 | 385.89 | 308.74 ± 7.05 |
| neo4j | 8 | 6000 | 47.32 | 66.31 | 84.45 | 170.79 ± 12.63 |
| neo4j | 32 | 6000 | 195.32 | 284.79 | 331.10 | 163.38 ± 2.15 |
| neo4j | 64 | 6000 | 402.92 | 561.86 | 637.33 | 157.66 ± 1.90 |

Throughput is formal successful events divided by the timed formal interval; baseline seeding and warmup are excluded. Each cell pools 3 runs × 2,000 events.

## Cross-backend effects

| Concurrency | Cassandra p50 | Neo4j p50 | Cassandra p50 reduction | Neo4j/Cassandra p50 | Cassandra/Neo4j throughput |
|---:|---:|---:|---:|---:|---:|
| 8 | 29.26 | 47.32 | 38.16% | 1.62× | 1.39× |
| 32 | 85.62 | 195.32 | 56.16% | 2.28× | 1.94× |
| 64 | 173.04 | 402.92 | 57.05% | 2.33× | 1.96× |

## Validity gates

- Manifest rows: expected 36000, actual 36000.
- Timeouts: 0.
- Fixed candidate count: 33/33 for every event.
- Sequential parity: PASS; logical mismatches=0, candidate mismatches=0, Top-K mismatches=0.
- Cross-backend event digests match for every concurrency/run pair.

## Interpretation

Under the same logical update and exactly fixed retrieval cohort, Cassandra has lower median update-to-final-TopK latency at c=8, c=32 and c=64. The advantage grows under higher concurrency. This supports a backend-serving claim for the RawERK BM25 reference retriever, not yet the full online Dense+RawERK CassMem fusion pipeline.
At c=8, Cassandra's p99 is 124.38 ms versus Neo4j's 84.45 ms even though Cassandra has lower p50 and p95. Therefore the result does not support a blanket claim that Cassandra dominates every tail-latency percentile.

## Limitations

- Candidate cohort is controlled at 33 documents to guarantee identical work; scale sensitivity must be measured separately.
- The unique probe token makes retrieval membership deterministic and tests visibility/serving rather than ranking difficulty.
- Online dense embedding generation and full CassMem Z-score fusion are not included.
- Results are from one local machine and one Cassandra/Neo4j deployment; environment details are in `environment_snapshot.json`.
- Only three repetitions were run per backend/concurrency; retain raw per-event and per-run files for uncertainty analysis.

## Canonical files

- `p5v31_formal_fixed_20260803_manifest.json`: protocol, hashes, exact counts and gates.
- `p5v31_formal_fixed_20260803_per_event.csv`: 36,000 raw event outcomes.
- `p5v31_formal_fixed_20260803_per_run.csv`: 18 run-level summaries.
- `p5v31_formal_fixed_20260803_summary.csv`: pooled backend/concurrency metrics.
- `p5_1_v31_main_table.csv` and `p5_1_v31_cross_backend_effects.csv`: publication-facing tables.
