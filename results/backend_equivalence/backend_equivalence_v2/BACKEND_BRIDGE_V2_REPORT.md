# Backend Bridge v2 Report

**Overall gate: PASS.** This report separates storage-input parity, exact ranking parity, retrieval-metric parity, and agreement with frozen CSV references.

## Protocol

All six methods use the canonical Cat1-4 query universe (1,540 questions). Retrieval effectiveness excludes the four official evidence-empty questions and therefore uses 1,536 questions. MRR is truncated at 10; Recall@10 is macro-averaged relevant-memory recall, not the historical Hit@10 label; nDCG@10 uses binary relevance.

## Exact backend Top-10 parity

| Method | Backend | Shared | Exact | Rate | Rank mismatches | Status |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | cassandra | 1540 | 1540 | 1.000000 | 0 | PASS |
| BM25 | neo4j | 1540 | 1540 | 1.000000 | 0 | PASS |
| Dense-bge | cassandra | 1540 | 1540 | 1.000000 | 0 | PASS |
| Dense-bge | neo4j | 1540 | 1540 | 1.000000 | 0 | PASS |
| Dense+GlobalKG | cassandra | 1540 | 1540 | 1.000000 | 0 | PASS |
| Dense+GlobalKG | neo4j | 1540 | 1540 | 1.000000 | 0 | PASS |
| RRF_compact | cassandra | 1540 | 1540 | 1.000000 | 0 | PASS |
| RRF_compact | neo4j | 1540 | 1540 | 1.000000 | 0 | PASS |
| ZScore-Raw | cassandra | 1540 | 1540 | 1.000000 | 0 | PASS |
| ZScore-Raw | neo4j | 1540 | 1540 | 1.000000 | 0 | PASS |
| ZScore-RawERK | cassandra | 1540 | 1540 | 1.000000 | 0 | PASS |
| ZScore-RawERK | neo4j | 1540 | 1540 | 1.000000 | 0 | PASS |

## Retrieval metrics

| Backend | Method | n | MRR@10 | Hit@10 | Recall@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| csv | BM25 | 1536 | 0.351734 | 0.551432 | 0.496920 | 0.373293 |
| csv | Dense-bge | 1536 | 0.488093 | 0.733724 | 0.666027 | 0.512138 |
| csv | Dense+GlobalKG | 1536 | 0.515486 | 0.760417 | 0.691911 | 0.537343 |
| csv | RRF_compact | 1536 | 0.520426 | 0.786458 | 0.714745 | 0.546753 |
| csv | ZScore-Raw | 1536 | 0.506249 | 0.770833 | 0.698523 | 0.532632 |
| csv | ZScore-RawERK | 1536 | 0.546828 | 0.798177 | 0.724882 | 0.569162 |
| cassandra | BM25 | 1536 | 0.351734 | 0.551432 | 0.496920 | 0.373293 |
| cassandra | Dense-bge | 1536 | 0.488093 | 0.733724 | 0.666027 | 0.512138 |
| cassandra | Dense+GlobalKG | 1536 | 0.515486 | 0.760417 | 0.691911 | 0.537343 |
| cassandra | RRF_compact | 1536 | 0.520426 | 0.786458 | 0.714745 | 0.546753 |
| cassandra | ZScore-Raw | 1536 | 0.506249 | 0.770833 | 0.698523 | 0.532632 |
| cassandra | ZScore-RawERK | 1536 | 0.546828 | 0.798177 | 0.724882 | 0.569162 |
| neo4j | BM25 | 1536 | 0.351734 | 0.551432 | 0.496920 | 0.373293 |
| neo4j | Dense-bge | 1536 | 0.488093 | 0.733724 | 0.666027 | 0.512138 |
| neo4j | Dense+GlobalKG | 1536 | 0.515486 | 0.760417 | 0.691911 | 0.537343 |
| neo4j | RRF_compact | 1536 | 0.520426 | 0.786458 | 0.714745 | 0.546753 |
| neo4j | ZScore-Raw | 1536 | 0.506249 | 0.770833 | 0.698523 | 0.532632 |
| neo4j | ZScore-RawERK | 1536 | 0.546828 | 0.798177 | 0.724882 | 0.569162 |

## CSV adapter versus frozen references

| Method | Frozen q | CSV q | Exact shared | Expected empty-gold extras | Status |
| --- | --- | --- | --- | --- | --- |
| BM25 | 1540 | 1540 | 1540 | 0 | PASS |
| Dense-bge | 1540 | 1540 | 1540 | 0 | PASS |
| Dense+GlobalKG | 1536 | 1540 | 1536 | 4 | PASS_WITH_EXPECTED_EMPTY_GOLD_CAVEAT |
| RRF_compact | 1540 | 1540 | 1540 | 0 | PASS |
| ZScore-Raw | 1540 | 1540 | 1540 | 0 | PASS |
| ZScore-RawERK | 1540 | 1540 | 1540 | 0 | PASS |

Dense+GlobalKG uses the corrected degree-centrality ranking in `results/retrieval/dense_global_kg_rerun/dense_global_kg_top10.csv`. That frozen artifact intentionally contains only the 1,536 evidence-bearing questions. The four evidence-empty rankings are tested for CSV/Cassandra/Neo4j parity but have no retrieval-effectiveness metric.

## Input scope digests

| Backend | Field | Comparable scopes | Matched | Status |
| --- | --- | --- | --- | --- |
| cassandra | candidate_count | 10 | 10 | PASS |
| cassandra | candidate_digest | 10 | 10 | PASS |
| cassandra | projection_digest | 10 | 10 | PASS |
| cassandra | corpus_order_digest | 10 | 10 | PASS |
| cassandra | edge_row_count | 10 | 10 | PASS |
| cassandra | edge_assignment_count | 10 | 10 | PASS |
| cassandra | edge_digest | 10 | 10 | PASS |
| neo4j | candidate_count | 10 | 10 | PASS |
| neo4j | candidate_digest | 10 | 10 | PASS |
| neo4j | projection_digest | 10 | 10 | PASS |
| neo4j | corpus_order_digest | 10 | 10 | PASS |
| neo4j | edge_row_count | 10 | 10 | PASS |
| neo4j | edge_assignment_count | 10 | 10 | PASS |
| neo4j | edge_digest | 10 | 10 | PASS |

## Gate interpretation

The detailed mismatch ledger contains **0** rows. A publication claim of lossless backend migration requires every non-N/A digest gate, every exact Top-10 gate, and every metric-difference gate to pass. Missing digest values are not silently treated as matches.

Machine-readable evidence is in `backend_parity_summary.csv`, `input_digest_parity_summary.csv`, `retrieval_metrics_by_backend.csv`, `retrieval_metric_differences.csv`, `csv_frozen_reference_summary.csv`, `densekg_empty_gold_ranking_parity.csv`, `mismatches.csv`, and `manifest.json`.

Dense+GlobalKG expected empty-gold reference exclusions recorded: 4.
