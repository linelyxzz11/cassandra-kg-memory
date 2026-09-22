# Table 6: Canonical Backend Bridge

## Table 6A. Exact Top-10 parity against the frozen CSV reference

| Method | Cassandra Exact Top-10 | Neo4j Exact Top-10 | Rank Mismatches | Missing / Extra Queries | Status |
|---|---:|---:|---:|---:|:---:|
| BM25 | 100.0% (1540/1540) | 100.0% (1540/1540) | 0 | 0 / 0 | PASS |
| Dense-bge | 100.0% (1540/1540) | 100.0% (1540/1540) | 0 | 0 / 0 | PASS |
| Dense+GlobalKG | 100.0% (1540/1540) | 100.0% (1540/1540) | 0 | 0 / 0 | PASS |
| RRF_compact | 100.0% (1540/1540) | 100.0% (1540/1540) | 0 | 0 / 0 | PASS |
| ZScore-Raw | 100.0% (1540/1540) | 100.0% (1540/1540) | 0 | 0 / 0 | PASS |
| **CassMem (ZScore-RawERK)** | **100.0% (1540/1540)** | **100.0% (1540/1540)** | **0** | **0 / 0** | **PASS** |

## Table 6B. Retrieval effectiveness preserved across backends

The values below are identical for CSV, Cassandra, and Neo4j. Effectiveness is evaluated on the 1,536 evidence-bearing Cat1-4 questions.

| Method | MRR@10 | Hit@1 | Hit@5 | Hit@10 | Recall@10 | nDCG@10 | Max. Absolute Backend Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3517 | 0.2585 | 0.4688 | 0.5514 | 0.4969 | 0.3733 | 0.0000 |
| Dense-bge | 0.4881 | 0.3757 | 0.6452 | 0.7337 | 0.6660 | 0.5121 | 0.0000 |
| Dense+GlobalKG | 0.5155 | 0.4043 | 0.6667 | 0.7604 | 0.6919 | 0.5373 | 0.0000 |
| RRF_compact | 0.5204 | 0.3978 | 0.6882 | 0.7865 | 0.7147 | 0.5468 | 0.0000 |
| ZScore-Raw | 0.5062 | 0.3783 | 0.6823 | 0.7708 | 0.6985 | 0.5326 | 0.0000 |
| **CassMem (ZScore-RawERK)** | **0.5468** | **0.4245** | **0.7129** | **0.7982** | **0.7249** | **0.5692** | **0.0000** |

## Input-parity gate

| Backend | Candidate / Projection / Corpus / Edge Digest Checks | Matched Scopes | Missing Values | Status |
|---|---:|---:|---:|:---:|
| Cassandra | 7 fields x 10 scopes = 70/70 | 10/10 per field | 0 | PASS |
| Neo4j | 7 fields x 10 scopes = 70/70 | 10/10 per field | 0 | PASS |

## Protocol note

- Query universe: 1,540 canonical LoCoMo Cat1-4 questions.
- Exact ranking parity uses all 1,540 questions; effectiveness excludes four official evidence-empty questions and therefore uses 1,536.
- CSV is the frozen reference backend. Candidate scope, corpus order, compact projection, and logical-edge assignments are fixed across backends.
- Dense+GlobalKG uses the corrected degree-centrality ranking. Its frozen effectiveness artifact contains the 1,536 evidence-bearing questions; the four evidence-empty questions are still included in the three-backend ranking-parity gate.
- MRR is truncated at 10. Recall@10 is macro-averaged relevant-memory recall and is not Hit@10. nDCG@10 uses binary relevance.
- This table establishes semantic equivalence, not a speed advantage. Backend latency, throughput, concurrency, and freshness belong to the 100K system tables.

## Supported claim

> Replacing the frozen CSV backend with Cassandra or Neo4j preserves the candidate inputs, exact Top-10 rankings, and retrieval effectiveness of all six canonical retrieval methods.

It does **not** support the claim that Cassandra improves retrieval effectiveness; Cassandra's contribution must be supported by the system-performance and online-freshness experiments.
