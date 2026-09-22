# LoCoMo-shaped multi-tenant online memory workload

Status: protocol draft V1; the profile and trace manifests are frozen, while the four live adapters and end-to-end runner remain to be implemented and gated.

## 1. Claim and scope

The experiment asks whether query-materialized Cassandra makes newly arrived structured memory become usable by the **actual CassMem Top-10 retriever** sooner and at higher sustainable concurrency than fair Neo4j alternatives.

The 100K dataset is a deterministic namespace replication of the local LoCoMo trace. It is called **LoCoMo-shaped trace replay**. It is not described as original 100K LoCoMo.

## 2. Frozen logical update event

One logical update event has identical semantic effects in every cell:

1. one versioned memory record (`scope_id`, `memory_id`, raw text, timestamp);
2. the precomputed structured fields used by RawERK (summary, entities, relations, keywords, triples);
3. the precomputed frozen 1024-dimensional BGE embedding;
4. all entity/edge records implied by the frozen triples;
5. one idempotency key and one monotonically increasing per-memory version.

LLM extraction and embedding computation are performed before `t_submit` and excluded from the primary database comparison. Their offline cost may be reported separately.

## 3. Four-cell design

| Cell | Stored/read design | Required query result |
|---|---|---|
| Cassandra-base | normalized memory + `edge_by_scope_src`; assemble structured view after partition-key reads | canonical RawERK projection |
| Cassandra-materialized | query-specific `by_scope_relation` plus materialized RawERK record | same canonical RawERK projection |
| Neo4j-native | indexed Memory/Entity nodes and typed relationships; bounded scope-local traversal | same canonical RawERK projection |
| Neo4j-materialized | RawERK property/node fetched by indexed `(scope_id,memory_id)` | same canonical RawERK projection |

Every event must pass a canonical projection digest and candidate/Top-10 parity gate. Neo4j receives constraints/indexes, connection pooling, JVM warm-up, a saved `PROFILE` plan and bounded traversal. No cell may do extra extraction or embedding work inside the timed interval.

## 4. Clock contract

All durations use a monotonic high-resolution clock in the load-generator process.

- `t_submit`: the event becomes eligible for backend execution.
- `t_commit`: backend acknowledges the complete logical update transaction/batch under the declared consistency level.
- `t_rawerk`: a fresh read returns the complete canonical structured projection with the expected version/digest.
- `t_sparse_index`: the updated RawERK document is visible to the shared incremental BM25 index.
- `t_dense_index`: the frozen embedding is visible to the shared incremental dense index.
- `t_index = max(t_sparse_index, t_dense_index)`.
- `t_top10`: a fresh query through candidate fetch, both scorers and query-wise ZScore fusion first returns the arriving gold memory in Top-10.

Primary latency is `t_top10 - t_submit`. Also report each adjacent stage and the complete query breakdown: backend fetch, structured expansion/materialized fetch, sparse scoring, dense scoring, fusion and total.

## 5. Retrieval contract

- Eligible freshness questions: Cat1-4 only, non-empty resolvable gold evidence.
- Before replay, withhold one designated gold memory from the selected namespace.
- Query scope is known; candidate retrieval is local to that conversation namespace.
- Sparse score: the same frozen RawERK BM25 definition used by CassMem.
- Dense score: cosine similarity over the frozen BGE-large vectors.
- Fusion: query-wise Z-score normalization, `0.6 * dense_z + 0.4 * sparse_z`, stable `memory_id` tie-break, `k=10`.
- A hit requires the withheld gold memory ID in the real fused Top-10, not merely edge or record visibility.

## 6. Workload matrix

- Scale: canonical 5,882-memory gate followed by a 100K main result. No 1M run is planned.
- Concurrency: 1, 8, 16, 32, 64. The 8-worker point aligns with P5-1 v3.1; the 16-worker point satisfies publication audit S04.
- Primary mix: 95% reads / 5% updates.
- Supplemental mixes: 90:10 and 99:1.
- At least three measured repetitions after warm-up; fixed trace seed and identical operation order per four-cell block.
- Runs are checkpointed by scale/cell/mix/concurrency/repetition and are independently resumable.

## 7. Metrics and gates

Effectiveness/freshness:

- FreshHit@10 and FreshRecall@10 at 50, 100, 250, 500 and 1000 ms;
- stale result rate;
- missed update rate and duplicate-application rate;
- per-stage and Time-to-Top10 p50/p95/p99.

Serving:

- QPS and update throughput;
- backlog size over time and drain time after a controlled burst;
- maximum sustainable throughput under the predeclared SLO;
- timeout/error rate and resource utilization.

Validity gates:

- zero sentinel latencies in citation-ready output;
- zero unresolved evidence in the selected event set;
- exact logical projection digest parity across four cells;
- identical index and fusion implementation across cells;
- candidate and fused Top-10 parity when all cells have reached the same version;
- manifest row/event counts equal actual completed records.

## 8. Relationship to P5-1 v3.1 and audit blockers

P5-1 v3.1 remains a valid fixed-cohort RawERK-BM25 backend microbenchmark. It already measures commit, complete logical-view visibility and BM25 Top-10 visibility. This protocol adds independent index clocks, real Dense+BM25 ZScore fusion, trace-shaped scale, workload mixes, backlog/recovery and a 2x2 materialization design.

The new successful evidence will replace the invalid legacy evidence for audit S01/S02 and directly satisfy S04. Legacy CSV files remain immutable and explicitly invalid/superseded.
