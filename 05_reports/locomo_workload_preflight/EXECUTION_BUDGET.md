# Execution budget and bottleneck forecast

Status: planning estimate based on the local 32-logical-CPU machine, P5-1 v3.1 live results, and the 50-event shared-index diagnostic. It is not a measured formal workload result.

## Measured anchors

- P5-1 v3.1 update-to-RawERK-BM25-Top10 p50 at c=8/32/64: Cassandra 29.26/85.62/173.05 ms; Neo4j 47.32/195.32/402.92 ms.
- Shared real RawERK BM25 + frozen BGE + query-wise ZScore diagnostic (50 isolated LoCoMo events):
  - sparse scope update p50/p95/p99: 19.43/22.60/25.03 ms;
  - dense scope update: 2.99/3.92/4.08 ms;
  - fused search: 2.29/2.71/2.91 ms;
  - index update + fused search: 24.48/28.31/31.41 ms;
  - FreshHit@10: 0.84 on this diagnostic sample.

The diagnostic excludes backend time and concurrency. Its purpose is runner sizing only.

## Expected wall-clock budget

| Stage | Expected time | Exit gate | Likely blocker |
|---|---:|---|---|
| Protocol/profile/manifests/shared index | completed | frozen hashes; tests pass | evidence normalization or protocol drift |
| Four live adapters + schemas + instrumentation | 6–12 h engineering | logical projection digest parity | unequal work between cells; Cassandra partition design; Cypher plan |
| Canonical 5,882-memory live preflight | 1–3 h | zero mismatches/timeouts/sentinels | version/idempotency bugs; index visibility race |
| Neo4j warm-up, indexes, JVM/pool tuning, PROFILE capture | 1–3 h | stable plan and no label scan in hot path | native traversal tail latency / GC |
| 100K data load + validation | 2–6 h | exact row/node/edge/version counts | Neo4j import transaction size; duplicate namespaces |
| 100K workload matrix | 6–18 h | all cells/repetitions complete | c=32/64 backlog and sparse-index lock contention |
| Supplemental 90:10 and 99:1 | 4–12 h | predeclared selected concurrency points | update-heavy index rebuild contention |
| Burst/recovery/backlog drain | 3–8 h | no loss/duplication; backlog reaches zero | recovery harness, checkpoints, container restart variance |
| Aggregation, figures, audit rerun | 2–5 h | S01/S02/S04 resolved by valid evidence | manifest/result count mismatch |

Expected total after the current completed stage: roughly **1–2 machine-days in the happy path, 2–3 wall-clock days including debugging and reruns**. The 5,882 gate is used to replace this range with a measured extrapolation before starting 100K.

## Bottleneck ranking

1. **Neo4j-native at high concurrency**: bounded traversal, page-cache misses, JVM GC and connection-pool queueing are the leading tail-latency risks.
2. **Sparse index update contention**: the current exact scoped BM25 rebuild costs about 19–23 ms per update. This is bounded by the 369–689-memory scope, but hot tenants can serialize writers. Copy-on-write and per-scope locks are required.
3. **100K loading and validation**: loading can dominate if every repetition reloads. Load once per cell, snapshot/checkpoint, and reuse between measured traces.
4. **Backlog at c=32/64**: if arrival rate exceeds materializer + index capacity, Time-to-Top10 measures queueing rather than only database latency. This is desired but must be reported with backlog/drain.
5. **Result volume and resumability**: per-event logs across the full matrix can be large. Stream compressed partitions and checkpoint each scale/cell/mix/concurrency/repetition.

## Run-control decisions

- Do not run the full 2x2 matrix before canonical parity passes.
- Do not reload 100K data for every repetition; use versioned namespaces or restored snapshots.
- Use a time-based measured window after warm-up and a fixed operation trace per four-cell block.
- Run the primary 95:5 matrix first. Supplemental mixes start only after the primary matrix is complete.
- A run with sentinel latency, incomplete drain, count mismatch or projection/Top-10 mismatch is diagnostic only and cannot close an audit blocker.
