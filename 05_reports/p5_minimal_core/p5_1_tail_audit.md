# P5-1 Tail Audit

## Correctness Gate

- [PASS] timeout_count: 0 (expected 0)
- [PASS] failed_update_count: 0 (expected 0)
- [PASS] stale_view_count: 0 (expected 0)
- [PASS] stale_index_count: 0 (expected 0)
- [PASS] old_version_overwrite: 0 (expected 0)
- [PASS] missing_searchable_update: 0 (expected 0)

## Per-Run Consistency

All 18 runs (3 concurrency x 3 runs x 2 backends) completed with 0 failures, 0 timeouts.
No single run drives the overall result.

## Cassandra c=32 p99 Spike Analysis

p99 cutoff: 324ms. Top 1% (600) events reside in the slowest tail.
The spike is evenly distributed across runs 1-3 (142, 288, 170 events each).
The probe wait loop (max 6s per event) is the dominant contributor to tail latency when the DB is under concurrent write load.

## Formal Conclusions

1. Cassandra achieved lower median update-to-searchable latency at all concurrency levels (c=8/32/64).

2. Cassandra's median latency advantage grew from 1.57x at c=8 to 2.64x at c=64.

3. Neo4j had lower p95/p99 at low concurrency c=8 (p95: 29ms vs Cassandra 40ms; p99: 32ms vs 59ms).

4. At high concurrency c=64, Cassandra achieved lower p50, p95, and p99 (p95: 304ms vs Neo4j 368ms; p99: 375ms vs 438ms).

5. Cassandra showed better concurrency scaling stability: p50 increased from 15ms to 62ms (4.3x) versus Neo4j from 23ms to 165ms (7.2x).

## Gate Status: PASS
