# 100K expansion protocol

Protocol ID: `locomo-shaped-100k-v1`

## Terminology

This dataset is a **100K LoCoMo-shaped trace replay corpus**. It is not original 100K LoCoMo and does not introduce new conversations or semantic facts.

## Frozen source

- 5,882 canonical memories in `data/locomo_memory_records.csv`.
- Aligned E/R/K/triples in `data/frozen_retrieval/p3_memory_features.csv`.
- Frozen 1,024-dimensional BGE vectors and ordered ID file.
- Source SHA-256 values are copied into `trace_manifest_100000.json` and must pass before expansion.

## Exact memory mapping

For global ordinal `g` in `[0, 99,999]`:

```text
replica_index   = g // 5,882
source_row_index = g % 5,882
scope_id        = "lr{replica_index:06d}::" + source_scope_id
memory_id       = "lr{replica_index:06d}::" + source_memory_id
```

- Replicas 0–16 are complete: `17 × 5,882 = 99,994` memories and 170 complete namespaces.
- Replica 17 contains source rows 0–5 only. Those rows all belong to `conv-26`, producing one partial namespace.
- Exact total: 100,000 memories and 171 namespaces.
- Every expanded ID has a unique reverse mapping to `(replica_index, source_row_index, source ID)`.

No text, timestamp, E/R/K/triples, embedding values or within-scope source order may be changed. The embedding SHA-256 stored with an expanded memory must equal its source vector hash.

## QA and evidence mapping

Only the 170 complete namespaces are eligible for measured QA and freshness operations. The partial namespace is load-only and cannot contribute a query, hit, miss or latency observation.

For each complete replica `r` and source QA/evidence ID:

```text
qa_id              = "lr{r:06d}::" + source_qa_id
scope_id           = "lr{r:06d}::" + source_scope_id
gold_memory_id     = "lr{r:06d}::" + source_gold_memory_id
```

This yields 33,762 mapped QA trace entries (`17 × 1,986`) and 26,061 unambiguous Cat1-4 freshness entries (`17 × 1,533`). Categories and evidence-count distributions are therefore preserved exactly in the measured query population.

## Operation trace

- Seed: `20260803`.
- Primary mix: exactly 95% reads / 5% updates in every measured operation list.
- Supplemental: exactly 90:10 and 99:1.
- Tenant/QA selection is deterministic from the frozen replicated QA population.
- Update order inside a namespace follows canonical source order; namespaces are deterministically interleaved by the seed.
- Each cell and repetition consumes the same ordered logical operation list.
- Warm-up operations are separate and never included in metrics.

Initial formal budget per cell/concurrency/repetition:

- 500 warm-up operations;
- 5,000 measured operations;
- concurrency 1/8/16/32/64;
- three repetitions for the 95:5 main result.

Supplemental mixes start only after the full 95:5 matrix passes.

## Pre-load gates

1. All source hashes match the manifest.
2. Expanded memory IDs and scope/memory pairs are unique.
3. Total memories = 100,000; namespaces = 171; complete namespaces = 170; partial namespaces = 1.
4. Every expanded record reverses to the declared source row.
5. Full-replica empirical distributions exactly equal 17 copies of canonical distributions.
6. Partial namespace is excluded from measured QA.

Any failed gate prevents database loading and makes the run non-citation-ready.

## Parallel preload policy

- Cassandra: 32 bounded client workers inside one cell; base and materialized cells load sequentially.
- Neo4j: 8 bounded batch workers inside one cell, 500 memories per transaction; cells load sequentially.
- Parallel completion order is not treated as logical update order. Formal update traces are replayed separately from the preload corpus.
- Worker counts, batch size and per-cell load duration are recorded in the load manifest.
