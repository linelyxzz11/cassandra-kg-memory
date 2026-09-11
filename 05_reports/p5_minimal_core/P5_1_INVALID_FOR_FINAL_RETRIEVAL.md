# Legacy P5-1 — Not Citation-Ready for Final Retrieval Visibility

## Verdict

The files named `p5_1_*` in this directory are preserved for historical
diagnosis but must not be used to claim that Cassandra makes a newly written
memory enter a final retrieval result faster than Neo4j.

## Validity failures

1. The terminal probe in `04_experiments/run_p5_1_v2.py` only reads the new KG
   edge. It never fetches a complete memory candidate set, runs a final ranker,
   or checks that the new `memory_id` occurs in Top-K.
2. Cassandra writes three rows (memory, edge, RawERK view), while Neo4j creates
   a Memory node, two Entity nodes, two MENTIONS relationships, one KG_EDGE and
   a RawERKView node. No common logical-event observable was frozen, so the
   timed work is not demonstrably equivalent.
3. The script executes 20,000 formal events × 3 runs × 3 concurrencies × 2
   backends = 360,000 recorded rows, with 2,000 warmup events per applicable
   run. `p5_1_run_manifest.json` instead states `n_runs=30` and
   `n_events_per_run=50000`.
4. The historical script writes to `reports/p5_minimal_core`, while the
   canonical files were later placed under `05_reports/p5_minimal_core`, which
   weakens direct provenance.

## Replacement

Use `04_experiments/p5_1_v3/` and
`05_reports/p5_1_retrieval_visibility_v3/`. Replacement v3 defines one common
logical memory event and ends latency only after a fresh backend candidate read
plus shared RawERK BM25 places the new memory in Top-K. Only a two-backend
benchmark manifest with `status=PASS` and `citation_ready=true` may replace this
warning with a paper claim.
