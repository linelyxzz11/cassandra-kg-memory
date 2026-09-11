# P5-1 v3.1 — Logical Memory Update to Final Retrieval Visibility

## Current canonical result

The citation-ready run is `formal_fixed_20260803/`: 36,000 formal events, zero
timeouts, exactly 33 candidates per event, every target at rank 1, and PASS for
logical-state/candidate/Top-K parity. Use its
`P5_1_V31_FORMAL_REPORT.md` and manifest for paper claims.

`formal_20260803/` is a superseded diagnostic because its shared-shard
candidate count varied with concurrent completion order; see its `STATUS.md`.

## Why v3 exists

Legacy P5-1 ended when a KG edge became readable. It did not establish that a
new memory entered a final retrieval result, the Cassandra and Neo4j writes did
not share an explicit logical-event contract, and the run manifest disagreed
with the executed scale. Legacy files under `05_reports/p5_minimal_core/p5_1_*`
must therefore be treated as historical diagnostics for this claim.

## Unified logical event

Every event has the same backend-neutral observable effects:

1. one memory document containing `raw_text`, `raw_erk_text`, version and IDs;
2. two entity records (`src_id`, `dst_id`);
3. one directed KG edge linked to the memory.

Cassandra materializes these effects as a logged batch over a memory table,
entity table and adjacency table. Neo4j materializes them in one transaction as
one Memory node, two Entity nodes and one relationship. Physical operations are
allowed to differ because that is the data-model comparison; the logical view,
input events and retrieval result must be equal.

## Terminal condition and metrics

Latency starts immediately before submitting the logical event and ends only
when both conditions pass:

- a fresh backend read reconstructs the complete logical event exactly;
- a fresh backend candidate read followed by the shared deterministic RawERK
  BM25 ranker contains the new `memory_id` in Top-K.

For strict per-event fairness, the candidate read uses a frozen cohort: the
same `baseline_per_shard` memory IDs plus only the current target memory. Other
concurrently completed updates are not part of that query's cohort. Thus both
backends fetch and rank exactly `baseline_per_shard + 1` documents per timed
event; the manifest aborts on any count mismatch.

The primary metric is `update_to_topk_ms` with p50/p95/p99. The output also
records commit acknowledgement, logical-view visibility, retrieval-after-commit,
candidate count, rank, probe attempts, errors and throughput.

This v3 protocol validates **RawERK BM25 final retrieval visibility**. It does
not yet validate full Dense+RawERK CassMem fusion because online dense embedding
materialization is not part of the event.

## Gates

- exact expected-vs-actual formal row count;
- zero Top-K visibility failures/timeouts;
- exact fixed candidate count for every event;
- identical logical-event digest for Cassandra and Neo4j for each configuration;
- sequential cross-backend equality of reconstructed logical views, candidate
  documents and final Top-K IDs (kept separate from concurrent timing);
- source file hash, configuration and output hashes frozen in the run manifest.

## Commands

From the project root:

```powershell
python 04_experiments/p5_1_v3/test_p5_1_protocol.py

# Put NEO4J_PASSWORD in the git-ignored project-root .env file, or export it
# in the environment. The harness never writes or logs the value.
python 04_experiments/p5_1_v3/run_p5_1_v3.py --mode preflight

python 04_experiments/p5_1_v3/run_p5_1_v3.py --mode benchmark `
  --run-tag p5v3_formal_YYYYMMDD `
  --formal-events 2000 --warmup-events 100 `
  --runs 3 --concurrencies 8 32 64
```

Do not call a run citation-ready unless it used `mode=benchmark`, both backends,
its manifest has `status=PASS` and `citation_ready=true`, and all gates pass.
A one-backend diagnostic receives `status=PASS_PARTIAL`.
