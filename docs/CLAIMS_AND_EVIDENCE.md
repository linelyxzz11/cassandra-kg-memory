# Claims and evidence

This file defines the claims supported by the public CassMem artifact. Paths are
relative to the repository root.

## Retrieval

CassMem combines dense retrieval over raw memory text with BM25 over the RawERK
view, using query-wise Z-score fusion. The canonical effectiveness table and the
development-set alpha sweep are in:

- `results/retrieval/retrieval_main_table/`
- `results/retrieval/zscore_alpha_sweep/`
- `results/representation_ablation/p1_compact_component_ablation/`

The selected fixed weight is an empirical development-set choice, not an adaptive
fusion method.

## Backend semantics

The backend comparison evaluates whether CSV, Cassandra, and Neo4j expose the same
logical projections, candidates, and ordered Top-10 under the frozen retrieval
protocol. Evidence is in `results/backend_equivalence/backend_equivalence_v2/`.

This supports semantic-preservation claims for the implemented access paths. It
does not establish that arbitrary physical layouts are equivalent.

## Serving under a fixed 100K workload

The resource-matched four-cell evaluation covers Cassandra base/materialized and
Neo4j native/materialized paths under the same fixed 100K logical workload. Its
formal results, per-event records, frozen configuration, resource samples, and
strict audit are in `results/serving_100k/resource_matched_v3/resmatch_20260924_v3r2/`.
The [results note](RESOURCE_MATCHED_SERVING_RERUN_RESULTS.md) defines the matched
container budget and timing boundaries. Earlier query-type, storage-work, and
application-worker recovery results remain in
`results/serving_100k/locomo_workload_graph_v2_100k/`; they are not substituted
for the new resource-matched main workload.

The supported wording is **concurrency scaling under a fixed 100K workload**. The
single-machine, single-instance experiment does not support distributed scale-out,
an intrinsic database-engine speed ranking, equal crash durability, cluster failover,
or disaster-recovery claims.

## Retrieval freshness

The resource-matched freshness records in
`results/serving_100k/resource_matched_v3/resmatch_20260924_v3r2/freshness/`
separate application worker queue wait, client-observed raw-write acknowledgement,
structured-view checks, index visibility, and completed post-update ranking.
The legacy field `t_commit_ms` includes queueing and must not be labeled database
commit latency. The older `results/freshness/experiment11_online_freshness_v2/`
is retained only as historical evidence.
An updated memory that does not enter Top-10 is not automatically a backend
freshness failure. The current artifact does not claim continuously monitored
first-hit latency or full Top-k convergence latency.

## Reader evaluation

Reader-facing F1, judge, and BLEU-1 summaries are under
`results/reader/reader_main_hingemem_style/`. Paid-model outputs are frozen artifacts;
rerunning them requires the corresponding local API configuration.
