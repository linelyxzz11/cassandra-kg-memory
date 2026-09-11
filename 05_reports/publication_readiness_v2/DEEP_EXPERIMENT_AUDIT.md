# CassMem deep experiment audit v2

Generated: 2026-09-11T12:59:34.317177+00:00

## Verdict

The three core axes are now internally complete: retrieval effectiveness, Reader answer quality, and graph-aware backend serving/freshness. The audit found two locally fillable presentation/evidence gaps and completed both: actual candidate-stage recall/ranking-loss decomposition and the 100K four-cell storage/write-amplification table.

- Completed evidence items: 21/28
- Core blockers remaining in the current local scope: 0
- Human adjudication remaining: nine ambiguous error cases
- External/generalization limits: second dataset, multi-node scale-out, and identical local reruns of quoted memory systems

## What is ready for the paper

1. Retrieval main table plus held-out significance, ERK ablation, fusion ablation, question-type analysis, and candidate-stage decomposition.
2. HingeMem-style Reader F1/J/B1 table with corrected Dense+GlobalKG and explicit quoted-vs-local provenance.
3. Six-method CSV/Cassandra/Neo4j backend bridge with exact ranking and input-digest parity.
4. Graph-aware 100K four-cell equal-work/storage table, 95:5 concurrency table, query-type table, actual Time-to-Top10 freshness table, and controlled update/materializer-worker recovery.

## What must not be claimed

- Do not call the current candidate stage arbitrary graph expansion; it is conversation-scoped frozen branch pruning.
- Do not claim Cassandra is universally faster: Neo4j is competitive at the light-load point in Experiment 11.
- Do not claim distributed scale-out or multi-node fault tolerance from single-instance Docker experiments.
- Do not claim significant Reader superiority over RRF_compact; the paired intervals do not support it.
- Do not describe the nine ambiguous error cases as independently human-resolved.

## Remaining optional experiments

Graph-aware 90:10/99:1 sensitivity remains an appendix-strengthening experiment. Controlled four-cell application-worker recovery is complete. Database restart, multi-node failover, and distributed fault tolerance remain outside the current evidence boundary.

## Machine-readable matrix

See `experiment_gap_matrix.csv`.
