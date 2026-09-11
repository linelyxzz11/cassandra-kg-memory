# Claims and Evidence

Only claims with a citation-ready evidence path are listed as validated.

## Retrieval Effectiveness

### R1 — CassMem improves held-out retrieval ranking over the strongest local baselines

- Protocol: LoCoMo Cat1–Cat4, canonical gold-memory v2, evidence-bearing held-out test `n=1,146`.
- CassMem: MRR@10 `0.5498`, Hit@1 `0.4241`, Hit@5 `0.7138`,
  Hit@10 `0.8124`, Recall@10 `0.7396`, nDCG@10 `0.5755`.
- Evidence: `05_reports/retrieval_main_table/retrieval_main_overall.csv`.
- Paired uncertainty: `05_reports/retrieval_main_table/retrieval_main_significance.csv`.
- Guardrail: the all-query row is diagnostic because parameters were selected on dev.

### R2 — Corrected Dense+GlobalKG improves over Dense-bge

- Selected fusion weight: `λ=0.2` on dev.
- Held-out MRR@10: `0.5235`; paired delta over Dense-bge: approximately `+0.028`.
- Evidence: `05_reports/dense_global_kg_rerun/`.
- Guardrail: the method uses a query-independent global degree prior. It is not
  a pruned graph-expansion pipeline.

### R3 — Gold-memory mapping is auditable

- 1,540 Cat1–Cat4 questions; 1,536 mapped; 4 official evidence-empty; 0
  unmapped evidence turns; 5 documented source corrections.
- Evidence: `02_artifacts/retrieval_gold_v2/`.

## Reader Answer Quality

### A1 — CassMem has the best local Setting-B reader scores among rerun local methods

- Full5 F1: `58.56%`.
- Cat1–Cat4 BLEU-1: approximately `0.4074`.
- Pure lexical Full5 BLEU-1 used in the HingeMem-style table: approximately `0.3503`.
- Full5 B1/Cat5 Accuracy hybrid: approximately `0.5087` (supplementary only; not labeled BLEU-1).
- HingeMem-compatible Full5 J: `73.06%` (displayed as `73.1`), using
  `gpt-4o-2024-08-06` and the Mem0/HingeMem binary judge prompt.
- Evidence: `05_reports/official_eval/gpt4o_dual_setting_locomo_corrected_v3/`.
- Judge evidence: `05_reports/llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected/`.
- Paired uncertainty: `reader_main_significance.csv`.
- Guardrail: the hybrid is not pure BLEU-1; Overall J is a micro mean over all
  1,986 Full5 questions, not an unweighted category macro mean.

### A2 — Corrected Dense+GlobalKG reader result is complete

- Both Cat settings contain 1,986/1,986 predictions, with no missing or
  duplicate QIDs and corrected degree-centrality ranking provenance.
- F1/B1 evidence:
  `05_reports/reader_offline_metrics_v4/`.
- GPT-4o Judge evidence:
  `05_reports/llm_judge_gpt4o_corrected_densekg_v2/`.
- The HingeMem-style table has already replaced the legacy row:
  `05_reports/reader_main_hingemem_style/`.

## System Layer

### S1 — Cassandra lowers update-to-final-TopK latency under the controlled P5-1 v3.1 protocol

- Protocol: one logical event = one memory document + two entities + one
  directed KG edge; terminal condition requires the exact logical view and the
  new memory in the shared RawERK BM25 Top-10.
- Fixed retrieval work: exactly 32 baseline memories + 1 target memory for
  every timed event on both backends.
- Scale: 36,000 formal events; 3 runs × 2,000 events × 3 concurrency levels ×
  2 backends; zero timeout; every target at rank 1.
- Pooled p50 update-to-TopK (Cassandra vs Neo4j): c=8 `29.26 vs 47.32 ms`,
  c=32 `85.62 vs 195.32 ms`, c=64 `173.05 vs 402.92 ms`.
- Cassandra p50 reductions: `38.16%`, `56.16%`, and `57.05%`, respectively.
- Cross-backend logical state, candidate documents and Top-10 parity: PASS,
  zero mismatches.
- Evidence:
  `05_reports/p5_1_retrieval_visibility_v3/formal_fixed_20260803/`.
- Guardrail: this supports final visibility for the shared RawERK BM25
  reference retriever. It does not yet include online dense embeddings or the
  full CassMem Dense+RawERK Z-score fusion pipeline.
- Tail-latency guardrail: at c=8 Cassandra p99 is `124.38 ms`, higher than
  Neo4j `84.45 ms`; do not claim universal tail-latency dominance.

### S2 — Legacy P5-1 cannot support final retrieval visibility

- Legacy P5-1 terminated when a KG edge was readable; it did not run a final
  retriever or require the new memory to enter Top-K.
- Cassandra and Neo4j did not share an explicit logical-event contract, and
  `p5_1_run_manifest.json` disagrees with the executed 360,000-row scale.
- The old latency table is historical diagnostic evidence only. See
  `05_reports/p5_minimal_core/P5_1_INVALID_FOR_FINAL_RETRIEVAL.md`.
- Replacement v3.1 is complete under
  `05_reports/p5_1_retrieval_visibility_v3/formal_fixed_20260803/`.

### S3 — Other frozen P5 workload/recovery evidence remains available

- P5-2: uniform/hot-entity results in
  `05_reports/p5_minimal_core/p5_2_hot_entity_summary.csv`.
- P5-3B/P5-3C: burst and restart gates under
  `05_reports/p5_minimal_core/`.

### S4 — P7-B backend visibility/recovery is not citation-ready

- `05_reports/backend_system_eval/update_visibility_results.csv` contains `-1`
  sentinels.
- `recovery_results.csv` has failed/sentinel fields and searchable ratio 0.
- See `05_reports/backend_system_eval/STATUS.md`.

### S5 — Six-method backend replacement preserves retrieval semantics

- Protocol: CSV reference, Cassandra, and Neo4j; six canonical methods; 1,540
  Cat1-Cat4 queries, with metrics on 1,536 evidence-bearing queries.
- Exact CSV-vs-Cassandra and CSV-vs-Neo4j Top-10 parity: 12/12 method-backend
  gates at 100%; candidate/projection/order/corrected-edge digest gates PASS.
- Evidence: `05_reports/backend_equivalence_v2/`.

### S6 — Graph-aware four-cell serving is complete at 100K

- Four cells: Cassandra-base, Cassandra-materialized, Neo4j-native, and
  Neo4j-materialized, all performing the same graph-aware logical event.
- Main 95:5 matrix: c=1/8/16/32/64, three repetitions, 300,000 measured ops.
- Query-type matrix: four query types at c=32, three repetitions, 240,000
  measured queries.
- Storage/work gate: 100,000 memories, 76,475 entities, 129,156 mentions, and
  42,944 semantic edges in every cell; zero semantic digest mismatches.
- Evidence: `05_reports/locomo_workload_graph_v2_100k/`.

### S7 — Online freshness reaches the real CassMem Top-10

- Protocol: graph-aware four-cell 100K workload, c=32, open-loop 95:5, update
  rates 1/2/5/10 per second, three repetitions, 2,400 staged updates.
- Stages include backend commit, structured-view visibility, sparse+dense index
  visibility, and actual Dense+BM25-RawERK+ZScore Top-10 completion.
- At 100 total ops/s, Cassandra-materialized p95 structured visibility is
  `69.815 ms` and p95 pipeline completion is `163.278 ms`.
- Evidence: `05_reports/experiment11_online_freshness_v2/`.

### S8 — Actual candidate-stage recall is measured separately from Top-10 recall

- Held-out CassMem actual fusion pool: average 86.2 candidates, 84.6% reduction
  from the conversation pool, Candidate Recall `0.911`, Top-10 Recall `0.740`.
- The `0.171` gap is ranking/truncation loss after candidate generation.
- Evidence: `05_reports/candidate_stage_analysis_v1/`.

### S9 — The graph-aware online pipeline recovers from a controlled worker failure

- Protocol: the same four 100K graph-aware cells, c=32, 95:5 open-loop load,
  20 arrivals/s, three repetitions, and a 10-second update/materializer-worker
  outage after raw-memory commit.
- All 12 cell-run gates pass with zero missed or duplicate updates and zero
  post-recovery Top-10 mismatches.
- Evidence: `05_reports/locomo_workload_graph_v2_100k/recovery_v2/`.
- Guardrail: the databases remain online. This supports application-pipeline
  replay and backlog recovery, not database restart, multi-node failover, or
  distributed fault tolerance.

## Claims Not Yet Supported

- Arbitrary/query-conditioned graph-expansion superiority. CassMem currently
  uses a full conversation scope followed by frozen branch candidate pruning;
  it is not an arbitrary-depth graph traversal method.
- Multi-node/distributed scale-out superiority. Current formal system evidence
  is single-machine, single-instance Cassandra and Neo4j.
- Database restart, multi-node failover, and distributed fault tolerance. The
  v2 recovery experiment stops the update/materializer worker while both
  databases remain online.
- Direct superiority over quoted external memory systems under an identical
  locally rerun protocol.
- Independent-human resolution of the nine ambiguous error-analysis cases.
