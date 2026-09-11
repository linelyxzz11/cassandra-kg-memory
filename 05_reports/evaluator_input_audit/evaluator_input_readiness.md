# Evaluator Input Readiness

## Per-Method Status

| Method | Per-query pred? | Query ID? | Category? | Gold answer? | Recoverable? | Status |
|---|---:|---|---|---|---|
| BM25 | Yes (reader_f1_full_scoped) | Yes (qa_id) | Yes | Yes | Full5 | READY |
| Dense-bge | Yes (same file) | Yes | Yes | Yes | Full5 | READY |
| Dense+GlobalKG | Yes (same file) | Yes | Yes | Yes | Full5 | READY |
| Dense+QueryKG | Yes (same file) | Yes | Yes | Yes | Full5 | READY |
| BM25_compact | No per-query | N/A | N/A | N/A | Partial | NEEDS_API |
| RRF_raw | No per-query | N/A | N/A | N/A | Partial | NEEDS_API |
| RRF_compact | No per-query | N/A | N/A | N/A | Partial | NEEDS_API |
| ZScore-Raw | No 1540 pred | N/A | N/A | N/A | No | NEEDS_API |
| ZScore-RawSummary | No 1540 pred | N/A | N/A | N/A | No | NEEDS_API |
| ZScore-RawTriples | No 1540 pred | N/A | N/A | N/A | No | NEEDS_API |
| ZScore-ERKOnly | No 1540 pred | N/A | N/A | N/A | No | NEEDS_API |
| ZScore-RawERK | Yes (P4 cache, 1540) | Yes (qa_id) | Yes | Yes | Cat1-4, Full5 | READY |
| P2 ZScore_Linear | Yes (P2, 1150) | Yes | Yes | N/A | Heldout1150 | PARTIALLY |
| P2 MinMax_Linear | Yes (same) | Yes | Yes | N/A | Heldout1150 | PARTIALLY |
| P2 WRRF | Yes (same) | Yes | Yes | N/A | Heldout1150 | PARTIALLY |
| RRF_ERK/ERKT | Yes (erk_vs_erkt, 1150) | Yes (qa_id) | Yes | Yes | Heldout1150 | PARTIALLY |
| P1-C E/R/K/ERKT | No per-query | N/A | N/A | N/A | Retrieval only | RETRIEVAL_ONLY |

## Files Ready for Evaluator

### With rF1 pre-computed:
1. results/final/reader_f1_memory_only_full_scoped_bm25_predictions.csv — BM25, Dense-bge, +GlobalKG, +QueryKG
2. 05_reports/p4_full5/reader_cache_all.jsonl — ZScore-RawERK (1540 canonical)
3. reports/p2_fusion_reader_final/reader_predictions.jsonl — ZScore_Linear, MinMax, WRRF (1150)
4. reports/erk_vs_erkt_reader_heldout/05_prediction_change_cases.jsonl — RRF_ERK/ERKT (1150)

### Raw predictions only (need evaluator run):
1. reports/p2_fusion_reader_final/reader_cache.jsonl — 3 methods (1150)
2. 05_reports/p3_reader_alignment/reader_cache.jsonl — ZScore_RawERK (1150)
3. 05_reports/p4_full5/canonical_reader_predictions.jsonl — ZScore_RawERK (1540)

### Aggregate only, no per-query:
1. new_methods_reader_overall.csv — BM25_compact, RRF_raw, RRF_compact (1540)

### Missing for canonical 1540:
1. ZScore-Raw, RawSummary, RawTriples, ERKOnly
2. P1-C E/R/K/ERKT variants
