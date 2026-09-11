# Canonical1540 Reaggregation — Method Comparison

**query_id_sha256**: 23c8a298433852f42149fc3aefa2da9cffc55eadbb311508f0254b7c19a76f35

| Method | Group | R@1 | R@10 | MRR | rF1 | rEM | WrongAbst | Status |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25                 | weak_baseline          | 0.2649 | 0.5619 | 0.3600 | 0.2648 | 0.1539 | 0.5929 | REAGGREGATED_AGGREGATE_ONLY_NO |
| Dense-bge            | weak_baseline          | 0.3419 | 0.7009 | 0.4534 | 0.3482 | 0.1831 | 0.4539 | REAGGREGATED_AGGREGATE_ONLY_NO |
| Dense+GlobalKG       | kg_baseline            | 0.3872 | 0.7095 | 0.4851 | 0.3222 | 0.1701 | 0.4877 | REAGGREGATED_AGGREGATE_ONLY_NO |
| Dense+QueryKG        | kg_baseline            | 0.3585 | 0.7185 | 0.4724 | 0.3516 | 0.1877 | 0.4494 | REAGGREGATED_AGGREGATE_ONLY_NO |
| BM25_compact         | weak_baseline          | 0.4273 | 0.6734 | 0.5083 | 0.3027 | 0.1721 | 0.5286 | REAGGREGATED_AGGREGATE_ONLY_NO |
| RRF_raw              | rrf_baseline           | 0.3539 | 0.7409 | 0.4763 | 0.3542 | 0.1968 | 0.4552 | REAGGREGATED_AGGREGATE_ONLY_NO |
| RRF_compact          | rrf_baseline           | 0.4279 | 0.7857 | 0.5432 | 0.3666 | 0.2065 | 0.4344 | REAGGREGATED_AGGREGATE_ONLY_NO |
| ZScore-Raw           | representation_ablation | NA | NA | NA | NA | NA | NA | NO_1540_READER_PREDICTIONS_AVA |
| ZScore-RawSummary    | representation_ablation | NA | NA | NA | NA | NA | NA | NO_1540_READER_PREDICTIONS_AVA |
| ZScore-RawTriples    | representation_ablation | NA | NA | NA | NA | NA | NA | NO_1540_READER_PREDICTIONS_AVA |
| ZScore-ERKOnly       | representation_ablation | NA | NA | NA | NA | NA | NA | NO_1540_READER_PREDICTIONS_AVA |
| ZScore-RawERK        | final_method           | NA | NA | NA | 0.3747 | 0.2149 | 0.4292 | REAGGREGATED_FROM_PER_QUERY_PR |

## Key Findings

1. **ZScore-RawERK (1540)**: Recomputed from P4 cache (reader_cache_all.jsonl). rF1=0.3747 matches cat1_4_overall.csv (0.3747), but canonical overview says 0.3579. Possible evaluator version difference.
2. **7 legacy methods**: Only overall aggregates exist. NO per-query predictions available. Cannot verify or refilter to canonical 1540.
3. **P3 ZScore representations**: Only heldout 1150 reader predictions exist. No 1540 predictions available.
4. **P1-C E/R/K/T variants**: No reader predictions at all. Retrieval-only.
