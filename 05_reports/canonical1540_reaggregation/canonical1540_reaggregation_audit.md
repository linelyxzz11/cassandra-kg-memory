# Canonical1540 Reaggregation Audit

## Data Sources

| Method | Source | Row count | Matched 1540 | Per-query rF1? |
|---|---|---:|---:|---:|
| ZScore-RawERK | P4 reader_cache_all.jsonl | 1986 | 1540 | ✅ |
| BM25_compact    | reports/locomo_new_methods_reader_legacy/new_methods_reader_overall.csv | 1 (aggregate) | 1540 (claimed) | ❌ No per-query |
| RRF_raw         | reports/locomo_new_methods_reader_legacy/new_methods_reader_overall.csv | 1 (aggregate) | 1540 (claimed) | ❌ No per-query |
| RRF_compact     | reports/locomo_new_methods_reader_legacy/new_methods_reader_overall.csv | 1 (aggregate) | 1540 (claimed) | ❌ No per-query |
| BM25            | reports/locomo_new_methods_reader_legacy/new_methods_reader_overall.csv | 1 (aggregate) | 1540 (claimed) | ❌ No per-query |
| Dense-bge       | reports/locomo_new_methods_reader_legacy/new_methods_reader_overall.csv | 1 (aggregate) | 1540 (claimed) | ❌ No per-query |
| Dense+GlobalKG  | reports/locomo_new_methods_reader_legacy/new_methods_reader_overall.csv | 1 (aggregate) | 1540 (claimed) | ❌ No per-query |
| Dense+QueryKG   | reports/locomo_new_methods_reader_legacy/new_methods_reader_overall.csv | 1 (aggregate) | 1540 (claimed) | ❌ No per-query |

## Discrepancies

### ZScore-RawERK canonical rF1
- P4 canonical_cat1_4_reader_overall.csv: rF1=0.3579
- P4 cat1_4_overall.csv (from cache): rF1=0.3747
- P4 reader_cache_all.jsonl recomputed: rF1=0.3747
- Discrepancy: cache reagg = cat1_4_overall.csv, but differs from canonical_overview by Δ=0.0168
Possible cause: different evaluator version or prediction set
