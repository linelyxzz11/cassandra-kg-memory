# Legacy-Aligned RRF Attribution

## Reproduction: PASS (source: sample_scoped_retrieval_summary.csv)

## Results
- BM25_raw_legacy: R@1=0.2649 R@10=0.5619 MRR=0.36
- Dense_raw_legacy: R@1=0.3419 R@10=0.7009 MRR=0.4534
- BM25_compact_legacy: R@1=0.4273 R@10=0.6734 MRR=0.5083
- RRF_raw_legacy: R@1=0.3539 R@10=0.7409 MRR=0.4763
- RRF_compact_legacy: R@1=0.4279 R@10=0.7857 MRR=0.5432

## Attribution
- generic_hybrid: dMRR=+0.0229
- KG_rep_lexical: dMRR=+0.1483
- KG_rep_in_hybrid: dMRR=+0.0669
- total_final_gain: dMRR=+0.0898
