# RRF Implementation Diff Audit

## Finding
Legacy and P2 RRF use IDENTICAL code (same alpha, k, candidate union, tie-breaking).
The R@1 values differ ONLY because of the query set:
- Legacy summary CSV value 0.3539 was computed on ALL 1986 queries (including cat5)
- Current P2 computed on cat1-4 (1540): R@1=0.3416
- Legacy recomputed on cat1-4: R@1=0.3416
- Delta 0.3539-0.3416 = 0.0123 comes from cat5 inclusion

## Per-query audit
- top10 set identical: 1540/1540
- gold rank identical: 1540/1540
- legacy-only correct: 0
- p2-only correct: 0

## RRF Parameters (identical)
- alpha=0.5, k=10, rank 1-based, missing=999
- candidate union: set union with order preservation
- tie-breaking: canonical memory ID ascending
- no sample filtering (Dense/BM25 already per-sample)
