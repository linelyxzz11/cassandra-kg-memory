# ERK vs ERKT Final Reader Decision

## Default representation: KEEP ERK as default, ERKT as extension

- Time retrieval effect (held-out RRF): dMRR=+0.0021
- Time reader effect (held-out): drF1=+0.0075, drEM=+0.0052, dWrongAbst=+0.0026
- Query-level F1 CI: [-0.0006, +0.0158]
- Cluster-level F1 CI: [-0.0026, +0.0168]
- Best category: see 03_reader_by_category.csv
- Decision reason: rF1 point positive (++0.0075) but cluster CI crosses zero
- API calls: 1892, cache hits: 408, failures: 0, runtime: 2013s
