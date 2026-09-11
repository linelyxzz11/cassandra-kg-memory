# Table 3: Fusion Ablation

Held-out test, LoCoMo Cat1-4, n=1,146. Dense and RawERK BM25 branches are fixed.

| Fusion | Formula | MRR@10 | Hit@10 | Recall@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| RawScore | 0.9*Dense + 0.1*BM25 (unnormalized) | 0.4705 | 0.6937 | 0.6275 | 0.4912 |
| Equal-RRF | 0.5 RRF(Dense) + 0.5 RRF(BM25), k=10 | 0.5260 | 0.7888 | 0.7183 | 0.5509 |
| Weighted-RRF | 0.6 RRF(Dense) + 0.4 RRF(BM25), k=10 | 0.5248 | 0.7949 | 0.7255 | 0.5531 |
| MinMax | 0.6 minmax(Dense) + 0.4 minmax(BM25) | 0.5447 | 0.8072 | 0.7349 | 0.5704 |
| **ZScore** | **0.6 z(Dense) + 0.4 z(BM25)** | **0.5498** | **0.8124** | **0.7396** | **0.5755** |

The historical `R@10` field was binary Hit@10. True multi-gold Recall@10 is reported separately.
ZScore exactly matches the CassMem row in Retrieval Table 1 on all six metrics.
