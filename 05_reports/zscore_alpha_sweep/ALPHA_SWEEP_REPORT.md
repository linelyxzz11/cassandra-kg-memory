# Query-wise Z-score Fusion Weight Sweep

Alpha was selected using development MRR@10 only. Held-out results are reported after selection and were not used to choose the weight.

## Development selection

| Dense alpha | n | MRR@10 | Hit@1 | Hit@5 | Hit@10 | Recall@10 | nDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 390 | 0.4497 | 0.3590 | 0.5769 | 0.6410 | 0.5842 | 0.4684 |
| 0.1 | 390 | 0.4655 | 0.3692 | 0.6026 | 0.6667 | 0.6079 | 0.4842 |
| 0.2 | 390 | 0.4779 | 0.3795 | 0.6103 | 0.6974 | 0.6343 | 0.4999 |
| 0.3 | 390 | 0.4894 | 0.3718 | 0.6513 | 0.7513 | 0.6786 | 0.5196 |
| 0.4 | 390 | 0.5061 | 0.3821 | 0.7000 | 0.7718 | 0.6986 | 0.5361 |
| 0.5 | 390 | 0.5293 | 0.4128 | 0.7051 | 0.7769 | 0.7004 | 0.5524 |
| **0.6** | **390** | **0.5380** | **0.4256** | **0.7103** | **0.7564** | **0.6815** | **0.5507** |
| 0.7 | 390 | 0.5183 | 0.4026 | 0.6795 | 0.7513 | 0.6812 | 0.5374 |
| 0.8 | 390 | 0.5029 | 0.3923 | 0.6513 | 0.7231 | 0.6541 | 0.5195 |
| 0.9 | 390 | 0.4898 | 0.3846 | 0.6308 | 0.7077 | 0.6428 | 0.5058 |
| 1.0 | 390 | 0.4662 | 0.3615 | 0.6128 | 0.6974 | 0.6270 | 0.4848 |

The unique development optimum is alpha=0.6 under MRR@10. Other metrics are reported rather than used as additional tuning objectives.

## Local robustness around the optimum

| Comparison | Delta MRR@10 | Clustered 95% CI | Wins / ties / losses |
|---|---:|---:|---:|
| alpha=0.6 vs alpha=0.5 | +0.0087 | [-0.0081, +0.0262] | 52 / 299 / 39 |
| alpha=0.6 vs alpha=0.7 | +0.0197 | [+0.0120, +0.0278] | 52 / 312 / 26 |

The development split contains 390 queries from only two conversations. The clustered interval is therefore a sensitivity check, not strong inferential evidence; the primary justification remains dev-only selection followed by one frozen held-out evaluation.

## Frozen held-out result

| Dense alpha | n | MRR@10 | Hit@1 | Hit@5 | Hit@10 | Recall@10 | nDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | 1146 | 0.5498 | 0.4241 | 0.7138 | 0.8124 | 0.7396 | 0.5755 |

The alpha=0.6 held-out row exactly matches the canonical CassMem row in Retrieval Table 1 on all six metrics.

### Post-selection sensitivity (diagnostic only)

| Dense alpha | MRR@10 | Hit@1 | Hit@5 | Hit@10 | Recall@10 | nDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.4395 | 0.3360 | 0.5663 | 0.6379 | 0.5783 | 0.4567 |
| 0.1 | 0.4658 | 0.3630 | 0.5977 | 0.6763 | 0.6170 | 0.4852 |
| 0.2 | 0.4845 | 0.3752 | 0.6230 | 0.7155 | 0.6492 | 0.5063 |
| 0.3 | 0.5073 | 0.3979 | 0.6545 | 0.7408 | 0.6738 | 0.5287 |
| 0.4 | 0.5246 | 0.4049 | 0.6780 | 0.7757 | 0.7097 | 0.5500 |
| 0.5 | 0.5448 | 0.4206 | 0.7042 | 0.7993 | 0.7264 | 0.5683 |
| **0.6** | **0.5498** | **0.4241** | **0.7138** | **0.8124** | **0.7396** | **0.5755** |
| 0.7 | 0.5391 | 0.4154 | 0.7112 | 0.8063 | 0.7352 | 0.5672 |
| 0.8 | 0.5236 | 0.4014 | 0.6867 | 0.7827 | 0.7150 | 0.5519 |
| 0.9 | 0.5109 | 0.3944 | 0.6702 | 0.7661 | 0.6972 | 0.5376 |
| 1.0 | 0.4955 | 0.3805 | 0.6562 | 0.7461 | 0.6793 | 0.5215 |

This table is not used to select alpha. It shows that the dev-selected value remains the best point estimate for all six held-out metrics among the tested weights.

## Protocol

- Dense(raw) Top-50 and BM25(RawERK) Top-50 within the known conversation scope.
- Query-wise population Z-score within each truncated branch.
- Missing branch scores use that branch's minimum normalized score.
- Candidate union and tie handling follow the frozen P1-C implementation.
- The four evidence-empty held-out questions are excluded; held-out n=1,146.
