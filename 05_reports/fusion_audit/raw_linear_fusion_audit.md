# Raw Linear Fusion Audit

## Answer: YES_COMPLETE, REJECTED_BASELINE

Formally run in the P2 fusion comparison. Results published.

## Experiment Details

| Field | Value |
|---|---|
| Formula | score = alpha * raw_Dense + (1-alpha) * raw_BM25 |
| Dev best alpha | 0.9 |
| Dev MRR (n=390) | 0.4708 |
| Dev R@1 | 0.3744 |
| Test MRR (n=1150) | 0.4672 |
| Test R@1 | 0.3600 |
| Status | Rejected: ZScore (0.5426) and MinMax (0.5383) achieved higher MRR |

## Confirmation

- NOT ZScore (no normalizer applied)
- NOT MinMax (no normalizer applied)
- NOT RRF (uses raw scores, not ranks)
- Formula verified: `alpha * score_Dense + (1-alpha) * score_BM25`

## Artifact

- Dev sweep: reports/p2_fusion_comparison/dev_parameter_grid.csv
- Test results: reports/p2_fusion_comparison/test_fusion_overall.csv
