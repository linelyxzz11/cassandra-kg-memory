# Fusion Experiment Artifact Audit — Summary

## Question A: 未归一化线性融合是否正式运行？

**Answer: YES_COMPLETE — REJECTED_BASELINE**

Formula: score = alpha * raw_Dense + (1-alpha) * raw_BM25
Best dev alpha: 0.9
Test MRR (n=1150): 0.4672, R@1: 0.3600
Status: Rejected — ZScore (0.5426) and MinMax (0.5383) achieved higher MRR.

## Question B: 0.6/0.4 是否有完整 dev-only sweep 表？

**Answer: YES_COMPLETE**

ZScore linear fusion alpha=0.6 uniquely best among 11 candidates.
Dev MRR: 0.6=0.5340, next=0.5=0.5214 (gap +0.0126, no tie-break needed).
No held-out leakage detected.

## Final Fusion Status

| Method | Status | Official |
|---|---|---|
| Raw unnormalized linear fusion | YES_COMPLETE, REJECTED_BASELINE | no |
| Equal RRF (alpha=0.5, k=10) | PILOT_BASELINE | no |
| Weighted RRF (alpha=0.6, k=10) | PILOT_BASELINE | no |
| MinMax normalized fusion | PILOT_BASELINE | no |
| **ZScore normalized fusion** | **OFFICIAL_SELECTED** | **yes** |
| Legacy mixed-scope RRF | QUARANTINED | no |

## Official Method

**ZScore-RawERK**: score = 0.6 * z(Dense_raw) + 0.4 * z(BM25_RawERK)
- Dev MRR: 0.5340 (alpha=0.6, dev n=390)
- Test MRR: 0.5426 (P2 test fusion, n=1150)
- Frozen P3 RawERK anchor MRR: 0.5462 (reader input, n=1150)
- R@10: 0.8070
- Config: `p3_config.json` — alpha_dense: 0.6

## Statistical Note

All comparisons are based on point estimates. No paired/bootstrap CI was computed.
Terms like "significantly better" should be read as "achieved higher MRR in point estimates."
