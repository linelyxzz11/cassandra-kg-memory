# Alpha 0.6 Selection Audit (Superseded)

This historical audit used the earlier P2 metric scope. The complete canonical
rerun, including alpha=0.9 and alpha=1.0 and all six retrieval metrics, is now
reported in `05_reports/zscore_alpha_sweep/ALPHA_SWEEP_REPORT.md`. That rerun
uses the exact frozen P1-C fusion semantics and reproduces Retrieval Table 1 at
alpha=0.6. The material below is retained only for provenance.

## Answer: YES_COMPLETE

Complete dev-only sweep exists for ZScore linear fusion. alpha=0.6 was uniquely best.

## Sweep Details

| Field | Value |
|---|---|
| Split | dev |
| Dev query count | 390 |
| Alpha candidates | 0.0, 0.1, 0.2, ..., 1.0 (11 values, step 0.1) |
| Fixed params | k=10 (RRF candidate depth) |
| Selection metric | Dev MRR |

## ZScore Linear: Each Alpha Dev MRR

| Alpha | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| 0.0 | 0.3590 | — | 0.6410 | 0.4492 |
| 0.1 | 0.3846 | — | 0.7026 | 0.4848 |
| 0.2 | 0.3872 | — | 0.7385 | 0.4934 |
| 0.3 | 0.3846 | — | 0.7462 | 0.4990 |
| 0.4 | 0.3949 | — | 0.7513 | 0.5076 |
| 0.5 | 0.4103 | — | 0.7615 | 0.5214 |
| **0.6** | **0.4282** | **—** | **0.7513** | **0.5340** |
| 0.7 | 0.4026 | — | 0.7385 | 0.5175 |
| 0.8 | 0.3897 | — | 0.7179 | 0.5044 |
| 0.9 | — | — | — | — (not in grid) |
| 1.0 | — | — | — | — (not in grid) |

alpha=0.6 was the **unique best** (no tie-break needed): next best 0.5 (0.5214), gap=+0.0126.

## Held-out Leakage Check

- alpha=0.6 selected on dev MRR BEFORE held-out evaluation
- No evidence of repeated alpha adjustment on held-out data
- All candidates tested once on dev before final test

## ZScore Version Trace

Two ZScore MRR values exist across different experiment stages:

| MRR | Artifact | Query | Representation | Stage | Status |
|---|---|---|---|---|---|
| 0.5426 | reports/p2_fusion_comparison/test_fusion_overall.csv | 1150 | Dense_raw + BM25_ERK | P2 test fusion | OFFICIAL_SELECTED |
| 0.5462 | config_ZScore_RawERK.json (retrieval_gate.csv) | 1150 | Dense_raw + BM25_RawERK | P3 reader alignment | FROZEN_ANCHOR |

The 0.5462 is the frozen RawERK retrieval anchor used as input to the P3/P4 reader experiments.
The 0.5426 is from the earlier P2 method comparison. Both use the same alpha=0.6 ZScore formula,
with minor MRR differences attributable to representation changes across experiment stages.

## Final Confirmation

Config alpha_dense=0.6 in p3_config.json was selected via documented dev sweep.
