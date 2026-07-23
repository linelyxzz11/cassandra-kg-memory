# P1-C Compact Component Ablation Summary

## Official-method parity

RawERK rebuilt Top-10 matched the frozen P3 ranking for every held-out query.

## ZScore overall — held-out 1150

| Variant | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Raw | 0.3748 | 0.6835 | 0.7817 | 0.5073 |
| RawE | 0.4061 | 0.7017 | 0.7957 | 0.5312 |
| RawR | 0.3852 | 0.6922 | 0.7939 | 0.5181 |
| RawK | 0.4035 | 0.6957 | 0.7930 | 0.5292 |
| RawER | 0.4157 | 0.7130 | 0.8017 | 0.5425 |
| RawEK | 0.4157 | 0.7087 | 0.7991 | 0.5385 |
| RawRK | 0.4139 | 0.7017 | 0.8009 | 0.5384 |
| RawERK | 0.4217 | 0.7087 | 0.8070 | 0.5462 |
| RawERKT | 0.4357 | 0.7200 | 0.8148 | 0.5587 |

## Leave-one-out deltas

- NoCompact: ΔMRR=-0.03889, ΔR@10=-0.02522
- WithoutEntity: ΔMRR=-0.00775, ΔR@10=-0.00609
- WithoutRelation: ΔMRR=-0.00763, ΔR@10=-0.00783
- WithoutKeyword: ΔMRR=-0.00368, ΔR@10=-0.00522
- TimeSensitivity: ΔMRR=+0.01254, ΔR@10=+0.00783

## Interpretation rule

Only use statistical-support language when the conversation-cluster bootstrap confidence interval does not cross zero.
RawERKT is a time-sensitivity test; it is not a w/o-time condition.

See p1c_paired_bootstrap.csv and category tables for final claims.

## Time Category Deltas (RawERKT - RawERK)

| Category | ΔMRR | ΔR@10 |
|---|---:|---:|
| cat1_multi-hop | +0.0135 | +0.0070 |
| cat2_temporal | +0.0095 | +0.0035 |
| cat3_open-domain | +0.0023 | +0.0000 |
| cat4_single-hop | +0.0146 | +0.0017 |

## Frozen Status
- **Status**: OFFICIAL_FROZEN
- RawERKT: POST_HOC_SENSITIVITY (not promoted)
- Time helps all categories, not only temporal; cat4 single-hop gains most.
