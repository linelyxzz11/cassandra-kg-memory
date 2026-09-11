# Question Type Breakdown V1

## Why this analysis is needed

The main tables establish effectiveness; this analysis tests where the gain comes from.
Dense-to-CassMem is the total pipeline gain. ZScore-Raw-to-RawERK is the cleaner incremental ERK comparison.

## Correct category counts

| Column | LoCoMo category | Reader n | Retrieval-evaluable n |
|---|---:|---:|---:|
| Single-Hop | Cat4 | 841 | 841 |
| Multi-Hop | Cat1 | 282 | 282 |
| Temporal | Cat2 | 321 | 321 |
| Open-Domain | Cat3 | 96 | 92 |
| Adversarial | Cat5 | 446 | n/a |

Four Cat3 questions have empty gold evidence and are excluded only from retrieval metrics.

## Main results

- Retrieval Dense-to-CassMem overall MRR@10: +5.87 points.
- Reader Dense-to-CassMem Overall F1: Cat.x +2.26 points; Cat.v +2.27 points.
- Under Cat.v, the largest incremental Raw-to-RawERK Reader gain is Open-Domain (+5.61 F1 points).

## Mechanism conclusion

The evidence does not support a claim that ERK helps mainly Multi-Hop. Retrieval gains are largest on Single-Hop and Temporal; Reader gains depend on prompt protocol and are strongest on Open-Domain under Cat.v. Therefore the defensible claim is that compact ERK signals improve several question types, not specifically multi-hop reasoning.

Do not infer mechanism from Dense-to-CassMem alone because that contrast changes both fusion and representation.
