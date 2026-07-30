# Senior Report: RRF Compact vs Baselines

## Table
| Comparison | dMRR | drF1 | dWrongAbst | dHit@10 |
|---|---|---|---|---|
| RRF_compact vs Dense-bge | 0.0898 | 0.0184 | -0.0195 | 0.0558 |
| RRF_compact vs Dense+GlobalKG | 0.0581 | 0.0444 | -0.0533 | 0.0519 |
| RRF_compact vs Dense+QueryKG | 0.0708 | 0.015 | -0.015 | 0.0363 |
| RRF_compact vs RRF_raw | 0.0669 | 0.0124 | -0.0208 | 0.0448 |
| BM25_compact vs BM25_raw | 0.1483 | 0.0379 | -0.0643 | 0.1247 |

## Answers
1. Retrieval -> Reader: YES
2. RRF_compact vs RRF_raw: YES
3. WrongAbst reduction: YES
4. Hit@10: YES
5. Best method: RRF_compact (Dense+bge + BM25_KG-enriched RRF)

## API Stats
- Calls: 2436
- Runtime: 4022s
