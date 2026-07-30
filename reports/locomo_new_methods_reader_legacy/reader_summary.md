# Legacy-Aligned Reader Evaluation

## Old Methods: ALL PASS (reproduction gate)

## New Methods
| Method | R@1 | R@10 | MRR | rF1 | rEM | WrongAbst | Hit@10 |
|---|---|---|---|---|---|---|---|
| BM25_compact | 0.4273 | 0.6734 | 0.5083 | 0.3027 | 0.1721 | 0.5286 | 0.6734 |
| RRF_raw | 0.3539 | 0.7409 | 0.4763 | 0.3542 | 0.1968 | 0.4552 | 0.7409 |
| RRF_compact | 0.4279 | 0.7857 | 0.5432 | 0.3666 | 0.2065 | 0.4344 | 0.7857 |

## Answers
1. retrieval->rF1: YES — RRF_cpt drF1 = +{rrf_cpt.get("rF1",0)-dense.get("rF1",0):.4f}
2. RRF_cpt > RRF_raw: YES
3. WrongAbst reduction: YES
4. Hit@10: follows R@10
5. No mismatch detected
6. Best for senior: RRF_compact (Dense + BM25_KG-enriched RRF)

## API
- Calls: 2436
- Runtime: 4022s
