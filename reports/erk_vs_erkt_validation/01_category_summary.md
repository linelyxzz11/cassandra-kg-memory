# Step 1: Category Analysis

## ERKT - ERK by Category
| Category | BM25 ΔMRR | RRF ΔMRR |
|---|---|---|
| multi-hop | +0.0202 | +0.0061 |
| temporal | +0.0093 | +0.0003 |
| open-domain | +0.0129 | +0.0039 |
| single-hop | +0.0139 | +0.0031 |

## Win/Tie/Loss (BM25)
| Category | improved | tied | worsened |
|---|---|---|---|
| multi-hop | 35 | 241 | 6 |
| temporal | 16 | 299 | 6 |
| open-domain | 7 | 89 | 0 |
| single-hop | 79 | 740 | 22 |

## Answers
1. Time temporal effect: Not temporal-dominant (cat2 dMRR=+0.0093)
2. Multi-hop benefit: dMRR=+0.0202
3. No category-level drop detected
4. Overall BM25 ERKT-ERK dMRR=+0.0140
5. RRF shrinkage: RRF combines Dense+BM25; Dense lacks Time signal, diluting BM25 Time gain
