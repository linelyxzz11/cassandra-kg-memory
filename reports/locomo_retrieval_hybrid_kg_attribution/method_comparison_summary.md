# Strong Hybrid Baseline + KG Attribution

## Best Method
- RRF+GlobalKG (k=10, lam_kg=0.1, lam_t=0)
- R@1=0.2708 R@10=0.6532 MRR=0.3952

## vs Baselines
- Dense-bge: R@10=0.6565 MRR=0.3776
- BM25: R@10=0.1039 MRR=0.0578
- RRF (k=10): R@10=0.6513 MRR=0.3179 (dMRR=-0.0597)

## Component Attribution
- RRF over Dense: dMRR=-0.0597
- GlobalKG over RRF: dMRR=0.0773
- TemporalKG over RRF: dMRR=0.0773
- TemporalKG over RRF+GlobalKG: dMRR=-0.0004

## Answers
1. RRF over Dense: NO (delta=-0.0597)
2. GlobalKG over RRF: YES, +0.0773
3. TemporalKG cat2 gain: see by_category
4. cat4 hurt: see rescue_hurt_analysis
5. Final recommendation: RRF+GlobalKG or RRF+GlobalKG+TemporalKG
6. KG-Native stopped: R@10=36.17%, MRR=0.2216, far below Dense.

## Runtime
- 19.4s
