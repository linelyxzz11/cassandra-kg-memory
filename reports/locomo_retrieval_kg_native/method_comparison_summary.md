# KG-Native Method Comparison

## Overall (cat1-4, sample-scoped)
- KG-Native: R@10=0.3617, MRR=0.2216
- BM25: R@10=0.0669, MRR=0.0485
- Dense-bge: R@10=0.7110, MRR=0.4738

## Anchor Coverage
- Entity anchor: 100.0%
- Any KG candidate: 99.6%
- Avg candidates: 145

## Verdict
- Beats BM25: True
- Anchor coverage >= 70%: True
- Continue to Cassandra serving? YES

## Runtime
- 96.6s
