# Final Method Recommendation v2

## 1. Best Compact Representation
- BM25 entities+relations+keywords (NO raw triples): MRR=0.4849, R@1=0.3935, R@10=0.6701
- BM25 full (includes triples): MRR=0.4939, R@1=0.4065
- Delta over BM25_raw: MRR +0.1019

## 2. Raw Triples Verdict
Triples show marginal positive when bundled with other enrichment (+0.009 MRR over no-triples compact),
but they increase token count significantly.
Keep as optional enrichment; NOT as standalone variant.

## 3. Dense-enriched
NEEDS BGE API RE-ENCODE.

## 4. Best Method: Weighted RRF (Dense_raw + BM25_compact)
- RRF alpha=0.5, k=10
- MRR=0.5226, R@1=0.4058, R@10=0.7805
- Over Dense_raw: MRR +0.0488 (+4.9%)
- Over BM25_compact: MRR +0.0377
- Net rescue over Dense: +60

## 5. KG Contribution Attribution
- Representation: PRIMARY (+0.1019 MRR in BM25)
- Ranking prior: MINOR (+0.0028)
- Candidate expansion: NEGLIGIBLE (+0.9pp)
- Reader packaging: NEGATIVE (-0.011 F1)

## 6. Cassandra-KG Backend
KG triples in Cassandra kg_edges_by_src serve both retrieval enrichment
(enriched memory text generation) and system-axis benchmarks (latency/correctness).

## 7. Negative Results
- GlobalKG++: negligible
- KG candidate expansion: +0.9pp ceiling
- KG-Native: R@10=36.17%
- Raw triples reader: F1 -0.011
