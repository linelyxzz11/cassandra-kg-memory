# Final Method Definition

The final retriever uses two complementary memory views:

1. Raw natural-language memory for dense semantic retrieval (Dense-bge).
2. A KG-derived compact structured memory view for lexical retrieval (BM25),
   consisting of entities, relations, and informative keywords.

The two rankings are combined using weighted reciprocal rank fusion.

## KG Contribution Attribution
KG contribution is attributed through:
```
RRF(Dense_raw + BM25_KG_enriched) - RRF(Dense_raw + BM25_raw)
```

## Compact Representation Format
```
<raw memory text>

Entities: <deduplicated canonical entity names from KG triples>
Relations: <deduplicated relation labels>
Keywords: <deduplicated informative keywords from KG tokens>
```

Raw spaCy KG triples are NOT included in the final reader prompt
(shown to add noise, F1 -0.011).

## Methods Included in Final Table
- BM25_raw
- Dense_raw (BGE-large)
- BM25_KG-Enriched-Compact
- RRF(Dense_raw + BM25_raw)
- RRF(Dense_raw + BM25_KG-Enriched-Compact) ← recommended
- Dense_KG-Enriched-Compact (pending BGE API)
- Dense+GlobalKG-Prior (ablation)
- BM25_full_with_triples (ablation)

## Methods EXCLUDED from Final Table
- QueryKG
- GlobalKG++
- KG-Native (R@10=36.17%)
- KG candidate expansion (+0.9pp ceiling only)
- Reader raw-triple packaging (F1 -0.011)
