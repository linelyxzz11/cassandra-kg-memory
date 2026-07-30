# locomo_pipeline/retrieval/

## Purpose
LoCoMo memory retrieval: TF-IDF baseline, BM25, Dense-bge, GlobalKG boost, QueryKG rerank, and sample-scoped evaluation.

## Status
PAPER_EVIDENCE — Core retrieval pipeline producing the paper's Table 1-3 results.

## Important Scripts
- `locomo_retrieval_bm25.py` — BM25 retrieval baseline
- `locomo_retrieval_dense_bge.py` — Dense-bge sentence transformer retrieval
- `locomo_retrieval_dense_kg_boost.py` — Dense-bge + GlobalKG prior boost (weight sweep)
- `locomo_retrieval_dense_bge_query_kg_rerank.py` — Dense-bge + QueryKG rerank (config_B, lam/topN sweep)
- `locomo_retrieval_sample_scoped.py` — Sample-scoped retrieval for all methods (no cross-sample leakage)
- `locomo_retrieval_tfidf.py` — TF-IDF baseline
- `locomo_cassandra_retrieval.py` — Cassandra KG-based retrieval backend
- `locomo_neo4j_retrieval.py` — Neo4j KG-based retrieval backend
- `locomo_retrieval_benchmark.py` — Retrieval benchmark runner
- `encode_bge_large_api.py` — BGE-large encoding (API-based)

## Safety
- Local retrieval scripts (bm25, tfidf): Safe to run.
- Dense retrieval: Requires `encode_bge_large_api.py` with API key.
- Cassandra/Neo4j retrieval: Do not run without explicit confirmation. Read-only but requires DB connection.

## Related Reports
- `results/locomo_bm25_results.csv`
- `results/locomo_dense_bge_results.csv`
- `results/locomo_dense_kg_boost_results.csv`
- `results/sample_scoped/`
- `results/query_kg_rerank/`
- `results/locomo_memory_bge_large.npy`, `results/locomo_qa_bge_large.npy` (embedding caches)
