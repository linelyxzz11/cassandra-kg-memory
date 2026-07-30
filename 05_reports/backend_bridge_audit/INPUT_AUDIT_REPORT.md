# INPUT AUDIT REPORT — LoCoMo Retrieval Methods
## Data Source Tracing & Backend Integration Feasibility
### Date: 2026-07-29 | No API calls | No experiment re-run

---

## Method-by-Method Summary

| # | Method | Memory View | KG Reliance | Score Source | Top-k Source |
|---|--------|------------|-------------|-------------|-------------|
| 1 | BM25 | Raw text | None | Online BM25 (CountVectorizer) | online ranking |
| 2 | Dense-bge | Embeddings(bge) | None | Precomputed frozen scores | precomputed top10 |
| 3 | Dense+GlobalKG | Embeddings + KG triples | Direct | Precomputed frozen scores + KG boost | precomputed top10 |
| 4 | RRF_compact | Dense(raw)+BM25(RawERK) | Indirect via BM25 | Precomputed both | fusion ranking |
| 5 | ZScore-Raw | Dense(raw)+BM25(raw) | None | Precomputed both | fusion ranking |
| 6 | ZScore-RawERK | Dense(raw)+BM25(RawERK) | Indirect via BM25 | Precomputed both | fusion ranking |

---

## Artifact Dependency Graph

```
locomo10.json
    │
    ├──→ memory_records.csv (5882)
    │       │
    │       ├──→ BM25 Raw index (conversation-scoped)
    │       ├──→ frozen memory embeddings (bge-large)
    │       │       └──→ frozen_dense_scores_long.csv
    │       │               ├──→ Dense-bge top10
    │       │               ├──→ Dense+GlobalKG top10 (w/ KG boost)
    │       │               ├──→ ZScore-Raw top10 (w/ BM25 raw)
    │       │               └──→ ZScore-RawERK top10 (w/ BM25 RawERK)
    │       │
    │       └──→ p3_memory_features.csv (5882, ERK)
    │               ├──→ BM25 RawERK index
    │               ├──→ Dense+GlobalKG KG triples
    │               └──→ RRF_compact (WRRF fusion Dense+BM25 RawERK)
    │
    └──→ selected_full5_questions.csv (1986 QIDs, all 6 methods)
```

---

## Findings

### Finding 1: All 6 methods use the same question set
**YES.** All 6 methods use `selected_full5_questions.csv` (1986 QIDs, Cat1-282, Cat2-321, Cat3-96, Cat4-841, Cat5-446). Same query_id universe. SHA256 aligned.

### Finding 2: All 6 methods use the same memory_id universe
**YES.** All 6 methods use `memory_records.csv` (5882 memory_ids). The memory_id set is identical across all 6. No method adds or subtracts memory entries.

### Finding 3: Differences are logical views over the same data
**YES.** The 6 methods are not 6 different retrievers; they are 6 different ways to score the SAME memory candidates:
- BM25 Raw = lexical view (bag-of-words over raw text)
- BM25 RawERK = lexical view (bag-of-words over ERK-augmented text)
- Dense-bge = semantic view (cosine in embedding space)
- Dense+GlobalKG = semantic view + graph view (embedding cosine plus KG adjacency)
- Hybrids (ZScore, RRF) = fused views of the above

### Finding 4: Only ONE method directly depends on KG triples
- **Dense+GlobalKG** = direct dependency (entity extraction → KG graph traversal → adjacency score)
- **BM25 RawERK / RRF_compact / ZScore-RawERK** have INDIRECT KG dependency (ERK features are stored as text in memory_features.csv, not as graph structure)
- **BM25 Raw / Dense-bge / ZScore-Raw** have zero KG dependency

### Finding 5: All but BM25 read precomputed scores/rankings
- **5 methods**: Dense-bge, Dense+GlobalKG, RRF_compact, ZScore-Raw, ZScore-RawERK all load precomputed frozen scores from `frozen_dense_scores_long.csv` and/or precomputed BM25 rankings
- **1 method**: BM25 rebuilds its index at runtime via CountVectorizer
- All 5 use `frozen_dense_scores_long.csv` which only covers 1540/1986 queries (Cat1-4 only)

### Finding 6: All 6 methods can directly enter CSV/Cassandra/Neo4j parity experiment at L2
**YES — with the right adapter.** Since all methods are:
- Python-side computation
- Storage-agnostic logic
- Input = memory text + optionally ERK features + optionally frozen embeddings

Replacing `pd.read_csv()` with `cassandra.query()` or `neo4j.query()` preserves identical computation. The parity test would:
1. Store memory_records.csv content in Cassandra `kg_memory_table` and Neo4j graph
2. Store memory_features.csv content in both backends
3. Read from each backend → rebuild BM25 index → recompute fusion → compare top10

### Finding 7: Adapter requirements per method

| Method | Adapter Needed | Complexity | Notes |
|--------|---------------|-----------|-------|
| BM25 | text reader | Low | Replace CSV read with DB fetch for memory text, rebuild BM25 index |
| Dense-bge | text reader + embedder | Medium | Need embedding model available; or pre-store embeddings in DB |
| Dense+GlobalKG | text + KG triple reader | High | Need graph adjacency query from DB; Dense side same as above |
| RRF_compact | dual score reader | Medium | Needs both Dense scores and BM25 scores from DB |
| ZScore-Raw | dual score reader | Medium | Same as RRF minus ERK text |
| ZScore-RawERK | dual score reader | Medium | Same as RRF |

---

## Cat5 Gap (Known)

All 6 methods use legacy global corpus fallback for Cat5 (446 adversarial queries). The fallback reads `reader_f1_memory_only_full_scoped_bm25_predictions.csv` (global corpus) rather than the conversation-scoped retrieval used for Cat1-4. This is acceptable because Cat5 evaluates abstention behavior, but for a clean parity experiment, Cat5 should also use conversation-scoped retrieval.

---

## Files Generated

```
05_reports/backend_bridge_audit/
├── method_input_audit.csv              ✅
├── artifact_alignment.csv              ✅
├── backend_integration_feasibility.csv ✅
├── input_audit_manifest.json           ✅
└── INPUT_AUDIT_REPORT.md               ✅
```
