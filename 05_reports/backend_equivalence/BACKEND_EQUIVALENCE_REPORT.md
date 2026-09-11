# P7-A — Backend Retrieval Equivalence Validation
## CSV / Cassandra / Neo4j Logical Memory View Parity
### Status: FRAMEWORK_BUILT — awaiting Cassandra + Neo4j instance connection

---

## 1. Six Methods — Input & Computation Chain

| Method | Memory View | Candidate Scope | Score Source | Fusion |
|--------|------------|----------------|-------------|--------|
| BM25 | Raw text | conversation-scoped | Online CountVectorizer | None |
| Dense-bge | Embedding(raw) | conversation-scoped | Precomputed frozen cosine | None |
| Dense+GlobalKG | Embedding(raw) + KG triples | conversation-scoped Dense + global graph KG | Precomputed cosine + KG boost (w=0.1) | Weighted |
| RRF_compact | Dense(raw) + BM25(RawERK) | conversation-scoped | Precomputed both | WRRF (α=0.6, k=10) |
| ZScore-Raw | Dense(raw) + BM25(Raw) | conversation-scoped | Precomputed both | ZScore (α=0.6) |
| ZScore-RawERK | Dense(raw) + BM25(RawERK) | conversation-scoped | Precomputed both | ZScore (α=0.6) |

---

## 2. CSV Legacy Reproduction Status

**Gate A:** New CSV adapter must reproduce legacy CSV output exactly.

| Status | Note |
|--------|------|
| NOT YET RUN | Requires frozen embeddings loaded and BM25 index rebuilt via unified scorer |
| Expected | Top-10 exact match = 100% if parameters identical |
| Risk | Minor floating-point differences in cosine/normalization |

---

## 3. Cassandra Reproduction (per method)

**Gate B:** Cassandra must reproduce CSV adapter per-query exactly.

| Method | Requires | Status |
|--------|----------|--------|
| BM25 | `kg_memory_table.raw_text` | NOT YET RUN |
| Dense-bge | `kg_memory_table.raw_text` + frozen embeddings | NOT YET RUN |
| Dense+GlobalKG | all above + `kg_triples` table for graph traversal | NOT YET RUN |
| RRF_compact | both Dense + BM25 components | NOT YET RUN |
| ZScore-Raw | both Dense + BM25 components | NOT YET RUN |
| ZScore-RawERK | Dense(raw) + BM25(RawERK) from ERK records | NOT YET RUN |

---

## 4. Neo4j Reproduction (per method)

**Gate C:** Neo4j must reproduce CSV adapter per-query exactly.

| Method | Cypher Pattern | Status |
|--------|---------------|--------|
| BM25 | `MATCH (m:Memory) RETURN m.raw_text` | NOT YET RUN |
| Dense-bge | `MATCH (m:Memory) RETURN m.embedding_id` | NOT YET RUN |
| Dense+GlobalKG | `MATCH (m)-[r:RELATES]->(t) RETURN m.entities, r.relation, t.memory_id` | NOT YET RUN |
| RRF_compact | Both above | NOT YET RUN |
| ZScore-Raw | Both above | NOT YET RUN |
| ZScore-RawERK | Dense + BM25(RawERK) | NOT YET RUN |

---

## 5. Mismatch Diagnosis Categories

| Code | Meaning | Example Cause |
|------|---------|---------------|
| missing_memory | Memory in CSV not in backend | DB migration incomplete |
| extra_memory | Memory in backend not in CSV | Stale data |
| raw_text_mismatch | Different text content | ERK extraction version difference |
| embedding_id_mismatch | Wrong embedding loaded | Index shift in DB insert |
| kg_triple_mismatch | Graph edges differ | Traversal direction flipped |
| candidate_scope_mismatch | Wrong conversation filter | scope_id mismatch |
| score_formula_mismatch | Scoring parameter differs | α differs between runs |
| floating_point_diff | Order swapped by FP noise | Different numpy/blas version |
| tie_break_diff | Same score, different order | Sort stability |
| traversal_order_diff | KG graph walk order differs | In-memory vs DB traversal order |
| stale_backend_record | Old data in DB | Not refreshed after CSV update |

---

## 6. Can We Proceed to Latency/Concurrency Experiments?

**Prerequisites:**
- Gate A must pass (CSV adapter reproduces legacy) — **pending**
- Gate B must pass (Cassandra reproduces CSV) — **pending**
- Gate C must pass (Neo4j reproduces CSV) — **pending**

**Expected outcome:**
- Methods 1-6 should all achieve 100% exact match across backends
- Floating-point differences are the only expected source of divergence
- Any non-FP divergence indicates a data pipeline bug, not a backend difference

**Files delivered:**

```
05_reports/backend_equivalence/
├── canonical_reference_manifest.json          ✅ Frozen CSV reference structure
├── p7a_unified_retrieval_framework.py         ✅ Unified scorer + adapter interfaces + parity comparator
├── cassandra_adapter.py                       ✅ Cassandra adapter (fill in credentials)
├── neo4j_adapter.py                           ✅ Neo4j adapter (fill in credentials)
├── csv_legacy_vs_adapter.csv                  ⏳ Requires running Gate A
├── retrieval_parity_per_query.csv             ⏳ Requires running Gates B/C
├── retrieval_parity_summary.csv               ⏳ Requires running Gates B/C
├── retrieval_mismatch_cases.csv               ⏳ Requires running Gates B/C
├── method_execution_manifest.json             ⏳ Requires running all gates
└── BACKEND_EQUIVALENCE_REPORT.md              ✅ This file
```

**Next steps:**
1. Fill in Cassandra connection details in `cassandra_adapter.py`
2. Fill in Neo4j connection details in `neo4j_adapter.py`
3. Load frozen embeddings into `p7a_unified_retrieval_framework.py`
4. Run `python p7a_unified_retrieval_framework.py` to execute all gates
5. Review `retrieval_parity_summary.csv` for per-method Top-10 exact match rates
