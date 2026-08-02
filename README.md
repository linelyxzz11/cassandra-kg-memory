# CassMem: Cassandra-Backed Structured Conversational Memory

> **Paper target**: DAI 2026 (8th Int'l Conference on Distributed Artificial Intelligence, 中国香港)
> **Deadline**: Abstract Jul 27, Full Paper Aug 3

CassMem is a research system that uses **Cassandra as the storage backend** for
long-term conversational memory in LLM-based assistants. It evaluates along
three orthogonal axes: **retrieval effectiveness**, **reader answer quality**,
and **system serving performance**.

---

## Table of Contents

1. [Overview](#overview)
2. [Axis 1: Retrieval Effectiveness](#axis-1-retrieval-effectiveness)
3. [Axis 2: Reader Answer Quality (F1 / EM / BLEU-1)](#axis-2-reader-answer-quality)
4. [Axis 3: System Serving (Cassandra vs Neo4j)](#axis-3-system-serving)
5. [Six Retrieval Methods](#six-retrieval-methods)
6. [Backend Equivalence (P7-A)](#backend-equivalence-p7-a)
7. [Directory Layout](#directory-layout)
8. [Reproduction](#reproduction)

---

## Overview

```
LoCoMo conversations
        │
        ▼
Memory Construction (ERK extraction: Entity / Relation / Keyword)
        │
        ├──► CSV (canonical reference / development)
        ├──► Cassandra (proposed production backend)
        └──► Neo4j (graph baseline)
        │
        ▼
Retrieval Algorithms (shared client-side scorer, backend-agnostic)
  BM25 │ Dense-bge │ Dense+GlobalKG │ RRF_compact │ ZScore-Raw │ ZScore-RawERK
        │
        ▼
GPT-4o Reader (top-k memories → short answer)
        │
        ▼
LoCoMo official evaluation (F1 / EM / BLEU-1 / abstention)
```

**Dataset**: LoCoMo, 10 long multi-session conversations
- 5,882 memory records
- 1,986 QA pairs: Cat1 multi-hop (282), Cat2 temporal (321), Cat3 commonsense (96), Cat4 single-hop (841), Cat5 adversarial (446)

**Two prompt settings (all six methods × both settings)**:
- **Cat.✗ (Setting A)**: unified generic short-answer prompt for all categories
- **Cat.✓ (Setting B)**: LoCoMo category-format prompt (Cat2 temporal-date instruction; Cat5 adversarial 2-option binarization with option-text mapping)

---

## Axis 1: Retrieval Effectiveness

### Sample-scoped retrieval (conversation-level candidates, R@K / MRR)

| Method | R@1 | R@5 | R@10 | MRR |
|---|:---:|:---:|:---:|:---:|
| BM25 | 0.2649 | 0.4809 | 0.5619 | 0.3600 |
| Dense-bge | 0.3419 | 0.6078 | 0.7009 | 0.4534 |
| Dense+GlobalKG (w=0.1) | **0.3872** | 0.6198 | 0.7095 | **0.4851** |
| Dense+QueryKG (config_B) | 0.3585 | **0.6234** | **0.7185** | 0.4724 |

- **GlobalKG = precision-oriented** (best R@1 / MRR)
- **QueryKG = recall-oriented** (best R@10)
- All methods: cross_sample_rate = 0 (conversation-scoped)

### KG Coverage Audit

| View | KG coverage |
|---|---|
| All memories | 2353/5882 = 40.0% |
| Gold evidence | 1060/1434 = 73.9% |
| Non-gold | 1293/4457 = 29.0% |
| Enrichment ratio | **2.55×** |

### Category-wise R@1

| Method | Cat1 multi | Cat2 temporal | Cat3 common | Cat4 single | Cat5 adv |
|---|:---:|:---:|:---:|:---:|:---:|
| BM25 | 0.1028 | 0.3209 | 0.0833 | 0.3008 | 0.2982 |
| Dense-bge | 0.2801 | 0.4766 | 0.1875 | 0.3876 | 0.2309 |
| GlobalKG | **0.3617** | 0.4953 | **0.2604** | **0.4174** | **0.2960** |
| QueryKG | 0.2908 | **0.5140** | 0.2083 | 0.4043 | 0.2354 |

---

## Axis 2: Reader Answer Quality

### Main table (GPT-4o-2024-08-06, max_tokens=64, full 1986 QA, offline corrected v3)

**Setting A (Cat.✗)** — unified prompt:

| Method | Cat1 | Cat2 | Cat3 | Cat4 | Cat5 | **Cat1-4** | **Full5** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BM25 | 0.1959 | 0.2618 | 0.1761 | 0.4572 | 0.0942 | 0.3511 | 0.2934 |
| Dense-bge | 0.3328 | 0.3124 | 0.1941 | 0.5632 | 0.0785 | 0.4457 | 0.3633 |
| Dense+GlobalKG | 0.3280 | 0.2951 | 0.2211 | 0.5720 | 0.0874 | 0.4477 | 0.3668 |
| RRF_compact | 0.3547 | 0.2890 | 0.2135 | **0.6033** | 0.1009 | 0.4680 | 0.3855 |
| ZScore-Raw | 0.3585 | 0.3041 | **0.2440** | 0.5875 | 0.0785 | 0.4651 | 0.3783 |
| **CassMem (ZScore-RawERK)** | **0.3702** | 0.3057 | 0.2351 | 0.6009 | 0.0807 | **0.4743** | **0.3859** |

**Setting B (Cat.✓)** — category-format prompt:

| Method | Cat1 | Cat2 | Cat3 | Cat4 | Cat5 | **Cat1-4** | **Full5** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BM25 | 0.2024 | 0.3654 | 0.1991 | 0.4573 | 0.8789 | 0.3753 | 0.4884 |
| Dense-bge | 0.3327 | 0.4389 | 0.2097 | 0.5638 | 0.8722 | 0.4734 | 0.5629 |
| Dense+GlobalKG | 0.3351 | 0.4520 | 0.2078 | 0.5675 | 0.8789 | 0.4784 | 0.5684 |
| RRF_compact | 0.3498 | 0.4363 | 0.2242 | 0.6030 | 0.8677 | 0.4983 | 0.5812 |
| ZScore-Raw | 0.3423 | 0.4299 | 0.2004 | 0.5851 | 0.8700 | 0.4843 | 0.5709 |
| **CassMem (ZScore-RawERK)** | **0.3661** | **0.4526** | **0.2564** | 0.6028 | 0.8587 | **0.5065** | **0.5856** |

### Key reader findings

- **Setting B (category-format) boosts Temporal (+0.10~0.15)** and **Cat5 abstention (0.08→0.86)**
- **Cat5 scoring fix**: Setting B (a)/(b) outputs must be mapped back to option text via per-query deterministic swap before scoring — old evaluator compared "(a)" against gold text and scored 0
- **CassMem (ZScore-RawERK) ranks #1 in Cat1-4 and Full5** under both settings among all six internal methods
- Full5 = weighted per-query mean (282/321/96/841/446), not simple category average

### BLEU-1 (Mem0/A-MEM protocol: w=(1,0,0,0), method1 smoothing)

| Method | Setting B Cat1-4 B1 | Setting B Full5 B1 |
|---|:---:|:---:|
| BM25 | 0.2982 | 0.4289 |
| Dense-bge | 0.3827 | 0.4933 |
| Dense+GlobalKG | 0.3829 | 0.4946 |
| RRF_compact | 0.4012 | 0.5066 |
| ZScore-Raw | 0.3842 | 0.4939 |
| **CassMem** | **0.4074** | **0.5096** |

> Note: Full5 B1 is inflated by Cat5 (canonical gold = fixed phrase "Not mentioned in the conversation", 22.5% weight). Report Cat1-4 B1 as headline; report Full5 B1 with the caveat.

---

## Axis 3: System Serving

### Layer framework

| Layer | Status | Result |
|---|---|---|
| **Layer A** correctness equivalence | ✅ done | Cassandra vs Neo4j retrieval disagreement = **0** |
| **Layer B1** parallel worker sweep | ✅ done | Hop=2 **10.68×**, Hop=4 **37.86×** |
| **Layer B2** relation-index | ⏳ | needs relation-selective workload |
| **Layer B3** cache effective-latency | ⏳ | warm upper bound ≠ realistic |
| **Layer C** trade-off map | ⏳ | future work |
| **P7-A** backend retrieval equivalence | ✅ done | CSV=Cassandra=Neo4j **100%** |
| **P7-B** serving performance | ✅ done | see below |

### B1: Parallel worker sweep (100K synthetic graph, graph_id=synth_100000_1781447372)

| Mode | Workers | Hop=2 mean ms | Speedup | Hop=4 mean ms | Speedup |
|---|:---:|:---:|:---:|:---:|:---:|
| naive | 1 | 312.38 | 1.00× | 11918.14 | 1.00× |
| w4 | 4 | 52.17 | 5.99× | 1160.37 | 10.27× |
| w8 | 8 | 33.23 | 9.40× | 502.61 | 23.71× |
| w16 | 16 | 30.19 | 10.35× | 359.17 | 33.18× |
| w32 | 32 | 29.25 | 10.68× | 314.80 | 37.86× |

All modes produce identical raw_edges — parallelism does not change semantics.

### P7-A: Backend equivalence (CSV reference)

| Check | Cassandra vs CSV | Neo4j vs CSV |
|---|:---:|:---:|
| 5882 memory_id set | 100% | 100% |
| raw_text per-record | 100% | 100% |
| ERK fields per-record | 100% | 100% |
| 2096 KG triples | 100% (after paren fix) | 100% |
| BM25 Top-10 exact match (1986 q) | **100%** | **100%** |

Conclusion: all six methods are **storage-agnostic** — the backend only provides the logical memory view; the shared client-side scorer guarantees identical retrieval. Therefore QA F1/B1/J results carry over unchanged to Cassandra/Neo4j deployments.

### P7-B: Serving performance (Cassandra vs Neo4j, same logical data)

| Workload | Cassandra p50/p95 (ms) | Neo4j p50/p95 (ms) | Neo4j speedup |
|---|:---:|:---:|:---:|
| Point lookup | 10.3 / 19.5 | 1.3 / 1.8 | ~8× |
| Entity neighbors | 9.8 / 18.1 | 0.9 / 2.0 | ~11× |
| Relation filter | 9.4 / 13.5 | 0.8 / 1.2 | ~11× |
| Multi-hop | 18.3 / 28.1 | 1.0 / 1.6 | ~19× |
| Hybrid serving | 9.9 / 18.5 | 1.0 / 1.3 | ~10× |

- Throughput: Cassandra ~104 QPS vs Neo4j ~1000-1190 QPS (sequential, single-node)
- Burst (500 writes+reads): Cassandra 10.1s vs Neo4j 1.3s
- Both backends preserve retrieval semantics; the difference is **serving trade-off**, not accuracy

### Legacy system results (Layer A-D, 100K graph)

| Experiment | Result |
|---|---|
| C0-A parallel (16w, hop4) | 42,908 ms → 1,413 ms = **30.4×** |
| C0-B parallel+cache | 42,203 ms → 1,161 ms = **36.3×** |
| C0-C relation-index | raw edges **-92.1%** on relation-selective workload |
| C0-D Neo4j vs Cassandra | 39 paths = 39 paths, **0 disagreements** |

---

## Six Retrieval Methods

All six share the same 5,882-record memory corpus, same 1,986 questions, same
conversation scope, and the same underlying facts. They differ only in the
**logical view** consumed:

| Method | Raw | ERK text | Dense vec | KG triples | Fusion |
|---|:---:|:---:|:---:|:---:|---|
| BM25 | ✅ | ❌ | ❌ | ❌ | — |
| Dense-bge | ✅ | ❌ | ✅ | ❌ | — |
| Dense+GlobalKG | ✅ | ~ | ✅ | ✅ | KG boost (w=0.1) |
| RRF_compact | ✅ | ✅ | ✅ | ❌ | WRRF (α=0.6, k=10) |
| ZScore-Raw | ✅ | ❌ | ✅ | ❌ | ZScore (α=0.6) |
| **ZScore-RawERK (CassMem)** | ✅ | ✅ | ✅ | ❌ | ZScore (α=0.6) |

**Logical memory record**: `(memory_id, graph_id, raw_text, entities, relations, keywords, timestamp, version, embedding_id)`
**Logical graph edge**: `(graph_id, src, relation, dst, memory_id, version)`

---

## Backend Equivalence (P7-A)

Unified `BackendAdapter` interface with three implementations (`CSVBackend`,
`CassandraBackend`, `Neo4jBackend`). One shared `RetrievalScorer` performs all
scoring/fusion/tie-breaking — backends only expose:

```
list_memories(scope_id)
get_raw_records(scope_id)
get_erk_records(scope_id)
get_memory_ids(scope_id)
get_embeddings(memory_ids)
get_triples(scope_id)
get_edges_by_src(scope_id, src)
get_edges_by_src_relation(scope_id, src, relation)
```

Key files:
- `03_src/p7a_unified_retrieval_framework.py`
- `03_src/cassandra_adapter.py`, `03_src/neo4j_adapter.py`
- `05_reports/backend_equivalence/` (results)
- `05_reports/backend_system_eval/` (P7-B results)

---

## Directory Layout

| Directory | Purpose |
|---|---|
| `00_project/` | Experiment registry, artifact hashes, claims and evidence |
| `01_data/` | Canonical LoCoMo memory/QA records and frozen embeddings |
| `02_artifacts/` | Reusable intermediate artifacts (p3_memory_features.csv, frozen events) |
| `03_src/` | Evaluation, retrieval, memory, and backend implementations |
| `04_experiments/` | Publication-facing experiment entrypoints and audits |
| `05_reports/` | Formal results, manifests, audits, and tables |
| `06_analysis/` | Exploratory analyses (not formal results) |
| `07_runtime/` | Local runtime/cache state; ignored by Git |
| `08_literature/` | Structured literature evidence |
| `09_archive/` | Superseded or failed runs |

Key subpaths:
- `01_data/locomo_memory_records.csv` — 5,882 canonical memories
- `01_data/locomo_qa_records.csv` — 1,986 QA
- `02_artifacts/p3_memory_features.csv` — ERK features + triples
- `05_reports/evaluator_input_audit/selected_full5_questions.csv` — 1,986 question set
- `05_reports/official_eval/gpt4o_dual_setting_locomo_corrected_v3/` — offline corrected scores (main tables)
- `05_reports/locomo_gpt4o_prompt_protocol/` — per-method per-setting reader predictions
- `05_reports/backend_equivalence/` — P7-A parity results
- `05_reports/backend_system_eval/` — P7-B latency/throughput/burst/recovery
- `05_reports/p1_compact_component_ablation/` — 16-variant compact ablation
- `reports/c0_correctness/` — Layer A-D system summaries
- `reports/sysaxis_*` — 1M-scale system benchmarks

---

## Reproduction

```powershell
# 1. Retrieval + reader (requires OpenAI API key)
python 04_experiments/retrieval/build_locomo_gold_memory_v2.py
python 04_experiments/retrieval/run_dense_global_kg_rerun.py
python 04_experiments/retrieval/build_retrieval_main_table.py

# 2. GPT-4o dual-setting prompt protocol (6 methods × 2 settings)
python 03_src/evaluation/run_locomo_prompt_protocol_gpt4o.py \
  --questions 05_reports/evaluator_input_audit/selected_full5_questions.csv \
  --memories 01_data/locomo_memory_records.csv \
  --method ZScore-RawERK \
  --output-dir 05_reports/locomo_gpt4o_prompt_protocol/setting_b_category/ZScore-RawERK \
  --setting b_category

# 3. Offline corrected scoring (no API)
python 03_src/evaluation/locomo_corrected_evaluator_v3.py

# 4. Backend equivalence (P7-A, requires Cassandra + Neo4j running)
python 03_src/p7a_unified_retrieval_framework.py

# 5. System serving benchmark (P7-B)
python 05_reports/backend_system_eval/p7b_benchmark.py
```

**Hard rules**:
- Never mix 1150 / 1540 / 1986 scopes silently
- Never mix global-corpus and sample-scoped retrieval in the same table
- Never compare Cassandra vs Neo4j alone as "equivalence" — both must be
  compared against the CSV reference
- Report per-query means, not category averages, for Overall F1
