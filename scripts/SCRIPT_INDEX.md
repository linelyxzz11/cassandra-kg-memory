# Script Index — cassandra-kg-memory

Generated: 2026-07-09 | Mode: dry-run | Reorg plan: `scripts/reorg_plan_20260709.txt`

## Overview

| Status | Count | Description |
|---|---|---|
| ACTIVE_FINAL | 16 | Sysaxis 1M final sweep scripts |
| PAPER_EVIDENCE | 25 | Core LoCoMo + Cassandra-KG evidence chain |
| SUPPORTING_TOOL | 4 | Inspection and utility tools |
| DANGEROUS_RECOVERY | 1 | Data recovery (do not run) |
| ARCHIVE_LEGACY | 32 | Early prototypes |
| **Total** | **~78** | |

## Layer Map (for paper)

| Paper Section | Script Dir | Key Scripts |
|---|---|---|
| **Quality: KG signal retrieval** | `locomo_pipeline/retrieval/` | `locomo_retrieval_dense_bge.py`, `locomo_retrieval_dense_kg_boost.py`, `locomo_retrieval_dense_bge_query_kg_rerank.py`, `locomo_retrieval_sample_scoped.py`, `locomo_retrieval_bm25.py` |
| **Quality: KG extraction** | `locomo_pipeline/extraction/` | `observation_to_kg_spacy.py`, `locomo_to_memory_records.py` |
| **Quality: Reader evaluation** | `locomo_pipeline/reader_eval/` | `reader_f1_memory_only_v2.py`, `locomo_llm_reader_full_v3.py` |
| **Quality: Audit** | `locomo_pipeline/audit/` | `audit_dense_kg_rescue.py`, `analysis_sample_scoped.py`, `retrieval_scope_audit.py` |
| **System: Cassandra-KG backend** | `cassandra_kg_backend/` | `benchmark_cassandra_internal_ablation.py`, `benchmark_cassandra_kg_latency.py` |
| **System: Correctness (Layer A)** | `cassandra_kg_backend/correctness/` | `c0_cassandra_vs_neo4j.py`, `c0_naive_vs_parallel.py` |
| **System: Hop-depth sweep** | `sysaxis/1m_hop_depth/` | `hop_cold_sweep.py`, `hop_warm_sweep.py`, `semantic_gates.py` |
| **System: Concurrency sweep** | `sysaxis/1m_concurrency/` | `concurrency_cold_sweep.py`, `concurrency_warm_sweep.py` |
| **System: Write-ratio sweep** | `sysaxis/1m_write_ratio/` | `sysaxis_cold_sweep.py`, `sysaxis_warm_sweep.py` |

## Risk Legend

| Risk | Meaning |
|---|---|
| **high** | Can destroy data. Do not run without explicit confirmation. |
| **medium** | Connects to Cassandra/Neo4j and may modify data. Requires review. |
| **low** | Connects to DB but read-only. Safe to run with caution. |
| **none** | No DB connection. Safe to run anytime. |

## Full Index

See `scripts/SCRIPT_INDEX.csv` for the complete machine-readable index with old_path, new_path, status, layer, purpose, inputs, outputs, related_reports, safe_to_run, destructive_risk, last_verified, and notes for all ~78 scripts.

## Current Status

- **Dry-run only**: No files have been moved. All scripts remain at original locations.
- **Reorg plan**: `scripts/reorg_plan_20260709.txt`
- **CSV index**: `scripts/SCRIPT_INDEX.csv`
