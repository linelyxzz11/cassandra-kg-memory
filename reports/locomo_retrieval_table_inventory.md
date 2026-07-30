# LoCoMo Retrieval Table Inventory

Generated: 2026-07-10 | No experiments re-run.

## Target Table Mapping

```text
Method | R@1   | R@10  | MRR   | Avg Latency | P95
BM25   |
Dense  |
Hybrid |
Neo4j-KG |
Cassandra-KG |
```

## Fillable Rows (sample-scoped, preferred)

| Method | R@1 | R@10 | MRR | Latency | Source |
|---|---|---|---|---|---|
| **BM25** | 0.2649 | 0.5619 | 0.3600 | **MISSING** | `results/sample_scoped/sample_scoped_retrieval_summary.csv` |
| **Dense-bge** | 0.3419 | 0.7009 | 0.4534 | **MISSING** | same |
| Dense-bge+GlobalKG | 0.3872 | 0.7095 | 0.4851 | **MISSING** | same |
| Dense-bge+QueryKG | 0.3585 | 0.7185 | 0.4724 | **MISSING** | same |

## For Hybrid row

Two candidates:

| Sub-method | R@1 | R@10 | MRR |
|---|---|---|---|
| Dense-bge+GlobalKG (precision) | 0.3872 | 0.7095 | 0.4851 |
| Dense-bge+QueryKG (recall) | 0.3585 | 0.7185 | 0.4724 |

GlobalKG wins precision (R@1/MRR), QueryKG wins recall (R@10). Choose one, or report both.

## For Neo4j-KG and Cassandra-KG rows

### Quality: Identical by construction

- `Cassandra-KG` and `Neo4j-KG` return identical retrieval results (0 disagreements in `audit_e2_backend_fair.py`).
- The **only LoCoMo KG retrieval quality numbers** are from old **TF-IDF + KG boost** (not pure KG):
  - R@1=0.2638, R@10=0.4899, MRR=0.3358
  - Source: `results/locomo_cassandra_kg_results.csv` / `results/locomo_neo4j_results.csv`
- Pure KG signal retrieval (has_KG only) was **not run as standalone** on LoCoMo. Audit shows both_ok=973/1986 (49% of QA have KG-covered gold evidence).

### Latency: NOT on LoCoMo data

LoCoMo-specific KG backend latency was never measured. The only latency data available is from system-axis synthetic graph benchmarks:

| Backend | 1M synth hop=2 cold | 1M synth hop=2 warm | 100K legacy cold | 100K legacy warm |
|---|---|---|---|---|
| Cassandra-KG | mean=15.4ms p95=32.7ms | mean=14.9ms p95=32.6ms | mean=50.3ms p95=93.3ms | mean=50.1ms p95=92.3ms |
| Neo4j-KG | mean=16.6ms p95=34.2ms | mean=15.6ms p95=31.5ms | mean=64.9ms p95=160.1ms | mean=64.2ms p95=154.2ms |

These use synthetic graphs (fanout=20, hop=2), not LoCoMo memory graph. They show relative performance but cannot be directly filled into a LoCoMo retrieval table without a caveat.

## Cassandra-KG variants found

| # | Variant | Has Quality? | Has Latency? | LoCoMo? |
|---|---|---|---|---|
| 1 | TF-IDF+Cassandra-KG boost (global) | R@1=0.264 R@10=0.490 | No | Yes |
| 2 | Pure KG signal only | No (not run) | No | N/A |
| 3 | 1M synthetic hop=2 (system axis) | N/A | mean=15ms p95=33ms | No |
| 4 | 100K legacy mixed rw (scale sweep) | N/A | mean=50ms p95=93ms | No |

## Gaps

| Gap | Details |
|---|---|
| **Latency for all LoCoMo methods** | BM25, Dense, Hybrid, KG backends all missing per-query latency on LoCoMo data. `locomo_latency_results.csv` only has old TF-IDF microbenchmark (200 queries). |
| **Pure KG retrieval quality** | Only run as TF-IDF+KG boost. Pure KG (has_KG signal only) not run standalone. Could be computed from `evidence_map.csv` + `locomo_memory_records.csv`. |
| **LoCoMo KG backend latency** | Never measured end-to-end on LoCoMo. System-axis synthetic numbers are qualitatively informative but different corpus. |
