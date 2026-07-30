# cassandra_kg_backend/

## Purpose
Cassandra-KG as structured memory backend: ablation experiments, latency benchmarks, correctness verification, and data recovery.

## Status
PAPER_EVIDENCE — System-side evidence for the Cassandra-KG backend.

## Subdirectories

- **`ablation/`** — Layer B internal ablation (parallel workers, cache, relation index)
- **`latency/`** — End-to-end latency benchmarks
- **`correctness/`** — Layer A correctness equivalence (Cassandra vs Neo4j)
- **`recovery/`** — Data recovery tools (DANGEROUS)

## Safety
- All subdirectories except `recovery/` require Cassandra/Neo4j connections.
- **`recovery/`**: DANGEROUS. Can modify Cassandra data. Do not run without explicit confirmation.

## Related Reports
- `reports/sysaxis_1m_hop_depth_final/`
- `reports/sysaxis_1m_concurrency_final/`
- `reports/sysaxis_1m_write_ratio_final/`
- `results/system/layerB_*.csv`
