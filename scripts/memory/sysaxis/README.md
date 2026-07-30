# sysaxis/

## Purpose
System-axis experiments: Large-scale Cassandra vs Neo4j comparison experiments under controlled conditions (hop-depth, concurrency, write-ratio).

## Status
ACTIVE_FINAL — All three final sweep experiments completed and sealed.

## Subdirectories

- **`1m_hop_depth/`** — 1M edge hop-depth sweep (cold + warm, 40 trials each). Core Cassandra vs Neo4j crossover evidence.
- **`1m_concurrency/`** — 1M edge concurrency sweep (cold + warm). Client count scaling.
- **`1m_write_ratio/`** — 1M edge write-ratio sweep (cold + warm). Mixed read/write interference.
- **`scale_sweep/`** — [ACTIVE_PENDING] Reserved for future scale sweeps.

## Safety
- All sweep scripts require Cassandra and Neo4j connections.
- All are **read-only** (except `neo4j_scale_import.py` which imports data).
- Do not run without explicit confirmation.

## Related Reports
- `reports/sysaxis_1m_hop_depth_final/`
- `reports/sysaxis_1m_concurrency_final/`
- `reports/sysaxis_1m_write_ratio_final/`
