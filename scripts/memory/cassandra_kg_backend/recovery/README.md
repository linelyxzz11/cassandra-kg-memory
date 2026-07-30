# cassandra_kg_backend/recovery/

## Purpose
Cassandra data recovery tool. Reconstructs KG edges from source CSV when Cassandra tables are corrupted or lost.

## Status
DANGEROUS_RECOVERY — This script writes to Cassandra. Do not run without explicit confirmation.

## Safety
- **destructive_risk = high**
- **safe_to_run = no**
- Connects to Cassandra and writes data.

## Important Scripts
- `recover_by_src_safe.py` — Safe recovery by source node

## Related Reports
- `reports/data_recovery/` (archived to `_archive_20260709/reports_data_recovery/`)
