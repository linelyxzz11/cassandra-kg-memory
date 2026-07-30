# Scripts — cassandra-kg-memory

## Structure

```
scripts/
  README.md                     ← this file
  SCRIPT_INDEX.csv              ← machine-readable full index
  SCRIPT_INDEX.md               ← human-readable summary + layer map
  reorg_plan_20260709.txt       ← dry-run reorganization plan

  memory/
    locomo_pipeline/
      extraction/               ← KG extraction + data import
      retrieval/                ← all retrieval methods (BM25/Dense/KG boost/QueryKG/sample-scoped)
      reader_eval/              ← LLM reader evaluation
      audit/                    ← rescue/hurt, scope, equivalence audits
      backend_compare/          ← Cassandra vs Neo4j comparative scripts

    cassandra_kg_backend/
      ablation/                 ← Layer B internal ablation (parallel/cache/index)
      latency/                  ← end-to-end latency benchmarks
      correctness/              ← Layer A correctness equivalence (c0)
      recovery/                 ← DANGEROUS: data recovery tools

    sysaxis/
      1m_hop_depth/             ← ACTIVE: hop-depth cold+warm sweep
      1m_concurrency/           ← ACTIVE: concurrency sweep
      1m_write_ratio/           ← ACTIVE: write-ratio sweep
      scale_sweep/              ← PENDING: future scale experiments

    archive_legacy/              ← already-archived old versions

  prototypes/
    core_cassandra_early/       ← 26 early Cassandra benchmark prototypes
    initial_step_early/          ← 6 very first exploration scripts
```

## Summary

| Category | Scripts | Status |
|---|---|---|
| LoCoMo pipeline | 25 | PAPER_EVIDENCE |
| Cassandra-KG backend | 7 | PAPER_EVIDENCE |
| Sysaxis final | 16 | ACTIVE_FINAL |
| Recovery | 1 | DANGEROUS |
| Prototypes | 32 | ARCHIVE_LEGACY |
| **Total** | **~81** | |

## Safety

Scripts connecting to Cassandra or Neo4j are marked:
- `safe_to_run = no` — Requires explicit confirmation
- `destructive_risk = high/medium/low` — Risk level

See `SCRIPT_INDEX.csv` for per-script safety flags.


## ⚠️ NOTE: Reorg is DRY-RUN ONLY

The directory structure above reflects the PLANNED organization.
No scripts have been moved. All files remain at original locations.

See `reorg_plan_20260709.txt` for the complete dry-run plan.
