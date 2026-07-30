# prototypes/

## Purpose
Early Cassandra benchmark prototypes. These scripts were developed in May-June 2026 to explore Cassandra graph traversal strategies before the formal sysaxis experiments.

## Status
ARCHIVE_LEGACY — Not part of the current active experiment pipeline. Retained for provenance and reproducibility.

## Subdirectories

- **`core_cassandra_early/`** — 26 scripts. Depth profile, parallel cache, relation index, synthetic graph benchmarks, bulk insert tools, Neo4j baseline comparisons. Some marked PAPER_EVIDENCE_LEGACY for scripts that informed the final design.
- **`initial_step_early/`** — 6 scripts. Very first Cassandra-KG exploration scripts (query, insert, sync).

## PAPER_EVIDENCE_LEGACY scripts
The following early scripts directly influenced the paper's system design and can be cited as foundational work:
- `benchmark_depth_profile.py` / `benchmark_depth_profile_cache.py` / `benchmark_depth_profile_parallel.py`
- `benchmark_high_degree.py`
- `benchmark_parallel_cache.py`
- `benchmark_path_relation_index.py` / `benchmark_path_relation_index_parallel.py`
- `benchmark_src_relation_index.py`
- `benchmark_neo4j_baseline.py` / `benchmark_neo4j_relation_path.py`

## Safety
- Most scripts require Cassandra/Neo4j connections.
- Do not run without explicit confirmation.
