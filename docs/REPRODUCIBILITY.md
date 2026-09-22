# Reproducibility

This document identifies the current experiment entry points and the boundaries of
the corresponding evidence. Run commands from the repository root.

## Environment

Create a virtual environment and install the declared dependencies:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
~~~

Cassandra and Neo4j are required only for backend experiments. Credentials stay in
the ignored `.env` file. Do not place API keys in commands, manifests, or reports.

## Fast validation

~~~powershell
python -m pytest -q `
  04_experiments/p5_1_v3/test_p5_1_protocol.py `
  04_experiments/locomo_workload/test_trace_manifest.py `
  04_experiments/locomo_workload/test_online_retrieval.py `
  04_experiments/locomo_workload/test_graph_event_v2.py

python 04_experiments/retrieval/audit_system_design_fusion.py
python 04_experiments/audit_publication_readiness.py
~~~

These checks do not start databases or call paid model APIs.

## Retrieval effectiveness

The current retrieval table is built from canonical Cat1-Cat4 gold-memory mappings
and frozen embeddings:

~~~powershell
python 04_experiments/retrieval/build_locomo_gold_memory_v2.py
python 04_experiments/retrieval/run_dense_global_kg_rerun.py
python 04_experiments/retrieval/build_retrieval_main_table.py
~~~

Canonical outputs are under `05_reports/retrieval_main_table/`. Do not overwrite a
frozen result with a protocol change; create a versioned report directory and update
the artifact manifest.

## Backend semantic preservation

Start Cassandra and Neo4j, then prepare the isolated Bridge V2 views:

~~~powershell
python 04_experiments/retrieval/backend_bridge_v2_prepare.py --backend both --reset
python 04_experiments/retrieval/backend_bridge_v2_run.py --backend csv
python 04_experiments/retrieval/backend_bridge_v2_run.py --backend cassandra
python 04_experiments/retrieval/backend_bridge_v2_run.py --backend neo4j
python 04_experiments/retrieval/backend_bridge_v2_aggregate.py
~~~

`--reset` clears only the isolated Bridge V2 tables and labels created by this
experiment. The aggregate compares candidate and projection digests, ordered Top-10
results, and retrieval metrics against the CSV reference.

## Four-cell serving workload

The graph-aware workload compares Cassandra base/materialized paths and Neo4j
native/materialized paths under the same logical event stream:

~~~powershell
python 04_experiments/locomo_workload/run_graph_100k_load_gate_v2.py
python 04_experiments/locomo_workload/run_graph_system_matrix_v2.py --experiment all
python 04_experiments/locomo_workload/build_graph_storage_work_table_v2.py
python 04_experiments/locomo_workload/build_graph_system_publication_tables_v2.py
~~~

This matrix is a fixed 100K, single-machine experiment. Concurrency scaling in this
workload must not be described as distributed scale-out.

## Online update stages

Run one cell in smoke mode before launching the complete rate matrix:

~~~powershell
python 04_experiments/locomo_workload/run_experiment11_freshness_v2.py `
  --cell cassandra-materialized --smoke
~~~

The full protocol runs each of the four cells with concurrency 32, update rates
1/2/5/10 per second, and three repetitions. The script records raw commit,
structured-view checks, sparse and dense index checks, completed ranking, and whether
the target memory appears in the observed Top-10.

## Evidence policy

- `00_project/EXPERIMENT_REGISTRY.csv` is the status authority.
- `00_project/CLAIMS_AND_EVIDENCE.md` defines citation-ready claims and guardrails.
- `00_project/ARTIFACT_MANIFEST.csv` records canonical hashes.
- `05_reports/` contains formal reports; `06_analysis/` is exploratory.
- `09_archive/` and unnumbered legacy namespaces are provenance only.
- Never replace a frozen artifact in place. Write a new version and document the
  mapping from the superseded artifact.
