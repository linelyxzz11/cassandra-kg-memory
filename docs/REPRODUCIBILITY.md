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
python -m pip install -e .
Copy-Item .env.example .env
~~~

Cassandra and Neo4j are required only for backend experiments. Credentials stay in
the ignored `.env` file. Do not place API keys in commands, manifests, or reports.

## Fast validation

~~~powershell
python -m pytest -q `
  experiments/serving_100k/test_trace_manifest.py `
  experiments/serving_100k/test_online_retrieval.py `
  experiments/serving_100k/test_graph_event_v2.py

python experiments/retrieval/audit_system_design_fusion.py
~~~

These checks do not start databases or call paid model APIs.

## Retrieval effectiveness

The current retrieval table is built from canonical Cat1-Cat4 gold-memory mappings
and frozen embeddings:

~~~powershell
python experiments/retrieval/build_locomo_gold_memory_v2.py
python experiments/retrieval/run_dense_global_kg_rerun.py
python experiments/retrieval/build_retrieval_main_table.py
~~~

Canonical outputs are under `results/retrieval/retrieval_main_table/`. Do not overwrite a
frozen result with a protocol change; create a versioned report directory and update
the artifact manifest.

## Backend semantic preservation

Start Cassandra and Neo4j, then prepare the isolated Bridge V2 views:

~~~powershell
python experiments/backend_equivalence/backend_bridge_v2_prepare.py --backend both --reset
python experiments/backend_equivalence/backend_bridge_v2_run.py --backend csv
python experiments/backend_equivalence/backend_bridge_v2_run.py --backend cassandra
python experiments/backend_equivalence/backend_bridge_v2_run.py --backend neo4j
python experiments/backend_equivalence/backend_bridge_v2_aggregate.py
~~~

`--reset` clears only the isolated Bridge V2 tables and labels created by this
experiment. The aggregate compares candidate and projection digests, ordered Top-10
results, and retrieval metrics against the CSV reference.

## Four-cell serving workload

The graph-aware workload compares Cassandra base/materialized paths and Neo4j
native/materialized paths under the same logical event stream:

~~~powershell
python experiments/serving_100k/run_graph_100k_load_gate_v2.py
python experiments/serving_100k/run_graph_system_matrix_v2.py --experiment all
python experiments/serving_100k/build_graph_storage_work_table_v2.py
python experiments/serving_100k/build_graph_system_publication_tables_v2.py
~~~

This matrix is a fixed 100K, single-machine experiment. Concurrency scaling in this
workload must not be described as distributed scale-out.

## Online update stages

Run one cell in smoke mode before launching the complete rate matrix:

~~~powershell
python experiments/freshness/run_experiment11_freshness_v2.py `
  --cell cassandra-materialized --smoke
~~~

The full protocol runs each of the four cells with concurrency 32, update rates
1/2/5/10 per second, and three repetitions. The script records raw commit,
structured-view checks, sparse and dense index checks, completed ranking, and whether
the target memory appears in the observed Top-10.

## Evidence policy

- `docs/EXPERIMENT_REGISTRY.csv` is the status authority.
- `docs/CLAIMS_AND_EVIDENCE.md` defines citation-ready claims and guardrails.
- `results/ARTIFACT_MANIFEST.csv` records canonical hashes.
- `results/` contains the canonical paper-facing outputs.
- Historical prototypes and superseded result trees are not part of this artifact.
- Never replace a frozen artifact in place. Write a new version and document the
  mapping from the superseded artifact.
