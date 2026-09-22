# CassMem: Persistent Structured Memory for Agentic Services

CassMem is a research prototype for the memory layer of persistent, continuously
updated agentic services. It connects a structured logical memory representation
to scope-local retrieval, Cassandra-native physical access paths, and an online
update pipeline.

The project is not an offline retrieval wrapper around a database. It studies how
long-term memories can remain retrievable while user- or conversation-scoped state
continues to grow, concurrent reads and updates share serving resources, and derived
views must stay aligned with the retrieval semantics used by the agent.

## System overview

~~~text
cross-session interactions
        |
        v
structured memory: raw text + ERK fields + local relations + version
        |
        +---- dense retrieval over raw text
        +---- BM25 retrieval over the RawERK view
        |
        v
query-wise Z-score fusion within a known scope
        |
        v
Cassandra-native serving state
  - scope-local memory access
  - memory-local mentions and relations
  - relation-conditioned candidates
  - read-time reconstruction or update-time materialization
        |
        v
update-to-retrieval path
arrival -> backend commit -> structured view -> indexes -> observed Top-k
~~~

### Logical memory and retrieval

Each memory keeps a stable identity, scope, version, raw text, extracted entity,
relation, and keyword fields, local relation facts, and an embedding-content
fingerprint. RawERK is a derived lexical view that appends the structured fields to
the original text without replacing the source memory.

Retrieval is restricted to the known user or conversation scope. A dense channel
matches the raw text, a BM25 channel matches RawERK, and query-wise Z-score
normalization combines their relative evidence before stable Top-10 selection.

### Cassandra-native serving

The Cassandra layout follows the access patterns of the memory workload rather than
treating Cassandra as a general graph engine. It separates scope-local memory rows,
per-memory features, mentions, and relation records, and relation-conditioned
candidate access. The base path reconstructs RawERK and filters relations at read
time; the materialized path maintains RawERK and scope-relation candidates during
updates.

Both paths expose the same candidate identities and retrieval projections to the
application-level scorer. Backend equivalence is evaluated against a canonical CSV
reference before latency or throughput differences are interpreted.

### Online updates

A committed row is not yet retrieval-ready. The online pipeline distinguishes raw
backend commit, structured-view visibility, sparse and dense index visibility, and
the result of a post-update ranking. The current measurements report update-stage
latency and observed Top-10 hits; a non-hit is not automatically classified as a
backend freshness failure.

## Evidence map

| Question | Canonical evidence |
|---|---|
| Retrieval effectiveness | [`results/retrieval/retrieval_main_table/`](results/retrieval/retrieval_main_table/) |
| Reader answer quality | [`results/reader/reader_main_hingemem_style/`](results/reader/reader_main_hingemem_style/) |
| Backend semantic preservation | [`results/backend_equivalence/backend_equivalence_v2/`](results/backend_equivalence/backend_equivalence_v2/) |
| Four-cell 100K serving workload | [`results/serving_100k/locomo_workload_graph_v2_100k/`](results/serving_100k/locomo_workload_graph_v2_100k/) |
| Online update stages and observed Top-10 visibility | [`results/freshness/experiment11_online_freshness_v2/`](results/freshness/experiment11_online_freshness_v2/) |
| Supported claims and limitations | [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md) |
| Experiment status | [`docs/EXPERIMENT_REGISTRY.csv`](docs/EXPERIMENT_REGISTRY.csv) |
| Artifact hashes | [`results/ARTIFACT_MANIFEST.csv`](results/ARTIFACT_MANIFEST.csv) |

The Chinese System Design draft is maintained in
[`docs/SYSTEM_DESIGN_ZH.md`](docs/SYSTEM_DESIGN_ZH.md). Formula-to-code checks are
recorded separately in
[`docs/SYSTEM_DESIGN_IMPLEMENTATION_AUDIT.md`](docs/SYSTEM_DESIGN_IMPLEMENTATION_AUDIT.md).

## Repository layout

| Path | Purpose |
|---|---|
| `src/cassmem/` | Shared representation, retrieval, backend, serving, and evaluation code |
| `experiments/` | Reproducible entry points grouped by paper evaluation question |
| `data/` | LoCoMo inputs, retrieval gold, and frozen retrieval artifacts |
| `results/` | Canonical tables, reports, summaries, and manifests used by the paper |
| `docs/` | System design, claims, experiment registry, and reproduction guide |

## Quick start

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
Copy-Item .env.example .env
~~~

Populate only the local services and API credentials required by the experiment you
intend to run. The `.env` file is ignored and must never be committed.

Run the service-independent protocol tests:

~~~powershell
python -m pytest -q `
  experiments/serving_100k/test_trace_manifest.py `
  experiments/serving_100k/test_online_retrieval.py `
  experiments/serving_100k/test_graph_event_v2.py
~~~

Detailed prerequisites and experiment commands are documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). For VS Code users, open
[`CassMem.code-workspace`](CassMem.code-workspace) and see
[`docs/VSCODE_WORKSPACE.md`](docs/VSCODE_WORKSPACE.md).

## Scope and limitations

- Formal serving results use a single machine with one Cassandra instance and one
  Neo4j instance. They do not establish distributed scale-out superiority.
- Multi-scope workloads model independent user or conversation namespaces; the
  project does not claim tenant authentication or resource isolation.
- Recovery experiments stop and replay the application update worker while both
  databases remain online. They do not test database restart, cluster failover, or
  disaster recovery.
- Online Top-10 measurements use a post-update query. They do not estimate a
  continuously monitored first-hit time or Top-k convergence latency.
- Historical prototypes and superseded runs are intentionally excluded from this
  public artifact; the evidence map above is the citation authority.

## Contributing and security

Repository conventions are described in [`CONTRIBUTING.md`](CONTRIBUTING.md).
Report exposed credentials or other sensitive findings according to
[`SECURITY.md`](SECURITY.md).
