# CassMem VS Code workspace

Open `CassMem.code-workspace` instead of opening an individual experiment
folder. The workspace treats the repository root as one Python project, loads
the ignored `.env`, and adds `03_src` and `04_experiments` to Python analysis.

## Experiment map

- `00_project/`: claim, evidence, experiment, and artifact registries.
- `01_data/`: canonical LoCoMo inputs and frozen embeddings.
- `02_artifacts/`: frozen derived artifacts and retrieval gold.
- `03_src/`: reusable retrieval, storage, and evaluation implementation.
- `04_experiments/retrieval/`: retrieval effectiveness and backend bridge.
- `04_experiments/reader/`: Reader, Judge, question-type, and error analysis.
- `04_experiments/locomo_workload/`: four-cell serving, freshness, and recovery.
- `05_reports/`: citation-facing reports, tables, manifests, and compact results.
- `08_literature/`: evaluation-protocol literature evidence.
- `09_archive/`: legacy artifacts; do not use as current paper evidence.

## Built-in tasks

Open **Terminal → Run Task** and select one of:

- `CassMem: protocol tests`: run the 13 lightweight protocol tests.
- `CassMem: check Cassandra connection`: verify the local Cassandra backend.
- `CassMem: check Neo4j connection`: verify the local Neo4j backend.
- `CassMem: rebuild publication audit`: update the paper-readiness matrix.
- `CassMem: refresh artifact manifest`: refresh canonical SHA-256 inventory.
- `CassMem: Git status`: show the current branch and pending changes.

Select the intended Python interpreter once after opening the workspace. The
large local model, cache, raw-run, trace, and per-event directories remain on
disk but are excluded from file watching and full-text search to keep VS Code
responsive.

## Evidence entry points

- `README.md`
- `00_project/CLAIMS_AND_EVIDENCE.md`
- `00_project/EXPERIMENT_REGISTRY.csv`
- `00_project/ARTIFACT_MANIFEST.csv`
- `docs/SYSTEM_DESIGN_ZH.md`
- `docs/REPRODUCIBILITY.md`

Credentials stay outside Git. The local credential note at
`D:\memorytable\LOCAL_SERVICE_CREDENTIALS.md` is intentionally not part of
this workspace.
