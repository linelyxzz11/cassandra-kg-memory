# Canonical Layout and Artifact Policy

## Namespace Policy

- New raw/canonical data: `01_data/`
- New reusable intermediate artifacts: `02_artifacts/`
- New source code: `03_src/`
- New experiment entrypoints: `04_experiments/`
- New formal results: `05_reports/<experiment_name>/`
- Superseded or failed material: `09_archive/`

The unnumbered `results/`, `reports/`, `scripts/`, `artifacts/`, and
`generated_data/` directories are legacy namespaces. They may be read for
provenance, but new outputs must not be written there.

## Citation-Ready Minimum

Every formal experiment directory should contain:

1. machine-readable summary CSV/JSON;
2. per-query or per-event evidence where feasible;
3. run/protocol manifest with input hashes;
4. sample counts, exclusions, and split definition;
5. metric definitions and units;
6. source/backend/model versions;
7. seed or deterministic-selection policy;
8. status note if the run failed or contains sentinel values.

## Frozen Artifact Rule

Do not overwrite frozen data or official results. A correction creates a new
versioned artifact and a mapping/audit that explains the difference. Downstream
tables must reference exact hashes.

## Comparability Rule

- Retrieval uses LoCoMo Cat1–Cat4 only.
- Reader answer quality may use Cat1–Cat5.
- Cat5 is adversarial abstention and has no positive gold-memory target in the
  current retrieval protocol.
- Locally rerun and externally quoted results must be labeled separately.
- Legacy Dense+GlobalKG reader results must not be relabeled as the corrected
  λ=0.2 retrieval rerun.

## Runtime and Secrets

Virtual environments, model caches, SQLite state, logs, `.env`, and credentials
are local runtime state and must remain untracked.
