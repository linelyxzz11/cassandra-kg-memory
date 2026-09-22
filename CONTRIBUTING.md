# Contributing

CassMem is organized as a research artifact. Changes should preserve the connection
between claims, protocols, code, and frozen evidence.

## Repository conventions

- Put reusable implementation in `03_src/` and publication-facing entry points in
  `04_experiments/`.
- Write new formal outputs to a versioned directory under `05_reports/`.
- Use `06_analysis/` for exploratory work and `09_archive/` for superseded runs.
- Do not add new files to the unnumbered legacy `scripts/`, `reports/`, or `results/`
  namespaces.
- Never overwrite frozen inputs or citation-ready results.
- Keep credentials, local databases, caches, raw traces, and virtual environments out
  of Git.

## Before submitting a change

1. Run the protocol tests listed in `docs/REPRODUCIBILITY.md`.
2. Confirm that new outputs include sample counts, metric definitions, input hashes,
   protocol parameters, and status.
3. Update `00_project/EXPERIMENT_REGISTRY.csv` and
   `00_project/ARTIFACT_MANIFEST.csv` when an experiment or canonical artifact
   changes.
4. State the supported claim and its limitations in
   `00_project/CLAIMS_AND_EVIDENCE.md`.
5. Check `git status --ignored` before committing to ensure no secret or runtime
   artifact is staged.

Changes to a retrieval formula must match the implementation that produced the
reported result. A documentation-only rule must not be presented as an implemented
protocol.
