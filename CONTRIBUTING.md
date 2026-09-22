# Contributing

CassMem is organized as a research artifact. Changes should preserve the connection
between claims, protocols, code, and frozen evidence.

## Repository conventions

- Put reusable implementation in `src/cassmem/` and publication-facing entry points
  in `experiments/`.
- Write formal outputs to a versioned, topic-specific directory under `results/`.
- Keep exploratory notebooks, superseded runs, and raw traces outside the public
  artifact.
- Never overwrite frozen inputs or citation-ready results.
- Keep credentials, local databases, caches, raw traces, and virtual environments out
  of Git.

## Before submitting a change

1. Run the protocol tests listed in `docs/REPRODUCIBILITY.md`.
2. Confirm that new outputs include sample counts, metric definitions, input hashes,
   protocol parameters, and status.
3. Update `docs/EXPERIMENT_REGISTRY.csv` and
   `results/ARTIFACT_MANIFEST.csv` when an experiment or canonical artifact
   changes.
4. State the supported claim and its limitations in
   `docs/CLAIMS_AND_EVIDENCE.md`.
5. Check `git status --ignored` before committing to ensure no secret or runtime
   artifact is staged.

Changes to a retrieval formula must match the implementation that produced the
reported result. A documentation-only rule must not be presented as an implemented
protocol.
