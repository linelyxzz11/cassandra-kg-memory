# CassMem Publication-Readiness Audit

Generated: 2026-08-10T02:42:41.965718+00:00

## Verdict

Retrieval gold, six Top-10 ranking artifacts, retrieval main metrics, reader F1/BLEU provenance, and paired significance outputs pass the automated integrity checks. The repository is not yet submission-ready because a real pruned KG candidate stage and the remaining CassMem end-to-end serving evidence remain incomplete. The existing backend_system_eval visibility and recovery CSVs are explicitly not citation-ready.

## Check Summary

- PASS: 32
- WARN: 3
- BLOCKED: 3
- FAIL: 0

## Completed in This Audit

- Removed active hard-coded API keys and database passwords; added environment-variable configuration.
- Repaired `.gitignore`, added `.env.example`, and documented dependency ranges.
- Repaired P5-3C config paths and aligned the 18K frozen-event manifest with the actual artifact.
- Added Holm-corrected paired bootstrap significance for retrieval and reader main comparisons.
- Recomputed BLEU-1 from frozen predictions with an executable protocol and corrected the Full5 label to a B1/Cat5-accuracy hybrid.
- Added machine-readable ranking, path, check, and experiment-gap audits.
- Validated P5-1 v3.1: 36K fixed-candidate update-to-final-TopK events with exact cross-backend parity.
- Regenerated corrected Dense+GlobalKG Reader outputs for both 1,986-question settings and rescored F1, B1, and GPT-4o J.

## Interpretation Rules

- Cat5 is answer-level adversarial abstention and is excluded from retrieval effectiveness.
- `Candidate Recall=1.0` for the full conversation pool is only an oracle sanity check, not evidence for graph expansion.
- Dense+GlobalKG retrieval and Reader rows now share the frozen degree-centrality lambda=0.2 ranking provenance.
- Quoted external memory-system results must be marked as published values, not locally rerun results.

See `audit_checks.csv`, `ranking_integrity.csv`, `experiment_gap_matrix.csv`, and `hardcoded_path_audit.csv` for machine-readable details.
