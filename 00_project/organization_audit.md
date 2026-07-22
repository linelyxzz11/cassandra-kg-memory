# Organization Audit

Generated: 2026-07-22 21:31:51

## Summary
- Files moved: 168
- Duplicates: 1
- Errors: 0
- Unresolved: 44

## Directory Tree
```
00_project/  — registry, manifest, claims
01_data/     — LoCoMo raw data, 1M graph, embedding artifacts
02_artifacts/ — frozen events, SHA manifests, prompt caches
03_src/      — reusable Python modules
04_experiments/ — run scripts, configs
05_reports/  — per-event, per-run, summary, audit outputs
06_analysis/ — (empty, plotting scripts remain in original location)
07_runtime/  — (pycache, SQLite, worker logs)
08_literature/ — (empty)
09_archive/  — legacy 100K, failed runs, superseded results
```

## Remaining Tasks
1. Update experiment script config paths to new locations
2. Move analysis/plotting scripts to 06_analysis/
3. Add literature notes to 08_literature/
