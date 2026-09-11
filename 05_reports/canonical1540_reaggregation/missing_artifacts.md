# Missing Artifacts — Reader Predictions for Canonical 1540

## Summary

| Method | rF1 | rEM | WrongAbst | Can recover? | What's needed |
|---|---|---|---|---|---|
| ZScore-RawERK | ✅ 0.3747 | ✅ 0.2149 | ✅ 0.4292 | YES — P4 cache | — |
| BM25 | 0.2648 | 0.1539 | 0.5929 | PARTIAL | Need per-query preds to verify |
| Dense-bge | 0.3482 | 0.1831 | 0.4539 | PARTIAL | Need per-query preds to verify |
| Dense+GlobalKG | 0.3222 | 0.1701 | 0.4877 | PARTIAL | Need per-query preds to verify |
| Dense+QueryKG | 0.3516 | 0.1877 | 0.4494 | PARTIAL | Need per-query preds to verify |
| BM25_compact | 0.3027 | 0.1721 | 0.5286 | PARTIAL | Need per-query preds to verify |
| RRF_raw | 0.3542 | 0.1968 | 0.4552 | PARTIAL | Need per-query preds to verify |
| RRF_compact | 0.3666 | 0.2065 | 0.4344 | PARTIAL | Need per-query preds to verify |
| ZScore-Raw | NA | NA | NA | NO | 1540 reader predictions |
| ZScore-RawSummary | NA | NA | NA | NO | 1540 reader predictions |
| ZScore-RawTriples | NA | NA | NA | NO | 1540 reader predictions |
| ZScore-ERKOnly | NA | NA | NA | NO | 1540 reader predictions |

## Required API calls to fill gaps

To obtain 1540 canonical reader results for all methods, needs:
1. **7 legacy methods**: Re-run 7x reader eval with frozen prompts on canonical 1540 queries
2. **P3 representations**: Re-run 5x reader eval on canonical 1540 (not just heldout 1150)
3. **P1-C E/R/K/T variants**: Re-run reader for each variant with frozen prompt
Total: ~15 LLM API reader evaluations
