# Fusion and Reader Rerun Time Estimate

## Measured retrieval time

The complete eleven-point alpha sweep rebuilds canonical RawERK BM25 Top-50,
uses frozen Dense Top-50 scores, ranks 390 development and 1,146 held-out
queries, and computes six metrics. The current measured wall time is recorded
in `manifest.json` and is under 15 seconds on this workstation. It makes no API
calls and does not recompute embeddings.

## Reader inference estimate

The closest measured reader run is `results/reader/p3_reader_alignment/api_run_summary.json`:
1,150 API calls, four workers, 99 retries, and 559.8 seconds wall time. This is
about 9.3 minutes per ranking method for the historical held-out reader setup.

| Requested rerun | API calls | Evidence-based wall-time estimate |
|---|---:|---:|
| Retrieval metrics for all 11 weights | 0 | under 1 minute; already complete |
| Historical held-out Reader F1 for one new method | 1,150 | about 10-15 minutes |
| Historical held-out Reader F1 for all 11 weights | 12,650 | about 1.7 hours ideal; budget 2-3 hours with retries/rate limits |
| Full 1,986-question reader, one setting, one new method | 1,986 | about 16 minutes ideal; budget 20-40 minutes |
| Full 1,986-question reader, two settings, one new method | 3,972 | about 32 minutes ideal; budget 45-75 minutes |
| Full two-setting reader for all 11 weights | 43,692 | about 5.9 hours ideal; budget 7-10 hours |

LLM-as-a-Judge is a separate API stage. A full judge pass for one method adds
roughly another 1,986 requests. Because no reliable wall-time record is stored
for that provider, a conservative end-to-end budget is 1-2 hours for one new
adaptive method (two reader settings plus judge), and 9-15 hours for all eleven
weights. Monetary cost depends on the model and current API pricing and is not
estimated here.

## Recommended protocol

Do not run the reader for all eleven weights. Select alpha on development
retrieval metrics, freeze the ranking, and run the reader once for the selected
method. The existing alpha=0.6 reader result does not need to be repeated when
the generated Top-10 rankings are unchanged. A future adaptive fusion method
would require one new reader and judge run after its gate is frozen on dev.
