# ZScore-RawERK Version Audit

## Version A: P4 reader_cache_all.jsonl (rF1 = 0.3747)
- Path: 05_reports/p4_full5/reader_cache_all.jsonl
- Rows: 1986, Cat1-4 matched: 1540
- Recomputed rF1: 0.3747 (matches cat1_4_overall.csv)
- Has per-query prediction: yes
- No method field — assumed ZScore_RawERK from P4 full5
- SHA256: 69e95c8f473c93b609b9c989f67065b52a4c3f3fe173cf10d3e17961e138670f

## Version B: canonical_cat1_4_reader_overall.csv (rF1 = 0.3579)
- Path: 05_reports/p4_full5/canonical_cat1_4_reader_overall.csv
- Reported rF1: 0.3579, rEM: 0.2013, WrongAbst: 0.4513
- No per-query artifact found for this version
- SHA256: 0280fcba5f367ee50131f4352271b66797fa94d82d3029fab5cfb0d159794146

## Version C: canonical_reader_predictions.jsonl
- Path: 05_reports/p4_full5/canonical_reader_predictions.jsonl
- Rows: 1540
- No rF1 computed (raw predictions only)
- SHA256: 867c4cd387907aeba9eafb8ba509d25c8e8b9b1566e6300206e0bc89bd6876cd

## Discrepancy
Two rF1 values for ZScore-RawERK on cat1-4:
- P4 cache reagg: 0.3747
- Canonical overview: 0.3579
- Difference = 0.0168 — possible evaluator version difference
- Both versions preserved; no selection made.
