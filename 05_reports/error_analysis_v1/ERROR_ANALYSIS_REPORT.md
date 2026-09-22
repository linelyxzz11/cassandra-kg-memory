# CassMem Error Analysis V1

## Definition

The primary funnel is mutually exclusive. Type 3 is an audit flag rather than an automatically asserted cause.

| CassMem outcome | Count |
|---|---:|
| Correct_Cat5_abstention | 383 |
| Correct_or_partial_answer | 1153 |
| Type1_Retrieval_miss | 52 |
| Type2_Ranking_error | 138 |
| Type4_Reader_failure | 193 |
| Type5_Cat5_leakage | 63 |

## Interpretation

- Type 1: no gold memory enters the method's Top-100 or fusion input pool.
- Type 2: a gold memory is in the diagnostic pool but not Top-10.
- Type 3: missing ERK coverage or Raw-hit/RawERK-miss; requires manual confirmation.
- Type 4: gold memory is in Top-10 but LoCoMo answer score is zero.
- Type 5: Cat5 should abstain but the normalized prediction does not.

The balanced `manual_review_sample_100.csv` contains 20 candidates per requested type.

## 100-case review status

The evidence-based first pass is complete for all 100 sampled cases. It uses
frozen Dense/BM25 diagnostic ranks, gold-memory text, the official Cat5
protocol, and the already-frozen GPT-4o Judge labels; no external API was
called. This is an internal evidence-based audit rather than an independent human
annotation study.

- 91 cases received an evidence-based first-pass resolution.
- 9 cases remain explicitly marked `needs_secondary_review`.
- Among the 20 Type-4 candidates, 12 are confirmed Reader failures, 5 are
  token-F1 false negatives, and 3 have insufficient gold/evidence support.

See `manual_review_sample_100_reviewed.csv`, `manual_review_summary.csv`, and
`MANUAL_REVIEW_REPORT.md` for the auditable decisions.
