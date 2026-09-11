# Error Analysis 100-Case Review

This is a Codex-assisted evidence-based first pass, not an independent human annotation study.

## Completion

- Reviewed rows: 100/100
- Evidence-based resolved: 91
- Secondary expert review retained: 9
- External API calls: 0

## Type 4 correction

- Metric_false_negative_not_reader_failure: 5
- Type4_reader_failure_confirmed: 12
- evidence_or_annotation_insufficient: 3

## Interpretation safeguards

- Type 1 branch attribution is based on gold ranks in Dense@100 and BM25@100.
- Type 2 candidate-depth is not blamed when gold is already present in the frozen fusion input pool.
- Type 3 empty ERK is a coverage flag; it becomes causal only when stronger counterfactual evidence exists.
- Type 4 Judge-correct cases are metric false negatives, not Reader failures.
- Type 5 follows the official Cat5 adversarial/unanswerable protocol.
