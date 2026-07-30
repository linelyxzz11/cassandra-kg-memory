# locomo_pipeline/reader_eval/

## Purpose
LLM-based reader evaluation: pass top-K memory to LLM, measure answer F1/EM, and aggregate by category and method.

## Status
PAPER_EVIDENCE — Produces the paper's Reader@10 results.

## Important Scripts
- `reader_f1_memory_only_v2.py` — Reader F1 evaluation v2. Supports pilot (30/50/200 QA) and full (1986 QA) runs. Produces per-QA predictions CSV + by-category summary.
- `locomo_llm_reader_full_v3.py` — Full 1986-QA LLM reader v3. Produces `llm_reader_full_v3_results.csv` (352MB).
- `final_results_summary.py` — Final results aggregation across methods.

## Safety
- All scripts call LLM APIs. Slow and potentially costly.
- Do not run full evaluation without explicit confirmation.

## Related Reports
- `results/final/reader_f1_memory_only_full_scoped_bm25_predictions.csv`
- `results/final/reader_f1_memory_only_full_scoped_bm25_by_category.csv`
- `results/llm_reader_full_v3_results.csv` (352MB raw output)
- `results/llm_reader_full_v3_summary.csv`
