# LoCoMo 离线统一评估运行说明

脚本不会调用任何 LLM 或外部 API。

## 1. ZScore-RawERK：P4 Full-5 1986 条

```powershell
python locomo_official_eval_v1.py `
  --questions "05_reports\evaluator_input_audit\selected_full5_questions.csv" `
  --predictions "05_reports\p4_full5\reader_cache_all.jsonl" `
  --default-method "ZScore-RawERK-P4A" `
  --output-dir "05_reports\official_eval\zscore_rawerk_p4a"
```

## 2. ZScore-RawERK：Canonical 1540 原始预测版本

```powershell
python locomo_official_eval_v1.py `
  --questions "05_reports\evaluator_input_audit\selected_full5_questions.csv" `
  --predictions "05_reports\p4_full5\canonical_reader_predictions.jsonl" `
  --default-method "ZScore-RawERK-P4C" `
  --output-dir "05_reports\official_eval\zscore_rawerk_p4c"
```

## 3. BM25 / Dense / GlobalKG / QueryKG：四方法 Full-5

该 CSV 已有 `method` 字段，因此不需要 `--default-method`。

```powershell
python locomo_official_eval_v1.py `
  --questions "05_reports\evaluator_input_audit\selected_full5_questions.csv" `
  --predictions "results\final\reader_f1_memory_only_full_scoped_bm25_predictions.csv" `
  --output-dir "05_reports\official_eval\legacy_four_methods"
```

## 4. P2 fusion（仅 held-out 1150，审计用途）

```powershell
python locomo_official_eval_v1.py `
  --questions "05_reports\evaluator_input_audit\selected_full5_questions.csv" `
  --predictions "reports\p2_fusion_reader_final\reader_cache.jsonl" `
  --allow-partial `
  --output-dir "05_reports\official_eval\p2_fusion_heldout1150"
```

1150 结果不能放入 canonical 1540 主表。

## 每次输出

- `per_query_scores.csv`
- `scores_by_category.csv`
- `scores_overall.csv`
- `evaluation_audit.json`

## 正式列名

- `rF1_cat14`：Cat1–4 的官方 LoCoMo F1 微平均
- `official_score`：Cat1–4 时等于 rF1；Cat5 时为官方拒答 0/1；Full-5 时为混合微平均
- `cat5_official_abstention`：官方 Cat5 正确拒答率
- `cat5_leakage_official`：1 - 官方正确拒答率
- `rEM_cat14_project`：项目诊断指标，不是 LoCoMo 官方主指标
- `WrongAbstention_cat14_project`：项目诊断指标
