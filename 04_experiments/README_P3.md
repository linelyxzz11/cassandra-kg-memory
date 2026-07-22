# P3 Memory Representation Comparison

## 1. 安装

```bash
python -m pip install -r requirements.txt
```

## 2. 合并 raw / summary / ERK / triples

```bash
cp merge_config.example.json merge_config.json
# 修改路径
python merge_memory_features.py --config merge_config.json
```

输出应包含：

```text
memory_id, conversation_id, raw_text, summary,
entities, relations, keywords, triples
```

## 3. 没有 summary artifact 时才生成

```bash
cp summary_config.example.json summary_config.json
export LLM_BASE_URL="https://YOUR-ENDPOINT/v1"
export LLM_API_KEY="YOUR-KEY"
export LLM_MODEL="YOUR-MODEL"
python generate_memory_summaries.py --config summary_config.json
```

生成摘要后，再执行第 2 步合并。

## 4. 配置主实验

```bash
cp p3_config.example.json p3_config.json
```

修改 `files` 和列名。必须复用 P2 的：

- frozen Dense score artifact
- per-conversation memory scope
- Dense/BM25 topN
- Z-score missing-candidate 规则
- alpha_dense=0.6

## 5. 先做 Gate

```bash
python run_p3_memory_representation.py \
  --config p3_config.json \
  --validate-only
```

只有出现 `Validation PASS` 才能继续。Anchor：

```text
ZScore_RawERK MRR  = 0.5462
ZScore_RawERK R@10 = 0.8070
```

## 6. 正式运行

```bash
python run_p3_memory_representation.py --config p3_config.json
```

## 7. 主要输出

```text
reports/p3_memory_representation_comparison/
├── alignment_checks.json
├── alignment_gate.csv
├── representation_manifest.csv
├── dev_representation_results.csv
├── test_representation_results.csv
├── test_representation_by_category.csv
├── test_representation_per_query.csv
├── representation_attribution.csv
├── representation_query_bootstrap.csv
├── representation_cluster_bootstrap.csv
├── representation_per_conversation.csv
├── representation_leave_one_conversation_out.csv
├── representation_cost_comparison.csv
└── p3_memory_representation_summary.md
```

正文重点比较：

```text
ZScore_RawERK
ZScore_Raw
ZScore_RawSummary
ZScore_RawTriples
ZScore_ERKOnly
```

Reader 阶段只补四种：Raw、RawERK、dev 最强 summary variant、dev 最强 triples variant。
