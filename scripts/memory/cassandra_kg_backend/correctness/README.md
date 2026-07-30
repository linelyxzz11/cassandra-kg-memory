# C0 Unified Benchmark Harness（现有 Cassandra schema 兼容版）

本目录为 Layer C 的 **C0 一致性 gate**。它在性能对比前固定：

- 一份 canonical triples；
- 一份固定 workload manifest；
- 一套跨 Cassandra / Neo4j 的 traversal 语义；
- 同一个 per-frontier-node relation filter、cycle policy、排序和 fanout。

它专门适配当前环境：`127.0.0.1:9042`、keyspace `ai_memory`、无认证、Cassandra `edge_id timeuuid`、没有 `visible_from_version`。

## 最重要的设计改变

Cassandra 的 `edge_id` 是插入时 `now()` 生成的 `timeuuid`，LoCoMo CSV 又没有该字段。它不能用作跨后端等价比较的键。本 harness 因此比较逻辑边：

```text
logical_edge_id = SHA-1(graph_id, src_id, relation, dst_id, source)
```

每一跳的确定性筛选规则是：

1. relation path 过滤；
2. `cycle_policy=path` 时排除当前 path 已访问节点；
3. 按 `(relation, dst_id, source)` 排序；
4. 每个 frontier node 只取前 `fanout` 条；
5. 以逻辑边路径比较三后端结果。

精确相同的 `(graph_id, src_id, relation, dst_id, source)` 记录会在 canonical 化时去重；C0 不把仅 `timeuuid` 不同的重复物理写入视为不同语义边。

## 三表默认；第四张 index 表可选

默认 `config.local.example.json` 只启用：

```text
kg_edges_by_src
kg_edges_by_dst
kg_edges_by_relation_bucket
```

此时：

- `cassandra_naive`: 单 worker、无 cache、查 `by_src` 后 Python relation filter；
- `cassandra_opt`: frontier parallel + high-degree cache，仍查 `by_src`。

若确认存在且已回填 `kg_edges_by_src_relation`，可开启：

```json
"tables": {"by_src_relation": "kg_edges_by_src_relation"},
"relation_index": {"enabled": true}
```

此时 Cassandra-opt 在 relation-constrained hop 使用 `by_src_relation`。不要在未建表时开启它。

## 0. 建立配置

```bash
copy config.local.example.json config.json
python -m c0bench.cli profile --config config.json
```

预期 profile 应显示：

```text
keyspace = ai_memory
profile = 3-table compatible (relation index disabled)
requires_visible_from_version = false
timeuuid_migration_required = false
```

## 1. LoCoMo CSV 生成 canonical triples

你的真实输入是 `results/locomo_kg_edges_spacy.csv`，包含：

```text
graph_id,src_id,relation,dst_id,source
```

运行：

```bash
python -m c0bench.cli canonicalize \
  --input results/locomo_kg_edges_spacy.csv \
  --output results/c0_canonical_edges.csv
```

先用现有导入脚本把**同一份数据**导入 Cassandra 与 Neo4j；关键是保留 `source` 字段。Cassandra 导入仍可继续使用 `edge_id = now()`，无需重建或迁移表。

## 2. 固定 workload manifest

不要让每个系统独立随机抽 query：

```bash
python -m c0bench.cli generate-manifest \
  --plan examples/query_plan.json \
  --output results/c0_workload_manifest.jsonl \
  --count 1000 \
  --seed 20260705
```

后续 C1–C4 都复用该文件。真实 LoCoMo query 分布应由你的 retrieval query/seed 统计替换 `examples/query_plan.json`，而不是持续使用示例 seed。

## 3. 静态等价校验

Neo4j 尚未启动前，可以先验证 Cassandra 双模式：

```bash
python -m c0bench.cli validate \
  --config config.json \
  --canonical results/c0_canonical_edges.csv \
  --manifest results/c0_workload_manifest.jsonl \
  --systems cassandra_naive cassandra_opt \
  --report-dir reports/c0_cassandra_gate
```

Neo4j 本地实例已启动、完成同 triples 导入后，再执行：

```bash
python -m c0bench.cli validate \
  --config config.json \
  --canonical results/c0_canonical_edges.csv \
  --manifest results/c0_workload_manifest.jsonl \
  --systems cassandra_naive cassandra_opt neo4j \
  --report-dir reports/c0_all_backends_gate
```

通过标准：所有 `disagreements=0` 且 `errors=0`。失败细节在 `mismatches.jsonl`。

## 4. C2 写操作如何保持读结果不变

当前 Cassandra schema 没有版本列，因此不要执行“同一 graph、同一查询 partition 的即时可见写入”。

C2 应在单独的写入 graph_id（如 `locomo_kg__write_sink`）执行 writes，或者保证写入分区不被读 trace 访问。这样读结果保持固定；同时 Cassandra 写入会真实维护已有的三张物理表（如启用 index，则是四张）。

写入使用现有字段：`edge_id=now()`、`source`、`created_at=toTimestamp(now())`。不需要 `visible_from_version`。

## 5. Neo4j 对齐

运行 `schema/neo4j_schema.cypher` 后，Neo4j 使用：

```text
(:KGNode {graph_id, node_id})-[:KG_EDGE {graph_id, relation, source}]->(:KGNode)
```

Neo4j query 同样按 `(relation, dst_id, source)` 做 per-node `ORDER BY`/`LIMIT`；返回后会用相同的 logical edge ID 比较。

## 本地语义测试

```bash
python -m unittest discover -s tests -v
```
