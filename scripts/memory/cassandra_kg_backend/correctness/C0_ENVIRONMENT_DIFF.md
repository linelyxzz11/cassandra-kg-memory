# C0 修订说明：与现有 Cassandra-KG 环境对齐

本版替代上一版 C0 harness 中与当前项目不相容的假设。

| 项目 | 上一版假设 | 本版实现 |
|---|---|---|
| Cassandra keyspace | `kg_memory` | `ai_memory` |
| 认证 | 示例带用户名/密码 | 默认 `127.0.0.1:9042`、无认证 |
| 表结构 | 必须有 `visible_from_version` | **不需要任何 ALTER TABLE** |
| `edge_id` | 跨后端 string ID | Cassandra `timeuuid` 保持原样，不参与 C0 结果比较 |
| 逻辑边标识 | `edge_id` | SHA-1(`graph_id,src_id,relation,dst_id,source`) |
| 物理表 | 默认强制 `by_src_relation` | 默认适配三表；relation index 可选启用 |
| 读写混合隔离 | 版本字段 | 写入独立 graph_id / 不访问读 trace 分区 |

## 3 张表还是 4 张表？

你提供的 briefing 同时出现了两种状态：文字说明称“现有 4-table schema”，而你的最新说明是“目前 Cassandra 只有三张表”。因此本版把四表索引设为**可选**：

- 三表模式（默认）：`by_src`、`by_dst`、`by_relation_bucket`；`cassandra_opt` = frontier parallel + high-degree cache，relation 仍在应用层过滤。
- 四表模式：确认 `kg_edges_by_src_relation` 已存在且已经 backfill 后，再把 `tables.by_src_relation` 填入表名，并把 `relation_index.enabled` 改为 `true`。

没有第四张表时，论文中不能把 C0 的 Cassandra-opt 叫作 “parallel + cache + index”；应标记为 **parallel + cache**。index 作为独立 Layer B 或后续四表 C0 扩展报告。
