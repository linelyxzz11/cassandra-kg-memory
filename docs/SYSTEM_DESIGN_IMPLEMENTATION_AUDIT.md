# System Design 公式与冻结实现核对记录

日期：2026-09-20。已核对 Eq.1 / Eq.2、3.4 后端实现，以及按实际检查和查询时刻重写的 Eq.3。

## 写作约束

所有计算公式及其候选范围、参数、边界情况和排序规则，必须可追溯到产生冻结结果的源码、配置与输入产物。不得为了补全论文公式而新增实现中不存在的规则。发现多个执行协议不一致时，应分别说明，不能由文档改写隐含地统一实现；源码更新也不自动替代已冻结实验协议。本次未修改检索实现、配置或冻结实验结果。

公式解释依次说明为什么需要这个定义、每项代表什么、它服务于哪个系统问题。应区分四类内容：表示定义、计算规则、待验证的 invariant、观测指标；不能以形式化符号替代实现或实验依据。

Eq.1 是已有字段及派生过程的数学记法；Eq.2 是经复算验证的计算规则。原 compiler 映射公式已删除。旧版基于 publication frontier 和连续时间下确界的公式，也已由当前 Eq.3 的实际观测时延定义替换。后端语义保持以文字定义 invariant，注明相同版本、访问条件及检索协议的前提，并由后续实验核验投影、候选及有序 Top-10；不声称对任意状态和查询已获得证明。

## 3.5 就绪检查与观测时延核对

依据：`experiments/freshness/run_experiment11_freshness_v2.py` 与当前聚合结果。

- execute() 顺序执行 raw commit、结构化写入及回读检查、稀疏/向量更新及版本检查，再执行一次 search_with_timings()。
- structured_projection_ok、relation_candidate_ok、sparse_index_visible、dense_index_visible 分别记录检查结果；index_ns 记录检查完成时间，不能仅凭这一时间值断言检查成功。
- completed_ns 在检索返回后取得；命中时 time_to_top10_ms 等于 pipeline_complete_ms，未命中时为空字符串。正文用 NA 表示没有数值观测，不改记为无穷。
- 原 publication frontier 不是代码中的全局读屏障。is_visible() 返回索引版本检查结果；search() 读取索引快照，没有等待全局最小版本的逻辑。
- 当前一次更新后查询不提供连续时间中的首次进入 Top-10 时刻，也不证明某次未命中的 memory 永远不能命中。

已只读检查 Experiment 11 四格保存的全部 2,400 条事件：1,772 条命中均满足 time_to_top10_ms = pipeline_complete_ms，628 条未命中均为空值；公式映射不匹配为 0，索引完成时间晚于检索完成时间的记录为 0。本批记录四项状态检查全部通过。这是对这些观测记录的核验，不是任意并发执行都满足 invariant 的证明。

论文工件保留聚合后的阶段时延、在线 freshness 表和运行清单；体积较大的逐事件运行日志不进入公开仓库。旧实验报告中的 First inclusion / first inclusion 应解释为更新后单次检索的观测命中时延。

## 3.4 后端设计核对

当前 CassandraGraphCells 通过手写 DDL、预备 CQL 语句与按 cell 分支的读写函数实现固定访问路径；未实现通用逻辑查询解析、物理计划自动生成或优化器。因此正文采用 Cassandra-native structured memory serving，删除自动编译映射与对任意支持查询的等式保证，语义保持保留为需由实现和实验核实的要求。

- _schema()：memory 以 scope 分区；feature/mention/edge 按 scope + memory 等键组织；物化关系候选表以 scope + relation 分区。
- candidate_projection()：base 读取原文和特征后调用 renderer；materialized 直接读取 rawerk 字段。
- ids_by_relation()：base 在 scope 内读取关系记录并由应用筛选；materialized 直接读取 scope–relation 分区。
- insert() / write_structured()：由应用显式维护冗余与派生记录，并非 Cassandra 自动 materialized-view 或通用增量视图引擎。
- online_retrieval.py：BM25、dense scoring 与 Z-score 在应用层执行，不声称 Cassandra 原生执行融合排序。

源码：`src/cassmem/backend/live_cells_graph.py`。本节记录静态实现核对边界，不将手写访问路径描述为自动 compiler。

## 依据与冻结状态

- P1-C：`experiments/retrieval/p1_compact_component_ablation/run_p1c_ablation_v5.py`。当前 SHA-256 与 `results/representation_ablation/p1_compact_component_ablation/p1c_run_manifest.json` 完全一致。
- Backend Bridge：`experiments/backend_equivalence/backend_bridge_v2_run.py`，结合 `results/backend_equivalence/backend_equivalence_v2/` 的聚合 parity 结果。
- 在线四格共享核心：`src/cassmem/retrieval/online.py`；graph 100K workload、Experiment 11、recovery 的入口均调用该实现。本次记录当前代码哈希；未找到等同于 P1-C 的历史在线核心源码哈希证明，不能声称已认证其历史源码版本。
- 逻辑字段：`src/cassmem/representation/graph_event.py` 的 `GraphRecord`、`memory_projection()` 和 `graph_projection()`。

| 项目 | 核对结果 | 代码依据 |
| --- | --- | --- |
| RawERK | 原始文本 + 非空 E/R/K；固定标签、换行分隔；不拼入 triples 或 timestamp | P1-C `build_document()`；online `render_rawerk()` |
| 原始信息 | raw_text、entities、relations、keywords、triples 独立保留，并附加 rawerk | `GraphRecord.memory_projection()` |
| Embedding | record 保存 embedding_sha256，向量由独立冻结数组及 ID 映射提供 | `GraphRecord`；`run_index_preflight.py` |
| Dense 输入 | raw memory 的冻结 BGE embedding 与 query embedding；离线主排名使用冻结 cosine 分数缓存 | P1-C `top_dense_for_query()`；Bridge 主循环；online `_dense_search()` |
| Lexical 输入 | BM25(RawERK)，按 scope 建立词项统计 | P1-C `BM25Retriever`；online `_build_sparse()` |
| BM25 参数 | k1=1.5、b=0.75；lowercase；English stopwords；1–2 grams；最多 50,000 features | P1-C manifest；online `_build_sparse()` |
| 候选深度 | Dense@50 + BM25@50；不足 50 时取现有条目 | P1-C `build_zscore_rankings()`；Bridge constants；online `search()` |
| 在线过滤 | scope 上打分，按 backend candidate_ids 过滤后截断 Top-50；BM25 统计不因子集过滤而重建 | online `search()` / `search_with_timings()` |
| Z-score 范围 | 每 query、每通道截断后的 Top-50；不是完整 scope 或候选并集 | P1-C `zscore_values()`；online `_zscore()` |
| 标准差 | 总体标准差 ddof=0；std <= 1e-12 时分母取 1.0，不是加 epsilon | 同上 |
| 缺失通道 | branch_min：该通道最低 Z-score；空通道取 0；不补算完整 scope 分数 | P1-C manifest 与三个 `zscore_fuse()` |
| 融合权重 | Dense 0.6，BM25 0.4 | 同上 |
| 有序并集 | Dense 列表在前，追加 BM25 尚未出现的 ID | 三个 `zscore_fuse()` |
| 融合同分 | 保持上述有序并集的次序 | 三个 `zscore_fuse()` |
| 通道同分 | P1-C / Bridge：冻结语料 ordinal；online：memory_id 字典序 | P1-C stable mergesort / `top_dense_for_query()`；Bridge `top_indices()`；online 两个 search 函数 |
| 最终输出 | Top-10；P1-C / Bridge 内部可先保留融合 Top-50，再截取 Top-10 | P1-C `query_metric_rows()`；Bridge 主循环；online `search()` |

P1-C 冻结 BM25 CSV 仅保存 Top-10，公式核对使用哈希匹配的 P1-C 实现恢复 Top-50，并先确认其 Top-10 与冻结导出一致。

P1-C 与 online 的 BM25 TF 分母含 1e-9，Bridge 的 BM25 类未加这一项。Dense 离线采用冻结分数，online 由 float32 向量重新计算，以 1e-12 保护 L2 归一化分母。因此正文将原始打分器记为 BM25 / cosine，未扩写一个声称各路径逐浮点数一致的底层打分公式。语义保持应固定具体 scorer、候选及 tie-breaking 协议。

## 可执行复核

脚本：`experiments/retrieval/audit_system_design_fusion.py`。

~~~powershell
& 'D:/memorytable/cassandra-kg-memory/.venv-onnx/Scripts/python.exe' 'D:/memorytable/cassandra-kg-memory/experiments/retrieval/audit_system_design_fusion.py'
~~~

- 5,882 条 memory：P1-C 与 online RawERK 渲染差异为 0。
- 1,540 条 Cat1–4 query：两路均为 50 条候选。
- Eq.2 独立计算、P1-C 函数、Bridge 融合函数、online 融合函数，相对冻结正式 Top-10 的不匹配均为 0 / 1,540。
- 相同候选输入下，各融合计算的最大分数差为 0.0。
- 累计 112,398 个仅出现于单一路径的 query–memory 项，实际覆盖缺失通道规则。
- 6 组边界用例：缺失候选、全同分、单候选、单路为空、两路为空、极小方差。

范围：对 online / Bridge 复用了冻结分支输入，验证的是融合函数；未据此声称 online 完整候选生成、实时数据库或 freshness 全路径与离线结果一致。

## 代码快照 SHA-256

| 文件 | SHA-256 |
| --- | --- |
| P1-C `run_p1c_ablation_v5.py` | `5a8e870c17a732ef322eb29d3e8e47acce2a7a905ac782d4db17644eea8c1101` |
| online `online_retrieval.py` | `5162fa21c366e125c072908d47d0a5f1640e48a79446606b3b75dc50f7d4fc33` |
| Bridge `backend_bridge_v2_run.py` | `51db875ae0d9734eb54ba36bd9900c414068ae70a5195a536a5c636be53169d8` |

冻结参考：`results/retrieval/official_eval/zscore_rawerk_ranking_canonical1540.csv`，SHA-256 `23e73e853da3c538fb6287b5f8d5b183a27b6d932f8852960c3060a78b50eb42`。
