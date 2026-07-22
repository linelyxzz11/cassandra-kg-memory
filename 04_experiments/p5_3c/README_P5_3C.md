# P5-3C Restart-Only Runner

这套脚本不调用任何 LLM/API。它直接连接已启动的 Cassandra 和 Neo4j Docker。

## 设计

```text
producer
  → durable SQLite event queue
  → raw/KG backend commit

materializer 独立进程
  → RawERK view
  → SQLite FTS5 BM25
  → top-10 visibility probe
  → atomic checkpoint
```

默认时间线：

```text
0–60s    正常输入 100 events/s
60–90s   materializer 被真实 terminate，producer 和后端写入继续
90–180s  materializer 从 checkpoint 恢复，输入仍为 100 events/s
之后      drain 到 backlog=0，再稳定检查 60s
```

SQLite 只承担持久化事件队列、checkpoint 和双方共用的 BM25 索引；raw/KG 与 RawERK view 都真实写入对应数据库。

## 安装

建议直接使用已经跑通 P5-1/P5-2 的 Python 环境：

```bash
python -m pip install -r requirements.txt
```

Neo4j 推荐使用环境变量：

```bash
export NEO4J_URI="bolt://127.0.0.1:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="你的密码"
export NEO4J_DATABASE="neo4j"
```

PowerShell：

```powershell
$env:NEO4J_URI="bolt://127.0.0.1:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="你的密码"
$env:NEO4J_DATABASE="neo4j"
```

## 配置

```bash
cp p5_3c_config.example.json p5_3c_config.json
```

必须修改：

```json
"source_artifact": "c3_source_scale_1M.csv 的真实路径",
"expected_sha256": "P5-1 已验证 SHA-256",
"validated_manifest": "P5-1 run manifest 真实路径"
```

## 先做短 Smoke

复制一份 `p5_3c_config_smoke.json`，改成：

```json
"event_count": 1200,
"input_rate_per_second": 20,
"producer_duration_seconds": 60,
"materializer_stop_at_seconds": 20,
"materializer_outage_seconds": 10,
"stable_verify_seconds": 10,
"drain_timeout_seconds": 60,
"concurrency": 8,
"overwrite_events": true
```

生成 smoke events：

```bash
python run_p5_3c_restart_only.py --config p5_3c_config_smoke.json --generate-events
```

分别运行：

```bash
python run_p5_3c_restart_only.py --config p5_3c_config_smoke.json --backend cassandra --run
python run_p5_3c_restart_only.py --config p5_3c_config_smoke.json --backend neo4j --run
python run_p5_3c_restart_only.py --config p5_3c_config_smoke.json --compare
```

两个 Gate 都 PASS 后，再恢复正式配置。

## 正式运行

只生成一次 frozen events：

```bash
python run_p5_3c_restart_only.py --config p5_3c_config.json --generate-events
```

然后分别运行：

```bash
python run_p5_3c_restart_only.py --config p5_3c_config.json --backend cassandra --run
python run_p5_3c_restart_only.py --config p5_3c_config.json --backend neo4j --run
python run_p5_3c_restart_only.py --config p5_3c_config.json --compare
```

## 主要输出

```text
reports/p5_minimal_core/p5_3c_restart_only/
├── p5_3c_cassandra_scale_guard.json
├── p5_3c_neo4j_scale_guard.json
├── p5_3c_cassandra_run_manifest.json
├── p5_3c_neo4j_run_manifest.json
├── p5_3c_cassandra_per_event.csv
├── p5_3c_neo4j_per_event.csv
├── p5_3c_cassandra_timeline.csv
├── p5_3c_neo4j_timeline.csv
├── p5_3c_cassandra_phase_summary.csv
├── p5_3c_neo4j_phase_summary.csv
├── p5_3c_cassandra_final_state_audit.csv
├── p5_3c_neo4j_final_state_audit.csv
├── p5_3c_cassandra_recovery_summary.json
├── p5_3c_neo4j_recovery_summary.json
├── p5_3c_cassandra_gate.json
├── p5_3c_neo4j_gate.json
├── p5_3c_cross_backend_comparison.csv
└── p5_3c_cross_backend_summary.md
```

## Gate 关注

```text
materializer_stopped = true
materializer_restarted = true
checkpoint_fixed_during_outage = true
backlog_formed = true
final_backlog_zero = true
backend_commit_failures = 0
missed_updates = 0
searchability_timeouts = 0
final_state_failures = 0
```

## 注意

- 脚本使用 P5 专用 Cassandra 表和 Neo4j 标签，不覆盖既有实验结果。
- `reset_runtime=true` 只清理本地运行目录，不删除数据库中的 P5 专用状态；版本条件更新保证重复运行幂等。
- P5-3C 是 restart-only isolation，不应写成 Burst + Restart 结果。
