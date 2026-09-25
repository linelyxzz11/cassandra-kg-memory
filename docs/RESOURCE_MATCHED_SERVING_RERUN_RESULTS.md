# Resource-matched 100K serving rerun: audited results

Run ID: `resmatch_20260924_v3r2` (2026-09-24). This note reports a **single-workstation, implementation-level** comparison of four physical serving cells. It does not assert that one database engine is intrinsically faster, or that the two write acknowledgements have identical crash-durability semantics. The earlier v2 results and the incomplete v3/v3r1 attempts are not substituted into these tables.

## Audit outcome and evidence

The strict [audit](../results/serving_100k/resource_matched_v3/resmatch_20260924_v3r2/audit/validation.json) is `PASS`: 60/60 main runs with 300,000 measured operations; 48/48 freshness scenarios with 2,400 updates and 45,600 reads. All four 100K-memory load gates passed (171 scopes per cell), with zero sampled graph-digest or relation-candidate mismatches. Main and freshness runs report zero operation errors; the resource audit reports zero cgroup OOM kills. The original `cassandra` and `neo4j-kg` containers were restarted after the isolated blocks, and their original volumes were not mounted by the new cells.

Evidence paths under `results/serving_100k/resource_matched_v3/resmatch_20260924_v3r2/`:

| Evidence | File |
|---|---|
| Frozen configuration, source/trace hashes, storage path and timing boundaries | `protocol.json` |
| Strict validation and resource peaks | `audit/validation.json` |
| Main 95:5 per-cell/per-concurrency statistics, median of three run-level values with min/max retained | `main/main_95_5_summary.csv` |
| Main run-level statistics and all measured events | `main/main_95_5_runs.csv`, `main/runs/*_events.csv` |
| Freshness rates, timing segments and backlog | `audit/freshness_by_rate.csv`, `freshness/runs/*_events.csv`, `freshness/runs/*_reads.csv`, `freshness/runs/*_scenarios.csv` |
| Load gates and semantic checks | `cassandra/load/`, `neo4j/load/` |
| Actual container limits, memory settings, cgroup peaks and sampled runtime usage | `<backend>/formal_container.json`, `formal_memory_config.json`, `formal_cgroup_after.json`, `formal_stats.jsonl` |

The audit program is `experiments/serving_100k/audit_resource_matched_v3.py`; the exact protocol and command sequence are in [RESOURCE_MATCHED_SERVING_RERUN_PROTOCOL.md](RESOURCE_MATCHED_SERVING_RERUN_PROTOCOL.md).

## Experimental setting and fairness boundary

| Item | Cassandra cells | Neo4j cells |
|---|---|---|
| Machine and storage | Same Windows workstation (build 26200), Intel Core i9-14900HX (32 logical processors), 32 GiB physical RAM; Docker Engine 29.4.2 under Docker Desktop/WSL2, data disk on the same D: NVMe SSD | Same |
| Database | Cassandra 5.0.8, fixed image ID in `protocol.json` | Neo4j 5.26.26 Community, fixed image ID in `protocol.json` |
| Topology | One Cassandra node; workload keyspace `SimpleStrategy`, RF=1 | One Neo4j Community server |
| Container allocation | 8 vCPU; 12 GiB RAM; memory-swap limit also 12 GiB | Identical |
| Engine memory | Maximum JVM heap 5 GiB, verified by `nodetool info` | Initial/max JVM heap 5 GiB plus 3 GiB page cache, verified by the effective settings query |
| Isolation | One new database container at a time, on its own new named data volume and non-default port | Identical policy |
| Formal runtime evidence | 817 Docker samples; cgroup memory peak 6.147 GiB; OOM=0 | 1,816 Docker samples; cgroup memory peak 6.295 GiB; OOM=0 |

The comparison equalizes **total CPU and memory budgets**, not the engines' internal allocation mechanisms. Both containers stayed below the common 12 GiB limit; equal budget does not imply equal cache hit rates or query plans. `formal_stats.jsonl` uses roughly two-second `docker stats --no-stream` sampling; cgroup `memory.peak` covers the complete container lifetime. Source and traces were frozen before formal measurement. The 100K corpus is the benchmark's deterministic scale-up of LoCoMo-derived memories, not 100K independent real users. The reorganized repository lacked historical v2 trace hashes, so v3 uses newly generated, hashed traces from the recorded seed and inputs; we do not claim byte-for-byte identity with the old v2 trace files.

The four cells implement the same logical memory projection and relation candidates with different physical layouts. Cassandra base reconstructs RawERK at read time and scans scope-local edges for relation filtering; Cassandra materialized stores RawERK and a `(scope, relation)` candidate table. Neo4j native uses graph nodes/relationships and reconstructs the projection; Neo4j materialized adds the retrieval text and relation-candidate nodes. The load gates compare each cell with the same canonical records before timing. Cassandra uses prepared CQL and logged batches for structured writes; Neo4j uses parameterized Cypher and a driver with `max_connection_pool_size=128`. Cassandra driver pool sizing is not explicitly overridden. The initial Neo4j 100K load uses one worker after a **non-timed** 12-worker load attempt failed; timed main/freshness concurrency is unchanged. These are application-implementation comparisons, not an attempt to equalize statement counts.

Additional configuration details relevant to reproducibility:

| Item | Recorded protocol |
|---|---|
| Client drivers | Python Cassandra driver 3.30.0 (protocol v4) and Neo4j Python driver 6.2.0, from the same `.venv-onnx` environment. |
| Cassandra consistency and durability | Driver default `LOCAL_ONE`, RF=1; fixed image's `cassandra.yaml` has `commitlog_sync: periodic` and `commitlog_sync_period: 10000ms`. The new container used that image without a config override. Structured writes are logged CQL batches; the freshness raw stage is a separate prepared statement. |
| Neo4j transactions | Raw and structured stages are separate Cypher calls; `Session.run` results are fully consumed by the adapter. The actual Neo4j heap/page-cache settings are in the run-bound memory snapshot. No equivalence to Cassandra's fsync boundary is claimed. |
| Schema and indexes | Cassandra uses query-specific primary-key table layouts rather than a generic graph traversal index. Neo4j creates composite uniqueness constraints for scoped memory/entity/candidate identities plus scope indexes, and calls `db.awaitIndexes(300)` before the workload. Exact DDL is in `src/cassmem/backend/live_cells_graph.py`. |
| Batching and pooling | Timed operations use one logical event per worker invocation; Cassandra structured state is written as one logged batch per event, whereas Neo4j executes the corresponding parameterized Cypher operation. The Neo4j driver pool cap is 128; Cassandra uses its unmodified driver pool defaults. Initial bulk loading is excluded from measured windows. |

The main benchmark uses 95% reads and 5% updates, 500 warm-up operations and 5,000 measured operations per run, five worker-concurrency levels (1/8/16/32/64) and three repetitions. It is closed-loop: the next task is submitted as workers progress, so a run's duration is determined by 5,000 completions rather than a fixed wall-clock interval. Freshness instead uses 32 workers with an open-loop 19:1 read/update stream at 1/2/5/10 updates/s (20/40/100/200 total operations/s), three repetitions, with 40/40/40/80 updates per rate and nominal injection windows of 40/20/8/8 seconds; it then waits for the queue to drain. The in-process lexical/dense index is built before freshness injection; no separate 500-operation freshness warm-up is claimed. Database caches are warmed by loading and preceding test activity, but an identical internal cache state was neither forced nor verified. GC, page-cache hit rates and driver acquisition wait were not isolated into independent measurements. Driver acquisition, network and application operations remain inside the client-observed paths as stated below.

## Main 95:5 workload

Median throughput across three repeats, operations/s (each cell passed at every point):

| Physical cell | C=1 | C=8 | C=16 | C=32 | C=64 |
|---|---:|---:|---:|---:|---:|
| Cassandra base | 46.5 | 223.9 | 221.6 | 227.3 | 233.8 |
| Cassandra materialized | 48.0 | 225.7 | 223.7 | 226.6 | 230.3 |
| Neo4j native | 56.6 | 31.3 | 31.2 | 30.7 | 29.4 |
| Neo4j materialized | 57.0 | 33.4 | 34.2 | 33.3 | 32.6 |

At C=1, Neo4j has higher throughput in this implementation. At C=32, Cassandra's two layouts sustain about 227 operations/s, while Neo4j's sustain about 31–33 operations/s. The corresponding read p95s are 219–224 ms for Cassandra and 1,157–1,238 ms for Neo4j; at C=64 they are 362–379 ms and 2,324–2,581 ms, respectively. These p95s are **worker-start-to-completion application latencies**, not database-internal execution times. The full table retains update and combined p50/p95/p99 and repeat ranges.

## Freshness: the critical timing distinction

The legacy field `t_commit_ms` is measured from **event submission** until the raw-memory write call returns; it includes waiting for an available application worker. It must not be called `database commit latency`. The new event log separates:

1. `queue_wait_ms`: submission to worker start;
2. `raw_write_call_ms`: immediately before `commit_raw` to client-observed acknowledgement, including driver acquisition, network and server work, but not executor wait;
3. `arrival_to_raw_ack`: submission to raw-write acknowledgement (the legacy queue-inclusive `t_commit_ms`);
4. `pipeline_complete_ms`: submission through structured-view check, sparse/dense index visibility and one post-update ranking execution.

The table shows the **median of three per-scenario p95s**, in milliseconds. `backlog` is the median peak number of pending update tasks. All values are from the complete, audited v3r2 run.

| Physical cell | Updates/s | Queue p95 | Raw-call p95 | Arrival-to-raw-ack p95 | Pipeline p95 | Update backlog peak |
|---|---:|---:|---:|---:|---:|---:|
| Cassandra base | 1 | 0.2 | 20.3 | 20.5 | 206.2 | 1 |
| Cassandra base | 2 | 0.2 | 17.7 | 17.8 | 179.0 | 1 |
| Cassandra base | 5 | 0.4 | 13.3 | 13.4 | 193.6 | 2 |
| Cassandra base | 10 | 5.9 | 23.8 | 27.6 | 318.3 | 5 |
| Cassandra materialized | 1 | 0.2 | 17.6 | 17.7 | 137.3 | 1 |
| Cassandra materialized | 2 | 0.1 | 17.5 | 17.5 | 160.4 | 1 |
| Cassandra materialized | 5 | 0.2 | 12.4 | 12.6 | 170.3 | 2 |
| Cassandra materialized | 10 | 7.3 | 28.3 | 34.6 | 337.2 | 5 |
| Neo4j native | 1 | 0.3 | 11.0 | 12.4 | 143.6 | 1 |
| Neo4j native | 2 | 4,758.2 | 10.0 | 4,764.6 | 5,999.0 | 10 |
| Neo4j native | 5 | 16,727.5 | 12.1 | 16,734.6 | 17,814.9 | 30 |
| Neo4j native | 10 | 42,945.2 | 11.4 | 42,952.3 | 44,450.3 | 70 |
| Neo4j materialized | 1 | 0.2 | 12.5 | 13.2 | 168.2 | 1 |
| Neo4j materialized | 2 | 2,785.8 | 13.9 | 2,795.5 | 3,973.6 | 8 |
| Neo4j materialized | 5 | 15,394.7 | 12.9 | 15,405.1 | 16,455.4 | 29 |
| Neo4j materialized | 10 | 37,682.4 | 10.4 | 37,688.4 | 39,068.6 | 69 |

At 1 update/s (20 total arrivals/s), both Neo4j cells have virtually no update queue, and their raw-write call p95 is **lower** than Cassandra's. At 2 updates/s (40 total arrivals/s) Neo4j starts accumulating an application-worker queue; at 10 updates/s (200 total arrivals/s), its queue p95 reaches 38–43 seconds, while its raw-write call p95 remains 10–11 ms. Cassandra's queue p95 at that rate is 6–7 ms. This is consistent with the closed-loop Neo4j throughput of about 30–34 operations/s falling below the offered 40–200 operations/s; it is not proof that raw Neo4j commits themselves take seconds. The main and freshness access mixes differ, so the throughput comparison is a capacity-consistency explanation rather than an engine-level causal decomposition.

The median-of-three 500 ms **pipeline-completion** fraction is 100% for both Cassandra cells at these rates, 100% for Neo4j at 1 update/s, 15% at 2 updates/s, and 0% at 5/10 updates/s. One Cassandra-materialized repetition at 10 updates/s was 97.5%, so this is not a claim that every repetition was perfect. This metric asks whether the full update path and one ranking call finish within 500 ms; it is **not** Top-10 inclusion. Top-10 hit timing is only meaningful for update–query pairs whose final reference ranking contains the update. The raw event and scenario files retain those denominators separately.

## What may and may not be claimed

- Supported: under this 100K, single-node, matched-container-budget protocol, the Cassandra-native serving implementation sustains higher throughput and much lower queue-inclusive update-to-retrieval latency at the higher offered loads while retaining the sampled canonical projection/relation semantics.
- Not supported: an intrinsic Cassandra-versus-Neo4j database speed ranking, identical cache state, identical internal query work, identical crash durability, server-internal commit service time, cluster failover, or a claim that every new memory must enter Top-10.
- The write stages use different engine persistence mechanisms. Cassandra's periodic commitlog synchronization and Neo4j's transaction-log path are not a durability-matched experiment. Report the measured value as **client-observed raw-write acknowledgement**, and the legacy `t_commit_ms` as **queue-inclusive arrival-to-raw-write acknowledgement**.
- The old v2 approximately 7 ms versus 8,000 ms comparison was obtained under unequal surviving engine-memory settings and without its per-event split. It should be replaced, not patched with the new numbers.

The earlier `resmatch_20260924_v3` load-preparation failure and `v3r1` telemetry-incomplete run remain on disk as diagnostic attempts. Only `v3r2` has the complete resource, event and semantic audit reported here.
