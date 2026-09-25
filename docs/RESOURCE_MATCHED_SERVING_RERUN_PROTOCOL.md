# Resource-matched serving rerun protocol (v3)

**Status: `resmatch_20260924_v3r2` complete; strict audit PASS.** The historical v2 results remain untouched. Docker Desktop's data disk was migrated to D and the original containers and volumes were verified readable and restarted after each block. The initial `resmatch_20260924_v3` attempt passed Cassandra preflight but Neo4j dataset preparation hit `Transaction.Outdated` under 12 parallel loading workers. The `v3r1` retry completed all 60 main and 48 freshness points, but its Docker stats stream produced empty files. Its event-level results are retained for diagnosis, not promoted to resource-complete evidence. The final `v3r2` rerun used serial Neo4j initial loading and a verified periodic `docker stats --no-stream` sampler plus cgroup memory peak; timed workload concurrency remained unchanged. See [audited results](RESOURCE_MATCHED_SERVING_RERUN_RESULTS.md).

## Frozen experimental setting

| Item | Cassandra | Neo4j |
|---|---|---|
| Image | Existing image ID for Cassandra 5.0.8 | Existing image ID for Neo4j 5.26.26 Community |
| Docker budget | 8 vCPU, 12 GiB RAM, memory-swap=12 GiB (no swap) | Identical |
| Internal memory | `MAX_HEAP_SIZE=5G`; effective maximum verified with `nodetool info` | `server.memory.heap.initial_size=5G`, `server.memory.heap.max_size=5G`, `server.memory.pagecache.size=3G`; settings verified after startup |
| Placement | One database container at a time on the same Docker Desktop/WSL2 VM and SSD | Identical |
| Data | Independently named, new Docker volume; original volume never mounted | Identical policy |
| Topology | One node, RF=1 | One Community server |
| Application client | Same Windows host, same Python environment and frozen source files | Identical |

The primary experiment matches the **total CPU/RAM budget**, not internal allocation mechanisms. Per-run Docker stats and cgroup CPU/memory counters must be retained. Memory allocation is frozen before formal runs; an OOM or failed config check invalidates the block. No performance-driven retuning after seeing formal results.

The source dataset and 95:5 traces are fixed before measurement. The reorganized checkout omitted the historical per-operation JSONL traces, so the deterministic builder regenerated them from the same seed (`20260803`) and available inputs. The new six file hashes are in `results/serving_100k/locomo_workload_100k/traces/trace_manifest.json`. No old per-file trace hash was retained; v3 is therefore a newly frozen rerun of the documented trace-generating protocol, not a claimed byte-for-byte verification of historical v2 trace files.

## Workload and outcome gates

- Four physical cells: Cassandra base/materialized and Neo4j native/materialized. Load 100,000 memories per cell. Check memory count, namespace count, graph projection digest and relation candidates before measuring; preserve the load-gate outputs.
- Main closed-loop workload: concurrency 1/8/16/32/64, three repetitions, 500 warm-up + 5,000 measured operations per point, 95% reads and 5% updates. Report throughput, read/update p50/p95/p99, errors, semantic checks and per-event stage timings.
- Open-loop freshness: 32 client workers, 1/2/5/10 updates/s with 19 reads per update, three repetitions and at least 40 updates per scenario. Report offered and completed rates, backlog peak/drain, errors/timeouts, candidate/view/index/ranking visibility and per-event read/update timings. Saturated points are retained as overload behavior, not relabeled database service time.
- Preserve exact run command, image IDs, source/trace hashes, configuration, index status, warm-up policy, storage settings and runtime resource samples. A failed gate or OOM makes that run invalid; it is not silently dropped.

## Timing vocabulary

| Reported quantity | Clock boundaries | Queue-inclusive? | Interpretation |
|---|---|---|---|
| Main `latency_ms` | Worker starts to completed operation | No | Application service path under the closed-loop client, not server-internal DB time. |
| Freshness `queue_wait_ms` | Task submission to worker start | Yes, isolated component | Client executor/admission wait. |
| Freshness `raw_write_call_ms` | Immediately before `commit_raw` to acknowledged return | No executor wait | Client-observed raw-write round trip; includes driver pool/network/server work. **Not** server-internal service time or equal durability. |
| Freshness `worker_start_to_raw_commit_ms` | Worker start to raw acknowledged return | No executor wait | Worker-local raw stage, including local dispatch overhead. Legacy name retained in CSV; call it worker-to-raw-ack in prose. |
| Freshness `t_commit_ms` | Task submission to raw acknowledged return | **Yes** | Legacy CSV field; call **queue-inclusive arrival-to-raw-write acknowledgement** in prose. |
| Freshness `t_structured_view_visible_ms` | Submission to structured write plus projection/relation check | Yes | Cumulative application-observed structured visibility. |
| Freshness `t_index_visible_ms` | Submission to sparse/dense version check | Yes | Cumulative retrieval-index visibility. |
| Freshness `pipeline_complete_ms` | Submission to one post-update ranking completion | Yes | Update-to-retrieval-completion; not first Top-10 inclusion or continuous convergence. |

Additional event fields separate raw ack to structured completion, structured to index visibility, and index to ranking completion. Freshness read events retain both worker-local and arrival-to-completion latency. The main closed-loop update events separate full structured write acknowledgement, projection fetch, relation check, sparse/dense upsert, candidate fetch and ranking. Database-side aggregate metrics may aid diagnosis, but Cassandra and Neo4j server timers do not share an identical internal boundary and will not be reported as a directly matched per-event metric.

## Durability and fairness scope

Cassandra's existing configuration uses periodic commitlog sync and acknowledges before the next periodic fsync. Neo4j uses transactions and its own transaction log. The v3 primary result is an **acknowledged-write comparison under the declared engine settings**, not a comparison of identical crash-durability guarantees. If a durability-matched sensitivity experiment is added, it must be a separately labeled configuration with both effective settings verified before timing; it must not be merged into the main series.

## Execution and preservation

The isolated runner is `experiments/serving_100k/run_resource_matched_v3.py`. It requires the explicit `--storage-migrated` flag and verifies Docker Desktop's data disk is on D, uses non-default local ports (`19042`, `17687`), never removes volumes, and restores original containers in a `finally` block. The resource-complete rerun writes to `results/serving_100k/resource_matched_v3/resmatch_20260924_v3r2/`, not to any v2 or prior v3 output directory. The preflight stage loads the new data, checks memory settings, and runs small smoke workloads; only then may formal runs begin. Cassandra and Neo4j preflight and formal blocks ran separately. Both formal manifests passed, resource samples were nonempty and the strict audit passed.

Example after verifying the Docker storage migration (from repository root, using `.venv-onnx/Scripts/python.exe`):

```powershell
.\.venv-onnx\Scripts\python.exe experiments\serving_100k\run_resource_matched_v3.py --phase preflight --backend cassandra --storage-migrated
.\.venv-onnx\Scripts\python.exe experiments\serving_100k\run_resource_matched_v3.py --phase preflight --backend neo4j --storage-migrated
.\.venv-onnx\Scripts\python.exe experiments\serving_100k\run_resource_matched_v3.py --phase formal --backend cassandra --storage-migrated
.\.venv-onnx\Scripts\python.exe experiments\serving_100k\run_resource_matched_v3.py --phase formal --backend neo4j --storage-migrated
```

These commands mutate **only the new v3 volumes**, temporarily stop the original containers, and restart them afterward. The original containers and Docker data volumes are not deleted or reset.
