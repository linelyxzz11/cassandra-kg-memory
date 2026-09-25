# Resource-matched serving evidence

`resmatch_20260924_v3r2/` is the only formal result published from this rerun. It
passed the strict audit: 60 main workload runs (300,000 measured operations), 48
freshness scenarios (2,400 updates and 45,600 reads), four 100K-memory load gates,
resource sampling, and zero OOM kills. Main and freshness event CSVs are retained so
that the aggregates can be checked from individual observations. The six fixed
workload traces and their SHA-256 manifest are under
`../locomo_workload_100k/traces/`.

From the repository root, run:

```powershell
python experiments/serving_100k/audit_resource_matched_v3.py `
  --base results/serving_100k/resource_matched_v3/resmatch_20260924_v3r2
```

The [protocol](../../../docs/RESOURCE_MATCHED_SERVING_RERUN_PROTOCOL.md) and
[results note](../../../docs/RESOURCE_MATCHED_SERVING_RERUN_RESULTS.md) give the
settings and the interpretation of each latency boundary. Runtime `.log` files
and two earlier diagnostic attempts (`v3`, `v3r1`) are kept locally, not promoted
as formal public evidence. The published results retain per-event CSVs, manifests,
container/configuration snapshots and resource samples. In particular, legacy
`t_commit_ms` includes application queueing and must not be called database commit
latency.
