# C1 Concurrency Sweep — Result Protocol

## Two independent tables, one trial store

All 51 trials reside in a single `trial_summary.csv`. The repeat column
splits them into two datasets that are reported separately and never merged.

| Table file | Source | Rows | Purpose |
|-----------|--------|------|---------|
| `final_concurrency_balanced_3repeats.csv` | repeat 1–3 | 12 | Primary balanced experiment |
| `stability_followup_5repeats.csv` | repeat 4–8 | 3 | Post-hoc targeted diagnostic |

## C1-main: balanced 3-repeat sweep

- Every `(system, clients)` pair has exactly 3 independent repeats.
- Run order was randomized per repeat block and saved as `run_order.jsonl`.
- All 36 trials completed without error (`failures.jsonl` is empty).
- 360 hash spot-checks (36 × 10) all passed.

### Statistics

- P95 / P99 are computed **per trial** from that trial's latency buffer.
- The per-system summary reports the **median** of the 3 trial-level
  P95 / P99 values, not a merged-latency P95.
- QPS is `successful_queries / actual_measurement_seconds`.
- IQR is Q3 − Q1 of the 3 trial-level QPS values.

## Stability follow-up

Three `(system, clients)` pairs showed QPS range/median > 20% across the
original 3 repeats. To diagnose whether the extremes were transient or
systematic, 5 additional independent repeats were run for each pair
with the same config and workload.

These runs are **not part of the balanced experiment table** because they
violate the equal-repeat design. They are diagnostic evidence that the
original extremes were transient:

| system | clients | original IQR | follow-up IQR | reduction |
|--------|--------:|-------------:|--------------:|----------:|
| cassandra_parallel | 32 | 401 | 11 | 97% |
| neo4j | 32 | 146 | 5 | 96% |
| neo4j | 64 | 57 | 7 | 87% |

### Resource notes (from stability runs only)

- Python RSS: stable ~151–153 MB across all trials.
- cassandra_parallel c=32: avg CPU ~290%, peak ~470%.
- neo4j c=32/64: avg CPU ~415%, peak ~830–940% (Neo4j Docker container).

## Aggregation code

```python
def split_trials(csv_path):
    rows = []
    with csv_path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    balanced = [r for r in rows if int(r["repeat"]) <= 3]
    stability = [r for r in rows if int(r["repeat"]) >= 4]
    return balanced, stability

def aggregate(group):
    qps = sorted([float(t["QPS"]) for t in group])
    mid = len(group) // 2
    def iqr(arr):
        return round(arr[3*len(arr)//4] - arr[len(arr)//4], 3)
    return {
        "median_QPS": qps[mid],
        "min_QPS": qps[0],
        "max_QPS": qps[-1],
        "iqr_QPS": iqr(qps),
        "repeats": len(group),
    }
```

The `repeat` column in `trial_summary.csv` is 1‑based. Original sweep uses
repeats 1‑3; stability follow-up uses repeats 4‑8.
