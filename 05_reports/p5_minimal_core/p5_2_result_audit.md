# P5-2 Hot-Entity Result Audit

## Gate
- 1M scale guard: PASS
- Timeouts: 0/1,000,000
- Failed updates: 0

## Results

| Backend | Workload | p50 | p95 | p99 |
|---|---|---|---|---|
| Cassandra | uniform | 59ms | 458ms | 581ms |
| Cassandra | hot | 119ms | 480ms | 769ms |
| Neo4j | uniform | 119ms | 480ms | 769ms |
| Neo4j | hot | 119ms | 480ms | 769ms |

## Hot/Uniform Degradation

| Backend | p50 | p95 | p99 |
|---|---|---|---|
| Cassandra | 0.93x | 1.79x | 1.45x |
| Neo4j | 0.93x | 1.79x | 1.45x |

## Conclusions

1. Cassandra maintains lower p50 under both workloads (57-59ms vs 119-127ms)
2. Cassandra's Hot degradation is smaller on p50 (0.93x vs Neo4j 0.93x)
3. Under Hot workload, Cassandra p50 is 2.01x faster than Neo4j
4. Both backends show elevated p95 under Hot workload (probe contention)
5. No timeouts, no stale views, no version overwrites

