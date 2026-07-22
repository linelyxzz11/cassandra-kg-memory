# Scale Sweep Supporting Summary — 100K → 1M

## Scope

Supporting evidence for Cassandra-KG vs Neo4j cross-scale qualitative consistency.

## Input Files

| Scale | Graph | Source |
|---|---|---|
| 100K_legacy_clean | sysaxis_100K_legacy_clean_20260709 (99,875 edges) | `reports/.../scale_100k_legacy_clean/` |
| 1M_scale_controlled | c3_scale_1M_seed42 (1,000,000 edges) | `reports/sysaxis_1m_write_ratio_final/` |

## Fixed Conditions

- clients = 32
- hop = 2
- fanout = 20
- write_ratio = 10%
- application cache = disabled
- relation index = disabled

## ⚠️ Caveat

This is a supporting scale comparison, not a strict same-generator scale law. The 100K point is a rebuilt legacy synthetic graph with 99,875 distinct logical edges, while the 1M point is a scale-controlled graph with exactly 1,000,000 distinct logical edges. Therefore, conclusions should be phrased as **qualitative consistency across scales** rather than strict scaling linearity.

## Results

| Scale | System | Mode | read QPS | read p95 | read p99 | write QPS | write p95 | write p99 |
|---|---|---|---|---|---|---|---|---|
| 100K | cassandra_opt | cold | 597 | 93.3ms | 114.0ms | 66.2 | 64.9ms | 81.2ms |
| 100K | neo4j | cold | 484 | 160.1ms | 456.9ms | 53.2 | 18.4ms | 34.0ms |
| 100K | cassandra_opt | warm | 599 | 92.3ms | 113.8ms | 66.8 | 64.6ms | 80.8ms |
| 100K | neo4j | warm | 490 | 154.2ms | 453.2ms | 54.6 | 18.2ms | 31.8ms |
| 1M | cassandra | cold | 425 | 137.1ms | 173.6ms | 47.1 | 97.7ms | 115.8ms |
| 1M | neo4j | cold | 322 | 241.9ms | 696.7ms | 35.8 | 28.2ms | 46.0ms |
| 1M | cassandra | warm | 586 | 86.4ms | 105.2ms | 65.2 | 58.0ms | 66.8ms |
| 1M | neo4j | warm | 428 | 171.0ms | 488.3ms | 47.4 | 18.2ms | 27.2ms |

## Key Observations

1. **Cassandra read advantage persists** at both scales: p99 4.0x (100K) and 4.0x (1M cold) faster vs Neo4j
2. **Neo4j write advantage persists**: write p99 2.4x (100K) and 2.5x (1M cold) faster vs Cassandra (4-table sync overhead)
3. **Cassandra warmup benefit larger at 1M**: cold→warm read QPS +38% vs +0.3% at 100K, suggesting page cache pressure only at scale
4. **Neo4j tail latency grows with scale**: p99 457ms (100K) → 697ms (1M cold), +52%

## Correctness

- 100K: CSV/Cassandra/Neo4j distinct=99875, guards pass, hash gate 256/256, 20/20 formal trials, 40/40 spotchecks, failures empty
- 1M: Cassandra guard 1M/1M, hash gates pass, write-ratio sweep 60/60 trials, failures empty
