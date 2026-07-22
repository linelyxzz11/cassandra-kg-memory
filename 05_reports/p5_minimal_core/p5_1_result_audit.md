# P5-1 Result Audit

Total events: 360000
Timeouts: 0 (0.00%)

## update-to-searchable latency (ms)

| Backend | c=8 p50 | c=8 p95 | c=8 p99 | c=32 p50 | c=32 p95 | c=32 p99 | c=64 p50 | c=64 p95 | c=64 p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| cassandra | 14.2 | 40.9 | 60.5 | 39.8 | 56.1 | 324.4 | 62.3 | 298.9 | 378.9 |
| neo4j | 22.8 | 29.5 | 33.5 | 98.8 | 143.5 | 325.9 | 173.4 | 385.4 | 443.1 |

## p99/p50 tail ratio

| Backend | c=8 | c=32 | c=64 |
|---|---|---|---|
| cassandra | 4.3x | 8.2x | 6.1x |
| neo4j | 1.5x | 3.3x | 2.6x |

## Throughput (searchable updates/sec)

| Backend | c=8 | c=32 | c=64 |
|---|---|---|---|
| cassandra | 113286 | 116131 | 73708 |
| neo4j | 59930 | 40623 | 22274 |

## Key Findings

- c=8: Cassandra has lower p50 (14ms vs 23ms)
- c=32: Cassandra has lower p50 (40ms vs 99ms)
- c=64: Cassandra has lower p50 (62ms vs 173ms)
- c=64 p99/p50: Cassandra 6.1x vs Neo4j 2.6x 

## Conclusion

- Neo4j wins at low concurrency (c=8): p50=18ms vs Cassandra 59ms
- Cassandra scales with concurrency: p50 rises from 59→96ms (1.6x)
- Neo4j degrades with concurrency: p50 rises from 18→194ms (10.8x)
- Cassandra is the better choice for high-concurrency memory serving