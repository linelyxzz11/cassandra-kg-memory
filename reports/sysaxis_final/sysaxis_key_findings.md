# Sysaxis Key Findings
Generated: 2026-07-09T13:12:36.214007

## 1M Write-Ratio
At 1M edges, clients=32, hop=2, under 0/10/30% write ratios, Cassandra-KG consistently maintains lower read P95/P99 latency than Neo4j, while Neo4j has lower per-write latency.

10% write cold: Cass read QPS=425, p99=174ms vs Neo read QPS=322, p99=697ms (4.0x). Write: Cass p99=116ms vs Neo p99=46ms (Neo 2.5x faster).
10% write warm: Cass read QPS=586, p99=105ms vs Neo read QPS=428, p99=488ms (4.6x). Write: Cass p99=67ms vs Neo p99=27ms (Neo 2.5x faster).

## 1M Concurrency
Neo4j is strong at low concurrency (clients=1), but tail latency grows sharply at high concurrency. Cassandra opt keeps p99 below 190ms across all client levels and cold/warm modes.

Warm clients=64: Cass opt p99=189.2ms vs Neo4j p99=1194.8ms. Cassandra naive sometimes has higher QPS but worse tails.

## 1M Hop-Depth
Crossover at hop=2. Neo4j wins hop=1; Cassandra-KG wins hop=2+, with ~3x lower p99 at hop=4 in both cold and warm modes.

Warm hop=4: Cass QPS=138, mean=58.0ms, p99=141.9ms vs Neo QPS=108, mean=74.4ms, p99=424.8ms (3.0x).
Cold crossover hop=2: Cass QPS=521, p99=51.8ms vs Neo QPS=482, p99=111.8ms.

## Scale Supporting: 100K -> 1M
The clean legacy 100K point (99,875 edges) shows qualitative consistency with 1M (hop=2, clients=32, wr=10%):
- Cass read p99 advantage: 4.0x (both scales)
- Neo write advantage: 2.4x (100K) and 2.5x (1M)
- Cass warmup benefit larger at 1M (+38% QPS) than 100K (+0.3%)

Qualitative consistency, not strict scaling linearity (different graph generators).
