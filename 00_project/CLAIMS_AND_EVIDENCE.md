# Claims and Evidence

## P5-1: Cassandra achieves lower median update-to-searchable latency at all concurrency levels
- Evidence: p5_1_summary.csv — c=8: C=14.6ms, N=22.9ms; c=32: C=39.7ms, N=98.4ms; c=64: C=62.3ms, N=164.7ms
- Gate: 360K events, 0 timeouts, 0 stale

## P5-2: Cassandra maintains advantage under hot-entity skew
- Evidence: p5_2_hot_entity_summary.csv — Hot p50: C=59ms, N=119ms (2.01x)
- Degradation p50: C=1.04x, N=0.93x

## P5-3A: Event log + checkpoint recovery smoke passes
- Evidence: p5_3a_gate.csv — 4100 events, 0 timeouts, 0 duplicate visible state

## P5-3B: Burst-only processing with 0 loss
- Evidence: p5_3b_gate.csv — 42K events, 0 timeouts
- Burst p50: C=10.2ms, N=30.2ms

## P5-3C: Materializer restart recovery — both backends fully recover
- Evidence: p5_3c_cross_backend_summary.md — 36K events, 0 missed, both PASS
- Neo4j faster drain (132s vs 216s), Cassandra larger backlog (8445 vs 3372)
