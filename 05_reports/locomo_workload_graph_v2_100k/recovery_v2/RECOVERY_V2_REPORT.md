# Controlled Recovery under the 100K Graph-Aware Workload

## Protocol

- Four cells use the same 100K LoCoMo-shaped graph-aware logical-event contract.
- c=32; 95% reads / 5% updates; 20 total arrivals/s; three repetitions.
- Failure: the update/materializer worker stops after a raw-memory commit and before structured/index visibility; arrivals continue for a 10-second outage; a new worker replays and drains the backlog.
- The database remains online. This is controlled application-pipeline recovery, not multi-node database failover.
- Recovery SLO: five consecutive completed updates at pipeline latency <= 2,000 ms, with zero update error.

## Main results

| Cell | Backlog peak | First replay (ms) | First Top-10 hit (ms) | Drain (ms) | Restore SLO (ms) | Post-restart p95 (ms) | Timeout | Missed | Duplicate | Top-10 parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cassandra-base | 11 | 1.1 | 136.5 | 1336.9 | 4105.0 | 8876.2 | 14.3% | 0.0% | 0.0% | PASS |
| cassandra-materialized | 11 | 0.4 | 109.1 | 1047.2 | 3093.5 | 8805.5 | 14.3% | 0.0% | 0.0% | PASS |
| neo4j-native | 11 | 0.7 | 98.5 | 1015.9 | 3123.5 | 8833.4 | 14.3% | 0.0% | 0.0% | PASS |
| neo4j-materialized | 11 | 0.9 | 88.4 | 873.7 | 3080.8 | 8756.2 | 14.3% | 0.0% | 0.0% | PASS |

## Interpretation guardrails

- These results support recovery of the CassMem application pipeline under a controlled worker failure.
- They do not establish Cassandra or Neo4j multi-node availability, failover, or distributed fault tolerance.
- Post-recovery Top-10 mismatches across cells: 0.
