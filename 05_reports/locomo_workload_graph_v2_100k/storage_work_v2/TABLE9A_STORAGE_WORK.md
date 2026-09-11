# Table 9A: Actual storage work at 100K

All four cells store the same 100,000 logical memories, 76,475 entity dictionary records, 129,156 mentions, and 42,944 semantic edges. Physical records differ because the native and query-materialized schemas encode that semantic work differently.

| Cell | Memory | Feature | Entity | Mentions | Semantic edges | Physical edge rows/rels | Materialized candidates | Total records | Writes/event mean | Writes/event p95 | Load s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cassandra-base | 100000 | 100000 | 76475 | 129156 | 42944 | 128832 | 0 | 534463 | 6.135 | 17.0 | 33.78 |
| Cassandra-materialized | 100000 | 0 | 76475 | 129156 | 42944 | 42944 | 42944 | 391519 | 4.706 | 14.0 | 31.08 |
| Neo4j-native | 100000 | 100000 | 76475 | 129156 | 42944 | 42944 | 0 | 548575 | 6.277 | 15.0 | 69.85 |
| Neo4j-materialized | 100000 | 0 | 76475 | 129156 | 42944 | 42944 | 42723 | 391298 | 4.704 | 14.0 | 69.16 |

Interpretation: `actual_storage_records` counts rows/nodes/relationships, not bytes on disk. Cross-engine byte footprint is not reported because Cassandra SSTable and Neo4j store files include different compaction, transaction-log, and allocation overheads; presenting raw directory size as a fair schema comparison would be misleading.
