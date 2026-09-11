# 100K LoCoMo-shaped 95:5 Stage-Timing Aggregate

## Protocol

- Four live cells × concurrency `{1,8,16,32,64}` × three repetitions.
- 250 measured updates per point; 750 events per cell/concurrency; 15,000 total.
- All 60 source summaries are PASS and all `projection_ok` values must equal 1.
- Percentiles pool the three frozen repetitions at the event level using NumPy linear quantiles.
- `TTTop10*` is conditional on successful Top-10 entry. `FreshHit@10` uses all events and must be interpreted jointly.
- Deadline FreshHit@10 counts a success only when the target both enters Top-10 and does so within 50/100/250/500/1000 ms.
- `Attempt` is the measured end-to-end update-and-retrieval attempt, including dense-index visibility plus candidate fetch and non-overlapping sparse/dense/fusion scoring.

## Four-cell main table (milliseconds)

| Cell | c | N | FreshHit@10 | Fresh@100ms | Fresh@250ms | Fresh@500ms | Fresh@1s | TTTop10 p50* | TTTop10 p95* | TTTop10 p99* | Commit p50 | Commit p95 | Commit p99 | RawERK p50 | RawERK p95 | RawERK p99 | Attempt p50 | Attempt p95 | Attempt p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cassandra-base | 1 | 750 | 0.7453 | 0.7173 | 0.7453 | 0.7453 | 0.7453 | 63.9534 | 92.0016 | 140.2317 | 7.0321 | 13.2573 | 20.2091 | 26.5199 | 40.8102 | 54.6047 | 62.1542 | 91.8033 | 139.8232 |
| cassandra-base | 8 | 750 | 0.7453 | 0.1053 | 0.6133 | 0.7453 | 0.7453 | 166.4026 | 316.2116 | 354.1619 | 7.034 | 27.1697 | 74.3466 | 23.8526 | 92.6905 | 120.7184 | 167.9869 | 315.3879 | 356.3307 |
| cassandra-base | 16 | 750 | 0.7453 | 0.0 | 0.0653 | 0.5347 | 0.7453 | 427.8546 | 603.4109 | 658.2795 | 11.3627 | 38.1188 | 74.9349 | 38.995 | 104.1246 | 132.9773 | 427.9209 | 600.9544 | 667.6284 |
| cassandra-base | 32 | 750 | 0.7453 | 0.0 | 0.0 | 0.016 | 0.3347 | 1032.1438 | 3504.2484 | 4131.3481 | 38.9772 | 185.214 | 414.3656 | 136.2652 | 614.2743 | 835.06 | 1016.8684 | 3489.3872 | 4097.6292 |
| cassandra-base | 64 | 750 | 0.7453 | 0.0 | 0.0 | 0.0013 | 0.0587 | 1467.0991 | 2192.6216 | 2667.2089 | 101.9013 | 216.875 | 418.8943 | 358.7015 | 570.2859 | 771.679 | 1433.9511 | 1891.0582 | 2362.0572 |
| cassandra-materialized | 1 | 750 | 0.7453 | 0.7093 | 0.7453 | 0.7453 | 0.7453 | 53.6273 | 89.83 | 131.0537 | 6.8563 | 15.3467 | 22.8256 | 17.1352 | 31.2841 | 44.8086 | 53.854 | 94.5083 | 131.1403 |
| cassandra-materialized | 8 | 750 | 0.7453 | 0.12 | 0.6307 | 0.7453 | 0.7453 | 165.3681 | 305.2439 | 366.6946 | 6.5705 | 25.5933 | 78.435 | 14.3735 | 41.845 | 94.1657 | 161.9544 | 304.1205 | 366.7628 |
| cassandra-materialized | 16 | 750 | 0.7453 | 0.0 | 0.084 | 0.5507 | 0.7453 | 428.8289 | 616.2626 | 701.9753 | 10.9428 | 40.8589 | 98.1096 | 24.1369 | 92.056 | 133.9525 | 420.3909 | 597.2728 | 666.5763 |
| cassandra-materialized | 32 | 750 | 0.7453 | 0.0 | 0.0 | 0.0227 | 0.5293 | 916.5822 | 1302.3778 | 1756.6261 | 27.1863 | 93.5517 | 170.3905 | 63.7698 | 175.4484 | 245.2415 | 906.6957 | 1162.9792 | 1560.1937 |
| cassandra-materialized | 64 | 750 | 0.7453 | 0.0 | 0.0 | 0.004 | 0.112 | 1328.6663 | 1961.6387 | 2456.9788 | 100.6062 | 239.8569 | 404.9643 | 230.1014 | 432.6173 | 572.7937 | 1299.4876 | 1789.9373 | 2221.8766 |
| neo4j-materialized | 1 | 750 | 0.7453 | 0.744 | 0.7453 | 0.7453 | 0.7453 | 41.2321 | 48.7643 | 53.4893 | 3.8384 | 5.6676 | 6.4388 | 5.0351 | 7.1132 | 7.7728 | 41.1875 | 48.7044 | 53.0898 |
| neo4j-materialized | 8 | 750 | 0.7453 | 0.0 | 0.096 | 0.7253 | 0.7453 | 335.2709 | 416.8045 | 602.5778 | 4.0419 | 5.5994 | 7.5241 | 6.2572 | 8.5439 | 11.84 | 335.5201 | 420.0652 | 585.034 |
| neo4j-materialized | 16 | 750 | 0.7453 | 0.0 | 0.0 | 0.064 | 0.5147 | 723.4595 | 2583.977 | 2808.3568 | 5.0793 | 31.82 | 36.6221 | 8.9636 | 44.9379 | 60.8092 | 715.3479 | 2591.1258 | 2817.4991 |
| neo4j-materialized | 32 | 750 | 0.7453 | 0.0 | 0.0 | 0.0 | 0.108 | 1358.9375 | 1617.0587 | 1700.9195 | 5.7406 | 9.051 | 37.2317 | 11.8768 | 18.3649 | 57.2386 | 1349.8166 | 1613.1197 | 1705.1028 |
| neo4j-materialized | 64 | 750 | 0.7453 | 0.0 | 0.0 | 0.0 | 0.0 | 2730.6524 | 3286.0775 | 4271.6623 | 9.0743 | 17.0967 | 32.7693 | 20.7359 | 39.6973 | 60.788 | 2707.1783 | 3297.1364 | 4124.7556 |
| neo4j-native | 1 | 750 | 0.7453 | 0.744 | 0.7453 | 0.7453 | 0.7453 | 41.2295 | 49.9274 | 57.5009 | 4.0583 | 5.7567 | 8.1009 | 5.304 | 7.206 | 9.363 | 41.2677 | 49.1841 | 55.0543 |
| neo4j-native | 8 | 750 | 0.7453 | 0.0 | 0.0933 | 0.744 | 0.7453 | 335.5135 | 424.9302 | 451.9901 | 4.1559 | 5.7343 | 6.8293 | 6.3674 | 8.969 | 11.9716 | 334.6468 | 423.0073 | 466.1987 |
| neo4j-native | 16 | 750 | 0.7453 | 0.0 | 0.0 | 0.0693 | 0.52 | 708.0521 | 2585.2134 | 2900.5539 | 5.1673 | 30.6083 | 36.4032 | 8.9924 | 42.1196 | 53.1112 | 706.2215 | 2573.2236 | 2866.4251 |
| neo4j-native | 32 | 750 | 0.7453 | 0.0 | 0.0 | 0.0 | 0.0947 | 1384.678 | 1655.3485 | 1948.1237 | 5.8717 | 9.9084 | 19.771 | 12.4855 | 20.7909 | 39.7356 | 1376.4511 | 1653.2546 | 1926.7594 |
| neo4j-native | 64 | 750 | 0.7453 | 0.0 | 0.0 | 0.0 | 0.0 | 2766.8794 | 3567.3101 | 3968.1638 | 9.4635 | 18.8682 | 37.3335 | 21.8269 | 43.0139 | 81.6017 | 2755.0977 | 3526.362 | 3964.8974 |

## Dominant stage by mean contribution

| Cell | c | Mean bottleneck | Mean ms | Share | Stage p95 | Stage p99 |
|---|---|---|---|---|---|---|
| cassandra-base | 1 | sparse_index_update_ms | 25.1457 | 0.373 | 41.5431 | 95.9364 |
| cassandra-base | 8 | dense_index_update_ms | 75.828 | 0.4234 | 202.8458 | 255.6182 |
| cassandra-base | 16 | dense_index_update_ms | 268.8337 | 0.643 | 451.1849 | 480.9756 |
| cassandra-base | 32 | dense_index_update_ms | 1041.5768 | 0.6713 | 2374.724 | 2800.1907 |
| cassandra-base | 64 | dense_index_update_ms | 736.2713 | 0.5114 | 1126.1783 | 1246.8289 |
| cassandra-materialized | 1 | sparse_index_update_ms | 24.9221 | 0.4248 | 43.8412 | 92.5587 |
| cassandra-materialized | 8 | dense_index_update_ms | 77.9477 | 0.456 | 197.5749 | 266.4386 |
| cassandra-materialized | 16 | dense_index_update_ms | 272.0765 | 0.6663 | 456.9134 | 525.4691 |
| cassandra-materialized | 32 | dense_index_update_ms | 642.2251 | 0.7219 | 874.3558 | 1008.3694 |
| cassandra-materialized | 64 | dense_index_update_ms | 715.2534 | 0.5525 | 1143.8896 | 1373.2099 |
| neo4j-materialized | 1 | sparse_index_update_ms | 19.6443 | 0.4861 | 24.2849 | 26.511 |
| neo4j-materialized | 8 | candidate_fetch_ms | 236.6532 | 0.7161 | 312.7812 | 448.0029 |
| neo4j-materialized | 16 | candidate_fetch_ms | 772.4871 | 0.7576 | 1982.4299 | 2215.2757 |
| neo4j-materialized | 32 | candidate_fetch_ms | 995.6121 | 0.772 | 1261.0355 | 1332.6367 |
| neo4j-materialized | 64 | candidate_fetch_ms | 2043.1406 | 0.7818 | 2611.2202 | 3336.7902 |
| neo4j-native | 1 | sparse_index_update_ms | 19.6708 | 0.4826 | 24.6879 | 29.3647 |
| neo4j-native | 8 | candidate_fetch_ms | 235.5559 | 0.7191 | 315.7112 | 352.7559 |
| neo4j-native | 16 | candidate_fetch_ms | 751.5346 | 0.7596 | 1930.7374 | 2257.553 |
| neo4j-native | 32 | candidate_fetch_ms | 1026.3343 | 0.7747 | 1304.9865 | 1576.43 |
| neo4j-native | 64 | candidate_fetch_ms | 2104.3439 | 0.7822 | 2839.4573 | 3216.2987 |

## Cassandra versus Neo4j at matched design

Values above 1.0 in latency ratios mean Neo4j is slower; values below 1.0 mean Cassandra is slower.

| Pair | c | FreshHit Δ Cass-Neo | Commit p95 Neo/Cass | RawERK p95 Neo/Cass | Attempt p95 Neo/Cass |
|---|---|---|---|---|---|
| base_vs_native | 1 | 0.0 | 0.4342 | 0.1766 | 0.5358 |
| base_vs_native | 8 | 0.0 | 0.2111 | 0.0968 | 1.3412 |
| base_vs_native | 16 | 0.0 | 0.803 | 0.4045 | 4.2819 |
| base_vs_native | 32 | 0.0 | 0.0535 | 0.0338 | 0.4738 |
| base_vs_native | 64 | 0.0 | 0.087 | 0.0754 | 1.8648 |
| materialized | 1 | 0.0 | 0.3693 | 0.2274 | 0.5153 |
| materialized | 8 | 0.0 | 0.2188 | 0.2042 | 1.3812 |
| materialized | 16 | 0.0 | 0.7788 | 0.4882 | 4.3383 |
| materialized | 32 | 0.0 | 0.0967 | 0.1047 | 1.3871 |
| materialized | 64 | 0.0 | 0.0713 | 0.0918 | 1.842 |

## Interpretation constraints

- Do not compare hit-conditional Time-to-Top10 without also reporting FreshHit@10.
- `rawerk_visible_ms`, sparse visibility, and dense visibility are cumulative timestamps, not additive stage durations.
- Backend and materialization effects are separated by the 2×2 design; this is not evidence that Cassandra dominates arbitrary graph traversal.
- Claims are limited to the frozen 100K LoCoMo-shaped, scope-known, 95:5 online memory workload.
