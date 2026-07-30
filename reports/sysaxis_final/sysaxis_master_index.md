# Sysaxis Master Index
Generated: 2026-07-09T13:12:36.214007

## 1. 1M Write-Ratio Sweep
- Dir: reports/sysaxis_1m_write_ratio_final/
- Graph: c3_scale_1M_seed42 (1M edges, scale-controlled)
- Systems: cassandra, neo4j
- Config: clients=32, hop=2, write_ratio=0/10/30%, modes=cold/warm, repeats=5
- Trials: 60 (2x3x2x5)
- Correctness: PASS (guard 1M/1M, hash gates pass, failures empty)
- Summary: final_write_ratio_summary_cold_warm.csv

## 2. 1M Concurrency Sweep
- Dir: reports/sysaxis_1m_concurrency_final/
- Graph: c3_scale_1M_seed42 (1M edges, scale-controlled)
- Systems: cassandra_opt, cassandra_naive, neo4j
- Config: hop=2, clients=1/4/8/16/32/64, modes=cold/warm, repeats=3
- Trials: 108
- Correctness: PASS (hash gates, semantic gates, failures empty)
- Summary: final_concurrency_summary_cold_warm.csv

## 3. 1M Hop-Depth Sweep
- Dir: reports/sysaxis_1m_hop_depth_final/
- Graph: c3_scale_1M_seed42 (1M edges, scale-controlled)
- Systems: cassandra_opt, neo4j
- Config: clients=8, hop=1/2/3/4, modes=cold/warm, repeats=5
- Trials: 80 (2x4x2x5)
- Correctness: PASS (80/80 spotchecks, 16/16 hash gates, guard 1M/1M, failures empty)
- Summary: final_hop_depth_summary_cold_warm.csv

## 4. Supporting Scale: 100K -> 1M
- Dir: reports/sysaxis_scale_sweep_final/
- Graphs: sysaxis_100K_legacy_clean_20260709 (99,875 edges) + c3_scale_1M_seed42 (1M)
- Systems: cassandra_opt/neo4j (100K); cassandra/neo4j (1M)
- Config: clients=32, hop=2, write_ratio=10%, modes=cold/warm
- Trials: 20 (100K) + 12 (1M wr=10% subset)
- Correctness: PASS_WITH_CAVEAT (100K rebuilt from legacy c1_source_100k.csv)
- Summary: scale_sweep_supporting_100K_to_1M_cold_warm.csv
