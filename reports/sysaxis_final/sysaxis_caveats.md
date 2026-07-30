# Sysaxis Caveats
Generated: 2026-07-09T13:12:36.214007

1. Cold mode is process-cold (fresh Python process), not strict OS/database cache flush.
2. Application cache is disabled in all system-axis main runs; cache_hit_rate=0 and effective_latency equals measured read latency.
3. 100K scale point is legacy synthetic rebuilt from c1_source_100k.csv. Actual distinct edges=99,875 (125 CSV source duplicates removed). Not scale-controlled.
4. 1M point is scale-controlled with exactly 1,000,000 distinct logical edges.
5. Scale comparison supports qualitative consistency, not strict scaling linearity (different generators).
6. Cassandra writes use denormalized multi-table writes (4 tables); Neo4j writes are lower per-edge latency.
7. Cassandra naive sometimes has higher QPS than Cassandra opt under concurrency, but with worse tail latency.
8. All hop-depth and concurrency experiments are read-only (write_ratio=0). Write-ratio experiments add mixed read/write.
9. Hop-depth uses 8 clients; write-ratio and concurrency use 32 clients (except concurrency client=1/4/8/16/64 variants).
