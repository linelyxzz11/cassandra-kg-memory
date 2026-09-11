# Status: SUPERSEDED DIAGNOSTIC

This 36,000-row run passed its original v3 gates, but it is not the final
citation-ready P5-1 result.

The probe fetched all memories currently visible in a shared shard. Concurrent
completion order therefore changed candidate-set size by backend: at c=64 the
mean was 103.594 for Cassandra and 101.254 for Neo4j. The logical event was
equal, and the difference conservatively disadvantaged the faster backend, but
the timed final retrieval did not perform an exactly fixed amount of candidate
work per event.

The replacement protocol uses a frozen cohort: identical baseline memory IDs
plus the current target memory. Its manifest must assert an exact candidate
count for every event. Do not cite the latency values in this directory.
