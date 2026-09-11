# Backend System Evaluation Status

This directory is **diagnostic and not citation-ready**.

- `update_visibility_results.csv` contains `-1` sentinel values for first-visible time and latency.
- `recovery_results.csv` contains `-1` restart times and `final_searchable_ratio=0`.
- `throughput_results.csv` contains QPS observations but its latency-under-load fields are `-1`.
- These files must not be used to support update visibility, recovery, or latency-under-load claims.

Use the frozen P5 artifacts under `05_reports/p5_minimal_core/` for currently validated update/recovery evidence. A replacement P7-B run must include a run manifest, environment/backend versions, timestamp provenance, successful gates, and end-to-end retrieval component timings.
