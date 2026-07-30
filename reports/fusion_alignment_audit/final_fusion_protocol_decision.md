# Final Fusion Protocol Decision

## Verdict: Current P2 plumbing IS the canonical protocol.

### Evidence
1. Legacy and P2 RRF use identical Python code for fusion.
2. The 0.3539 value in `sample_scoped_retrieval_summary.csv` was computed on
   ALL 1986 queries, not cat1-4.
3. When recomputed on cat1-4, both legacy and P2 give the SAME result.
4. The only difference is the query set (1540 vs 1986).

### Decision: Situation A — legacy protocol confirmed
The legacy RRF implementation and current P2 plumbing are the SAME.
No code fix needed.
The canonical cat1-4 RRF_raw anchor IS 0.3539 on 1986 queries,
and 0.3416 on cat1-4.

### Action
Since the code is identical:
- P2 minmax/zscore results are valid as-is
- Gate anchor for RRF_raw cat1-4 should use 0.3416
- All score fusion methods use same candidate plumbing → no sensitivity issue
- Proceed to reader on held-out test queries

### Alignment Check
- all_pass: True
- protocol_frozen: True
- reader_gate: OPEN
