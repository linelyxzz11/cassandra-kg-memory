# Hit@10 Audit
## Root Cause
Old reader_f1_memory_only_v2.py computes Hit@10 with full alias resolution:
1. Resolves each raw memory_id through alias_to_mid (evidence_map + memory_aliases_from_row)
2. Builds gold_set from evidence_map with same alias resolution
3. Checks intersection: gold_set & resolved_top10_set

Our bug: simple string match without alias resolution.
For BM25/Dense which use pre-computed IDs that include alias forms, Hit@10 can differ from R@10.
For new methods which use canonical full memory IDs, Hit@10 ≈ R@10 (no alias gap).

## Corrected Hit@10
- BM25: 0.5487
- Dense-bge: 0.7299
- Dense+GlobalKG: 0.7338
- Dense+QueryKG: 0.7494
- BM25_compact: 0.6734
- RRF_raw: 0.7409
- RRF_compact: 0.7857
