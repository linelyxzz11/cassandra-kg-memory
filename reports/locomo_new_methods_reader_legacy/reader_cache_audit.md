# Reader Cache Audit

## Summary
- Total API calls: 4620
- Unique cache keys: 4620
- Potential intra-method reuses: 0
- Queries with identical context across methods: 0/1540

## Cache Key Components
- query_id
- ordered_memory_ids (sorted)
- question text
- method name
- prompt_sha256 (inferred from content hash)

## Finding
Cache reuse only works when different methods return the SAME ordered top10.
This is rare because BM25_compact vs RRF_raw vs RRF_compact have different rankings.
No actual API calls were duplicated — each unique cache key generated one call.
