# kg_edges_by_src Recovery Log

## Action
TRUNCATE kg_edges_by_src, then concurrent import from c1_source_100k.csv

## 100K
- Imported: 100000 edges in 29s
- Concurrent: 64 workers
- CSV logical: 100000  Cass distinct: 100000
- Duplicates: 96  Missing: 0  Extra: 0
- C1 gate: 256 checked, empty=0, mismatch=0

## Deduplication
- 1261 dup groups, 1261 rows deleted
- Raw: 99993 Distinct: 99875 Dup: 118 Missing: 125 Extra: 0
