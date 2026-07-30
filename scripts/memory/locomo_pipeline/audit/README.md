# locomo_pipeline/audit/

## Purpose
Cross-check and audit retrieval quality: KG rescue/hurt analysis, retrieval scope audit, Cassandra-KG equivalence verification, and category-wise breakdown.

## Status
PAPER_EVIDENCE — Produces audit evidence for paper claims.

## Important Scripts
- `audit_cassandra_kg_equivalence.py` — Cassandra online vs offline KG signal equivalence (Layer A)
- `audit_dense_kg_rescue.py` — Per-query rescue/hurt analysis: which queries benefit/suffer from KG signal
- `audit_e2_backend_fair.py` — End-to-end backend fairness: Cassandra vs Neo4j retrieval comparison
- `audit_kg_boost_top10.py` — KG boost top-10 audit: detailed per-query analysis
- `analysis_sample_scoped.py` — Sample-scoped analysis pipeline: rescue/hurt, category-wise, KG coverage audit
- `retrieval_scope_audit.py` — Retrieval scope audit: cross-sample rate analysis

## Safety
- Local audit scripts: Safe to run. Read CSV results only.
- `audit_cassandra_kg_equivalence.py`: Do not run without confirmation. Connects to Cassandra.

## Related Reports
- `results/audit/cassandra_online_vs_offline_kg_signal_audit.csv`
- `results/audit_dense_kg_rescue_results.csv`
- `results/e2_backend_fair_*.csv`
- `results/sample_scoped/analysis_rescue_hurt.csv`
- `results/sample_scoped/analysis_category_wise.csv`
- `results/sample_scoped/analysis_kg_coverage.csv`
- `results/final/retrieval_scope_audit_global.csv`
