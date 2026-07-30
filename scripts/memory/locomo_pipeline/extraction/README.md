# locomo_pipeline/extraction/

## Purpose
KG triple extraction from LoCoMo conversation observations, and data import into Cassandra/Neo4j.

## Status
PAPER_EVIDENCE — These scripts produce the KG that feeds retrieval and evaluation.

## Important Scripts
- `observation_to_kg_spacy.py` — spaCy-based KG triple extraction
- `locomo_to_memory_records.py` — Convert locomo10.json to memory_records.csv
- `locomo_cassandra_import.py` — Import into Cassandra
- `locomo_neo4j_import.py` — Import into Neo4j
- `inspect_locomo.py` — Inspect locomo10.json structure
- `inspect_locomo_categories.py` — Inspect QA categories
- `inspect_locomo_conversation.py` — Inspect single conversation

## Safety
- `locomo_cassandra_import.py`, `locomo_neo4j_import.py`: Do not run without explicit confirmation. Connects to Cassandra/Neo4j and may modify data.
- `locomo_to_memory_records.py`: Safe to run. No DB connection.
- Inspection scripts: Safe to run. Read-only.

## Related Reports
- `results/locomo_memory_records.csv`
- `results/locomo_kg_edges_spacy.csv`
- `results/locomo_kg_edges_extended.csv`
