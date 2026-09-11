# Canonical live gate validation

Status: **PASS**

- Four live cells each contain exactly 5,882 canonical memories.
- 23,528 canonical projection SHA-256 comparisons completed with zero mismatch.
- 1,533 unambiguous Cat1-4 questions produced 6,132 isolated freshness events (four cells per question).
- Projection visibility: 1,533/1,533 PASS in every cell.
- Scope candidate parity: 1,533/1,533 PASS in every cell.
- Cross-cell fused Top-10 mismatch queries: 0/1,533.
- Cross-cell designated-gold rank mismatch queries: 0/1,533.
- Designated fresh-gold Hit@10: 0.738421 in every cell.

The latency values are canonical correctness-gate measurements at isolated per-question operation, not the final 100K mixed-workload paper result. They include synchronous backend commit, projection read, scoped sparse/dense index update, backend candidate enumeration and real ZScore fusion; they exclude LLM extraction and embedding generation.
