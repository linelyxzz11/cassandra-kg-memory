# P1-C Component Audit

- RawERK parity gate: **PASS**
- Held-out queries: 1150
- Canonical Cat1–4 queries: 1540
- Reader/API calls: **none**
- Dense cache: frozen
- Fusion: 0.6 × z(Dense) + 0.4 × z(BM25)

## Leave-one-out definition

- Full: RawERK
- w/o Entity: RawRK
- w/o Relation: RawEK
- w/o Keyword: RawER
- Time sensitivity: RawERKT − RawERK
