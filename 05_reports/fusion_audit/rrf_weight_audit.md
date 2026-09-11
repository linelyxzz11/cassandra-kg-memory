# RRF Weight Audit

## 1. Equal-weight RRF (alpha=0.5, k=10)

- **Status**: PILOT_BASELINE
- **official**: no | evaluated but not selected
- Dev best MRR (n=390): 0.5121
- Test MRR (n=1150): 0.5201
- Used as a retrieval baseline in an earlier validation stage.
- Superseded by the official ZScore-RawERK method.

## 2. Weighted RRF (alpha=0.6, k=10)

- **Status**: PILOT_BASELINE
- **official**: no | evaluated but not selected
- Dev best MRR: 0.5182 (selected from sweep of alpha=0.0-1.0 x k=10,30,60,100)
- Test MRR: 0.5222
- Outperformed in point estimates by ZScore (0.5426) and MinMax (0.5383).

## 3. KG-Weighted RRF

- **Status**: SUPERSEDED
- Config: 3-way WRRF(BM25+Dense+KG), w_kg∈{0.25,0.5,1.0}, k=60
- KG component later removed because it hurt pure retrieval

## 4. Legacy RRF (mixed-scope)

- **Status**: QUARANTINED
- R@1=0.3539 computed on 1986 queries (incl cat5 adversarial).
- Correct cat1-4 value: 0.3416.
- See: reports/p2_fusion_reader_final/legacy_summary_quarantine.md
