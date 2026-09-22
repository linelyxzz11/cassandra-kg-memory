# Table 4: Reader QA Quality

Local results use the frozen GPT-4o reader predictions on all 1,986 LoCoMo
questions. Cat5 option labels are restored to their deterministic option text
before F1 and BLEU-1 scoring. External rows are quoted from HingeMem Table 1.

| Method | Cat. | Single-Hop F1 | J | Multi-Hop F1 | J | Temporal F1 | J | Open-Domain F1 | J | Adversarial F1 | J | Overall F1 | J | B1 |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LOCOMO | ✓ | 12.7 | 16.5 | 19.7 | 20.9 | 10.4 | 11.5 | 20.1 | 30.2 | 66.8 | 90.1 | 25.8 | 33.5 | 0.132 |
| RAG (Top-10) | ✓ | 36.4 | 50.7 | 29.6 | 38.6 | 27.7 | 22.7 | 21.2 | 37.5 | 86.8 | 86.5 | 44.6 | 51.9 | 0.306 |
| BM25 | ✗ | 45.7 | 58.4 | 19.6 | 28.7 | 26.2 | 42.7 | 17.6 | 27.1 | 9.4 | 41.9 | 29.3 | 46.4 | 0.241 |
| + Cat. format | ✓ | 45.7 | 58.1 | 20.2 | 29.4 | 36.5 | 50.5 | 19.9 | 33.3 | 87.9 | 87.9 | 48.8 | 58.3 | 0.429 |
| Dense-bge | ✗ | 56.3 | 72.7 | 33.3 | 53.2 | 31.2 | 52.6 | 19.4 | 42.7 | 7.8 | 38.8 | 36.3 | 57.6 | 0.299 |
| + Cat. format | ✓ | 56.4 | 73.1 | 33.3 | 54.3 | 43.9 | 63.9 | 21.0 | 42.7 | 87.2 | 87.2 | 56.3 | 70.6 | 0.493 |
| RRF_compact | ✗ | 60.3 | 77.3 | 35.5 | 55.7 | 28.9 | 54.2 | 21.3 | 40.6 | 10.1 | 41.5 | 38.6 | 60.7 | 0.315 |
| + Cat. format | ✓ | 60.3 | 77.5 | 35.0 | 56.0 | 43.6 | 65.7 | 22.4 | 45.8 | 86.8 | 86.8 | 58.1 | 73.1 | 0.507 |
| Mem0 | ✗ | 45.1 | 56.7 | 42.7 | 48.2 | 49.7 | 50.1 | 27.7 | 47.2 | 6.5 | 56.6 | 36.0 | 53.7 | 0.254 |
| + Cat. format | ✓ | 44.0 | 59.0 | 38.6 | 47.8 | 45.6 | 47.6 | 20.8 | 42.0 | 84.3 | 72.7 | 51.4 | 59.6 | 0.351 |
| Dense+GlobalKG | ✗ | 58.0 | 76.1 | 36.4 | 51.8 | 30.7 | 53.6 | 21.5 | 41.7 | 8.7 | 38.8 | 37.7 | 59.0 | 0.306 |
| + Cat. format | ✓ | 57.2 | 74.1 | 37.2 | 54.3 | 41.5 | 61.1 | 24.1 | 44.8 | 87.9 | 87.9 | 57.1 | 70.8 | 0.499 |
| Mem0g | ✗ | 45.3 | 55.1 | 40.4 | 48.5 | 47.9 | 52.0 | 28.4 | 41.0 | 6.7 | 54.1 | 35.5 | 51.7 | 0.258 |
| + Cat. format | ✓ | 43.4 | 62.0 | 38.3 | 45.7 | 44.4 | 48.5 | 23.4 | 44.1 | 82.7 | 68.5 | 50.7 | 59.7 | 0.348 |
| HippoRAG2 | ✗ | 54.4 | 78.5 | 35.4 | 52.8 | 55.5 | 61.0 | 23.4 | 35.2 | 4.3 | 69.5 | 39.1 | 68.5 | 0.289 |
| + Cat. format | ✓ | 59.2 | 75.5 | 38.0 | 46.4 | 44.9 | 66.6 | 21.2 | 35.4 | 87.7 | 87.2 | 58.4 | 70.6 | 0.396 |
| HingeMem | ✗ | 61.1 | 78.8 | 53.6 | 62.8 | 57.4 | 66.9 | 30.7 | 46.4 | 87.4 | 87.8 | 63.9 | 75.1 | 0.404 |
| ZScore-Raw | ✗ | 58.8 | 75.5 | 35.9 | 52.5 | 30.4 | 53.9 | 24.4 | 46.9 | 7.8 | 39.2 | 37.8 | 59.2 | 0.307 |
| + Cat. format | ✓ | 58.5 | 75.6 | 34.2 | 52.5 | 43.0 | 65.7 | 20.0 | 41.7 | 87.0 | 87.0 | 57.1 | 71.7 | 0.494 |
| **CassMem (Ours)** | **✗** | **60.1** | **77.1** | **37.0** | **55.7** | **30.6** | **55.5** | **23.5** | **40.6** | **8.1** | **39.9** | **38.6** | **60.4** | **0.313** |
| **+ Cat. format** | **✓** | **60.3** | **77.3** | **36.6** | **56.0** | **45.3** | **67.6** | **25.6** | **45.8** | **85.9** | **85.9** | **58.6** | **73.1** | **0.510** |

Category mapping: Single-Hop=Cat4, Multi-Hop=Cat1, Temporal=Cat2,
Open-Domain=Cat3, and Adversarial=Cat5. Overall is the micro mean over all
1,986 questions. `B1` is pure lexical Full5 BLEU-1, not the historical
BLEU-1/Cat5-accuracy hybrid.

External rows are reported results from HingeMem Table 1, not local reruns.
HingeMem reports the same 1,986-question LoCoMo scope, GPT-4o answering,
category-wise LoCoMo F1, and the Mem0 judge prompt, but uses
`text-embedding-3-small`; the exact GPT-4o snapshot and BLEU tokenizer/smoothing
are not fully specified. Therefore external F1/J rows are contextual
comparators, while controlled claims, paired significance tests, and ablations
must use the local rows only. Bold identifies CassMem rather than the global
best value.
