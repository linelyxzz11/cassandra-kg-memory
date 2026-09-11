# Table 5: Question Type Breakdown

## Panel A: Reader F1 by question type

GPT-4o Reader, both prompt settings, all 1,986 LoCoMo questions. Values are F1
points. The five selected methods match the analysis protocol fixed before
inspection of the category results.

| Method | Cat. | Single-Hop (841) | Multi-Hop (282) | Temporal (321) | Open-Domain (96) | Adversarial (446) | Overall (1,986) |
|---|:---:|---:|---:|---:|---:|---:|---:|
| BM25 | ✗ | 45.7 | 19.6 | 26.2 | 17.6 | 9.4 | 29.3 |
| + Cat. format | ✓ | 45.7 | 20.2 | 36.5 | 19.9 | 87.9 | 48.8 |
| Dense-bge | ✗ | 56.3 | 33.3 | 31.2 | 19.4 | 7.8 | 36.3 |
| + Cat. format | ✓ | 56.4 | 33.3 | 43.9 | 21.0 | 87.2 | 56.3 |
| RRF_compact | ✗ | 60.3 | 35.5 | 28.9 | 21.3 | 10.1 | 38.6 |
| + Cat. format | ✓ | 60.3 | 35.0 | 43.6 | 22.4 | 86.8 | 58.1 |
| ZScore-Raw | ✗ | 58.8 | 35.9 | 30.4 | 24.4 | 7.8 | 37.8 |
| + Cat. format | ✓ | 58.5 | 34.2 | 43.0 | 20.0 | 87.0 | 57.1 |
| **CassMem (ZScore-RawERK)** | **✗** | **60.1** | **37.0** | **30.6** | **23.5** | **8.1** | **38.6** |
| **+ Cat. format** | **✓** | **60.3** | **36.6** | **45.3** | **25.6** | **85.9** | **58.6** |

## Panel B: Retrieval MRR@10 by question type

Publication held-out test scope, Cat1-4 only, 1,146 mapped-evidence questions.
Cat5 is excluded because it has no positive retrieval target.

| Method | Single-Hop (612) | Multi-Hop (224) | Temporal (239) | Open-Domain (71) | Overall (1,146) |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.3981 | 0.1747 | 0.4232 | 0.1847 | 0.3465 |
| Dense-bge | 0.5103 | 0.4244 | 0.5896 | 0.2757 | 0.4955 |
| RRF_compact | 0.5433 | 0.4365 | 0.6268 | 0.2755 | 0.5232 |
| ZScore-Raw | 0.5371 | 0.4150 | 0.5940 | 0.3060 | 0.5108 |
| **CassMem (ZScore-RawERK)** | **0.5784** | **0.4495** | **0.6417** | **0.3111** | **0.5498** |

## Defensible interpretation

- Without Cat. format, the cleaner ZScore-Raw to CassMem representation
  contrast changes Reader F1 by +1.3 Single-Hop, +1.2 Multi-Hop, +0.2
  Temporal, -0.9 Open-Domain, and +0.2 Adversarial points; Overall improves by
  +0.8 points.
- With Cat. format, ZScore-Raw to CassMem changes Reader F1 by +1.8 Single-Hop,
  +2.4 Multi-Hop, +2.3 Temporal, +5.6 Open-Domain, and -1.1 Adversarial
  points; Overall improves by +1.5 points.
- Retrieval MRR improves over ZScore-Raw in every Cat1-4 category. The largest
  gain is Temporal (+0.0477), followed by Single-Hop (+0.0413), Multi-Hop
  (+0.0345), and Open-Domain (+0.0051) on the held-out split.
- These results do not support the narrow claim that ERK mainly helps
  multi-hop reasoning. The evidence supports broader gains across question
  types, with the largest Reader point gain on Open-Domain under Cat. format.
