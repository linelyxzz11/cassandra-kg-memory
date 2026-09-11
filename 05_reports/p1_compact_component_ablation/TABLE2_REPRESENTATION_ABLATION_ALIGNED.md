# Table 2: Representation Ablation

Publication-aligned held-out test scope: LoCoMo Cat1-4, 1,146 questions with
mapped canonical gold memories. Rankings are frozen P1-C outputs; only metric
aggregation was redone against `retrieval_gold_v2`.

| Variant | Representation | MRR@10 | Hit@1 | Hit@5 | Hit@10 | Recall@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Raw | raw memory | 0.5108 | 0.3770 | 0.6885 | 0.7871 | 0.7152 | 0.5409 |
| RawE | raw + entity | 0.5348 | 0.4084 | 0.7068 | 0.8010 | 0.7301 | 0.5626 |
| RawR | raw + relation | 0.5216 | 0.3874 | 0.6972 | 0.7993 | 0.7265 | 0.5516 |
| RawK | raw + keyword | 0.5322 | 0.4049 | 0.7007 | 0.7984 | 0.7282 | 0.5606 |
| RawER | raw + entity + relation | 0.5461 | 0.4180 | 0.7182 | 0.8072 | 0.7344 | 0.5715 |
| RawEK | raw + entity + keyword | 0.5422 | 0.4180 | 0.7138 | 0.8054 | 0.7333 | 0.5686 |
| RawRK | raw + relation + keyword | 0.5419 | 0.4162 | 0.7068 | 0.8063 | 0.7353 | 0.5690 |
| **RawERK (CassMem)** | **raw + entity + relation + keyword** | **0.5498** | **0.4241** | **0.7138** | **0.8124** | **0.7396** | **0.5755** |

## Post-hoc time sensitivity (not the official CassMem representation)

| Variant | Representation | MRR@10 | Hit@1 | Hit@5 | Hit@10 | Recall@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| RawERKT* | raw + entity + relation + keyword + time | 0.5623 | 0.4380 | 0.7251 | 0.8211 | 0.7475 | 0.5858 |

`RawERKT` is a post-hoc sensitivity result and must not be presented as the
official CassMem method. If Table 2 must remain compact, report MRR@10 and
Hit@10 only. The historical P1-C label `R@10` was binary hit rate and should be
renamed `Hit@10`; true multi-gold `Recall@10` is reported separately above.
