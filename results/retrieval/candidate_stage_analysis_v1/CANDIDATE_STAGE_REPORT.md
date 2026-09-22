# Candidate-stage recall and ranking-loss decomposition

## Protocol

This evaluates the actual frozen candidate stage used by each method. It does not claim a graph-expansion stage. Single-branch methods use their frozen Top-100 diagnostic pool; fusion methods use the union of their frozen Top-50 branches. Retrieval metrics exclude the four evidence-empty Cat3 questions.

## Held-out test results

| Method | Candidate definition | n | Avg pool | Pool reduction | Candidate Hit | Candidate Recall | Top-10 Hit | Top-10 Recall | Ranking loss | Retention |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | full_rank_top100 | 1146 | 100.0 | 0.821 | 0.756 | 0.684 | 0.552 | 0.500 | 0.184 | 0.731 |
| Dense-bge | full_rank_top100 | 1146 | 100.0 | 0.821 | 0.954 | 0.921 | 0.746 | 0.679 | 0.242 | 0.737 |
| Dense+GlobalKG | full_rank_top100 | 1146 | 100.0 | 0.821 | 0.962 | 0.930 | 0.777 | 0.706 | 0.224 | 0.759 |
| RRF_compact | fusion_input_union | 1146 | 55.1 | 0.902 | 0.942 | 0.896 | 0.797 | 0.724 | 0.172 | 0.808 |
| ZScore-Raw | fusion_input_union | 1146 | 87.3 | 0.845 | 0.942 | 0.899 | 0.787 | 0.715 | 0.184 | 0.795 |
| ZScore-RawERK | fusion_input_union | 1146 | 86.2 | 0.846 | 0.952 | 0.911 | 0.812 | 0.740 | 0.171 | 0.812 |

## Interpretation

- CassMem reduces the conversation-scoped pool by 84.6% on average before final ranking.
- Its candidate-stage macro Recall is 0.911; final Top-10 Recall is 0.740. The 0.171 absolute gap is attributable to ranking/truncation after candidate generation.
- Candidate Recall and final Top-10 Recall answer different questions and must not be labeled interchangeably.
- This table supports the existing Dense/BM25 fusion pipeline. It does not support a claim that Cassandra performs arbitrary KG expansion.
