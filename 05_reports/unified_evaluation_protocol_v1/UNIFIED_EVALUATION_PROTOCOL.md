# CassMem Unified Evaluation Protocol V1

## Frozen experiment contract

| Field | Value |
|---|---|
| Dataset | LoCoMo, 10 conversations, 5,882 memories, 1,986 QA |
| Retrieval scope | Cat1-4=1,540; 1,536 evidence-bearing; held-out primary=1,146 |
| Reader scope | Full5=1,986; Cat5=446 |
| Retrieval methods | BM25, Dense-bge, corrected Dense+GlobalKG, RRF_compact, ZScore-Raw, ZScore-RawERK |
| Reader | GPT-4o-2024-08-06, temperature=0, max_tokens=64 |
| Context | Top-10 frozen memory IDs |
| Embedding | Frozen bge-large 1024d artifacts, identified by SHA-256 |
| Retrieval metrics | MRR@10, Hit@1/5/10, Recall@10, nDCG@10 |
| Answer metrics | LoCoMo F1, Full5 BLEU-1, GPT-4o Judge |

## Non-negotiable rules

1. Cat5 is excluded from retrieval effectiveness and evaluated as answer-level abstention.
2. Cat5 `(a)/(b)` is restored to option text before F1, BLEU-1, or Judge.
3. Overall means the per-query micro mean over the named scope, never an unweighted mean of category means.
4. Raw prediction caches are immutable. Offline normalization is versioned separately.
5. API regeneration is required only when the actual prompt/context changes and no identical prompt cache exists.
6. Every paper number must point to a manifest and input hashes.

## Cache registry

See `prediction_cache_registry.csv` for all 12 Reader caches and SHA-256 values.

## Machine-readable protocol

See `protocol_manifest.json`.
