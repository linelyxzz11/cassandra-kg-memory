# HingeMem-Style Reader Main Table

This directory contains the publication-facing Reader main table. It combines
the six local retrieval methods with selected external rows quoted from
HingeMem Table 1.

## Current metric source

The local F1 and B1 values are rebuilt from the original frozen GPT-4o Reader
predictions by `reader_offline_metrics_v4`. No Reader API was called again.
Before scoring, Cat5 `(a)/(b)` outputs are deterministically restored to their
option text. This fixes the former offline aggregation error without changing
any model prediction.

- Overall F1: official LoCoMo category-aware score, micro mean over all 1,986
  Full5 questions.
- Overall B1: lowercase NLTK `word_tokenize` + sentence BLEU-1, weights
  `(1,0,0,0)`, method1 smoothing, over the same 1,986 questions.
- J: frozen GPT-4o Mem0/HingeMem-style judge result. Dense+GlobalKG uses its
  corrected ranking-specific judge cache.

The former `dual_setting_main_table.csv` and Dense+GlobalKG special-case BLEU
files are historical inputs and are superseded for this table.

## Category mapping

- Single-Hop = LoCoMo Cat4
- Multi-Hop = LoCoMo Cat1
- Temporal = LoCoMo Cat2
- Open-Domain = LoCoMo Cat3
- Adversarial = LoCoMo Cat5

## Layout policy

- Headers and metric grouping follow HingeMem Table 1.
- Each method's Cat.✗ and Cat.✓ rows are adjacent; the method name appears once.
- External and local methods are interleaved by retrieval/representation
  similarity rather than separated into artificial “classic” and “local” blocks.
- Only CassMem is bold. No result-cell coloring is used.
- CassMem is the final method.

## Outputs

- `reader_main_data.csv`: exact source data for the rendered table.
- `reader_main_hingemem_style_with_j.png`: corrected publication-facing table.
- `manifest.json`: source paths, protocol, status, and output SHA-256 values.
