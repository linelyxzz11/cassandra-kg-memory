# Literature Evidence for Evaluation Design

`metric_protocol_matrix.csv` separates peer-reviewed evidence from preprints and
records comparability limitations. It is a protocol guide, not a table of
numbers to paste into the CassMem main table.

## Decisions Supported by the Literature

- Keep retrieval and answer quality in separate tables.
- Report LoCoMo answer results by category and overall.
- Use F1 and BLEU-1 for answer overlap, but keep Cat5's adversarial abstention
  semantics explicit.
- Use MRR/Hit/Recall for retrieval ranking, with paired uncertainty where
  per-query outputs exist.
- Report query latency and token/serving cost separately from response quality.
- For system claims, use repeated runs and disclose the retrieval depth,
  backend/model versions, and candidate universe.

## CassMem-Specific Consequences

1. The retrieval main table remains Cat1–Cat4 only.
2. Cat5 is evaluated only at the reader layer.
3. `Recall@10` cannot be directly compared with a paper's top-15 recall.
4. HingeMem and ConvMemory v2 are useful protocol references but are preprints;
   they must not be described as peer-reviewed A/B-conference evidence.
5. External numbers remain quoted results until reproduced under CassMem's
   frozen query universe, model, prompt, and retrieval depth.
