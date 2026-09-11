# Reader LLM-Judge protocol decision (v3)

## Decision

Answers and primary judgments use `gpt-4o-2024-08-06` with the Mem0/HingeMem binary semantic-correctness prompt. This is the main-table protocol because HingeMem states that all experiments use GPT-4o unless otherwise specified and that its judge follows the Mem0 prompt. The exact snapshot is frozen and disclosed for reproducibility.

The corrected Cat5 evaluation resumes from the existing cache. Category scores are arithmetic means within each category. Overall J is the micro mean over all 1,986 LoCoMo questions, including Cat5, matching the scope and weighting implied by the HingeMem main table rather than an unweighted mean of five category means.

## Robustness checks

1. Optionally run an independent model-family judge on a fixed, category- and method-stratified subset.
2. Double-label a fixed stratified human subset and report human--GPT-4o agreement, cross-judge agreement, and Cohen's kappa.
3. Keep all six local methods blind and identically formatted during judging. Cache by question, reference answer, candidate answer, judge model snapshot, prompt hash, and trial.

## LoCoMo category handling

- Cat1 = Multi-Hop
- Cat2 = Temporal
- Cat3 = Open-Domain
- Cat4 = Single-Hop
- Cat5 = Adversarial

Cat5 reference semantics are `Not mentioned in the conversation`. For Cat-format predictions, deterministic `(a)/(b)` outputs must first be restored to their option text using the Reader run's recorded option ordering. Cat5 is never scored against the plausible adversarial distractor answer.

## Comparability rule

Published J values are labeled `reported`; HingeMem discloses GPT-4o but not its exact model snapshot. Local methods are compared under the same frozen GPT-4o snapshot and prompt. Cross-paper claims acknowledge the unresolved snapshot difference.

## Literature basis

- Mem0 reports a binary LLM judge, ten independent runs, and mean plus/minus one standard deviation; its paper and released configuration use GPT-4o-mini extensively.
- HingeMem follows the Mem0 judge prompt and states that experiments use GPT-4o unless otherwise specified, so it is a useful table-format reference but not an independent-judge precedent.
- MRAgent uses Gemini-2.5-Flash and Claude-Sonnet-4.5 answer backbones with GPT-4o-mini as the judge, temperature 0, and repeated trials.
- LoCoMo-Plus uses Gemini-2.5-Flash as judge and validates it against two human annotators and a GPT-4o cross-judge.
- APEX-MEM uses GPT-5 as judge across Claude and GPT answer agents and reports three trials.
