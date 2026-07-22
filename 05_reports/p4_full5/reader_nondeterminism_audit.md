# Reader Model Non-Determinism Audit

## Setup
- Model: deepseek-chat, temperature=0, max_tokens=128
- Same prompt SHA, same topK, same rendered context
- Two independent API runs: P3 frozen vs. P4 replication

## Result
- Total identical prompts: 1150
- Different predictions: 236
- Non-determinism rate: 0.2052

## Conclusion
The hosted reader model produced different outputs for 236 of 1150 identical prompts
across two independent API runs (both with temperature=0).

Therefore, all formal reader metrics are tied to immutable prompt-hash-keyed prediction artifacts.
The P3 frozen artifact (reader_predictions.jsonl) is the official canonical source.

## Metrics Comparison
| Source | rF1 | rEM | WrongAbst |
|---|---|---|---|
| P3 frozen (official) | 0.3675 | 0.2113 | 0.4383 |
| P4 replication 2 | 0.3896 | 0.2296 | 0.4087 |
