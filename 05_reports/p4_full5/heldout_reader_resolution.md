# Held-Out Reader Provenance Resolution

## Source
- Gate rF1=0.3675: P3 reader_predictions.jsonl (GPT reader script, DeepSeek API)
- Table 1 rF1=0.3896: P4 run_p4_full5.py (custom reader, DeepSeek API)

## Root Cause
Different API runs produced different predictions (236/1150 queries differ).
Both use temperature=0 and same prompt, but DeepSeek outputs vary between runs.
The P3 run had separate timing/network conditions from the P4 run.

## Resolution
- Gate rF1=0.3675 is the canonical frozen value (matches P2/P3 held-out anchor)
- Table 1 held-out rF1=0.3896 is from a fresh API run with different outputs
- The Gate predictions (P3 reader_predictions.jsonl) are the authoritative held-out result
- Table 1 must use the same P3 predictions, not freshly generated ones

## Action
- Replace Table 1 held-out row with P3 reader_predictions.jsonl values
- Or regenerate Table 1 held-out by loading P3 predictions directly
