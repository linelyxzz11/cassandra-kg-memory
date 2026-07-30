# Legacy Metric Definitions (from reader_f1_memory_only_v2.py)

## Model
- deepseek-chat, temperature=0, max_tokens=128, timeout=60

## Prompt
Answer the question using only the evidence below.
If the evidence does not contain the answer, respond exactly with 'Cannot answer'.
Return only the shortest answer. Do not explain.

Evidence:
[N] memory_id=X | sample=X | session=N | turn=D:N | time=... | speaker=X
Text: ...

Question: ...
Answer:

## Normalization
- Strict: lowercase, remove punctuation, remove a/an/the, collapse whitespace
- Relaxed: strict + map number words to digits (one->1, two->2, etc.)

## Metrics
- rEM: relaxed normalization exact match (pred == gold after relaxed norm)
- rF1: token-level F1 after relaxed normalization (Counter intersection)
- is_abstain: pred contains "cannot answer" / "not enough information" / etc.
- WrongAbst: is_abstain=1 but gold answer exists (non-adversarial)
  i.e., model refused to answer when answer WAS in evidence

## Hit@10
- 1 if any retrieved memory_id is in the gold evidence set
