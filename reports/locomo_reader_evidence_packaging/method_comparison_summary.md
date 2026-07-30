# Reader Evidence Packaging — Pilot Results

## Overall (80 queries, 20 per cat1-4, Dense+GlobalKG top5)
- text_only: mean F1=0.1735, EM=0.0125, abstain=50.0%
- text_plus_triples: mean F1=0.1625, EM=0.0125, abstain=53.8%
- Delta F1: -0.0110 (-1.1%)

## Verdict
- text+triples HURTS reader. KG noise may confuse the LLM.

## Next step
- If delta positive, run full 1540 queries
- If delta neutral/negative, stop evidence packaging direction

## Runtime
- 214.0s
