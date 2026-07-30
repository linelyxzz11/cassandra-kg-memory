# LoCoMo Retrieval: New Directions from A-MEM/Mem0/MemORAI

## Experiment 1: A-MEM-lite Enriched Memory
- BM25_raw R@10=0.5883 MRR=0.3830
- BM25_enriched R@10=0.6747 MRR=0.4939
- Delta: +0.1109 (+11.1%) 
- Verdict: Enriched memory helps BM25. Worth exploring Dense_enriched next.

## Experiment 2: Reader Evidence Packaging
- Status: Reader LLM API not available in this run.
- Format: text + KG triples + provenance generated for 20 examples.
- Saved to: reports/locomo_reader_evidence_packaging/reader_packaging_examples.jsonl
- Verdict: Need API key to run reader. Format is ready.

## Experiment 3: Selective Memory Filtering
- Best selective: Dense_selective_top50pct R@10=0.7110 MRR=0.4878
- Selective pools retain different amounts based on percentile.
- Verdict: Selective filtering maintains quality while reducing pool size. Worth developing further.

## Recommended Next Direction
1. A-MEM-lite: Continue with Dense_enriched if API available
2. Reader evidence: Build and test with API key
3. Selective memory: Develop salience index as lightweight noise reducer

## Runtime
- 37.9s
