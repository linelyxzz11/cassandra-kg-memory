# Final Negative Results

## Confirmed Negative
- **GlobalKG++ structured prior**: negligible gain over binary has_KG flag
- **KG candidate expansion**: only +0.9pp ceiling at top200 on top of Dense
- **KG-Native retriever**: R@10=36.17%, MRR=0.2216, far below Dense
- **Raw triples in reader prompt**: F1 -0.011 over text-only
- **Raw triples in BM25 representation**: marginal/neutral contribution
- **Selective memory filtering**: truth retention only 55%, too aggressive

## Not Negative (Retained)
- **KG-enriched memory representation**: +0.10 MRR on BM25, primary KG contribution
- **GlobalKG-Prior (binary)**: +0.0028 MRR over Dense_raw, minor but consistent
- **Salience scheduling**: soft rerank, positive at low lambda
