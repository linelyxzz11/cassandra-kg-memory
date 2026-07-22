import os
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

memory_csv = r'.\results\locomo_memory_records.csv'
qa_csv = r'.\results\locomo_qa_records.csv'
evidence_csv = r'.\results\locomo_evidence_map.csv'
out_csv = r'.\results\locomo_retrieval_tfidf_results.csv'

if not all(os.path.exists(f) for f in [memory_csv, qa_csv, evidence_csv]):
    raise FileNotFoundError('One or more CSV files not found in results/')

memory_df = pd.read_csv(memory_csv).fillna('')
qa_df = pd.read_csv(qa_csv).fillna('')
evidence_df = pd.read_csv(evidence_csv).fillna('')

memory_ids = memory_df['memory_id'].astype(str).tolist()
memory_texts = memory_df['text'].astype(str).tolist()

evidence_map = defaultdict(set)
for _, row in evidence_df.iterrows():
    evidence_map[str(row['qa_id'])].add(str(row['memory_id']))

vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), max_features=50000)
memory_matrix = vectorizer.fit_transform(memory_texts)

results = []

for _, qa in qa_df.iterrows():
    qa_id = str(qa['qa_id'])
    question = str(qa['question'])
    category = qa.get('category', '')

    query_vec = vectorizer.transform([question])
    scores = cosine_similarity(query_vec, memory_matrix).flatten()
    ranked_idx = scores.argsort()[::-1]

    gold_ids = evidence_map.get(qa_id, set())
    top1 = [memory_ids[i] for i in ranked_idx[:1]]
    top5 = [memory_ids[i] for i in ranked_idx[:5]]
    top10 = [memory_ids[i] for i in ranked_idx[:10]]

    def recall_at_k(topk):
        return int(any(mid in gold_ids for mid in topk))
    def reciprocal_rank(topk):
        for rank, mid in enumerate(topk, 1):
            if mid in gold_ids:
                return 1.0/rank
        return 0.0

    results.append({
        'qa_id': qa_id,
        'category': category,
        'question': question,
        'gold_memory_ids': ';'.join(sorted(gold_ids)),
        'top1_memory_ids': ';'.join(top1),
        'top5_memory_ids': ';'.join(top5),
        'top10_memory_ids': ';'.join(top10),
        'hit1': recall_at_k(top1),
        'hit5': recall_at_k(top5),
        'hit10': recall_at_k(top10),
        'rr': reciprocal_rank(top10)
    })

results_df = pd.DataFrame(results)
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')

print('LoCoMo TF-IDF retrieval completed.')
print(f'QA count: {len(results_df)}')
print(f'Recall@1: {results_df["hit1"].mean():.4f}')
print(f'Recall@5: {results_df["hit5"].mean():.4f}')
print(f'Recall@10: {results_df["hit10"].mean():.4f}')
print(f'MRR@10: {results_df["rr"].mean():.4f}')

print('\nCategory-level metrics:')
for category, group in results_df.groupby('category'):
    print(f'category {category}: n={len(group)}, R@1={group["hit1"].mean():.4f}, R@5={group["hit5"].mean():.4f}, R@10={group["hit10"].mean():.4f}, MRR@10={group["rr"].mean():.4f}')
print(f'Results saved to: {out_csv}')