import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_csv_rows(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_gold_evidence(file_path):
    mapping = defaultdict(set)
    with Path(file_path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["qa_id"]].add(row["evidence_id"])
    return mapping


def build_memory_index(memory_file):
    memory_texts = []
    memory_sample = []
    memory_evidence = []

    with Path(memory_file).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mid = row["memory_id"]
            ev_match = re.search(r"D\d+:\d+$", mid)
            if not ev_match:
                continue
            memory_texts.append(row["text"])
            memory_sample.append(row["sample_id"])
            memory_evidence.append(ev_match.group())

    return memory_texts, memory_sample, memory_evidence


def build_neo4j_kg_set(driver):
    kg_set = set()
    with driver.session() as session:
        result = session.run(
            "MATCH ()-[r:KG_EDGE]->() RETURN r.graph_id AS graph_id, r.source AS source"
        )
        for record in result:
            source_str = str(record["source"]) if record["source"] else ""
            parts = source_str.split("|")
            evidence = parts[-1] if len(parts) >= 2 else ""
            if evidence:
                kg_set.add((record["graph_id"], evidence))
    return kg_set


def eval_metrics(pred_evidence, gold_evidence):
    if not gold_evidence:
        return {"hit1": 0, "hit5": 0, "hit10": 0, "rr": 0.0}
    hits = [1 if e in gold_evidence else 0 for e in pred_evidence]
    rr = 0.0
    for rank, h in enumerate(hits, start=1):
        if h:
            rr = 1.0 / rank
            break
    return {
        "hit1": 1 if hits[0] == 1 else 0,
        "hit5": 1 if any(hits[:5]) else 0,
        "hit10": 1 if any(hits) else 0,
        "rr": rr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Neo4j graph-enhanced retrieval: TF-IDF + Neo4j boost."
    )
    parser.add_argument("--qa", default="results/locomo_qa_records.csv")
    parser.add_argument("--evidence", default="results/locomo_evidence_map.csv")
    parser.add_argument("--memory", default="results/locomo_memory_records.csv")
    parser.add_argument("--output", default="results/locomo_neo4j_results.csv")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password123")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--kg-boost", type=float, default=0.5)
    args = parser.parse_args()

    print("Loading gold evidence...")
    gold_map = load_gold_evidence(args.evidence)
    qa_rows = load_csv_rows(args.qa)

    print("Loading memory index...")
    all_texts, all_samples, all_evidences = build_memory_index(args.memory)
    n_memories = len(all_texts)
    print(f"  {n_memories} indexed memory turns")

    print("Loading Neo4j KG edge set...")
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    kg_set = build_neo4j_kg_set(driver)
    driver.close()
    print(f"  {len(kg_set)} KG edges loaded from Neo4j")

    kg_coverage = sum(1 for s, e in zip(all_samples, all_evidences) if (s, e) in kg_set)
    print(f"  KG covers {kg_coverage}/{n_memories} ({100*kg_coverage/n_memories:.1f}%) memory turns")

    print("Building TF-IDF...")
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=50000
    )
    mem_matrix = vectorizer.fit_transform(all_texts)
    print(f"  vocabulary: {len(vectorizer.vocabulary_)}")

    results = []
    total_kg_hits = 0
    total_preds = 0

    print(f"\nRunning {len(qa_rows)} queries (TF-IDF + Neo4j boost={args.kg_boost})...")

    for idx, qa in enumerate(qa_rows):
        qa_id = qa["qa_id"]
        question = qa["question"]
        category = qa.get("category", "")
        gold_ids = gold_map.get(qa_id, set())

        query_vec = vectorizer.transform([question])
        scores = cosine_similarity(query_vec, mem_matrix).flatten()

        candidate_scores = []
        for i in range(n_memories):
            base_score = float(scores[i])
            if (all_samples[i], all_evidences[i]) in kg_set:
                boosted = base_score + args.kg_boost
                candidate_scores.append((i, boosted, True))
            else:
                candidate_scores.append((i, base_score, False))

        candidate_scores.sort(key=lambda x: -x[1])
        top_k = candidate_scores[:args.top_k]

        pred_evidence = [all_evidences[i] for i, _, _ in top_k]
        kg_count = sum(1 for _, _, is_kg in top_k if is_kg)
        total_kg_hits += kg_count
        total_preds += len(top_k)

        metrics = eval_metrics(pred_evidence, gold_ids)
        results.append({
            "qa_id": qa_id, "category": category, "question": question,
            "gold_memory_ids": ";".join(sorted(gold_ids)),
            "retrieved_memory_ids": ";".join(pred_evidence),
            "hit1": metrics["hit1"], "hit5": metrics["hit5"],
            "hit10": metrics["hit10"], "rr": metrics["rr"],
        })

        if (idx + 1) % 200 == 0:
            r1 = sum(r["hit1"] for r in results) / len(results)
            r10 = sum(r["hit10"] for r in results) / len(results)
            print(f"  {idx+1}/{len(qa_rows)}  R@1={r1:.4f}  R@10={r10:.4f}")

    r1 = sum(r["hit1"] for r in results) / len(results)
    r5 = sum(r["hit5"] for r in results) / len(results)
    r10 = sum(r["hit10"] for r in results) / len(results)
    mrr = sum(r["rr"] for r in results) / len(results)

    fieldnames = ["qa_id", "category", "question", "gold_memory_ids",
                  "retrieved_memory_ids", "hit1", "hit5", "hit10", "rr"]
    with Path(args.output).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n=== Neo4j Results ===")
    print(f"Method: TF-IDF + Neo4j boost (boost={args.kg_boost})")
    print(f"KG coverage: {100*kg_coverage/n_memories:.1f}%")
    print(f"Avg KG memories in top-{args.top_k}: {total_kg_hits/total_preds*args.top_k:.1f}/{args.top_k}")
    print(f"Recall@1:  {r1:.4f}")
    print(f"Recall@5:  {r5:.4f}")
    print(f"Recall@10: {r10:.4f}")
    print(f"MRR@10:    {mrr:.4f}")

    by_cat = defaultdict(list)
    for r in results:
        by_cat[str(r["category"])].append(r)

    print("\nCategory-level:")
    for cat in sorted(by_cat.keys()):
        grp = by_cat[cat]
        n = len(grp)
        print(f"  cat {cat}: n={n}  R@1={sum(r['hit1'] for r in grp)/n:.4f}  "
              f"R@10={sum(r['hit10'] for r in grp)/n:.4f}  "
              f"MRR={sum(r['rr'] for r in grp)/n:.4f}")

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
