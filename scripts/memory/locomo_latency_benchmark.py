import argparse
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_csv_rows(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def build_kg_set(edge_file):
    kg_set = set()
    with Path(edge_file).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kg_set.add((row["graph_id"], row["evidence"]))
    return kg_set


def build_neo4j_kg_set():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))
    kg_set = set()
    with driver.session() as session:
        result = session.run("MATCH ()-[r:KG_EDGE]->() RETURN r.graph_id AS graph_id, r.source AS source")
        for record in result:
            source_str = str(record["source"]) if record["source"] else ""
            parts = source_str.split("|")
            evidence = parts[-1] if len(parts) >= 2 else ""
            if evidence:
                kg_set.add((record["graph_id"], evidence))
    driver.close()
    return kg_set


def measure_tfidf(questions, vectorizer, mem_matrix, all_samples, all_evidences, kg_set, boost, top_k):
    q_vecs = vectorizer.transform(questions)
    scores = cosine_similarity(q_vecs, mem_matrix)
    n_mem = len(all_evidences)
    total = 0.0
    for qi in range(len(questions)):
        t0 = time.perf_counter()
        cand = []
        for i in range(n_mem):
            base = float(scores[qi][i])
            if (all_samples[i], all_evidences[i]) in kg_set:
                cand.append((all_evidences[i], base + boost))
            else:
                cand.append((all_evidences[i], base))
        cand.sort(key=lambda x: -x[1])
        cand[:top_k]
        total += time.perf_counter() - t0
    return total / len(questions)


def main():
    parser = argparse.ArgumentParser(description="Retrieval latency benchmark.")
    parser.add_argument("--qa", default="results/locomo_qa_records.csv")
    parser.add_argument("--memory", default="results/locomo_memory_records.csv")
    parser.add_argument("--edges", default="results/locomo_kg_edges_spacy.csv")
    parser.add_argument("--output", default="results/locomo_latency_results.csv")
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--boost", type=float, default=0.5)
    parser.add_argument("--skip-neo4j", action="store_true")
    args = parser.parse_args()

    qa_rows = load_csv_rows(args.qa)[:args.n_queries]
    questions = [r["question"] for r in qa_rows]

    print("Loading memory index...")
    all_texts, all_samples, all_evidences = build_memory_index(args.memory)
    print(f"  {len(all_texts)} turns")

    print("Loading KG edge set...")
    kg_set = build_kg_set(args.edges)
    print(f"  {len(kg_set)} edges")

    print("Building TF-IDF (one-time)...")
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=50000)
    mem_matrix = vectorizer.fit_transform(all_texts)
    tfidf_build = (time.perf_counter() - t0) * 1000

    results = []

    print(f"TF-IDF baseline ({args.n_queries} queries)...")
    t0 = time.perf_counter()
    q_vecs = vectorizer.transform(questions)
    scores = cosine_similarity(q_vecs, mem_matrix)
    n_mem = len(all_evidences)
    total = 0.0
    for qi in range(len(questions)):
        ti = time.perf_counter()
        cand = [(all_evidences[i], float(scores[qi][i])) for i in range(n_mem)]
        cand.sort(key=lambda x: -x[1])
        cand[:args.top_k]
        total += time.perf_counter() - ti
    tfidf_avg = (total / len(questions)) * 1000
    results.append({
        "method": "TF-IDF (pure)",
        "index_build_ms": f"{tfidf_build:.1f}",
        "avg_query_ms": f"{tfidf_avg:.2f}",
        "note": f"{args.n_queries} queries x {n_mem} memories",
    })
    print(f"  index_build: {tfidf_build:.1f} ms, avg_query: {tfidf_avg:.2f} ms")

    print(f"TF-IDF + KG boost={args.boost} ({args.n_queries} queries)...")
    t0 = time.perf_counter()
    q_vecs = vectorizer.transform(questions)
    scores = cosine_similarity(q_vecs, mem_matrix)
    n_mem = len(all_evidences)
    total = 0.0
    boost = args.boost
    for qi in range(len(questions)):
        ti = time.perf_counter()
        cand = []
        for i in range(n_mem):
            base = float(scores[qi][i])
            if (all_samples[i], all_evidences[i]) in kg_set:
                cand.append((all_evidences[i], base + boost))
            else:
                cand.append((all_evidences[i], base))
        cand.sort(key=lambda x: -x[1])
        cand[:args.top_k]
        total += time.perf_counter() - ti
    kg_avg = (total / len(questions)) * 1000
    results.append({
        "method": "TF-IDF + KG boost",
        "index_build_ms": f"{tfidf_build:.1f}",
        "avg_query_ms": f"{kg_avg:.2f}",
        "note": f"boost={boost}, {args.n_queries} queries, KG dict lookup only",
    })
    print(f"  avg_query: {kg_avg:.2f} ms")

    if not args.skip_neo4j:
        print(f"Loading Neo4j KG edge set...")
        try:
            t0 = time.perf_counter()
            neo4j_kg = build_neo4j_kg_set()
            neo4j_load = (time.perf_counter() - t0) * 1000
            print(f"  {len(neo4j_kg)} edges loaded in {neo4j_load:.0f} ms")

            print(f"TF-IDF + Neo4j boost={boost} ({args.n_queries} queries)...")
            q_vecs = vectorizer.transform(questions)
            scores = cosine_similarity(q_vecs, mem_matrix)
            n_mem = len(all_evidences)
            total = 0.0
            for qi in range(len(questions)):
                ti = time.perf_counter()
                cand = []
                for i in range(n_mem):
                    base = float(scores[qi][i])
                    if (all_samples[i], all_evidences[i]) in neo4j_kg:
                        cand.append((all_evidences[i], base + boost))
                    else:
                        cand.append((all_evidences[i], base))
                cand.sort(key=lambda x: -x[1])
                cand[:args.top_k]
                total += time.perf_counter() - ti
            neo4j_avg = (total / len(questions)) * 1000
            results.append({
                "method": "TF-IDF + Neo4j boost",
                "index_build_ms": f"{tfidf_build:.1f}",
                "avg_query_ms": f"{neo4j_avg:.2f}",
                "note": f"KG load: {neo4j_load:.0f}ms, {args.n_queries} queries",
            })
            print(f"  avg_query: {neo4j_avg:.2f} ms")
        except Exception as e:
            results.append({
                "method": "TF-IDF + Neo4j boost",
                "index_build_ms": "-",
                "avg_query_ms": "-",
                "note": f"Neo4j unavailable: {e}",
            })
            print(f"  Neo4j unavailable: {e}")

    with Path(args.output).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "index_build_ms", "avg_query_ms", "note"])
        w.writeheader()
        w.writerows(results)

    print(f"\n=== Latency Summary ===")
    print(f"{'Method':30s} {'Index(ms)':>10s} {'Query(ms)':>10s} {'Note'}")
    print("-" * 80)
    for r in results:
        print(f"{r['method']:30s} {r['index_build_ms']:>10s} {r['avg_query_ms']:>10s} {r['note']}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
