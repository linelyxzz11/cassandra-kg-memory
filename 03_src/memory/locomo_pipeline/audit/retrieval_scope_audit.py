import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("D:/memorytable/cassandra-kg-memory")
RESULTS = ROOT / "results"
FINAL = RESULTS / "final"

def parse_qa_sample_id(qa_id):
    m = re.match(r"conv-(\d+)_qa_\d+", qa_id)
    if m:
        return f"conv-{m.group(1)}"
    return None

def parse_memory_sample_id(memory_id):
    if memory_id.startswith("conv-"):
        m = re.match(r"conv-(\d+)_", memory_id)
        if m:
            return f"conv-{m.group(1)}"
    return None

def parse_dia_id_sample_map():
    dia_to_samples = defaultdict(set)
    dia_to_full_ids = defaultdict(list)
    with open(RESULTS / "locomo_memory_records.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mid = row["memory_id"]
            sid = row["sample_id"]
            dia = mid.split("_")[-1]
            dia_to_samples[dia].add(sid)
            dia_to_full_ids[dia].append(mid)
    return dia_to_samples, dia_to_full_ids

def load_qa_sample_map():
    qa_to_sample = {}
    with open(RESULTS / "locomo_qa_records.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            qa_to_sample[row["qa_id"]] = row["sample_id"]
    return qa_to_sample

def load_retrieval_results(filepath, top10_col="retrieved_memory_ids"):
    rows = []
    with open(filepath, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            qa_id = row["qa_id"]
            top10_str = row[top10_col]
            top10_ids = [x.strip() for x in top10_str.split(";") if x.strip()]
            rows.append({"qa_id": qa_id, "top10_memory_ids": top10_ids})
    return rows

def audit_method(rows, qa_sample_map, method_name, id_format="conv_prefix"):
    per_query = []
    total_cross_count = 0
    total_memories_checked = 0
    affected_queries = 0
    n_queries = len(rows)

    for row in rows:
        qa_id = row["qa_id"]
        top10_ids = row["top10_memory_ids"]
        query_sample = qa_sample_map.get(qa_id) or parse_qa_sample_id(qa_id)
        if not query_sample:
            continue

        cross_count = 0
        for mid in top10_ids:
            total_memories_checked += 1
            if id_format == "conv_prefix":
                mem_sample = parse_memory_sample_id(mid)
                if mem_sample and mem_sample != query_sample:
                    cross_count += 1
            elif id_format == "dia_id":
                if mid in dia_to_samples:
                    if query_sample not in dia_to_samples[mid]:
                        cross_count += 1

        total_cross_count += cross_count
        if cross_count > 0:
            affected_queries += 1

        per_query.append({
            "qa_id": qa_id,
            "query_sample_id": query_sample,
            "top10_memory_ids": ";".join(top10_ids),
            "top10_count": len(top10_ids),
            "top10_cross_sample_count": cross_count,
            "top10_cross_sample_rate": cross_count / len(top10_ids) if len(top10_ids) > 0 else 0.0,
        })

    avg_cross_per_query = total_cross_count / n_queries if n_queries > 0 else 0
    overall_cross_rate = total_cross_count / total_memories_checked if total_memories_checked > 0 else 0
    affected_rate = affected_queries / n_queries if n_queries > 0 else 0

    summary = {
        "method": method_name,
        "n_queries": n_queries,
        "total_memories_checked": total_memories_checked,
        "total_cross_sample_count": total_cross_count,
        "overall_cross_sample_rate": round(overall_cross_rate, 4),
        "affected_query_count": affected_queries,
        "affected_query_rate": round(affected_rate, 4),
        "avg_cross_sample_memories_per_query": round(avg_cross_per_query, 4),
    }

    return per_query, summary

dia_to_samples, dia_to_full_ids = parse_dia_id_sample_map()
qa_sample_map = load_qa_sample_map()

methods = [
    ("Dense-bge", RESULTS / "locomo_dense_bge_results.csv", "retrieved_memory_ids", "conv_prefix"),
    ("Dense-bge+GlobalKG", RESULTS / "locomo_dense_kg_boost_best_results.csv", "retrieved_memory_ids", "conv_prefix"),
    ("Dense-bge+QueryKG", RESULTS / "query_kg_rerank/dense_bge_query_kg_rerank_best_results.csv", "retrieved_memory_ids", "conv_prefix"),
    ("BM25", RESULTS / "locomo_bm25_results.csv", "retrieved_memory_ids", "dia_id"),
]

all_per_query_rows = []
all_summaries = []

for method_name, filepath, top10_col, id_format in methods:
    print(f"\nAuditing: {method_name}")
    print(f"  File: {filepath}")
    print(f"  ID format: {id_format}")

    rows = load_retrieval_results(filepath, top10_col)
    per_query, summary = audit_method(rows, qa_sample_map, method_name, id_format)

    print(f"  n_queries: {summary['n_queries']}")
    print(f"  cross_sample_count: {summary['total_cross_sample_count']}")
    print(f"  overall_cross_sample_rate: {summary['overall_cross_sample_rate']}")
    print(f"  affected_query_count: {summary['affected_query_count']}")
    print(f"  affected_query_rate: {summary['affected_query_rate']}")
    print(f"  avg_cross_per_query: {summary['avg_cross_sample_memories_per_query']}")

    for pq in per_query:
        pq["method"] = method_name
    all_per_query_rows.extend(per_query)
    all_summaries.append(summary)

detail_fields = ["method", "qa_id", "query_sample_id", "top10_memory_ids",
                 "top10_count", "top10_cross_sample_count", "top10_cross_sample_rate"]
with open(FINAL / "retrieval_scope_audit_global.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=detail_fields)
    w.writeheader()
    w.writerows(all_per_query_rows)
print(f"\nDetail output: {FINAL / 'retrieval_scope_audit_global.csv'}")

summary_fields = ["method", "n_queries", "total_memories_checked",
                   "total_cross_sample_count", "overall_cross_sample_rate",
                   "affected_query_count", "affected_query_rate",
                   "avg_cross_sample_memories_per_query"]
with open(FINAL / "retrieval_scope_audit_global_summary.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_fields)
    w.writeheader()
    w.writerows(all_summaries)
print(f"Summary output: {FINAL / 'retrieval_scope_audit_global_summary.csv'}")

print("\n=== SCOPE AUDIT CONCLUSION ===")
for s in all_summaries:
    flag = "GLOBAL-CROSS-SESSION" if s["overall_cross_sample_rate"] > 0 else "SAMPLE-SCOPED"
    print(f"  {s['method']}: cross_rate={s['overall_cross_sample_rate']}, "
          f"affected={s['affected_query_rate']}, avg_cross/q={s['avg_cross_sample_memories_per_query']} "
          f"-> {flag}")

print("\nDone.")