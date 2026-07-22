import csv
import json
import re
from collections import defaultdict
from pathlib import Path

METHODS = {
    "BM25": ("results/locomo_bm25_results.csv", ["retrieved_memory_ids", "top10_memory_ids", "top10_evidence_ids"]),
    "Dense-ONNX-MiniLM": ("results/locomo_dense_onnx_results.csv", ["retrieved_memory_ids", "top10_memory_ids", "top10_evidence_ids"]),
    "BM25-Dense-RRF": ("results/locomo_fusion_bm25_dense_results.csv", ["fusion_top10_memory_ids", "retrieved_memory_ids", "top10_memory_ids"]),
    "KG-boost": ("results/locomo_cassandra_kg_results.csv", ["retrieved_memory_ids", "top10_memory_ids", "top10_evidence_ids"]),
    "KG-aware-RRF": ("results/locomo_kg_aware_fusion_best_results.csv", ["retrieved_memory_ids", "kg_aware_top10_memory_ids"]),
}

QA_CSV = "results/locomo_qa_records.csv"
MEMORY_CSV = "results/locomo_memory_records.csv"
EVIDENCE_CSV = "results/locomo_evidence_map.csv"
KG_EDGE_CSV = "results/locomo_kg_edges_spacy.csv"
V3_READER_CSV = "results/llm_reader_pilot_v3_results.csv"

OUT_SUMMARY = "results/audit_kg_boost_top10_summary.csv"
OUT_QA = "results/audit_kg_boost_top10_qa.csv"
OUT_DUMP = "results/audit_kg_boost_top10_dump.csv"

def load_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def split_ids(raw):
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            vals = json.loads(raw)
            return [str(x).strip() for x in vals if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in re.split(r"[;,\|]", raw) if x.strip()]

def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def load_memory():
    mem_by_id = {}
    mem_by_dia = {}
    for row in load_csv(MEMORY_CSV):
        mid = str(row.get("memory_id", "")).strip()
        sample_id = str(row.get("sample_id", "")).strip()
        dia_id = str(row.get("dia_id", "")).strip()
        if mid:
            mem_by_id[mid] = row
        if sample_id and dia_id:
            mem_by_dia[(sample_id, dia_id)] = row
    return mem_by_id, mem_by_dia

def resolve_memory_id(raw_id, qa_id, mem_by_id, mem_by_dia):
    raw_id = str(raw_id).strip()
    if not raw_id:
        return ""
    if raw_id in mem_by_id:
        return raw_id
    sample_id = qa_id.split("_qa_")[0] if "_qa_" in qa_id else qa_id
    return resolve_memory_id_with_sample(raw_id, sample_id, mem_by_id, mem_by_dia)

def resolve_memory_id_with_sample(raw_id, sample_id, mem_by_id, mem_by_dia):
    raw_id = str(raw_id).strip()
    sample_id = str(sample_id).strip()
    if not raw_id:
        return ""
    if raw_id in mem_by_id:
        return raw_id
    m = re.match(r"D(\d+):(\d+)$", raw_id)
    if m:
        candidate = f"{sample_id}_session_{m.group(1)}_{raw_id}"
        if candidate in mem_by_id:
            return candidate
        row = mem_by_dia.get((sample_id, raw_id))
        if row:
            return row.get("memory_id", "")
    if raw_id.startswith("session_"):
        candidate = f"{sample_id}_{raw_id}"
        if candidate in mem_by_id:
            return candidate
    candidate = f"{sample_id}_{raw_id}"
    if candidate in mem_by_id:
        return candidate
    return ""

def pick_column(rows, candidates, method, path):
    if not rows:
        raise ValueError(f"Empty ranking file for {method}: {path}")
    cols = rows[0].keys()
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"No ranking column for {method} in {path}. Columns={list(cols)}")

def load_rankings(mem_by_id, mem_by_dia):
    rankings = {}
    used_cols = {}
    for method, (path, candidates) in METHODS.items():
        rows = load_csv(path)
        col = pick_column(rows, candidates, method, path)
        used_cols[method] = col
        method_map = {}
        for row in rows:
            qid = str(row.get("qa_id", "")).strip()
            if not qid:
                continue
            raw_ids = unique_keep_order(split_ids(row.get(col, "")))[:10]
            resolved_ids = []
            for rid in raw_ids:
                mid = resolve_memory_id(rid, qid, mem_by_id, mem_by_dia)
                resolved_ids.append(mid if mid else rid)
            method_map[qid] = {
                "raw_ids": raw_ids,
                "resolved_ids": resolved_ids,
            }
        rankings[method] = method_map
        print(f"{method}: loaded {len(method_map)} rankings from {path}, column={col}")
    return rankings, used_cols

def load_gold_map():
    gold = defaultdict(set)
    for row in load_csv(EVIDENCE_CSV):
        qid = str(row.get("qa_id", "")).strip()
        evidence_id = str(row.get("evidence_id", "")).strip()
        memory_id = str(row.get("memory_id", "")).strip()
        if qid and evidence_id:
            gold[qid].add(evidence_id)
        if qid and memory_id:
            gold[qid].add(memory_id)
    return gold

def load_qa_rows():
    out = []
    for row in load_csv(QA_CSV):
        qid = str(row.get("qa_id", "")).strip()
        if qid:
            out.append({
                "qa_id": qid,
                "category": str(row.get("category", "")).strip(),
                "question": str(row.get("question", "")).strip(),
                "answer": str(row.get("answer", "")).strip(),
            })
    return out

def load_kg_covered(mem_by_id, mem_by_dia):
    covered = set()
    rows = load_csv(KG_EDGE_CSV)
    candidate_cols = [
        "memory_id", "evidence_memory_id", "evidence_id", "evidence",
        "dia_id", "source_dia_id", "turn_id"
    ]
    sample_cols = ["sample_id", "graph_id", "conversation_id", "conv_id"]
    for row in rows:
        sample_id = ""
        for sc in sample_cols:
            if row.get(sc):
                sample_id = str(row.get(sc)).strip()
                break
        for col in candidate_cols:
            val = str(row.get(col, "")).strip()
            if not val:
                continue
            for item in split_ids(val):
                if item in mem_by_id:
                    covered.add(item)
                elif sample_id:
                    mid = resolve_memory_id_with_sample(item, sample_id, mem_by_id, mem_by_dia)
                    if mid:
                        covered.add(mid)
    return covered

def load_v3_reader():
    rows = load_csv(V3_READER_CSV)
    out = {}
    for row in rows:
        qid = str(row.get("qa_id", "")).strip()
        method = str(row.get("method", "")).strip()
        if qid and method:
            out[(qid, method)] = row
    return out

def is_hit(qid, raw_ids, resolved_ids, gold_map):
    gold = gold_map.get(qid, set())
    if not gold:
        return 0
    return 1 if (set(raw_ids) | set(resolved_ids)) & gold else 0

def avg(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0

def main():
    mem_by_id, mem_by_dia = load_memory()
    qa_rows = load_qa_rows()
    gold_map = load_gold_map()
    kg_covered = load_kg_covered(mem_by_id, mem_by_dia)
    rankings, used_cols = load_rankings(mem_by_id, mem_by_dia)
    v3 = load_v3_reader()

    print(f"KG-covered memory ids: {len(kg_covered)}")

    qa_metrics = []

    for qa in qa_rows:
        qid = qa["qa_id"]
        bm25_set = set(rankings["BM25"].get(qid, {}).get("resolved_ids", []))
        kg_set = set(rankings["KG-boost"].get(qid, {}).get("resolved_ids", []))
        for method in METHODS:
            entry = rankings[method].get(qid, {"raw_ids": [], "resolved_ids": []})
            raw_ids = entry["raw_ids"]
            resolved_ids = entry["resolved_ids"]
            top10 = resolved_ids[:10]
            kg_count = sum(1 for mid in top10 if mid in kg_covered)
            hit10 = is_hit(qid, raw_ids, resolved_ids, gold_map)
            overlap_bm25 = len(set(top10) & bm25_set)
            overlap_kg = len(set(top10) & kg_set)
            reader = v3.get((qid, method), {})
            qa_metrics.append({
                "qa_id": qid,
                "category": qa["category"],
                "question": qa["question"],
                "answer": qa["answer"],
                "method": method,
                "top10_count": len(top10),
                "kg_covered_in_top10": kg_count,
                "kg_covered_ratio": f"{kg_count / len(top10):.4f}" if top10 else "0.0000",
                "retrieval_hit10": hit10,
                "overlap_with_bm25_top10": overlap_bm25,
                "overlap_with_kg_top10": overlap_kg,
                "reader_relaxed_f1": reader.get("relaxed_f1", ""),
                "reader_predicted_answer": reader.get("predicted_answer", ""),
                "top10_ids": ";".join(raw_ids),
                "resolved_top10_ids": ";".join(resolved_ids),
            })

    summary_rows = []
    for method in METHODS:
        rows = [r for r in qa_metrics if r["method"] == method]
        summary_rows.append({
            "method": method,
            "n": len(rows),
            "avg_kg_covered_in_top10": f"{avg(float(r['kg_covered_in_top10']) for r in rows):.4f}",
            "avg_kg_covered_ratio": f"{avg(float(r['kg_covered_ratio']) for r in rows):.4f}",
            "pct_top10_all_kg_covered": f"{avg(1 if int(r['kg_covered_in_top10']) == int(r['top10_count']) and int(r['top10_count']) > 0 else 0 for r in rows):.4f}",
            "retrieval_hit10": f"{avg(float(r['retrieval_hit10']) for r in rows):.4f}",
            "avg_overlap_with_bm25_top10": f"{avg(float(r['overlap_with_bm25_top10']) for r in rows):.4f}",
            "avg_overlap_with_kg_top10": f"{avg(float(r['overlap_with_kg_top10']) for r in rows):.4f}",
            "avg_reader_relaxed_f1_if_available": f"{avg(float(r['reader_relaxed_f1']) for r in rows if str(r['reader_relaxed_f1']).strip() != ''):.4f}",
        })

    qa_fields = [
        "qa_id", "category", "question", "answer", "method",
        "top10_count", "kg_covered_in_top10", "kg_covered_ratio",
        "retrieval_hit10", "overlap_with_bm25_top10", "overlap_with_kg_top10",
        "reader_relaxed_f1", "reader_predicted_answer",
        "top10_ids", "resolved_top10_ids",
    ]
    write_csv(OUT_QA, qa_metrics, qa_fields)

    summary_fields = [
        "method", "n", "avg_kg_covered_in_top10", "avg_kg_covered_ratio",
        "pct_top10_all_kg_covered", "retrieval_hit10",
        "avg_overlap_with_bm25_top10", "avg_overlap_with_kg_top10",
        "avg_reader_relaxed_f1_if_available",
    ]
    write_csv(OUT_SUMMARY, summary_rows, summary_fields)

    selected_qids = []
    kg_rows = [r for r in qa_metrics if r["method"] == "KG-boost"]
    hard_cases = [
        r for r in kg_rows
        if int(r["kg_covered_in_top10"]) >= 9
        and int(r["overlap_with_bm25_top10"]) <= 3
        and str(r["reader_relaxed_f1"]).strip() not in {"", "nan"}
        and float(r["reader_relaxed_f1"]) <= 0.1
    ]
    for r in hard_cases[:20]:
        selected_qids.append(r["qa_id"])

    kg_hit_cases = [r for r in kg_rows if int(r["retrieval_hit10"]) == 1 and r["qa_id"] not in selected_qids]
    for r in kg_hit_cases[:20]:
        selected_qids.append(r["qa_id"])

    if not selected_qids:
        selected_qids = [r["qa_id"] for r in kg_rows[:20]]

    selected_qids = unique_keep_order(selected_qids)[:40]

    dump_rows = []
    for qid in selected_qids:
        qa = next((x for x in qa_rows if x["qa_id"] == qid), {})
        for method in METHODS:
            entry = rankings[method].get(qid, {"raw_ids": [], "resolved_ids": []})
            raw_ids = entry["raw_ids"]
            resolved_ids = entry["resolved_ids"]
            for i, mid in enumerate(resolved_ids[:10], start=1):
                mem = mem_by_id.get(mid, {})
                raw = raw_ids[i - 1] if i - 1 < len(raw_ids) else ""
                dump_rows.append({
                    "qa_id": qid,
                    "category": qa.get("category", ""),
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "method": method,
                    "rank": i,
                    "raw_id": raw,
                    "memory_id": mid,
                    "is_kg_covered": 1 if mid in kg_covered else 0,
                    "is_gold": 1 if raw in gold_map.get(qid, set()) or mid in gold_map.get(qid, set()) else 0,
                    "speaker": mem.get("speaker", ""),
                    "timestamp": mem.get("timestamp", ""),
                    "session_id": mem.get("session_id", ""),
                    "dia_id": mem.get("dia_id", ""),
                    "text": mem.get("text", ""),
                    "v3_predicted_answer": v3.get((qid, method), {}).get("predicted_answer", ""),
                    "v3_relaxed_f1": v3.get((qid, method), {}).get("relaxed_f1", ""),
                })

    dump_fields = [
        "qa_id", "category", "question", "answer", "method", "rank",
        "raw_id", "memory_id", "is_kg_covered", "is_gold",
        "speaker", "timestamp", "session_id", "dia_id", "text",
        "v3_predicted_answer", "v3_relaxed_f1",
    ]
    write_csv(OUT_DUMP, dump_rows, dump_fields)

    print("")
    print("=== KG Boost Top10 Audit ===")
    print(f"QA count: {len(qa_rows)}")
    print(f"KG-covered memory ids: {len(kg_covered)}")
    print("")
    print(f"{'Method':20s} {'kg/top10':>10s} {'allKG%':>8s} {'hit10':>8s} {'ovBM25':>8s} {'ovKG':>8s} {'v3F1':>8s}")
    print("-" * 82)
    for r in summary_rows:
        print(
            f"{r['method']:20s} "
            f"{float(r['avg_kg_covered_in_top10']):10.4f} "
            f"{float(r['pct_top10_all_kg_covered']):8.4f} "
            f"{float(r['retrieval_hit10']):8.4f} "
            f"{float(r['avg_overlap_with_bm25_top10']):8.4f} "
            f"{float(r['avg_overlap_with_kg_top10']):8.4f} "
            f"{float(r['avg_reader_relaxed_f1_if_available']):8.4f}"
        )

    print("")
    print(f"Summary: {OUT_SUMMARY}")
    print(f"QA-level audit: {OUT_QA}")
    print(f"Top10 text dump: {OUT_DUMP}")
    print("")
    print("Columns used:")
    for method, col in used_cols.items():
        print(f"  {method}: {col}")

if __name__ == "__main__":
    main()