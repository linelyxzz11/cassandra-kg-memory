import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path("D:/memorytable/cassandra-kg-memory")
SCOPED = ROOT / "results/sample_scoped"
QA_CSV = ROOT / "results/locomo_qa_records.csv"
EVIDENCE_CSV = ROOT / "results/locomo_evidence_map.csv"
MEMORY_CSV = ROOT / "results/locomo_memory_records.csv"
KG_EDGES_CSV = ROOT / "results/locomo_kg_edges_spacy.csv"


def load_csv(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def compute_metrics(retrieved_ids, gold_ids):
    r1 = int(bool(set(retrieved_ids[:1]) & gold_ids))
    r5 = int(bool(set(retrieved_ids[:5]) & gold_ids))
    r10 = int(bool(set(retrieved_ids) & gold_ids))
    rr = 0.0
    for i, mid in enumerate(retrieved_ids[:10], 1):
        if mid in gold_ids:
            rr = 1.0 / i
            break
    return {"R@1": r1, "R@5": r5, "R@10": r10, "MRR": rr}


def load_gold_map():
    gold = defaultdict(set)
    for r in load_csv(EVIDENCE_CSV):
        gold[r["qa_id"].strip()].add(r["memory_id"].strip())
    return dict(gold)


def load_qa_cat_map():
    qa_cat = {}
    qa_q = {}
    for r in load_csv(QA_CSV):
        qa_cat[r["qa_id"].strip()] = r["category"].strip()
        qa_q[r["qa_id"].strip()] = r["question"].strip()
    return qa_cat, qa_q


def load_retrieval(filepath):
    rows = {}
    for r in load_csv(filepath):
        qid = r["qa_id"].strip()
        top_ids = [x.strip() for x in r["retrieved_memory_ids"].split(";") if x.strip()]
        rows[qid] = {
            "qa_id": qid,
            "retrieved_memory_ids": top_ids,
            "metric": compute_metrics(top_ids, gold_map.get(qid, set())),
        }
    return rows


gold_map = load_gold_map()
qa_cat, qa_q = load_qa_cat_map()

method_files = [
    ("BM25", SCOPED / "locomo_bm25_sample_scoped_results.csv"),
    ("Dense-bge", SCOPED / "locomo_dense_bge_sample_scoped_results.csv"),
    ("Dense-bge+GlobalKG", SCOPED / "locomo_dense_global_kg_sample_scoped_results.csv"),
    ("Dense-bge+QueryKG", SCOPED / "locomo_dense_query_kg_sample_scoped_results.csv"),
]

method_data = {}
for name, fp in method_files:
    data = load_retrieval(fp)
    method_data[name] = data
    print(f"Loaded {name}: {len(data)} queries")

cats = sorted(set(qa_cat.values()))
all_qa_ids = sorted(data.keys())

print(f"\n=== 1. CATEGORY-WISE RETRIEVAL METRICS ===")

cat_summary = []
for method, data in method_data.items():
    by_cat = defaultdict(list)
    for qid, info in data.items():
        cat = qa_cat.get(qid, "?")
        by_cat[cat].append(info["metric"])

    for cat in cats:
        metrics = by_cat.get(cat, [])
        n = len(metrics)
        if n == 0:
            continue
        cat_summary.append({
            "method": method,
            "category": cat,
            "n": n,
            "R@1": round(sum(m["R@1"] for m in metrics) / n, 4),
            "R@5": round(sum(m["R@5"] for m in metrics) / n, 4),
            "R@10": round(sum(m["R@10"] for m in metrics) / n, 4),
            "MRR": round(sum(m["MRR"] for m in metrics) / n, 4),
        })

cat_fields = ["method", "category", "n", "R@1", "R@5", "R@10", "MRR"]
write_csv(SCOPED / "analysis_category_wise.csv", cat_summary, cat_fields)

print(f"{'Method':<25s} {'Cat':<6s} {'n':>5s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s} {'MRR':>8s}")
print("-" * 70)
for row in cat_summary:
    print(f"{row['method']:<25s} {row['category']:<6s} {row['n']:>5d} "
          f"{row['R@1']:>8.4f} {row['R@5']:>8.4f} {row['R@10']:>8.4f} {row['MRR']:>8.4f}")

print(f"\nOutput: {SCOPED / 'analysis_category_wise.csv'}")

print(f"\n=== 2. RESCUE/HURT ANALYSIS ===")

pairs = [
    ("Dense-bge+GlobalKG", "Dense-bge"),
    ("Dense-bge+QueryKG", "Dense-bge"),
    ("Dense-bge+QueryKG", "Dense-bge+GlobalKG"),
]

rescue_hurt_rows = []
for method_a, method_b in pairs:
    data_a = method_data[method_a]
    data_b = method_data[method_b]

    rescue_hurt = {"rescue@1": 0, "hurt@1": 0, "rescue@5": 0, "hurt@5": 0, "rescue@10": 0, "hurt@10": 0}
    rescue_examples = []
    hurt_examples = []

    for qid in all_qa_ids:
        if qid not in data_a or qid not in data_b:
            continue
        gold = gold_map.get(qid, set())
        ma = data_a[qid]["metric"]
        mb = data_b[qid]["metric"]

        for k in [1, 5, 10]:
            if ma[f"R@{k}"] and not mb[f"R@{k}"]:
                rescue_hurt[f"rescue@{k}"] += 1
                if k == 1 or k == 10:
                    rescue_examples.append({"qid": qid, "k": k, "cat": qa_cat.get(qid, "?"), "q": qa_q.get(qid, "")})
            elif mb[f"R@{k}"] and not ma[f"R@{k}"]:
                rescue_hurt[f"hurt@{k}"] += 1
                if k == 1 or k == 10:
                    hurt_examples.append({"qid": qid, "k": k, "cat": qa_cat.get(qid, "?"), "q": qa_q.get(qid, "")})

    print(f"\n  {method_a} vs {method_b}:")
    for k in [1, 5, 10]:
        net = rescue_hurt[f"rescue@{k}"] - rescue_hurt[f"hurt@{k}"]
        print(f"    @{k}: rescue={rescue_hurt[f'rescue@{k}']}, hurt={rescue_hurt[f'hurt@{k}']}, net={net:+d}")

    rescue_hurt_rows.append({
        "method_a": method_a, "method_b": method_b,
        **{f"rescue@{k}": rescue_hurt[f"rescue@{k}"] for k in [1, 5, 10]},
        **{f"hurt@{k}": rescue_hurt[f"hurt@{k}"] for k in [1, 5, 10]},
        **{f"net@{k}": rescue_hurt[f"rescue@{k}"] - rescue_hurt[f"hurt@{k}"] for k in [1, 5, 10]},
    })

    rh_examples = []
    for ex in rescue_examples[:5]:
        rh_examples.append({"type": "rescue", "k": ex["k"], "qa_id": ex["qid"],
                            "category": ex["cat"], "question": ex["q"]})
    for ex in hurt_examples[:5]:
        rh_examples.append({"type": "hurt", "k": ex["k"], "qa_id": ex["qid"],
                            "category": ex["cat"], "question": ex["q"]})

    print(f"    rescue examples (@1/@10): {len(rescue_examples)} total")
    print(f"    hurt examples (@1/@10): {len(hurt_examples)} total")

rh_fields = ["method_a", "method_b",
             "rescue@1", "hurt@1", "net@1",
             "rescue@5", "hurt@5", "net@5",
             "rescue@10", "hurt@10", "net@10"]
write_csv(SCOPED / "analysis_rescue_hurt.csv", rescue_hurt_rows, rh_fields)
print(f"\nOutput: {SCOPED / 'analysis_rescue_hurt.csv'}")

rescue_hurt_detail = []
for qid in all_qa_ids:
    if qid not in method_data["Dense-bge+QueryKG"] or qid not in method_data["Dense-bge"]:
        continue
    gold = gold_map.get(qid, set())
    ma = method_data["Dense-bge+QueryKG"][qid]["metric"]
    mb = method_data["Dense-bge"][qid]["metric"]
    for k in [1, 5, 10]:
        is_rescue = ma[f"R@{k}"] and not mb[f"R@{k}"]
        is_hurt = mb[f"R@{k}"] and not ma[f"R@{k}"]
        if is_rescue or is_hurt:
            rescue_hurt_detail.append({
                "qa_id": qid,
                "type": "rescue" if is_rescue else "hurt",
                "k": k,
                "category": qa_cat.get(qid, "?"),
                "question": qa_q.get(qid, ""),
                "method_a": "Dense-bge+QueryKG",
                "method_b": "Dense-bge",
            })

rh_detail_fields = ["qa_id", "type", "k", "category", "question", "method_a", "method_b"]
write_csv(SCOPED / "analysis_rescue_hurt_detail.csv", rescue_hurt_detail, rh_detail_fields)
print(f"Detail output: {SCOPED / 'analysis_rescue_hurt_detail.csv'}")

print(f"\n=== 3. KG COVERAGE AUDIT ===")

mid_by_dia = defaultdict(list)
all_memory_ids = set()
with open(MEMORY_CSV, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        mid = row["memory_id"].strip()
        sid = row["sample_id"].strip()
        dia = row["dia_id"].strip()
        all_memory_ids.add(mid)
        mid_by_dia[(sid, dia)].append(mid)

kg_memory_ids = set()
with open(KG_EDGES_CSV, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        gid = row["graph_id"].strip()
        ev = row["evidence"].strip()
        for mid in mid_by_dia.get((gid, ev), []):
            kg_memory_ids.add(mid)

all_gold_memories = set()
for qid, gold_set in gold_map.items():
    all_gold_memories |= gold_set

non_gold_memories = all_memory_ids - all_gold_memories

gold_has_kg = all_gold_memories & kg_memory_ids
non_gold_has_kg = non_gold_memories & kg_memory_ids

print(f"  Total memories: {len(all_memory_ids)}")
print(f"  Memories with KG edges: {len(kg_memory_ids)} ({len(kg_memory_ids)/len(all_memory_ids)*100:.1f}%)")
print(f"  Gold evidence memories: {len(all_gold_memories)}")
print(f"  Gold memories with KG: {len(gold_has_kg)} ({len(gold_has_kg)/max(1, len(all_gold_memories))*100:.1f}%)")
print(f"  Non-gold memories: {len(non_gold_memories)}")
print(f"  Non-gold memories with KG: {len(non_gold_has_kg)} ({len(non_gold_has_kg)/max(1, len(non_gold_memories))*100:.1f}%)")
print(f"  P(hasKG | gold) = {len(gold_has_kg)/max(1, len(all_gold_memories)):.4f}")
print(f"  P(hasKG | non-gold) = {len(non_gold_has_kg)/max(1, len(non_gold_memories)):.4f}")

kg_coverage_rows = [{
    "metric": "total_memories",
    "value": len(all_memory_ids),
    "with_kg": len(kg_memory_ids),
    "without_kg": len(all_memory_ids) - len(kg_memory_ids),
    "kg_rate": round(len(kg_memory_ids) / len(all_memory_ids), 4),
}, {
    "metric": "gold_evidence_memories",
    "value": len(all_gold_memories),
    "with_kg": len(gold_has_kg),
    "without_kg": len(all_gold_memories) - len(gold_has_kg),
    "kg_rate": round(len(gold_has_kg) / max(1, len(all_gold_memories)), 4),
}, {
    "metric": "non_gold_memories",
    "value": len(non_gold_memories),
    "with_kg": len(non_gold_has_kg),
    "without_kg": len(non_gold_memories) - len(non_gold_has_kg),
    "kg_rate": round(len(non_gold_has_kg) / max(1, len(non_gold_memories)), 4),
}, {
    "metric": "P(hasKG|gold)",
    "value": 0,
    "with_kg": len(gold_has_kg),
    "without_kg": len(all_gold_memories) - len(gold_has_kg),
    "kg_rate": round(len(gold_has_kg) / max(1, len(all_gold_memories)), 4),
}, {
    "metric": "P(hasKG|non_gold)",
    "value": 0,
    "with_kg": len(non_gold_has_kg),
    "without_kg": len(non_gold_memories) - len(non_gold_has_kg),
    "kg_rate": round(len(non_gold_has_kg) / max(1, len(non_gold_memories)), 4),
}]

kg_fields = ["metric", "value", "with_kg", "without_kg", "kg_rate"]
write_csv(SCOPED / "analysis_kg_coverage.csv", kg_coverage_rows, kg_fields)
print(f"\nOutput: {SCOPED / 'analysis_kg_coverage.csv'}")

print("\nDone.")