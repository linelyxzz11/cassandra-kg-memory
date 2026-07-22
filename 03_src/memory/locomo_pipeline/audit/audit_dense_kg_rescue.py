import csv
from collections import defaultdict
from pathlib import Path

BASE = Path("D:/memorytable/cassandra-kg-memory/results")

DENSE_CSV = str(BASE / "locomo_dense_bge_results.csv")
KG_CSV = str(BASE / "locomo_dense_kg_boost_best_results.csv")
QA_CSV = str(BASE / "locomo_qa_records.csv")
EVIDENCE_CSV = str(BASE / "locomo_evidence_map.csv")

OUT_RESULTS = str(BASE / "audit_dense_kg_rescue_results.csv")
OUT_SUMMARY = str(BASE / "audit_dense_kg_rescue_summary.csv")


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def parse_ids(raw):
    if not raw:
        return []
    return [x.strip() for x in str(raw).split(";") if x.strip()]


def check_hit(top_ids, gold_ids, k):
    ids = parse_ids(top_ids)
    ids_k = set(ids[:k])
    return int(bool(ids_k & gold_ids))


def main():
    print("Loading Dense-bge results...")
    dense_rows = {r["qa_id"]: r for r in load_csv(DENSE_CSV)}
    print(f"  {len(dense_rows)} queries")

    print("Loading Dense+KG best results...")
    kg_rows_all = load_csv(KG_CSV)
    kg_rows = {}
    for r in kg_rows_all:
        qid = r["qa_id"]
        if qid not in kg_rows:
            kg_rows[qid] = r
    print(f"  {len(kg_rows)} queries")

    print("Loading QA records...")
    qa_rows = load_csv(QA_CSV)
    qa_cat = {}
    qa_question = {}
    for r in qa_rows:
        qa_cat[r["qa_id"]] = r["category"].strip()
        qa_question[r["qa_id"]] = r["question"].strip()
    print(f"  {len(qa_cat)} queries")

    print("Loading gold evidence...")
    gold_map = defaultdict(set)
    for r in load_csv(EVIDENCE_CSV):
        qid = r["qa_id"].strip()
        eid = r.get("evidence_id", "").strip()
        mid = r.get("memory_id", "").strip()
        if eid:
            gold_map[qid].add(eid)
        if mid:
            gold_map[qid].add(mid)
    print(f"  {len(gold_map)} queries with gold evidence")

    result_rows = []

    for qid in sorted(dense_rows.keys()):
        if qid not in kg_rows:
            continue

        dr = dense_rows[qid]
        kr = kg_rows[qid]
        category = qa_cat.get(qid, "?")
        question = qa_question.get(qid, "?")
        gold_ids = gold_map.get(qid, set())

        dense_top10 = dr.get("retrieved_memory_ids", "")
        kg_top10 = kr.get("retrieved_memory_ids", "")

        for k in [1, 5, 10]:
            dense_hit = check_hit(dense_top10, gold_ids, k)
            kg_hit = check_hit(kg_top10, gold_ids, k)

            if dense_hit and kg_hit:
                outcome = "both_hit"
            elif dense_hit and not kg_hit:
                outcome = "dense_hit_kg_miss"
            elif not dense_hit and kg_hit:
                outcome = "kg_rescue"
            else:
                outcome = "both_miss"

            result_rows.append({
                "qa_id": qid,
                "k": k,
                "category": category,
                "question": question,
                "gold_memory_ids": ";".join(sorted(gold_ids)),
                "dense_top10": dense_top10,
                "kg_top10": kg_top10,
                "dense_hit": dense_hit,
                "kg_hit": kg_hit,
                "outcome": outcome,
            })

    result_fields = [
        "qa_id", "k", "category", "question", "gold_memory_ids",
        "dense_top10", "kg_top10", "dense_hit", "kg_hit", "outcome",
    ]
    write_csv(OUT_RESULTS, result_rows, result_fields)
    print(f"\nPer-query results: {OUT_RESULTS} ({len(result_rows)} rows)")

    summary_rows = []

    for k in [1, 5, 10]:
        k_rows = [r for r in result_rows if int(r["k"]) == k]
        total = len(k_rows)

        both_hit = len([r for r in k_rows if r["outcome"] == "both_hit"])
        dense_hit_kg_miss = len([r for r in k_rows if r["outcome"] == "dense_hit_kg_miss"])
        kg_rescue = len([r for r in k_rows if r["outcome"] == "kg_rescue"])
        both_miss = len([r for r in k_rows if r["outcome"] == "both_miss"])

        net_rescue = kg_rescue - dense_hit_kg_miss
        rescue_rate = kg_rescue / total if total else 0
        hurt_rate = dense_hit_kg_miss / total if total else 0

        summary_rows.append({
            "level": "ALL",
            "k": k,
            "n": total,
            "both_hit": both_hit,
            "both_hit_pct": f"{100*both_hit/total:.2f}%",
            "kg_rescue": kg_rescue,
            "kg_rescue_pct": f"{100*rescue_rate:.2f}%",
            "dense_hit_kg_miss": dense_hit_kg_miss,
            "dense_hit_kg_miss_pct": f"{100*hurt_rate:.2f}%",
            "both_miss": both_miss,
            "both_miss_pct": f"{100*both_miss/total:.2f}%",
            "net_rescue": net_rescue,
            "net_rescue_pct": f"{100*net_rescue/total:.2f}%",
            "rescue_gt_hurt": "YES" if kg_rescue > dense_hit_kg_miss else "NO",
        })

        for cat in sorted(set(r["category"] for r in k_rows), key=lambda x: int(x) if x.isdigit() else 999):
            cat_rows = [r for r in k_rows if r["category"] == cat]
            n = len(cat_rows)
            ch = len([r for r in cat_rows if r["outcome"] == "both_hit"])
            dhkm = len([r for r in cat_rows if r["outcome"] == "dense_hit_kg_miss"])
            kres = len([r for r in cat_rows if r["outcome"] == "kg_rescue"])
            bm = len([r for r in cat_rows if r["outcome"] == "both_miss"])
            net = kres - dhkm
            rr = kres / n if n else 0
            hr = dhkm / n if n else 0

            summary_rows.append({
                "level": f"cat_{cat}",
                "k": k,
                "n": n,
                "both_hit": ch,
                "both_hit_pct": f"{100*ch/n:.2f}%",
                "kg_rescue": kres,
                "kg_rescue_pct": f"{100*rr:.2f}%",
                "dense_hit_kg_miss": dhkm,
                "dense_hit_kg_miss_pct": f"{100*hr:.2f}%",
                "both_miss": bm,
                "both_miss_pct": f"{100*bm/n:.2f}%",
                "net_rescue": net,
                "net_rescue_pct": f"{100*net/n:.2f}%",
                "rescue_gt_hurt": "YES" if kres > dhkm else "NO",
            })

    summary_fields = [
        "level", "k", "n",
        "both_hit", "both_hit_pct",
        "kg_rescue", "kg_rescue_pct",
        "dense_hit_kg_miss", "dense_hit_kg_miss_pct",
        "both_miss", "both_miss_pct",
        "net_rescue", "net_rescue_pct",
        "rescue_gt_hurt",
    ]
    write_csv(OUT_SUMMARY, summary_rows, summary_fields)
    print(f"Summary: {OUT_SUMMARY}")

    print(f"\n{'='*80}")
    print(f"Dense-bge vs Dense-bge+KG Rescue Analysis")
    print(f"{'='*80}")

    for k in [1, 5, 10]:
        print(f"\n--- k={k} ---")
        sr = [r for r in summary_rows if int(r["k"]) == k and r["level"] == "ALL"][0]
        print(f"  Total queries: {sr['n']}")
        print(f"  both_hit:          {sr['both_hit']:5d}  ({sr['both_hit_pct']})")
        print(f"  KG rescue (+):     {sr['kg_rescue']:5d}  ({sr['kg_rescue_pct']})")
        print(f"  KG hurt (-):       {sr['dense_hit_kg_miss']:5d}  ({sr['dense_hit_kg_miss_pct']})")
        print(f"  both_miss:         {sr['both_miss']:5d}  ({sr['both_miss_pct']})")
        print(f"  Net rescue:        {sr['net_rescue']:+5d}  ({sr['net_rescue_pct']})")
        print(f"  Rescue > Hurt:     {sr['rescue_gt_hurt']}")

    print(f"\n--- Per Category (k=10) ---")
    print(f"{'cat':>6s}  {'n':>5s}  {'rescue':>8s}  {'hurt':>8s}  {'net':>8s}  {'rescue>hurt':>12s}")
    print("-" * 58)
    for cat in sorted(set(r["level"] for r in summary_rows if r["level"] != "ALL" and int(r["k"]) == 10),
                      key=lambda x: int(x.replace("cat_", ""))):
        sr = [r for r in summary_rows if r["level"] == cat and int(r["k"]) == 10][0]
        print(f"  {cat:>6s}  {sr['n']:>5d}  {sr['kg_rescue']:>8d}  {sr['dense_hit_kg_miss']:>8d}  {sr['net_rescue']:>8d}  {sr['rescue_gt_hurt']:>12s}")


if __name__ == "__main__":
    main()
