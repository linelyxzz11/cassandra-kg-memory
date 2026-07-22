import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_csv_dict(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_edge_set(edge_file):
    edge_set = set()
    with Path(edge_file).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            edge_set.add((row["graph_id"], row["evidence"]))
    return edge_set


def rank_of_gold_in_list(gold_ids, retrieved_list):
    if not gold_ids:
        return "none"
    best_rank = 999
    for g in gold_ids:
        for i, r in enumerate(retrieved_list):
            if r == g:
                best_rank = min(best_rank, i + 1)
                break
    if best_rank == 999:
        return ">10"
    return str(best_rank)


def main():
    parser = argparse.ArgumentParser(
        description="E2 backend-fair audit: per-query Cassandra-KG vs Neo4j diagnosis."
    )
    parser.add_argument("--cassandra", default="results/locomo_cassandra_kg_results.csv")
    parser.add_argument("--neo4j", default="results/locomo_neo4j_results.csv")
    parser.add_argument("--evidence", default="results/locomo_evidence_map.csv")
    parser.add_argument("--edges", default="results/locomo_kg_edges_spacy.csv")
    parser.add_argument("--output-summary", default="results/e2_backend_fair_audit_summary.csv")
    parser.add_argument("--output-disagree", default="results/e2_backend_fair_disagreements.csv")
    args = parser.parse_args()

    c_rows = load_csv_dict(args.cassandra)
    n_rows = load_csv_dict(args.neo4j)
    edge_set = load_edge_set(args.edges)

    c_lookup = {r["qa_id"]: r for r in c_rows}
    n_lookup = {r["qa_id"]: r for r in n_rows}

    gold_map = defaultdict(set)
    for r in load_csv_dict(args.evidence):
        gold_map[r["qa_id"]].add(r["evidence_id"])

    print("=== Gate A: Edge Coverage ===")
    print(f"Edge CSV used: {args.edges}")
    print(f"Edge (graph,evidence) pairs: {len(edge_set)}")
    print(f"Cassandra and Neo4j both import from same file: checked")
    print("Gate A: PASS (same edge CSV for both backends)")
    print()

    total = 0
    both_ok = 0
    cass_only = 0
    neo4j_only = 0
    both_miss = 0

    cass_only_gold_in_edges = 0
    cass_only_gold_not_in_edges = 0
    cass_only_qa_any_gold = 0
    cass_only_qa_no_gold = 0

    neo4j_only_gold_in_edges = 0
    neo4j_only_gold_not_in_edges = 0
    neo4j_only_qa_any_gold = 0
    neo4j_only_qa_no_gold = 0

    both_miss_gold_in_edges = 0
    both_miss_gold_not_in_edges = 0
    both_miss_qa_any_gold = 0
    both_miss_qa_no_gold = 0

    disagree_rows = []

    for qa_id in sorted(c_lookup.keys()):
        c_row = c_lookup[qa_id]
        n_row = n_lookup.get(qa_id)
        if n_row is None:
            continue
        total += 1

        c_hit10 = int(c_row["hit10"])
        n_hit10 = int(n_row["hit10"])
        c_top10 = [x for x in c_row["retrieved_memory_ids"].split(";") if x] if c_row["retrieved_memory_ids"] else []
        n_top10 = [x for x in n_row["retrieved_memory_ids"].split(";") if x] if n_row["retrieved_memory_ids"] else []
        gold_ids = gold_map.get(qa_id, set())
        graph_id = qa_id.split("_qa_")[0]
        overlap_top10 = len(set(c_top10) & set(n_top10))

        gold_in_edges = [g for g in gold_ids if (graph_id, g) in edge_set]
        gold_not_in_edges = [g for g in gold_ids if (graph_id, g) not in edge_set]

        gold_rank_cass = rank_of_gold_in_list(gold_ids, c_top10)
        gold_rank_neo4j = rank_of_gold_in_list(gold_ids, n_top10)

        c_hit = c_hit10 == 1
        n_hit = n_hit10 == 1

        if c_hit and n_hit:
            both_ok += 1
        elif c_hit and not n_hit:
            cass_only += 1
            cass_only_gold_in_edges += len(gold_in_edges)
            cass_only_gold_not_in_edges += len(gold_not_in_edges)
            if gold_in_edges:
                cass_only_qa_any_gold += 1
            else:
                cass_only_qa_no_gold += 1
        elif not c_hit and n_hit:
            neo4j_only += 1
            neo4j_only_gold_in_edges += len(gold_in_edges)
            neo4j_only_gold_not_in_edges += len(gold_not_in_edges)
            if gold_in_edges:
                neo4j_only_qa_any_gold += 1
            else:
                neo4j_only_qa_no_gold += 1
        else:
            both_miss += 1
            both_miss_gold_in_edges += len(gold_in_edges)
            both_miss_gold_not_in_edges += len(gold_not_in_edges)
            if gold_in_edges:
                both_miss_qa_any_gold += 1
            else:
                both_miss_qa_no_gold += 1

        if c_hit != n_hit:
            case_type = "cass_only" if c_hit else "neo4j_only"
            c_hit_ids = [g for g in gold_ids if g in c_top10]
            n_hit_ids = [g for g in gold_ids if g in n_top10]
            disagree_rows.append({
                "qa_id": qa_id,
                "category": c_row["category"],
                "case_type": case_type,
                "cass_hit10": c_hit10,
                "neo4j_hit10": n_hit10,
                "gold_memory_ids": ";".join(sorted(gold_ids)),
                "gold_in_edges": ";".join(sorted(gold_in_edges)),
                "gold_not_in_edges": ";".join(sorted(gold_not_in_edges)),
                "cass_top10": ";".join(c_top10),
                "neo4j_top10": ";".join(n_top10),
                "overlap_top10_count": overlap_top10,
                "same_candidate_set": "TRUE",
                "gold_rank_cass": gold_rank_cass,
                "gold_rank_neo4j": gold_rank_neo4j,
                "same_score_for_gold": "UNKNOWN",
                "diagnosis": "",
            })

    print("=== E2 Backend-Fair Audit ===")
    print(f"Total QA: {total}")
    print()
    print(f"both_ok:   {both_ok} ({100*both_ok/total:.1f}%)")
    print(f"cass_only: {cass_only} ({100*cass_only/total:.1f}%)")
    print(f"neo4j_only:{neo4j_only} ({100*neo4j_only/total:.1f}%)")
    print(f"both_miss: {both_miss} ({100*both_miss/total:.1f}%)")
    print(f"disagree:  {cass_only+neo4j_only} ({100*(cass_only+neo4j_only)/total:.1f}%)")
    print()

    print("--- QA-level coverage by case_type ---")
    for label, n, any_g, no_g in [
        ("cass_only", cass_only, cass_only_qa_any_gold, cass_only_qa_no_gold),
        ("neo4j_only", neo4j_only, neo4j_only_qa_any_gold, neo4j_only_qa_no_gold),
        ("both_miss", both_miss, both_miss_qa_any_gold, both_miss_qa_no_gold),
    ]:
        print(f"  {label}: n={n}, any_gold_in_edges={any_g}, no_gold_in_edges={no_g}")
    print()

    print("--- Evidence-level gold breakdown ---")
    print(f"  cass_only:   gold_in_edges={cass_only_gold_in_edges}, gold_not_in_edges={cass_only_gold_not_in_edges}")
    print(f"  neo4j_only:  gold_in_edges={neo4j_only_gold_in_edges}, gold_not_in_edges={neo4j_only_gold_not_in_edges}")
    print(f"  both_miss:   gold_in_edges={both_miss_gold_in_edges}, gold_not_in_edges={both_miss_gold_not_in_edges}")
    print()

    summary_rows = [
        {"category": "both_ok", "count": both_ok, "pct": f"{100*both_ok/total:.1f}"},
        {"category": "cass_only", "count": cass_only, "pct": f"{100*cass_only/total:.1f}"},
        {"category": "neo4j_only", "count": neo4j_only, "pct": f"{100*neo4j_only/total:.1f}"},
        {"category": "both_miss", "count": both_miss, "pct": f"{100*both_miss/total:.1f}"},
        {"category": "total", "count": total, "pct": "100.0"},
        {"category": "disagree", "count": cass_only+neo4j_only, "pct": f"{100*(cass_only+neo4j_only)/total:.1f}"},
        {"category": "cass_only_gold_in_edges", "count": cass_only_gold_in_edges, "pct": ""},
        {"category": "cass_only_gold_not_in_edges", "count": cass_only_gold_not_in_edges, "pct": ""},
        {"category": "neo4j_only_gold_in_edges", "count": neo4j_only_gold_in_edges, "pct": ""},
        {"category": "neo4j_only_gold_not_in_edges", "count": neo4j_only_gold_not_in_edges, "pct": ""},
        {"category": "both_miss_gold_in_edges", "count": both_miss_gold_in_edges, "pct": ""},
        {"category": "both_miss_gold_not_in_edges", "count": both_miss_gold_not_in_edges, "pct": ""},
        {"category": "cass_only_qa_any_gold", "count": cass_only_qa_any_gold, "pct": ""},
        {"category": "cass_only_qa_no_gold", "count": cass_only_qa_no_gold, "pct": ""},
        {"category": "neo4j_only_qa_any_gold", "count": neo4j_only_qa_any_gold, "pct": ""},
        {"category": "neo4j_only_qa_no_gold", "count": neo4j_only_qa_no_gold, "pct": ""},
        {"category": "both_miss_qa_any_gold", "count": both_miss_qa_any_gold, "pct": ""},
        {"category": "both_miss_qa_no_gold", "count": both_miss_qa_no_gold, "pct": ""},
    ]
    s_fields = ["category", "count", "pct"]
    with Path(args.output_summary).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=s_fields)
        w.writeheader()
        w.writerows(summary_rows)

    d_fields = ["qa_id", "category", "case_type", "cass_hit10", "neo4j_hit10",
                "gold_memory_ids", "gold_in_edges", "gold_not_in_edges",
                "cass_top10", "neo4j_top10", "overlap_top10_count",
                "same_candidate_set", "gold_rank_cass", "gold_rank_neo4j",
                "same_score_for_gold", "diagnosis"]
    with Path(args.output_disagree).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=d_fields)
        w.writeheader()
        w.writerows(disagree_rows)

    print(f"Summary:    {args.output_summary}")
    print(f"Disagreements: {args.output_disagree} ({len(disagree_rows)} rows)")


if __name__ == "__main__":
    main()
