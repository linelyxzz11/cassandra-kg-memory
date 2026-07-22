import csv, json
from pathlib import Path

PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT = PROJ / "reports"
OUT.mkdir(parents=True, exist_ok=True)

candidates = []

# === SAMPLE-SCOPED (preferred) ===
ss = {
    "BM25": (0.2649, 0.5619, 0.3600),
    "Dense-bge": (0.3419, 0.7009, 0.4534),
    "Dense-bge+GlobalKG": (0.3872, 0.7095, 0.4851),
    "Dense-bge+QueryKG": (0.3585, 0.7185, 0.4724),
}
for m, (r1, r10, mrr) in ss.items():
    candidates.append(dict(
        method=m, candidate_display_name=m+" (sample-scoped)",
        R1=r1, R10=r10, MRR=mrr, avg_latency="", p95_latency="",
        source_file="results/sample_scoped/sample_scoped_retrieval_summary.csv",
        source_script="locomo_retrieval_sample_scoped.py",
        scope="sample_scoped", notes="cross_sample_rate=0, 5882 corpus, 1986 QA",
        usable_for_main_table=True, latency_missing=True, quality_missing=False))

# === GLOBAL retrieval ===
gm = [
    ("BM25", 0.2160, 0.4708, 0.2947, "results/locomo_bm25_results.csv"),
    ("Dense-bge", 0.3343, 0.6803, 0.4405, "results/locomo_dense_bge_results.csv"),
    ("Dense+GlobalKG_best", 0.3389, 0.6833, 0.4438, "results/locomo_dense_kg_boost_best_results.csv"),
    ("Dense+QueryKG_best", 0.3474, 0.6803, 0.4522, "results/query_kg_rerank/dense_bge_query_kg_rerank_best_results.csv"),
]
for m, r1, r10, mrr, f in gm:
    candidates.append(dict(
        method=m, candidate_display_name=m+" (global)",
        R1=r1, R10=r10, MRR=mrr, avg_latency="", p95_latency="",
        source_file=f, source_script="locomo_retrieval_*.py",
        scope="global", notes="Global-corpus, cross-sample contamination possible",
        usable_for_main_table=False, latency_missing=True, quality_missing=False))

# === CASSANDRA-KG variants ===
kg_v = [
    ("TF-IDF+KG boost (global)", 0.2638, 0.4899, 0.3358, "", "",
     "results/locomo_cassandra_kg_results.csv", "locomo_cassandra_retrieval.py", "global",
     "TF-IDF+Cassandra-KG boost=0.5. NOT pure KG. Identical to Neo4j (0 disagreements).",
     False, True, False),
    ("pure KG signal (not in retrieval results)", "", "", "", "", "",
     "results/e2_backend_fair_audit_summary.csv", "audit_e2_backend_fair.py", "unknown",
     "Pure KG only retrieval not run standalone. Audit: both_ok=973/1986, disagree=0.",
     False, True, True),
    ("1M synthetic hop=2 cold latency (NOT LoCoMo)", "", "", "", "15.4", "32.7",
     "reports/sysaxis_1m_hop_depth_final/", "hop_cold_sweep.py", "system_axis",
     "1M synthetic, clients=8, hop=2. Do NOT mix with LoCoMo quality.",
     False, False, True),
    ("1M synthetic hop=2 warm latency (NOT LoCoMo)", "", "", "", "14.9", "32.6",
     "reports/sysaxis_1m_hop_depth_final/", "hop_warm_sweep.py", "system_axis",
     "1M synthetic, clients=8, hop=2 warm. Do NOT mix with LoCoMo quality.",
     False, False, True),
    ("100K legacy scale sweep cold (NOT LoCoMo)", "", "", "", "50.3", "93.3",
     "reports/sysaxis_scale_sweep_final/", "scale_100k_legacy_clean_fixed.py", "system_axis",
     "100K, clients=32, hop=2, wr=10%. Mixed read/write. Do NOT mix with LoCoMo.",
     False, False, True),
    ("100K legacy scale sweep warm (NOT LoCoMo)", "", "", "", "50.1", "92.3",
     "reports/sysaxis_scale_sweep_final/", "scale_100k_legacy_clean_fixed.py", "system_axis",
     "100K, clients=32, hop=2, wr=10% warm. Do NOT mix with LoCoMo.",
     False, False, True),
]
for m, r1, r10, mrr, lat, p95, sf, ss, sc, notes, usable, lm, qm in kg_v:
    candidates.append(dict(
        method="Cassandra-KG", candidate_display_name="Cassandra-KG "+m,
        R1=r1, R10=r10, MRR=mrr, avg_latency=lat, p95_latency=p95,
        source_file=sf, source_script=ss, scope=sc, notes=notes,
        usable_for_main_table=usable, latency_missing=lm, quality_missing=qm))

# === NEO4J-KG variants ===
neo_v = [
    ("TF-IDF+KG boost (global)", 0.2638, 0.4899, 0.3358, "", "",
     "results/locomo_neo4j_results.csv", "locomo_neo4j_retrieval.py", "global",
     "Identical to Cassandra-KG TF-IDF+boost. 0 disagreements.", False, True, False),
    ("1M synthetic hop=2 cold latency (NOT LoCoMo)", "", "", "", "16.6", "34.2",
     "reports/sysaxis_1m_hop_depth_final/", "hop_cold_sweep.py", "system_axis",
     "1M synthetic, hop=2 cold.", False, False, True),
    ("1M synthetic hop=2 warm latency (NOT LoCoMo)", "", "", "", "15.6", "31.5",
     "reports/sysaxis_1m_hop_depth_final/", "hop_warm_sweep.py", "system_axis",
     "1M synthetic, hop=2 warm.", False, False, True),
    ("100K legacy scale sweep cold (NOT LoCoMo)", "", "", "", "64.9", "160.1",
     "reports/sysaxis_scale_sweep_final/", "scale_100k_legacy_clean_fixed.py", "system_axis",
     "100K, clients=32, hop=2, wr=10%.", False, False, True),
    ("100K legacy scale sweep warm (NOT LoCoMo)", "", "", "", "64.2", "154.2",
     "reports/sysaxis_scale_sweep_final/", "scale_100k_legacy_clean_fixed.py", "system_axis",
     "100K, clients=32, hop=2, wr=10% warm.", False, False, True),
]
for m, r1, r10, mrr, lat, p95, sf, ss, sc, notes, usable, lm, qm in neo_v:
    candidates.append(dict(
        method="Neo4j-KG", candidate_display_name="Neo4j-KG "+m,
        R1=r1, R10=r10, MRR=mrr, avg_latency=lat, p95_latency=p95,
        source_file=sf, source_script=ss, scope=sc, notes=notes,
        usable_for_main_table=usable, latency_missing=lm, quality_missing=qm))

# Sort: main table candidates first
candidates.sort(key=lambda c: (not c["usable_for_main_table"], c["candidate_display_name"]))

fields = ["method","candidate_display_name","R@1","R@10","MRR","avg_latency_ms","p95_latency_ms",
          "source_file","source_script","scope","notes","usable_for_main_table","latency_missing","quality_missing"]
field_map = {"R1":"R@1","R10":"R@10","avg_latency":"avg_latency_ms","p95_latency":"p95_latency_ms"}

with (OUT/"locomo_retrieval_table_candidates.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for c in candidates:
        row = {}
        for k in fields:
            v = c.get(k) or c.get({"R@1":"R1","R@10":"R10","avg_latency_ms":"avg_latency","p95_latency_ms":"p95_latency"}.get(k)) or ""
            row[k] = v
        w.writerow(row)

print(f"Wrote {len(candidates)} rows to reports/locomo_retrieval_table_candidates.csv")
for c in candidates:
    q = f"R@1={c['R1']} R@10={c['R10']} MRR={c['MRR']}" if c['R1'] != '' else "quality=N/A"
    l = f"avg={c['avg_latency']}ms p95={c['p95_latency']}ms" if c['avg_latency'] else "latency=N/A"
    mark = "[MAIN]" if c['usable_for_main_table'] else ""
    print(f"  {mark} {c['candidate_display_name']:60s} {q:50s} {l}")
