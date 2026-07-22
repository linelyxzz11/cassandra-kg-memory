import csv, json, os
from pathlib import Path
from datetime import datetime

PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT = PROJ / "reports/sysaxis_final"
OUT.mkdir(parents=True, exist_ok=True)
NOW = datetime.utcnow().isoformat()

wr_rows = []
with (PROJ / "reports/sysaxis_1m_write_ratio_final/final_write_ratio_summary_cold_warm.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): wr_rows.append(r)

conc_rows = []
with (PROJ / "reports/sysaxis_1m_concurrency_final/final_concurrency_summary_cold_warm.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): conc_rows.append(r)

hop_rows = []
with (PROJ / "reports/sysaxis_1m_hop_depth_final/final_hop_depth_summary_cold_warm.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): hop_rows.append(r)

scale_rows = []
with (PROJ / "reports/sysaxis_scale_sweep_final/scale_sweep_supporting_100K_to_1M_cold_warm.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): scale_rows.append(r)

print(f"Read: wr={len(wr_rows)} conc={len(conc_rows)} hop={len(hop_rows)} scale={len(scale_rows)}")

MF = ["axis","scale_label","actual_edges","graph_id","graph_type","mode","system","clients","hop","write_ratio","n",
      "read_QPS","write_QPS","mean_ms","p50_ms","p95_ms","p99_ms","write_mean_ms","write_p95_ms","write_p99_ms",
      "cache_enabled","cache_hit_rate","effective_latency_ms","correctness_status","source_summary_file","notes"]

def sf(v, d=0):
    try: return float(v) if v else d
    except: return d

master = []

for r in wr_rows:
    master.append({"axis":"write_ratio","scale_label":"1M_scale_controlled","actual_edges":1000000,
        "graph_id":"c3_scale_1M_seed42","graph_type":"scale_controlled","mode":r["mode"],"system":r["system"],
        "clients":32,"hop":2,"write_ratio":float(r["write_ratio"]),"n":r["n"],
        "read_QPS":sf(r.get("median_read_QPS")),"write_QPS":sf(r.get("median_write_QPS")),
        "mean_ms":sf(r.get("median_read_mean_ms")),"p50_ms":None,
        "p95_ms":sf(r.get("median_read_p95_ms")),"p99_ms":sf(r.get("median_read_p99_ms")),
        "write_mean_ms":sf(r.get("median_write_mean_ms")),"write_p95_ms":sf(r.get("median_write_p95_ms")),
        "write_p99_ms":sf(r.get("median_write_p99_ms")),"cache_enabled":False,"cache_hit_rate":0,
        "effective_latency_ms":sf(r.get("median_read_mean_ms")),"correctness_status":"PASS",
        "source_summary_file":"reports/sysaxis_1m_write_ratio_final/final_write_ratio_summary_cold_warm.csv",
        "notes":"wr="+r["write_ratio"]})

for r in conc_rows:
    master.append({"axis":"concurrency","scale_label":"1M_scale_controlled","actual_edges":1000000,
        "graph_id":"c3_scale_1M_seed42","graph_type":"scale_controlled","mode":r["mode"],"system":r["system"],
        "clients":int(float(r["clients"])),"hop":2,"write_ratio":0,"n":r["n"],
        "read_QPS":sf(r.get("median_QPS")),"write_QPS":None,"mean_ms":sf(r.get("median_mean_ms")),
        "p50_ms":sf(r.get("median_p50_ms")),"p95_ms":sf(r.get("median_p95_ms")),
        "p99_ms":sf(r.get("median_p99_ms")),"write_mean_ms":None,"write_p95_ms":None,"write_p99_ms":None,
        "cache_enabled":False,"cache_hit_rate":0,"effective_latency_ms":sf(r.get("median_mean_ms")),
        "correctness_status":"PASS",
        "source_summary_file":"reports/sysaxis_1m_concurrency_final/final_concurrency_summary_cold_warm.csv","notes":""})

for r in hop_rows:
    master.append({"axis":"hop_depth","scale_label":"1M_scale_controlled","actual_edges":1000000,
        "graph_id":"c3_scale_1M_seed42","graph_type":"scale_controlled","mode":r["mode"],"system":r["system"],
        "clients":8,"hop":int(r["hop"]),"write_ratio":0,"n":r["n"],
        "read_QPS":sf(r.get("median_QPS")),"write_QPS":None,"mean_ms":sf(r.get("median_mean_ms")),
        "p50_ms":sf(r.get("median_p50_ms")),"p95_ms":sf(r.get("median_p95_ms")),
        "p99_ms":sf(r.get("median_p99_ms")),"write_mean_ms":None,"write_p95_ms":None,"write_p99_ms":None,
        "cache_enabled":False,"cache_hit_rate":0,"effective_latency_ms":sf(r.get("median_mean_ms")),
        "correctness_status":"PASS",
        "source_summary_file":"reports/sysaxis_1m_hop_depth_final/final_hop_depth_summary_cold_warm.csv",
        "notes":"hop="+r["hop"]})

for r in scale_rows:
    master.append({"axis":"scale_supporting","scale_label":r["scale_label"],
        "actual_edges":int(r["actual_edges"]),"graph_id":r["graph_id"],"graph_type":r["graph_type"],
        "mode":r["mode"],"system":r["system"],"clients":int(r["clients"]),"hop":int(r["hop"]),
        "write_ratio":int(r["write_ratio"]),"n":r["n"],"read_QPS":sf(r["read_QPS"]),
        "write_QPS":sf(r["write_QPS"]),"mean_ms":sf(r["read_mean_ms"]),"p50_ms":None,
        "p95_ms":sf(r["read_p95_ms"]),"p99_ms":sf(r["read_p99_ms"]),
        "write_mean_ms":sf(r["write_mean_ms"]),"write_p95_ms":sf(r["write_p95_ms"]),
        "write_p99_ms":sf(r["write_p99_ms"]),"cache_enabled":False,"cache_hit_rate":0,
        "effective_latency_ms":sf(r["effective_latency_ms"]),"correctness_status":"PASS_WITH_CAVEAT",
        "source_summary_file":"reports/sysaxis_scale_sweep_final/scale_sweep_supporting_100K_to_1M_cold_warm.csv",
        "notes":r["notes"]})

with (OUT/"sysaxis_master_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=MF,extrasaction="ignore"); w.writeheader(); w.writerows(master)
with (OUT/"sysaxis_master_summary.json").open("w") as f: json.dump(master,f,indent=2)

# Index
with (OUT/"sysaxis_master_index.md").open("w") as f:
    f.write(f"""# Sysaxis Master Index
Generated: {NOW}

## 1. 1M Write-Ratio Sweep
- Dir: reports/sysaxis_1m_write_ratio_final/
- Graph: c3_scale_1M_seed42 (1M edges, scale-controlled)
- Systems: cassandra, neo4j
- Config: clients=32, hop=2, write_ratio=0/10/30%, modes=cold/warm, repeats=5
- Trials: 60 (2x3x2x5)
- Correctness: PASS (guard 1M/1M, hash gates pass, failures empty)
- Summary: final_write_ratio_summary_cold_warm.csv

## 2. 1M Concurrency Sweep
- Dir: reports/sysaxis_1m_concurrency_final/
- Graph: c3_scale_1M_seed42 (1M edges, scale-controlled)
- Systems: cassandra_opt, cassandra_naive, neo4j
- Config: hop=2, clients=1/4/8/16/32/64, modes=cold/warm, repeats=3
- Trials: 108
- Correctness: PASS (hash gates, semantic gates, failures empty)
- Summary: final_concurrency_summary_cold_warm.csv

## 3. 1M Hop-Depth Sweep
- Dir: reports/sysaxis_1m_hop_depth_final/
- Graph: c3_scale_1M_seed42 (1M edges, scale-controlled)
- Systems: cassandra_opt, neo4j
- Config: clients=8, hop=1/2/3/4, modes=cold/warm, repeats=5
- Trials: 80 (2x4x2x5)
- Correctness: PASS (80/80 spotchecks, 16/16 hash gates, guard 1M/1M, failures empty)
- Summary: final_hop_depth_summary_cold_warm.csv

## 4. Supporting Scale: 100K -> 1M
- Dir: reports/sysaxis_scale_sweep_final/
- Graphs: sysaxis_100K_legacy_clean_20260709 (99,875 edges) + c3_scale_1M_seed42 (1M)
- Systems: cassandra_opt/neo4j (100K); cassandra/neo4j (1M)
- Config: clients=32, hop=2, write_ratio=10%, modes=cold/warm
- Trials: 20 (100K) + 12 (1M wr=10% subset)
- Correctness: PASS_WITH_CAVEAT (100K rebuilt from legacy c1_source_100k.csv)
- Summary: scale_sweep_supporting_100K_to_1M_cold_warm.csv
""")

# Key findings
with (OUT/"sysaxis_key_findings.md").open("w") as f:
    f.write(f"""# Sysaxis Key Findings
Generated: {NOW}

## 1M Write-Ratio
At 1M edges, clients=32, hop=2, under 0/10/30% write ratios, Cassandra-KG consistently maintains lower read P95/P99 latency than Neo4j, while Neo4j has lower per-write latency.

10% write cold: Cass read QPS=425, p99=174ms vs Neo read QPS=322, p99=697ms (4.0x). Write: Cass p99=116ms vs Neo p99=46ms (Neo 2.5x faster).
10% write warm: Cass read QPS=586, p99=105ms vs Neo read QPS=428, p99=488ms (4.6x). Write: Cass p99=67ms vs Neo p99=27ms (Neo 2.5x faster).

## 1M Concurrency
Neo4j is strong at low concurrency (clients=1), but tail latency grows sharply at high concurrency. Cassandra opt keeps p99 below 190ms across all client levels and cold/warm modes.

Warm clients=64: Cass opt p99=189.2ms vs Neo4j p99=1194.8ms. Cassandra naive sometimes has higher QPS but worse tails.

## 1M Hop-Depth
Crossover at hop=2. Neo4j wins hop=1; Cassandra-KG wins hop=2+, with ~3x lower p99 at hop=4 in both cold and warm modes.

Warm hop=4: Cass QPS=138, mean=58.0ms, p99=141.9ms vs Neo QPS=108, mean=74.4ms, p99=424.8ms (3.0x).
Cold crossover hop=2: Cass QPS=521, p99=51.8ms vs Neo QPS=482, p99=111.8ms.

## Scale Supporting: 100K -> 1M
The clean legacy 100K point (99,875 edges) shows qualitative consistency with 1M (hop=2, clients=32, wr=10%):
- Cass read p99 advantage: 4.0x (both scales)
- Neo write advantage: 2.4x (100K) and 2.5x (1M)
- Cass warmup benefit larger at 1M (+38% QPS) than 100K (+0.3%)

Qualitative consistency, not strict scaling linearity (different graph generators).
""")

# Caveats
with (OUT/"sysaxis_caveats.md").open("w") as f:
    f.write(f"""# Sysaxis Caveats
Generated: {NOW}

1. Cold mode is process-cold (fresh Python process), not strict OS/database cache flush.
2. Application cache is disabled in all system-axis main runs; cache_hit_rate=0 and effective_latency equals measured read latency.
3. 100K scale point is legacy synthetic rebuilt from c1_source_100k.csv. Actual distinct edges=99,875 (125 CSV source duplicates removed). Not scale-controlled.
4. 1M point is scale-controlled with exactly 1,000,000 distinct logical edges.
5. Scale comparison supports qualitative consistency, not strict scaling linearity (different generators).
6. Cassandra writes use denormalized multi-table writes (4 tables); Neo4j writes are lower per-edge latency.
7. Cassandra naive sometimes has higher QPS than Cassandra opt under concurrency, but with worse tail latency.
8. All hop-depth and concurrency experiments are read-only (write_ratio=0). Write-ratio experiments add mixed read/write.
9. Hop-depth uses 8 clients; write-ratio and concurrency use 32 clients (except concurrency client=1/4/8/16/64 variants).
""")

# Artifact manifest
am = {"created_at":NOW,"input_directories":["reports/sysaxis_1m_write_ratio_final/","reports/sysaxis_1m_concurrency_final/","reports/sysaxis_1m_hop_depth_final/","reports/sysaxis_scale_sweep_final/"],
    "input_files":["final_write_ratio_summary_cold_warm.csv","final_concurrency_summary_cold_warm.csv","final_hop_depth_summary_cold_warm.csv","scale_sweep_supporting_100K_to_1M_cold_warm.csv"],
    "output_files":["sysaxis_master_index.md","sysaxis_master_summary.csv","sysaxis_master_summary.json","sysaxis_key_findings.md","sysaxis_caveats.md","sysaxis_artifact_manifest.json"],
    "axis_status":{"write_ratio":"PASS","concurrency":"PASS","hop_depth":"PASS","scale_supporting":"PASS_WITH_CAVEAT"},
    "trial_counts":{"write_ratio":60,"concurrency":108,"hop_depth":80,"scale_supporting_100K":20,"total":268},
    "correctness_gates":{"graph_guards":"PASS (1M 1M/1M, 100K 99875/99875)","hash_gates":"PASS (all 256/256, 0 mismatch, 0 empty)","semantic_gates":"PASS","spotchecks":"PASS (all trials 10/10)","failures":"empty (all axes)"},
    "known_caveats":["100K is legacy synthetic, not scale-controlled","Scale comparison qualitative only, not same generator","Cache disabled, cold is process-cold","Hop-depth uses 8 clients, others use 32"]}
with (OUT/"sysaxis_artifact_manifest.json").open("w") as f: json.dump(am,f,indent=2)

print(f"Done. {len(master)} rows in master summary. 6 files in reports/sysaxis_final/")
for fn in sorted(os.listdir(OUT)):
    print(f"  {fn} ({ (OUT/fn).stat().st_size} bytes)")
