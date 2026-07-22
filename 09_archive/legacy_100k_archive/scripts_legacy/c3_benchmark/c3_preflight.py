"""
C3-0: graph-scale sweep preflight. Generate 1M graph, import to Cassandra+Neo4j,
generate manifests, run semantic preflight.
"""
import csv
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cassandra.cluster import Cluster
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark_cassandra_internal_ablation import SyntheticGraph

GR_100K = "synth_100000_1781447372"
GR_1M = "c3_synth_1M_seed42"
FANOUT = 20; HOP = 2; MANIFEST_SEED = 20260707
N_QUERIES = 256; SPOTCHECK_N = 64

PROJ = Path("D:/memorytable/cassandra-kg-memory")
OUT_DIR = PROJ / "reports/c3_preflight_scale"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NEO_PWD = os.environ["NEO4J_PASSWORD"]

_cass_session = None; _cass_executor = None; _neo_driver = None
FRONTIER_WORKERS = 16


def init():
    global _cass_session, _cass_executor, _neo_driver
    c = Cluster(["127.0.0.1"], port=9042)
    _cass_session = c.connect("ai_memory")
    _cass_executor = ThreadPoolExecutor(max_workers=FRONTIER_WORKERS)
    _neo_driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", NEO_PWD))


def shutdown():
    _cass_executor.shutdown(wait=True)
    _cass_session.cluster.shutdown()
    _neo_driver.close()


def generate_graph(n_entities, n_edges, graph_id):
    g = SyntheticGraph(n_entities=n_entities, n_edges=n_edges, high_degree_frac=0.02, high_degree_mult=20)
    g.generate()
    edges = []
    for e in g.edges:
        edges.append({"graph_id": graph_id, "src_id": e["src_id"], "relation": e["relation"],
                      "dst_id": e["dst_id"], "source": e["source"]})
    # Deduplicate
    seen = set()
    dedupe = []
    for e in edges:
        lk = (e["graph_id"], e["src_id"], e["relation"], e["dst_id"], e["source"])
        if lk not in seen: seen.add(lk); dedupe.append(e)
    return dedupe


def dataset_summary(edges, graph_id):
    srcs = Counter(e["src_id"] for e in edges)
    deg = sorted(srcs.values())
    high = sum(1 for v in deg if v > 100)
    return {"graph_id": graph_id, "total_edges": len(edges), "distinct_sources": len(srcs),
            "min_outdegree": deg[0], "max_outdegree": deg[-1], "p50_outdegree": deg[len(deg)//2],
            "high_degree_nodes": high}


def import_cassandra(edges):
    ins = _cass_session.prepare(
        "INSERT INTO kg_edges_by_src (graph_id,src_id,relation,dst_id,edge_id,src_type,dst_type,confidence,source,created_at) "
        "VALUES (?,?,?,?,now(),'ENTITY','ENTITY',1.0,?,toTimestamp(now()))")
    for i, e in enumerate(edges):
        _cass_session.execute(ins, (e["graph_id"], e["src_id"], e["relation"], e["dst_id"], e["source"]))
        if (i+1) % 100000 == 0: print(f"  Cass: {i+1}/{len(edges)}", flush=True)
    print(f"  Cass: {len(edges)} done", flush=True)


def import_neo4j(edges, label, rel_type, neo_graph_id):
    with _neo_driver.session() as s:
        s.run(f"MATCH (n:{label}) DETACH DELETE n")
        s.run(f"CREATE CONSTRAINT {label.lower()}_uq IF NOT EXISTS FOR (n:{label}) REQUIRE (n.graph_id, n.node_id) IS UNIQUE")
    nodes = set()
    for e in edges: nodes.add(e["src_id"]); nodes.add(e["dst_id"])
    nl = [{"g": neo_graph_id, "nid": n} for n in nodes]
    for i in range(0, len(nl), 5000):
        with _neo_driver.session() as s:
            s.run(f"UNWIND $rows AS r MERGE (n:{label} {{graph_id: r.g, node_id: r.nid}})", rows=nl[i:i+5000]).consume()
    print(f"  Neo4j: {len(nl)} nodes", flush=True)
    for i in range(0, len(edges), 5000):
        batch = [{"g": neo_graph_id, "s": e["src_id"], "d": e["dst_id"], "rel": e["relation"], "src": e["source"]} for e in edges[i:i+5000]]
        with _neo_driver.session() as s:
            s.run(f"UNWIND $rows AS r MATCH (s:{label} {{graph_id: r.g, node_id: r.s}}) MATCH (d:{label} {{graph_id: r.g, node_id: r.d}}) CREATE (s)-[:{rel_type} {{relation: r.rel, source: r.src, graph_id: r.g}}]->(d)", rows=batch).consume()
    print(f"  Neo4j: {len(edges)} edges", flush=True)


def neo4j_preflight(edges, label, rel_type, neo_graph_id):
    cass_set = set()
    for e in edges:
        cass_set.add((neo_graph_id, e["src_id"], e["relation"], e["dst_id"], e["source"]))
    neo_set = set()
    with _neo_driver.session() as s:
        rows = s.run(f"MATCH (s:{label})-[r:{rel_type}]->(d:{label}) RETURN s.node_id AS sid, r.relation AS rel, d.node_id AS did, r.source AS src")
        for row in rows:
            neo_set.add((neo_graph_id, row["sid"], row["rel"], row["did"], row["src"] or ""))
    missing = sorted(cass_set - neo_set); extra = sorted(neo_set - cass_set)
    m = len(missing) + len(extra)
    print(f"  Preflight: cass={len(cass_set)} neo={len(neo_set)} mismatch={m} {'PASS' if m==0 else 'FAIL'}", flush=True)
    return {"cassandra": len(cass_set), "neo4j": len(neo_set), "mismatches": m, "all_consistent": m == 0}, missing, extra


def cass_fetch(graph_id, src):
    rows = _cass_session.execute("SELECT src_id,relation,dst_id,source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s", (graph_id, src))
    return [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]


def cass_hop(graph_id, sources, rel):
    se = {}
    futures = {_cass_executor.submit(cass_fetch, graph_id, s): s for s in sources}
    for f in as_completed(futures):
        s = futures[f]; se[s] = [e for e in f.result() if e[1] == rel]
    return se


def cass_read(graph_id, q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({s[0] for s in frontier})
        se = cass_hop(graph_id, sources, rel)
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FANOUT]:
                if e[2] not in np: nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({s[2] for s in frontier}))


def neo_read(label, rel_type, neo_graph_id, q):
    frontier = {(q["seed_id"], (q["seed_id"],), ())}
    for rel in q["relation_path"]:
        sources = sorted({s[0] for s in frontier})
        se = {}
        for src in sources:
            with _neo_driver.session() as s:
                rows = list(s.run(
                    f"MATCH (n:{label} {{graph_id: $g, node_id: $n}})-[r:{rel_type} {{relation: $rel}}]->(m:{label} {{graph_id: $g}}) "
                    "RETURN n.node_id AS s,r.relation AS r,m.node_id AS d,coalesce(r.source,'') AS src ORDER BY r,d,src",
                    g=neo_graph_id, n=src, rel=rel))
            se[src] = [(row["s"], row["r"], row["d"], row["src"]) for row in rows]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FANOUT]:
                if e[2] not in np: nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier = nf
        if not frontier: break
    return tuple(sorted({s[2] for s in frontier}))


def path_hash(paths):
    return hashlib.sha256(json.dumps(sorted([list(p) for p in sorted(paths)]), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def generate_manifest(edges, graph_id, n_queries):
    by_src = defaultdict(list)
    for e in edges: by_src[e["src_id"]].append(e)
    all_sources = sorted(by_src.keys())
    rng = random.Random(MANIFEST_SEED + hash(graph_id) % 10000)
    queries = []
    seeds_used = set()
    for _ in range(n_queries * 20):
        if len(queries) >= n_queries: break
        seed = rng.choice(all_sources)
        if seed in seeds_used: continue
        h1 = sorted(by_src.get(seed, []), key=lambda e: (e["relation"], e["dst_id"], e["source"]))[:FANOUT]
        if not h1: continue
        rel_paths = set()
        for e1 in h1:
            h2 = sorted(by_src.get(e1["dst_id"], []), key=lambda e: (e["relation"], e["dst_id"], e["source"]))[:FANOUT]
            for e2 in h2: rel_paths.add((e1["relation"], e2["relation"]))
        if not rel_paths: continue
        seeds_used.add(seed)
        rp = list(rel_paths)[0]
        q = {"query_id": f"c3-{graph_id[-6:]}-{len(queries):06d}", "graph_id": graph_id,
             "seed_id": seed, "relation_path": list(rp), "hop": HOP, "fanout": FANOUT, "cycle_policy": "path"}
        q["expected_path_hash"] = path_hash(cass_read(graph_id, q))
        queries.append(q)
    return queries[:n_queries]


def manifest_summary(edges, queries):
    by_src = defaultdict(list)
    for e in edges: by_src[e["src_id"]].append(e)
    seeds = set(q["seed_id"] for q in queries)
    rel_counts = Counter(str(q["relation_path"]) for q in queries)
    pc = []
    for q in queries:
        frontier = {(q["seed_id"], (q["seed_id"],), ())}
        for rel in q["relation_path"]:
            sources = sorted({s[0] for s in frontier})
            se = {}
            for src in sources:
                all_e = sorted(by_src.get(src, []), key=lambda e: (e["relation"], e["dst_id"], e["source"]))
                se[src] = [e for e in all_e if e["relation"] == rel]
            nf = set()
            for src, np, ep in frontier:
                for e in se.get(src, [])[:FANOUT]:
                    if e[2] not in np: nf.add((e[2], np+(e[2],), ep+(e[3],)))
            frontier = nf
        pc.append(len(frontier))
    pc.sort()
    return {"query_count": len(queries), "distinct_seeds": len(seeds),
            "relation_path_top5": rel_counts.most_common(5),
            "result_path_min": pc[0], "result_path_max": pc[-1],
            "result_path_p50": pc[len(pc)//2], "result_path_p95": pc[int(len(pc)*0.95)]}


def semantic_preflight(edges, label, rel_type, neo_graph_id, queries, n_check):
    rng = random.Random(42)
    checks = rng.sample(queries, min(n_check, len(queries)))
    mp = 0; mn = 0
    for q in checks:
        cass = cass_read(q["graph_id"], q)
        neo = neo_read(label, rel_type, neo_graph_id, q)
        if set(cass) != set(neo): mn += 1
        if len(cass) == 0: mp += 1
    sp = {"checked": len(checks), "empty_queries": mp, "disagreements": mn, "all_pass": mp == 0 and mn == 0}
    print(f"  {len(checks)} checked: empty={mp} disagree={mn} {'PASS' if sp['all_pass'] else 'FAIL'}", flush=True)
    return sp


def main():
    print("=== C3-0 Graph-Scale Preflight ===", flush=True)
    init()

    # --- 1M graph: generate + import ---
    print("\n[1] Generating 1M graph (n_entities=5000, n_edges=1000000)...", flush=True)
    edges_1m = generate_graph(5000, 1000000, GR_1M)
    ds_1m = dataset_summary(edges_1m, GR_1M)
    print(f"  {ds_1m['total_edges']} distinct edges, {ds_1m['distinct_sources']} sources, "
          f"outdegree {ds_1m['min_outdegree']}-{ds_1m['max_outdegree']}", flush=True)
    with (OUT_DIR/"dataset_summary_1m.json").open("w") as f: json.dump(ds_1m, f, indent=2)

    # Save CSV for future reference
    csv_path_1m = PROJ / "results/c3_source_1M.csv"
    with csv_path_1m.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["graph_id","src_id","relation","dst_id","source"])
        w.writeheader(); w.writerows(edges_1m)

    print("\n[2] Import 1M to Cassandra...", flush=True)
    t0 = time.perf_counter()
    import_cassandra(edges_1m)
    print(f"  Cassandra done in {time.perf_counter()-t0:.0f}s", flush=True)

    print("\n[3] Import 1M to Neo4j (C3KGNode/C3KG_EDGE)...", flush=True)
    t0 = time.perf_counter()
    import_neo4j(edges_1m, "C3KGNode", "C3KG_EDGE", GR_1M)
    print(f"  Neo4j done in {time.perf_counter()-t0:.0f}s", flush=True)

    print("\n[4] 1M full-graph preflight...", flush=True)
    fp_1m, miss_1m, ext_1m = neo4j_preflight(edges_1m, "C3KGNode", "C3KG_EDGE", GR_1M)
    with (OUT_DIR/"full_graph_import_preflight_1m.json").open("w") as f: json.dump(fp_1m, f, indent=2)
    if miss_1m or ext_1m:
        with (OUT_DIR/"full_graph_import_mismatches_1m.jsonl").open("w") as f:
            for m in miss_1m[:50]: f.write(json.dumps({"type":"missing","key":list(m)})+"\n")
            for e in ext_1m[:50]: f.write(json.dumps({"type":"extra","key":list(e)})+"\n")

    if fp_1m["mismatches"] > 0:
        print("ABORT: 1M preflight failed", flush=True); shutdown(); return

    # --- 100K dataset summary ---
    print("\n[5] 100K dataset summary...", flush=True)
    # Use the same generator to get 100K stats
    edges_100k = generate_graph(5000, 100000, GR_100K)
    ds_100k = dataset_summary(edges_100k, GR_100K)
    with (OUT_DIR/"dataset_summary_100k.json").open("w") as f: json.dump(ds_100k, f, indent=2)
    print(f"  {ds_100k['total_edges']} edges, {ds_100k['distinct_sources']} sources, "
          f"outdegree {ds_100k['min_outdegree']}-{ds_100k['max_outdegree']}", flush=True)

    # --- Manifests ---
    print("\n[6] Generating manifests...", flush=True)
    q_100k = generate_manifest(edges_100k, GR_100K, N_QUERIES)
    q_1m = generate_manifest(edges_1m, GR_1M, N_QUERIES)
    ms_100k = manifest_summary(edges_100k, q_100k)
    ms_1m = manifest_summary(edges_1m, q_1m)

    for qs, path, ms in [(q_100k, "results/c3_manifest_100k_h2.jsonl", ms_100k),
                          (q_1m, "results/c3_manifest_1m_h2.jsonl", ms_1m)]:
        with (PROJ/path).open("w") as f:
            for q in qs: f.write(json.dumps(q, ensure_ascii=False)+"\n")

    with (OUT_DIR/"manifest_summary_100k.json").open("w") as f: json.dump(ms_100k, f, indent=2)
    with (OUT_DIR/"manifest_summary_1m.json").open("w") as f: json.dump(ms_1m, f, indent=2)

    comp = {
        "100K": {"queries": ms_100k["query_count"], "distinct_seeds": ms_100k["distinct_seeds"],
                  "path_min": ms_100k["result_path_min"], "path_p50": ms_100k["result_path_p50"],
                  "path_p95": ms_100k["result_path_p95"], "path_max": ms_100k["result_path_max"]},
        "1M": {"queries": ms_1m["query_count"], "distinct_seeds": ms_1m["distinct_seeds"],
               "path_min": ms_1m["result_path_min"], "path_p50": ms_1m["result_path_p50"],
               "path_p95": ms_1m["result_path_p95"], "path_max": ms_1m["result_path_max"]},
        "notes": "1M graph has same entities (5000) with 200 edges each (vs 20). 2-hop path cardinality scales accordingly."
    }
    with (OUT_DIR/"manifest_difficulty_comparison.json").open("w") as f: json.dump(comp, f, indent=2)

    print(f"  100K manifest: {ms_100k}")
    print(f"  1M manifest: {ms_1m}")

    # --- Semantic preflight ---
    print("\n[7] Semantic preflight (64 queries each)...", flush=True)
    sp_100k = semantic_preflight(edges_100k, "C1KGNode", "C1KG_EDGE", GR_100K, q_100k, SPOTCHECK_N)
    sp_1m = semantic_preflight(edges_1m, "C3KGNode", "C3KG_EDGE", GR_1M, q_1m, SPOTCHECK_N)

    with (OUT_DIR/"semantic_preflight_100k.json").open("w") as f: json.dump(sp_100k, f, indent=2)
    with (OUT_DIR/"semantic_preflight_1m.json").open("w") as f: json.dump(sp_1m, f, indent=2)

    shutdown()
    all_ok = fp_1m["all_consistent"] and sp_100k["all_pass"] and sp_1m["all_pass"]
    print(f"\n{'='*60}")
    print(f"C3-0 {'ALL PASS' if all_ok else 'ISSUES'}")
    print(f"  1M import: mismatch={fp_1m['mismatches']}")
    print(f"  100K semantic: empty={sp_100k['empty_queries']} disagree={sp_100k['disagreements']}")
    print(f"  1M semantic: empty={sp_1m['empty_queries']} disagree={sp_1m['disagreements']}")


if __name__ == "__main__":
    main()
