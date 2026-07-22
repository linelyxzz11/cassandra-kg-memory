import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20260707
N_QUERIES = 256
HOP = 2
FANOUT = 20
GRAPH_ID = "c1_synth_100k_seed42"

CSV_PATH = Path("D:/memorytable/cassandra-kg-memory/results/c1_source_100k.csv")
MANIFEST_PATH = Path("D:/memorytable/cassandra-kg-memory/results/c1_manifest_100k_h2.jsonl")
REPORT_DIR = Path("D:/memorytable/cassandra-kg-memory/reports/c1_preflight_100k")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_graph():
    by_src = defaultdict(list)
    with CSV_PATH.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            by_src[row["src_id"]].append({
                "relation": row["relation"],
                "dst_id": row["dst_id"],
                "source": row["source"],
            })
    return by_src


def trace_two_hop(by_src, seed_id, max_fanout=FANOUT):
    hop1 = by_src.get(seed_id, [])
    if not hop1:
        return None, None, 0
    hop1_sorted = sorted(hop1, key=lambda e: (e["relation"], e["dst_id"], e["source"]))
    hop1_fanout = hop1_sorted[:max_fanout]

    paths = []
    for e1 in hop1_fanout:
        rel1 = e1["relation"]
        hop2 = by_src.get(e1["dst_id"], [])
        if not hop2:
            continue
        hop2_sorted = sorted(hop2, key=lambda e: (e["relation"], e["dst_id"], e["source"]))
        hop2_fanout = hop2_sorted[:max_fanout]
        for e2 in hop2_fanout:
            rel2 = e2["relation"]
            path_key = (rel1, e1["source"], e1["dst_id"], rel2, e2["source"], e2["dst_id"])
            paths.append(path_key)

    if not paths:
        return None, None, 0

    # Deduplicate paths (same edge ID path)
    path_set = sorted(set(paths))
    # Extract relations used
    rel_paths_used = set()
    for p in paths:
        rel_paths_used.add((p[0], p[3]))

    return path_set, rel_paths_used, len(hop1_fanout)


def path_hash(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(json.dumps(p, sort_keys=True).encode())
    return h.hexdigest()[:16]


def main():
    print(f"Loading graph from {CSV_PATH}...")
    by_src = load_graph()
    all_sources = sorted(by_src.keys())
    print(f"  {len(all_sources)} source nodes")

    rng = random.Random(SEED)
    queries = []
    seeds_tried = set()
    rel_path_counts = defaultdict(int)
    path_counts = []

    for _ in range(N_QUERIES * 10):
        if len(queries) >= N_QUERIES:
            break
        seed = rng.choice(all_sources)
        if seed in seeds_tried:
            continue
        paths, rels, _ = trace_two_hop(by_src, seed)
        if not paths or not rels:
            continue
        seeds_tried.add(seed)
        # Pick first available relation pair
        rel_pair = list(rels)[0]
        queries.append({
            "query_id": f"c1-read-{len(queries):06d}",
            "graph_id": GRAPH_ID,
            "seed_id": seed,
            "relation_path": list(rel_pair),
            "hop": HOP,
            "fanout": FANOUT,
            "cycle_policy": "path",
            "expected_path_hash": path_hash(paths),
        })
        rel_path_counts[str(rel_pair)] += 1
        path_counts.append(len(paths))

    if len(queries) < N_QUERIES:
        print(f"  WARNING: only generated {len(queries)}/{N_QUERIES} queries")
        N = len(queries)
    else:
        N = N_QUERIES

    # Shuffle queries for final output order
    rng.shuffle(queries)
    queries = queries[:N]

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    path_counts_sorted = sorted(path_counts)
    mn, mx = path_counts_sorted[0], path_counts_sorted[-1]
    p50 = path_counts_sorted[len(path_counts_sorted) // 2]
    p95 = path_counts_sorted[int(len(path_counts_sorted) * 0.95)]

    manifest_summary = {
        "manifest_path": str(MANIFEST_PATH.resolve()),
        "query_count": N,
        "distinct_seeds": len(seeds_tried),
        "relation_path_distribution": dict(rel_path_counts.most_common() if hasattr(rel_path_counts, 'most_common') else sorted(rel_path_counts.items(), key=lambda x: -x[1])[:10]),
        "result_path_count_min": mn,
        "result_path_count_max": mx,
        "result_path_count_p50": p50,
        "result_path_count_p95": p95,
    }
    with (REPORT_DIR / "manifest_summary.json").open("w") as f:
        json.dump(manifest_summary, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {N} queries")
    print(f"  Distinct seeds: {len(seeds_tried)}")
    print(f"  Result paths: min={mn} p50={p50} p95={p95} max={mx}")
    print(json.dumps(manifest_summary, indent=2, ensure_ascii=False))

    print(f"\nSample queries from {MANIFEST_PATH}:")
    with MANIFEST_PATH.open() as f:
        lines = f.readlines()
        for line in lines[:3]:
            print(json.dumps(json.loads(line), indent=2))


if __name__ == "__main__":
    main()
