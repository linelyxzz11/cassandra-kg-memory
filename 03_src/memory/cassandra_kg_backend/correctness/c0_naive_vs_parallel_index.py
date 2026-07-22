import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cassandra.cluster import Cluster


SOURCE_QUERY = (
    "SELECT src_id, relation, dst_id, source FROM kg_edges_by_src "
    "WHERE graph_id=%s AND src_id=%s"
)
INDEX_QUERY = (
    "SELECT src_id, relation, dst_id, source FROM kg_edges_by_src_relation "
    "WHERE graph_id=%s AND src_id=%s AND relation=%s"
)


def fetch_all(session, graph_id, src_id):
    rows = session.execute(SOURCE_QUERY, (graph_id, src_id))
    edges = [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]
    edges.sort(key=lambda e: (e[1], e[2], e[3]))
    return edges


def fetch_filtered(session, graph_id, src_id, relation):
    rows = session.execute(INDEX_QUERY, (graph_id, src_id, relation))
    edges = [(r.src_id, r.relation, r.dst_id, str(r.source or "")) for r in rows]
    edges.sort(key=lambda e: (e[1], e[2], e[3]))
    return edges


def logical_key(edge):
    return (edge[0], edge[1], edge[2], edge[3])


def preflight_check(session, graph_id, seed, relation_path, fanout):
    print("\n=== INDEX PREFLIGHT ===")
    mismatches = []
    frontier = {(seed, (seed,), ())}
    all_queries = defaultdict(set)

    for depth, relation in enumerate(relation_path):
        sources = sorted({s[0] for s in frontier})
        print(f"  Hop {depth}: relation={relation}, {len(sources)} frontier nodes")
        for src in sources:
            all_queries[src].add(relation)

        src_edges = {}
        for src in sources:
            base_all = fetch_all(session, graph_id, src)
            base_filtered = [e for e in base_all if e[1] == relation]
            index_result = fetch_filtered(session, graph_id, src, relation)
            base_set = {logical_key(e) for e in base_filtered}
            idx_set = {logical_key(e) for e in index_result}
            if base_set != idx_set:
                missing = sorted(base_set - idx_set)
                extra = sorted(idx_set - base_set)
                print(f"    MISMATCH {src}/{relation}: base={len(base_set)} idx={len(idx_set)} missing={len(missing)} extra={len(extra)}")
                mismatches.append({
                    "src_id": src,
                    "relation": relation,
                    "base_count": len(base_set),
                    "index_count": len(idx_set),
                    "missing": [list(m) for m in missing[:10]],
                    "extra": [list(e) for e in extra[:10]],
                })
            src_edges[src] = base_filtered

        next_frontier = set()
        for src, node_path, edge_path in frontier:
            edges = src_edges.get(src, [])
            candidates = [e for e in edges if e[2] not in node_path]
            for _, rel, dst, source in candidates[:fanout]:
                next_frontier.add((dst, node_path + (dst,), edge_path + (source,)))
        frontier = next_frontier
        if not frontier:
            break

    n_checked = sum(len(v) for v in all_queries.values())
    preflight = {
        "partitions_checked": n_checked,
        "distinct_sources": len(all_queries),
        "mismatches": len(mismatches),
        "all_consistent": len(mismatches) == 0,
        "detail": mismatches[:50],
    }
    print(f"  Checked: {n_checked} partitions, {len(all_queries)} distinct sources")
    print(f"  Mismatches: {len(mismatches)}")
    return preflight, all_queries


def traverse(session, seed, graph_id, relation_path, max_depth, fanout, workers, use_index):
    t0 = time.perf_counter()
    frontier = {(seed, (seed,), ())}
    total_raw = 0
    n_queries = 0

    for depth in range(max_depth):
        relation = relation_path[depth]
        sources = sorted({s[0] for s in frontier})
        src_edges = {}
        if workers == 1:
            for src in sources:
                if use_index:
                    edges = fetch_filtered(session, graph_id, src, relation)
                else:
                    edges_all = fetch_all(session, graph_id, src)
                    total_raw += len(edges_all)
                    edges = [e for e in edges_all if e[1] == relation]
                if not use_index:
                    pass  # raw already counted above
                else:
                    total_raw += len(edges)
                n_queries += 1
                src_edges[src] = edges
        else:
            if use_index:
                def _do(src):
                    edges = fetch_filtered(session, graph_id, src, relation)
                    return src, edges, len(edges)
            else:
                def _do(src):
                    edges_all = fetch_all(session, graph_id, src)
                    edges = [e for e in edges_all if e[1] == relation]
                    return src, edges, len(edges_all)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_do, src): src for src in sources}
                for f in as_completed(futures):
                    src, edges, raw = f.result()
                    total_raw += raw
                    n_queries += 1
                    src_edges[src] = edges

        next_frontier = set()
        for src, node_path, edge_path in frontier:
            edges = src_edges.get(src, [])
            candidates = [e for e in edges if e[2] not in node_path]
            for _, rel, dst, source in candidates[:fanout]:
                next_frontier.add((dst, node_path + (dst,), edge_path + (source,)))
        frontier = next_frontier
        if not frontier:
            break

    elapsed = (time.perf_counter() - t0) * 1000
    paths = tuple(sorted({s[2] for s in frontier}))
    return {
        "paths": paths,
        "latency_ms": round(elapsed, 3),
        "raw_edges": total_raw,
        "one_hop_queries": n_queries,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-id", required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--relation-path", required=True)
    p.add_argument("--hop", type=int, default=2)
    p.add_argument("--fanout", type=int, default=20)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    relation_path = args.relation_path.split(",")
    if len(relation_path) != args.hop:
        print(f"ERROR: relation-path length ({len(relation_path)}) != hop ({args.hop})")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Graph: {args.graph_id}  Seed: {args.seed}  Hop: {args.hop}  Fanout: {args.fanout}")
    print(f"Relation path: {' -> '.join(relation_path)}")
    print(f"Workers: {args.workers}")

    cluster = Cluster(["127.0.0.1"], port=9042)
    session = cluster.connect("ai_memory")

    # Step 1: Verify seed
    check = list(session.execute(
        "SELECT count(*) FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
        (args.graph_id, args.seed),
    ))
    if check[0].count == 0:
        print(f"ERROR: seed '{args.seed}' has no edges")
        cluster.shutdown()
        return
    print(f"Seed outdegree: {check[0].count}")

    # Step 2: Index preflight
    preflight, _ = preflight_check(session, args.graph_id, args.seed, relation_path, args.fanout)

    with (out_dir / "index_preflight_summary.json").open("w") as f:
        json.dump(preflight, f, indent=2, ensure_ascii=False)

    if preflight["mismatches"]:
        with (out_dir / "index_preflight_mismatches.jsonl").open("w") as f:
            for m in preflight["detail"]:
                f.write(json.dumps(m) + "\n")

    if not preflight["all_consistent"]:
        print("\n*** PREFLIGHT FAILED: index table incomplete. See index_preflight_mismatches.jsonl ***")
        cluster.shutdown()
        return

    # Step 3: Traversal comparisons
    print("\n=== C0-C TRAVERSALS ===")

    print("\n[1] Naive (workers=1, by_src+filter)...")
    r_naive = traverse(session, args.seed, args.graph_id, relation_path, args.hop, args.fanout, 1, False)
    print(f"  paths={len(r_naive['paths'])}  latency={r_naive['latency_ms']}ms  raw_edges={r_naive['raw_edges']}  queries={r_naive['one_hop_queries']}")

    print("\n[2] Parallel baseline (workers=16, by_src+filter)...")
    r_para = traverse(session, args.seed, args.graph_id, relation_path, args.hop, args.fanout, args.workers, False)
    print(f"  paths={len(r_para['paths'])}  latency={r_para['latency_ms']}ms  raw_edges={r_para['raw_edges']}  queries={r_para['one_hop_queries']}")

    print("\n[3] Parallel index (workers=16, by_src_relation)...")
    r_idx = traverse(session, args.seed, args.graph_id, relation_path, args.hop, args.fanout, args.workers, True)
    print(f"  paths={len(r_idx['paths'])}  latency={r_idx['latency_ms']}ms  raw_edges={r_idx['raw_edges']}  queries={r_idx['one_hop_queries']}")

    cluster.shutdown()

    def diff(a, b, label):
        a_set = set(a["paths"])
        b_set = set(b["paths"])
        missing = sorted(a_set - b_set)
        extra = sorted(b_set - a_set)
        match = len(missing) == 0 and len(extra) == 0
        print(f"\n{label}: {'MATCH' if match else 'MISMATCH'}")
        if not match:
            print(f"  missing={len(missing)}  extra={len(extra)}")
        return match, missing, extra

    match_para, miss_para, ext_para = diff(r_naive, r_para, "naive vs parallel baseline")
    match_idx, miss_idx, ext_idx = diff(r_naive, r_idx, "naive vs parallel index")

    if miss_para or ext_para:
        with (out_dir / "mismatches_parallel.jsonl").open("w") as f:
            f.write(json.dumps({"missing": miss_para[:50], "extra": ext_para[:50]}) + "\n")
    if miss_idx or ext_idx:
        with (out_dir / "mismatches_index.jsonl").open("w") as f:
            f.write(json.dumps({"missing": miss_idx[:50], "extra": ext_idx[:50]}) + "\n")

    summary = {
        "experiment": "C0-C naive vs parallel+index semantic gate",
        "graph_id": args.graph_id,
        "seed": args.seed,
        "hop": args.hop,
        "fanout": args.fanout,
        "relation_path": relation_path,
        "index_preflight_mismatches": preflight["mismatches"],
        "naive": {
            "mode": "workers=1, by_src+filter",
            "paths": len(r_naive["paths"]),
            "latency_ms": r_naive["latency_ms"],
            "raw_edges_from_cassandra": r_naive["raw_edges"],
            "one_hop_queries": r_naive["one_hop_queries"],
        },
        "parallel_baseline": {
            "mode": "workers=16, by_src+filter",
            "paths": len(r_para["paths"]),
            "latency_ms": r_para["latency_ms"],
            "raw_edges_from_cassandra": r_para["raw_edges"],
            "one_hop_queries": r_para["one_hop_queries"],
        },
        "parallel_index": {
            "mode": "workers=16, by_src_relation",
            "paths": len(r_idx["paths"]),
            "latency_ms": r_idx["latency_ms"],
            "raw_edges_from_cassandra": r_idx["raw_edges"],
            "one_hop_queries": r_idx["one_hop_queries"],
        },
        "naive_vs_parallel_disagreements": 0 if match_para else 1,
        "naive_vs_index_disagreements": 0 if match_idx else 1,
        "index_raw_edge_reduction_pct": round(
            (1 - r_idx["raw_edges"] / max(r_para["raw_edges"], 1)) * 100, 1
        ),
        "all_pass": match_para and match_idx and preflight["all_consistent"],
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n===== C0-C SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
