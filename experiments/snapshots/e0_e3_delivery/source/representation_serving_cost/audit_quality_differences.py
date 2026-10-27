"""Explain historical/online ranking differences without changing either scorer."""
import importlib.util
import sys
from collections import defaultdict
from common import HERE, ROOT, rows, write_json, lexical, build_graph_records, vectors
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex, zscore_fuse

path = ROOT / "experiments/retrieval/p1_compact_component_ablation/run_p1c_ablation_v5.py"
spec = importlib.util.spec_from_file_location("historical_p1", path)
p1 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p1
spec.loader.exec_module(p1)
cfg_path = path.with_name("p1c_config.json")
cfg = p1.load_json(cfg_path)
directory = HERE / "results/quality_20261012_v1"
differences = rows(directory / "rawerk_reference_differences.csv")
qids = {r["query_id"] for r in differences}
inputs = p1.load_inputs(cfg, ROOT, qids)
queries = [q for q in p1.load_queries(ROOT / "data/frozen_retrieval/p3_queries.csv") if q.query_id in qids]
docs = {mid:p1.build_document(mid, ("E","R","K"), inputs, cfg["renderer"]) for mid in inputs.memory_records}
old_sparse = p1.build_bm25_rankings_for_variant("RawERK", queries, inputs, cfg, docs)
index = ScopedOnlineRetrievalIndex()
grouped = defaultdict(list)
for r in build_graph_records():
    grouped[r.scope_id].append(r)
mv, qv = vectors("memory"), vectors("qa")
for s, recs in grouped.items():
    index.load_scope(s, {r.memory_id:lexical(r,"RawERK") for r in recs}, {r.memory_id:mv[r.memory_id] for r in recs})
reference = {r["query_id"]:r for r in rows(ROOT / "results/retrieval/fusion_ablation_aligned_v2/fusion_ablation_per_query_heldout1146.csv") if r["method"].lower().startswith("zscore")}
output = []
for q in queries:
    old_d = p1.top_dense_for_query(q, inputs, 50)
    old_b = old_sparse[q.query_id]
    old_rank = p1.zscore_fuse(old_d, old_b, .6, "branch_min", 10)
    online = index.search(q.conversation_id, q.question, qv[q.query_id])
    hybrids = {
        "historical_dense_historical_sparse": old_rank,
        "historical_dense_online_sparse": zscore_fuse(old_d, online.sparse_top50),
        "online_dense_historical_sparse": zscore_fuse(online.dense_top50, old_b),
        "online_dense_online_sparse": list(online.top10),
    }
    expected = reference[q.query_id]["top10_memory_ids"].split(";")
    assert [m for m,_ in old_rank] == expected, f"Historical result not reproduced: {q.query_id}"
    changed = {mid for rank,mid in enumerate(expected) if online.top10[rank][0] != mid}
    output.append({"query_id":q.query_id, "historical_reproduced":True,
                   "rankings":{name:[mid for mid,_ in value] for name,value in hybrids.items()},
                   "changed_items": {mid: {"dense_frozen":dict(old_d).get(mid), "dense_online":dict(online.dense_top50).get(mid),
                                           "sparse_historical":dict(old_b).get(mid), "sparse_online":dict(online.sparse_top50).get(mid),
                                           "fusion_historical":dict(old_rank).get(mid), "fusion_online":dict(online.top10).get(mid)} for mid in sorted(changed)}})
write_json(directory / "difference_audit.json", {"status":"PASS", "queries":output,
           "decision":"Keep exact current online scorer for all cost/quality arms; retain historical table untouched. New experiment uses newly computed quality, not copied historical values."})
print(output)
