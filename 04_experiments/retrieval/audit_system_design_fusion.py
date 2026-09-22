"""Read-only audit of paper Eq.1/2 against frozen artifacts and actual functions.

AST extraction avoids importing experiment entrypoints or requiring databases.
No retrieval code or frozen artifact is modified. Prints its evidence as JSON.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "04_experiments/retrieval/p1_compact_component_ablation/run_p1c_ablation_v5.py"
ONLINE = ROOT / "04_experiments/locomo_workload/online_retrieval.py"
BRIDGE = ROOT / "04_experiments/retrieval/backend_bridge_v2_run.py"


def functions(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected, type_ignores=[])
    ns = {"np": np, "CountVectorizer": CountVectorizer, "ALPHA_DENSE": 0.6, "CANDIDATE_DEPTH": 50}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equation(dense, sparse):
    normalized, floors = [], []
    for branch in (dense, sparse):
        if not branch:
            normalized.append({})
            floors.append(0.0)
            continue
        values = np.asarray([score for _, score in branch], dtype=np.float64)
        mean = float(np.mean(values))
        sigma = float(np.sqrt(np.mean((values - mean) ** 2)))
        denominator = sigma if sigma > 1e-12 else 1.0
        scores = {mid: (score - mean) / denominator for mid, score in branch}
        normalized.append(scores)
        floors.append(min(scores.values()))
    ids = list(dict.fromkeys([mid for mid, _ in dense + sparse]))
    fused = [(mid, 0.6 * normalized[0].get(mid, floors[0]) + 0.4 * normalized[1].get(mid, floors[1])) for mid in ids]
    return sorted(fused, key=lambda item: -item[1])[:10]


def main():
    p1 = functions(P1, ["nonempty", "build_document", "BM25Retriever", "zscore_values", "zscore_fuse"])
    online = functions(ONLINE, ["render_rawerk", "_zscore", "zscore_fuse"])
    bridge = functions(BRIDGE, ["zscore", "zscore_fuse", "top_pairs", "top_indices"])
    manifest_path = ROOT / "05_reports/p1_compact_component_ablation/p1c_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert sha(P1) == manifest["script"]["sha256"]
    assert manifest["fusion"]["candidate_depth"] == 50
    assert manifest["fusion"]["alpha_dense"] == 0.6
    assert manifest["fusion"]["missing_candidate_rule"] == "branch_min"
    memories = list(rows(ROOT / "01_data/locomo_memory_records.csv"))
    features = {row["memory_id"]: row for row in rows(ROOT / "02_artifacts/p3_memory_features.csv")}
    inputs = SimpleNamespace(memory_records={row["memory_id"]: row for row in memories}, memory_features=features)
    renderer_mismatches = []
    documents = {}
    for row in memories:
        mid = row["memory_id"]
        feature = features.get(mid, {})
        a = p1["build_document"](mid, ("E", "R", "K"), inputs, manifest["renderer"])
        b = online["render_rawerk"](row["text"], feature.get("entities", ""), feature.get("relations", ""), feature.get("keywords", ""))
        documents[mid] = a
        if a != b:
            renderer_mismatches.append(mid)

    ordinal = {row["memory_id"]: i for i, row in enumerate(memories)}
    dense = defaultdict(list)
    for row in rows(ROOT / "scripts/experiments/artifacts/frozen_dense_scores_long.csv"):
        dense[row["query_id"]].append((row["memory_id"], float(row["score"])))
    for qid in dense:
        dense[qid].sort(key=lambda item: (-item[1], ordinal[item[0]]))
        dense[qid] = dense[qid][:50]
    sparse = defaultdict(list)
    sparse_path = ROOT / "05_reports/p1_compact_component_ablation/p1c_bm25_rankings_cat1_4_1540.csv"
    for row in rows(sparse_path):
        if row["variant"] == "RawERK":
            sparse[row["query_id"]].append((int(row["rank"]), row["memory_id"], float(row["score"])))
    # The persisted BM25 CSV contains Top-10 only. Recover the actual Top-50
    # with the unmodified frozen scorer; never normalize the exported Top-10.
    sparse_reference = {qid: [(mid, score) for _, mid, score in sorted(branch)[:10]] for qid, branch in sparse.items()}
    scope_ids = defaultdict(list)
    for row in memories:
        scope_ids[row["sample_id"]].append(row["memory_id"])
    scorers = {}
    for scope, ids in scope_ids.items():
        scorer = p1["BM25Retriever"](**manifest["bm25"])
        scorer.fit([documents[mid] for mid in ids])
        scorers[scope] = scorer
    sparse = {}
    for row in rows(ROOT / "02_artifacts/retrieval_gold_v2/locomo_cat1_4_gold_memory.csv"):
        qid, scope = row["query_id"], row["conversation_id"]
        indices, values = scorers[scope].search(row["question"], top_k=50)
        sparse[qid] = [(scope_ids[scope][index], score) for index, score in zip(indices, values)]
        assert [mid for mid, _ in sparse[qid][:10]] == [mid for mid, _ in sparse_reference[qid]]
    reference_path = ROOT / "05_reports/official_eval/zscore_rawerk_ranking_canonical1540.csv"
    reference = defaultdict(list)
    for row in rows(reference_path):
        reference[row["query_id"]].append((int(row["rank"]), row["memory_id"]))
    reference = {qid: [mid for _, mid in sorted(branch)[:10]] for qid, branch in reference.items()}
    assert len(reference) == 1540
    mismatches = {name: [] for name in ("equation", "p1", "online_fusion", "bridge_fusion")}
    max_score_delta = 0.0
    single_branch_candidate_count = 0
    for qid, expected in reference.items():
        d, s = dense[qid], sparse[qid]
        assert len(d) == len(s) == 50
        single_branch_candidate_count += len({mid for mid, _ in d} ^ {mid for mid, _ in s})
        outputs = {
            "equation": equation(d, s),
            "p1": p1["zscore_fuse"](d, s, 0.6, "branch_min", 10),
            "online_fusion": online["zscore_fuse"](d, s, 0.6, 10),
            "bridge_fusion": bridge["zscore_fuse"](d, s, ordinal)[:10],
        }
        for name, output in outputs.items():
            if [mid for mid, _ in output] != expected:
                mismatches[name].append(qid)
        expected_scores = dict(outputs["p1"])
        for output in outputs.values():
            for mid, score in output:
                if mid in expected_scores:
                    max_score_delta = max(max_score_delta, abs(score - expected_scores[mid]))

    cases = [
        ([("b", 4.0), ("a", 1.0)], [("c", 10.0), ("a", 0.0)]),
        ([("z", 1.0), ("a", 1.0)], [("c", 2.0), ("a", 2.0)]),
        ([("z", 1.0)], []),
        ([], [("a", 3.0), ("b", 1.0)]),
        ([], []),
        ([("a", 1.0), ("b", 1.0 + 1e-13)], [("c", 2.0)]),
    ]
    for d, s in cases:
        expected = equation(d, s)
        for output in (p1["zscore_fuse"](d, s, 0.6, "branch_min", 10), online["zscore_fuse"](d, s, 0.6, 10)):
            assert [mid for mid, _ in output] == [mid for mid, _ in expected]
            assert np.allclose([score for _, score in output], [score for _, score in expected], rtol=0, atol=1e-12)
    assert [mid for mid, _ in equation(*cases[1])] == ["z", "a", "c"]
    offline_tie = bridge["top_pairs"](np.asarray([1.0, 1.0]), ["z", "a"], 2, {"z": 0, "a": 1})
    assert [mid for mid, _ in offline_tie] == ["z", "a"]

    result = {
        "status": "PASS" if not renderer_mismatches and not any(mismatches.values()) else "FAIL",
        "p1_script_matches_frozen_manifest": True,
        "rawerk_documents_checked": len(memories),
        "rawerk_mismatches": len(renderer_mismatches),
        "queries_checked": len(reference),
        "branch_depths": [50, 50],
        "top10_mismatches": {key: len(value) for key, value in mismatches.items()},
        "mismatch_examples": {key: value[:5] for key, value in mismatches.items()},
        "max_fusion_score_difference": max_score_delta,
        "single_branch_candidates_across_queries": single_branch_candidate_count,
        "synthetic_edge_cases": len(cases),
        "validation_boundary": "Frozen dense scores plus Top-50 rebuilt by hash-matched P1 BM25; same branch inputs passed to each fusion function. Not online candidate generation, live backends, or freshness.",
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (P1, ONLINE, BRIDGE)},
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (manifest_path, sparse_path, reference_path)},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert result["status"] == "PASS"


if __name__ == "__main__":
    main()
