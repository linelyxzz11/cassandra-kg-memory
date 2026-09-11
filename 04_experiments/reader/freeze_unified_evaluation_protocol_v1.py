#!/usr/bin/env python3
"""Freeze the cross-layer CassMem evaluation contract and artifact hashes."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "05_reports/unified_evaluation_protocol_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, role: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    core_files = [
        (ROOT / "external_data/locomo10.json", "original_dataset"),
        (ROOT / "01_data/locomo_memory_records.csv", "canonical_memory_corpus"),
        (ROOT / "01_data/locomo_qa_records.csv", "canonical_full5_questions"),
        (ROOT / "02_artifacts/retrieval_gold_v2/locomo_cat1_4_gold_memory.csv", "retrieval_gold"),
        (ROOT / "02_artifacts/p3_memory_features.csv", "structured_features"),
        (ROOT / "01_data/locomo_memory_bge_large.npy", "frozen_memory_embeddings"),
        (ROOT / "01_data/locomo_qa_bge_large.npy", "frozen_query_embeddings"),
        (ROOT / "scripts/experiments/artifacts/frozen_dense_scores_long.csv", "frozen_dense_scores"),
        (ROOT / "results/locomo_kg_edges_spacy.csv", "corrected_global_kg"),
        (ROOT / "03_src/evaluation/run_locomo_prompt_protocol_gpt4o.py", "reader_prompt_implementation"),
        (ROOT / "03_src/evaluation/locomo_official_eval_v1.py", "official_f1_evaluator"),
        (ROOT / "05_reports/reader_offline_metrics_v4/manifest.json", "offline_reader_metrics_manifest"),
        (ROOT / "05_reports/backend_equivalence_v2/manifest.json", "backend_bridge_manifest"),
    ]
    artifacts = [artifact(path, role) for path, role in core_files]

    cache_rows: list[dict[str, Any]] = []
    methods = ("BM25", "Dense-bge", "Dense+GlobalKG", "RRF_compact", "ZScore-Raw", "ZScore-RawERK")
    settings = (("setting_a_unified", "a_unified", "Cat.x"), ("setting_b_category", "b_category", "Cat.v"))
    for setting_dir, setting, label in settings:
        for method in methods:
            base = ROOT / ("05_reports/locomo_gpt4o_prompt_protocol_corrected_densekg_v2" if method == "Dense+GlobalKG" else "05_reports/locomo_gpt4o_prompt_protocol")
            path = base / setting_dir / method / "reader_predictions.jsonl"
            rows = sum(1 for line in path.open(encoding="utf-8-sig") if line.strip())
            cache_rows.append({
                "experiment": "reader_full5",
                "method": method,
                "setting": setting,
                "setting_label": label,
                "query_scope": "Full5",
                "backend": "frozen ranking cache",
                "reader": "gpt-4o-2024-08-06",
                "temperature": 0,
                "top_k": 10,
                "prediction_rows": rows,
                "cache_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "cache_sha256": sha256(path),
                "status": "PASS" if rows == 1986 else "FAIL",
            })
    write_csv(OUT / "prediction_cache_registry.csv", cache_rows)

    protocol = {
        "protocol_version": "cassmem-unified-eval-v1",
        "dataset": {
            "name": "LoCoMo canonical 10-conversation release",
            "conversations": 10,
            "memories": 5882,
            "questions_full5": 1986,
            "category_counts": {"1_multi_hop": 282, "2_temporal": 321, "3_open_domain": 96, "4_single_hop": 841, "5_adversarial": 446},
        },
        "query_scopes": {
            "retrieval_all": "Cat1-4 n=1540; metrics on 1536 evidence-bearing queries",
            "retrieval_heldout": "test split n=1146 evidence-bearing queries; primary effectiveness claim",
            "reader_full5": "Cat1-5 n=1986; natural micro weighting",
            "cat5": "n=446; answer-level abstention only, excluded from retrieval metrics",
        },
        "retrieval": {
            "methods": list(methods),
            "candidate_scope": "all memories in the query conversation",
            "top_k": 10,
            "fusion_candidate_depth": 50,
            "embedding": "project-frozen bge-large 1024d arrays; exact identity is the artifact SHA-256",
            "metrics": ["MRR@10", "Hit@1", "Hit@5", "Hit@10", "Recall@10", "nDCG@10"],
            "cat5_policy": "excluded because no positive retrieval target",
        },
        "reader": {
            "model": "gpt-4o-2024-08-06",
            "temperature": 0,
            "max_tokens": 64,
            "context_top_k": 10,
            "setting_a": "single unified short-answer prompt for Cat1-5",
            "setting_b": "ordinary prompt for Cat1/3/4, temporal date instruction for Cat2, deterministic binary option prompt for Cat5",
            "cat5_reference": "Not mentioned in the conversation",
            "cat5_normalization": "restore (a)/(b) to deterministic option text before every evaluator",
        },
        "answer_metrics": {
            "F1": "official LoCoMo category-aware per-query score; micro mean over scope",
            "B1": "lowercase NLTK word_tokenize sentence BLEU-1, weights=(1,0,0,0), method1; Cat5 uses abstention reference after option restoration",
            "J": "Mem0/HingeMem binary correctness prompt, judge gpt-4o-2024-08-06, micro mean over 1986",
        },
        "backends": {
            "effectiveness_reference": "CSV",
            "bridge": ["CSV", "Cassandra", "Neo4j"],
            "semantic_gate": "candidate/projection/edge digest plus exact Top-10 and metric parity",
        },
        "cache_policy": {
            "immutable_raw_predictions": True,
            "cache_key_fields": ["question", "reference", "candidate_answer", "reader_or_judge_model", "prompt_sha256", "ranking_sha256", "setting"],
            "offline_repairs_allowed": ["option-label restoration", "deterministic scoring", "aggregation", "manifest regeneration"],
            "api_rerun_required_only_if": "prompt or Top-10 context changed and no byte-identical prompt cache exists",
        },
        "artifacts": artifacts,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "protocol_manifest.json"
    manifest_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CassMem Unified Evaluation Protocol V1", "",
        "## Frozen experiment contract", "",
        "| Field | Value |", "|---|---|",
        "| Dataset | LoCoMo, 10 conversations, 5,882 memories, 1,986 QA |",
        "| Retrieval scope | Cat1-4=1,540; 1,536 evidence-bearing; held-out primary=1,146 |",
        "| Reader scope | Full5=1,986; Cat5=446 |",
        "| Retrieval methods | BM25, Dense-bge, corrected Dense+GlobalKG, RRF_compact, ZScore-Raw, ZScore-RawERK |",
        "| Reader | GPT-4o-2024-08-06, temperature=0, max_tokens=64 |",
        "| Context | Top-10 frozen memory IDs |",
        "| Embedding | Frozen bge-large 1024d artifacts, identified by SHA-256 |",
        "| Retrieval metrics | MRR@10, Hit@1/5/10, Recall@10, nDCG@10 |",
        "| Answer metrics | LoCoMo F1, Full5 BLEU-1, GPT-4o Judge |", "",
        "## Non-negotiable rules", "",
        "1. Cat5 is excluded from retrieval effectiveness and evaluated as answer-level abstention.",
        "2. Cat5 `(a)/(b)` is restored to option text before F1, BLEU-1, or Judge.",
        "3. Overall means the per-query micro mean over the named scope, never an unweighted mean of category means.",
        "4. Raw prediction caches are immutable. Offline normalization is versioned separately.",
        "5. API regeneration is required only when the actual prompt/context changes and no identical prompt cache exists.",
        "6. Every paper number must point to a manifest and input hashes.", "",
        "## Cache registry", "",
        "See `prediction_cache_registry.csv` for all 12 Reader caches and SHA-256 values.", "",
        "## Machine-readable protocol", "",
        "See `protocol_manifest.json`.",
    ]
    (OUT / "UNIFIED_EVALUATION_PROTOCOL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "caches": len(cache_rows), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
