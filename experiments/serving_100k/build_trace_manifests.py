"""Build the compact, deterministic manifest for 100K trace replay.

The manifests describe a lazy namespace replication.  They deliberately do
not materialize million-row CSV files, avoiding an unnecessary large artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "results" / "serving_100k" / "locomo_workload_profile" / "locomo_workload_profile.json"
DEFAULT_OUTPUT = ROOT / "results" / "serving_100k" / "locomo_workload_manifests"


def canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(profile: dict, target_memories: int, seed: int) -> dict:
    source_memories = int(profile["counts"]["memories"])
    source_scopes = int(profile["counts"]["conversation_scopes"])
    full_replicas, remainder = divmod(target_memories, source_memories)
    partial_scope_count = 0
    remaining = remainder
    scope_profile = profile.get("conversation_scope_memory_count", {})
    for scope_id in scope_profile.get("source_row_order", []):
        if remaining <= 0:
            break
        partial_scope_count += 1
        remaining -= int(scope_profile["by_scope"][scope_id])
    if remainder and not partial_scope_count:
        # Backward-compatible fallback for minimal unit-test profiles.
        partial_scope_count = min(source_scopes, remainder)
    manifest = {
        "manifest_version": "locomo-shaped-trace-v1",
        "workload_name": "LoCoMo-shaped multi-tenant online memory workload",
        "terminology_guardrail": "Synthetic namespace replication preserving empirical LoCoMo distributions; not original scaled LoCoMo.",
        "seed": seed,
        "target_memory_count": target_memories,
        "source_memory_count": source_memories,
        "source_scope_count": source_scopes,
        "full_namespace_replicas": full_replicas,
        "partial_replica_memory_count": remainder,
        "namespace_count": full_replicas * source_scopes + partial_scope_count,
        "partial_replica_scope_count": partial_scope_count,
        "lazy_mapping": {
            "source_row_index": "global_memory_ordinal % source_memory_count",
            "replica_index": "global_memory_ordinal // source_memory_count",
            "scope_id": "lr{replica_index:06d}::{source_sample_id}",
            "memory_id": "lr{replica_index:06d}::{source_memory_id}",
            "qa_id": "lr{replica_index:06d}::{source_qa_id}",
            "order": "preserve canonical per-scope memory order; deterministically interleave namespaces with seed",
        },
        "preserved_empirical_dimensions": [
            "conversation length",
            "session structure",
            "KG coverage",
            "triples per memory",
            "relation frequency",
            "QA category",
            "evidence count",
            "within-scope update order",
        ],
        "operation_mix": {
            "primary": {"read_fraction": 0.95, "update_fraction": 0.05},
            "supplemental": [
                {"read_fraction": 0.90, "update_fraction": 0.10},
                {"read_fraction": 0.99, "update_fraction": 0.01},
            ],
        },
        "concurrency": [1, 8, 16, 32, 64],
        "freshness_deadlines_ms": [50, 100, 250, 500, 1000],
        "source_sha256": profile["source"]["sha256"],
        "estimated_replication_factor": target_memories / source_memories,
        "minimum_full_replicas_required": math.ceil(target_memories / source_memories),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--scales", type=int, nargs="+", default=[100_000])
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for scale in args.scales:
        manifest = build_manifest(profile, scale, args.seed)
        path = args.output / f"trace_manifest_{scale}.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
