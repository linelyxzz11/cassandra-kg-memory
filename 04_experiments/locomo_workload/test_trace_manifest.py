from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("trace", HERE / "build_trace_manifests.py")
trace = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = trace
spec.loader.exec_module(trace)


def test_manifest_is_deterministic_and_exact() -> None:
    profile = {
        "counts": {"memories": 5882, "conversation_scopes": 10},
        "source": {"sha256": {"memory_csv": "abc"}},
    }
    first = trace.build_manifest(profile, 100_000, 7)
    second = trace.build_manifest(profile, 100_000, 7)
    assert first == second
    assert first["target_memory_count"] == 100_000
    assert first["full_namespace_replicas"] == 17
    assert first["partial_replica_memory_count"] == 6
    assert first["namespace_count"] == 176  # fallback profile treats six rows as six scopes
    assert first["manifest_sha256"]


def test_manifest_hash_changes_with_scale() -> None:
    profile = {
        "counts": {"memories": 5882, "conversation_scopes": 10},
        "source": {"sha256": {"memory_csv": "abc"}},
    }
    assert trace.build_manifest(profile, 100_000, 7)["manifest_sha256"] != trace.build_manifest(profile, 1_000_000, 7)["manifest_sha256"]
