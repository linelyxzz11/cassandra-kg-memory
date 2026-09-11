from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("online", HERE / "online_retrieval.py")
online = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = online
spec.loader.exec_module(online)


def test_rawerk_renderer_matches_frozen_labels() -> None:
    assert online.render_rawerk("raw", "alice", "likes", "tea") == "raw\nE: alice\nR: likes\nK: tea"


def test_new_gold_enters_real_fused_top10() -> None:
    index = online.ScopedOnlineRetrievalIndex()
    docs = {f"m{i}": f"ordinary memory number {i}" for i in range(20)}
    embeddings = {f"m{i}": np.array([0.0, 1.0], dtype=np.float32) for i in range(20)}
    index.load_scope("scope", docs, embeddings)
    before = index.search("scope", "uniquegold", [1.0, 0.0])
    assert "gold" not in {memory_id for memory_id, _ in before.top10}

    assert index.upsert_sparse("scope", "gold", "uniquegold exact evidence", 2)
    assert index.upsert_dense("scope", "gold", [1.0, 0.0], 2)
    sparse_visible, dense_visible = index.is_visible("scope", "gold", 2)
    assert sparse_visible and dense_visible
    after = index.search("scope", "uniquegold", [1.0, 0.0])
    assert "gold" in {memory_id for memory_id, _ in after.top10}


def test_duplicate_version_is_idempotent() -> None:
    index = online.ScopedOnlineRetrievalIndex()
    index.load_scope("scope", {"m": "text"}, {"m": [1.0, 0.0]})
    assert not index.upsert_sparse("scope", "m", "changed", 1)
    assert not index.upsert_dense("scope", "m", [0.0, 1.0], 1)
    assert index.duplicate_sparse_updates == 1
    assert index.duplicate_dense_updates == 1
