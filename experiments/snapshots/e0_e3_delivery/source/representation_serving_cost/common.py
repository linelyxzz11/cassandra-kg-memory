"""Shared, isolated support for the representation/serving experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "cassandra-kg-memory"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments" / "serving_100k")]

import numpy as np
from cassmem.retrieval.online import render_rawerk
from run_graph_canonical_gate_v2 import build_graph_records

ARMS = ("Raw-only", "RawERK", "ERK-only")


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def lexical(record, arm):
    if arm == "Raw-only":
        return str(record.raw_text or "").strip()
    if arm == "RawERK":
        return record.rawerk
    if arm == "ERK-only":
        return render_rawerk("", record.entities, record.relations, record.keywords)
    raise ValueError(arm)


def vectors(kind):
    names = (ROOT / "data" / f"locomo_{kind}_ids_bge.txt").read_text(encoding="utf-8-sig").splitlines()
    array = np.load(ROOT / "data" / f"locomo_{kind}_bge_large.npy", mmap_mode="r")
    assert len(names) == len(array) and len(set(names)) == len(names)
    assert array.dtype == np.float32 and array.shape[1] == 1024
    return dict(zip(names, array))


def source_manifest():
    paths = [
        ROOT / "src/cassmem/retrieval/online.py",
        ROOT / "src/cassmem/backend/live_cells_graph.py",
        ROOT / "src/cassmem/representation/graph_event.py",
        ROOT / "data/locomo_memory_records.csv",
        ROOT / "data/frozen_retrieval/p3_memory_features.csv",
        ROOT / "data/frozen_retrieval/p3_queries.csv",
        ROOT / "data/retrieval_gold/locomo_cat1_4_gold_memory.csv",
        ROOT / "data/locomo_memory_bge_large.npy",
        ROOT / "data/locomo_qa_bge_large.npy",
        HERE.parent / "paper/main.tex",
        *sorted(HERE.glob("*.py")),
    ]
    return {str(path): sha(path) for path in paths}


def index_space(index):
    sparse = list(index._sparse.values())
    dense = list(index._dense.values())
    return {
        "scopes": len(sparse),
        "memories": sum(len(s.memory_ids) for s in sparse),
        "lexical_text_utf8_bytes": sum(len(t.encode("utf-8")) for s in sparse for t in s.documents),
        "vocabulary_entries_sum_scopes": sum(len(s.vectorizer.vocabulary_) for s in sparse),
        "sparse_nnz": sum(s.doc_tf.nnz for s in sparse),
        "sparse_arrays_bytes": sum(sum(a.nbytes for a in (s.doc_tf.data, s.doc_tf.indices, s.doc_tf.indptr, s.doc_len, s.idf, s.len_norm)) for s in sparse),
        "normalized_dense_arrays_bytes": sum(s.normalized_embeddings.nbytes for s in dense),
        "note": "Array bytes exclude Python dictionaries, texts, source vector mappings, and temporary rebuild copies; not total RSS.",
    }
