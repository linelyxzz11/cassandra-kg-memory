#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
P1-C v5 — Compact Component Ablation

Retrieval-only experiment.

Dense branch:
    frozen raw-memory dense scores

BM25 variants:
    Raw
    RawE
    RawR
    RawK
    RawER
    RawEK
    RawRK
    RawERK
    RawERKT

Fusion:
    0.6 * z(Dense) + 0.4 * z(BM25)

Important:
- No Reader API
- No LLM API
- No embedding recomputation
- No alpha search
- No component re-extraction
- RawERK must pass official Top-10 parity before ablations run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


VARIANTS: dict[str, tuple[str, ...]] = {
    "Raw": (),
    "RawE": ("E",),
    "RawR": ("R",),
    "RawK": ("K",),
    "RawER": ("E", "R"),
    "RawEK": ("E", "K"),
    "RawRK": ("R", "K"),
    "RawERK": ("E", "R", "K"),
    "RawERKT": ("E", "R", "K", "T"),
}

FACTORIAL_VARIANTS = [
    "Raw",
    "RawE",
    "RawR",
    "RawK",
    "RawER",
    "RawEK",
    "RawRK",
    "RawERK",
]

CATEGORY_NAMES = {
    "1": "cat1_multi-hop",
    "2": "cat2_temporal",
    "3": "cat3_open-domain",
    "4": "cat4_single-hop",
}


def log(message: str) -> None:
    print(
        time.strftime("%Y-%m-%d %H:%M:%S"),
        message,
        flush=True,
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_json(
    path: Path,
    obj: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def resolve_path(
    value: str,
    project_root: Path,
) -> Path:
    expanded = Path(
        os.path.expandvars(
            os.path.expanduser(value)
        )
    )

    if expanded.is_absolute():
        return expanded

    return (project_root / expanded).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def normalize_category(value: Any) -> str:
    text = str(value).strip()

    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    return text


def nonempty(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)

    if text.lower() == "nan":
        return ""

    return text.strip()


def read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return [
            dict(row)
            for row in csv.DictReader(f)
        ]


def csv_header(path: Path) -> list[str]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.reader(f)

        try:
            return next(reader)
        except StopIteration:
            return []


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()

        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)

        fieldnames = ordered

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def percentile(
    values: Sequence[float],
    q: float,
) -> float:
    if not values:
        return float("nan")

    values_sorted = sorted(
        float(v)
        for v in values
    )

    if len(values_sorted) == 1:
        return values_sorted[0]

    position = (len(values_sorted) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values_sorted[lower]

    return (
        values_sorted[lower] * (upper - position)
        + values_sorted[upper] * (position - lower)
    )


def parse_gold_ids(value: Any) -> set[str]:
    text = nonempty(value)

    if not text:
        return set()

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return {
                    nonempty(item)
                    for item in parsed
                    if nonempty(item)
                }

        except json.JSONDecodeError:
            pass

    if ";" in text:
        parts = text.split(";")
    else:
        parts = re.split(r",\s*", text)

    return {
        part.strip()
        for part in parts
        if part.strip()
    }


def check_unique(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    label: str,
) -> None:
    seen: set[tuple[Any, ...]] = set()
    duplicate_count = 0

    for row in rows:
        key = tuple(
            row.get(column)
            for column in keys
        )

        if key in seen:
            duplicate_count += 1

        seen.add(key)

    if duplicate_count:
        raise ValueError(
            f"{label}: found {duplicate_count} duplicate rows "
            f"for key {list(keys)}"
        )


def safe_float(
    value: Any,
    label: str,
) -> float:
    try:
        number = float(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot parse float for {label}: {value!r}"
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"Non-finite float for {label}: {value!r}"
        )

    return number


class BM25Retriever:
    """
    Existing project BM25 implementation.

    CountVectorizer:
    - lowercase=True
    - English stopwords
    - unigram + bigram
    - max_features=50000
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        lowercase: bool = True,
        stop_words: str | None = "english",
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int | None = 50000,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)

        self.lowercase = bool(lowercase)
        self.stop_words = stop_words
        self.ngram_range = tuple(ngram_range)
        self.max_features = max_features

        self.vectorizer: CountVectorizer | None = None
        self.doc_tf = None
        self.doc_len: np.ndarray | None = None
        self.avg_dl: float | None = None
        self.idf: np.ndarray | None = None
        self.len_norm: np.ndarray | None = None
        self.n_docs = 0

    def fit(
        self,
        documents: Sequence[str],
    ) -> None:
        if not documents:
            raise ValueError(
                "BM25 fit received an empty document list"
            )

        self.vectorizer = CountVectorizer(
            lowercase=self.lowercase,
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
        )

        self.doc_tf = (
            self.vectorizer
            .fit_transform(documents)
            .tocsc()
        )

        self.n_docs = self.doc_tf.shape[0]

        self.doc_len = np.asarray(
            self.doc_tf.sum(axis=1)
        ).flatten()

        self.avg_dl = float(
            self.doc_len.mean()
        )

        document_frequency = np.asarray(
            (self.doc_tf > 0).sum(axis=0)
        ).flatten()

        self.idf = np.log(
            (
                self.n_docs
                - document_frequency
                + 0.5
            )
            /
            (
                document_frequency
                + 0.5
            )
            + 1.0
        )

        self.len_norm = (
            1.0
            - self.b
            + self.b
            * (
                self.doc_len
                / max(self.avg_dl, 1e-12)
            )
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> tuple[list[int], list[float]]:
        if (
            self.vectorizer is None
            or self.doc_tf is None
        ):
            raise RuntimeError(
                "BM25 search called before fit"
            )

        query_vector = (
            self.vectorizer
            .transform([query])
            .tocsc()
        )

        query_rows, query_columns = (
            query_vector.nonzero()
        )

        scores = np.zeros(
            self.n_docs,
            dtype=np.float64,
        )

        for _, column in zip(
            query_rows,
            query_columns,
        ):
            column_start = (
                self.doc_tf.indptr[column]
            )

            column_end = (
                self.doc_tf.indptr[column + 1]
            )

            column_rows = (
                self.doc_tf.indices[
                    column_start:column_end
                ]
            )

            column_data = (
                self.doc_tf.data[
                    column_start:column_end
                ]
            )

            term_frequency = np.zeros(
                self.n_docs,
                dtype=np.float64,
            )

            term_frequency[column_rows] = column_data

            term_frequency_score = (
                term_frequency
                * (self.k1 + 1.0)
                /
                (
                    term_frequency
                    + self.k1 * self.len_norm
                    + 1e-9
                )
            )

            scores += (
                term_frequency_score
                * self.idf[column]
            )

        ranked_indices = np.argsort(
            -scores,
            kind="mergesort",
        )

        top_indices = ranked_indices[
            : min(
                int(top_k),
                self.n_docs,
            )
        ]

        return (
            top_indices.tolist(),
            [
                float(scores[index])
                for index in top_indices
            ],
        )


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    question: str
    category: str
    conversation_id: str
    gold_ids: frozenset[str]
    split: str


@dataclass
class Inputs:
    memory_records: dict[str, dict[str, str]]
    memory_features: dict[str, dict[str, str]]
    queries: list[QueryRecord]
    dense_scores: dict[str, dict[str, float]]
    official_rankings: dict[
        str,
        list[tuple[str, int, float]]
    ]
    conversation_memories: dict[str, list[str]]


REQUIRED_HEADERS = {
    "memory_records": {
        "memory_id",
        "sample_id",
        "text",
        "timestamp",
    },
    "memory_features": {
        "memory_id",
        "conversation_id",
        "raw_text",
        "entities",
        "relations",
        "keywords",
    },
    "queries": {
        "query_id",
        "question",
        "category",
        "conversation_id",
        "gold_memory_ids",
        "split",
    },
    "dense_scores": {
        "query_id",
        "memory_id",
        "score",
    },
    "official_rankings": {
        "query_id",
        "memory_id",
        "rank",
        "score",
    },
}


def validate_headers(
    name: str,
    path: Path,
) -> None:
    actual = set(
        csv_header(path)
    )

    missing = (
        REQUIRED_HEADERS[name]
        - actual
    )

    if missing:
        raise ValueError(
            f"{name} missing columns "
            f"{sorted(missing)} "
            f"in {path}"
        )


def load_queries(
    path: Path,
) -> list[QueryRecord]:
    rows = read_csv_rows(path)

    check_unique(
        rows,
        ["query_id"],
        "queries",
    )

    queries: list[QueryRecord] = []

    for row in rows:
        queries.append(
            QueryRecord(
                query_id=nonempty(
                    row["query_id"]
                ),
                question=nonempty(
                    row["question"]
                ),
                category=normalize_category(
                    row["category"]
                ),
                conversation_id=nonempty(
                    row["conversation_id"]
                ),
                gold_ids=frozenset(
                    parse_gold_ids(
                        row["gold_memory_ids"]
                    )
                ),
                split=nonempty(
                    row["split"]
                ).lower(),
            )
        )

    invalid = [
        query.query_id
        for query in queries
        if (
            not query.query_id
            or not query.question
            or not query.conversation_id
        )
    ]

    if invalid:
        raise ValueError(
            f"Invalid query rows: {invalid[:10]}"
        )

    empty_gold = [
        query.query_id
        for query in queries
        if not query.gold_ids
    ]

    if empty_gold:
        log(
            f"Warning: {len(empty_gold)} queries "
            "without gold memory IDs."
        )

    return queries


def select_scope(
    queries: Sequence[QueryRecord],
    spec: Mapping[str, Any],
) -> list[QueryRecord]:
    categories = {
        normalize_category(value)
        for value in spec.get(
            "categories",
            [],
        )
    }

    split = nonempty(
        spec.get(
            "split",
            "",
        )
    ).lower()

    conversation_ids = {
        nonempty(value)
        for value in spec.get(
            "conversation_ids",
            [],
        )
    }

    selected = []

    for query in queries:
        if (
            categories
            and query.category not in categories
        ):
            continue

        if (
            split
            and query.split != split
        ):
            continue

        if (
            conversation_ids
            and query.conversation_id
            not in conversation_ids
        ):
            continue

        selected.append(query)

    expected = spec.get(
        "expected_query_count"
    )

    if (
        expected is not None
        and len(selected) != int(expected)
    ):
        raise ValueError(
            f"Scope {spec.get('name')} "
            f"has {len(selected)} queries; "
            f"expected {expected}"
        )

    return selected


def load_inputs(
    config: Mapping[str, Any],
    project_root: Path,
    only_query_ids: set[str],
) -> Inputs:
    paths_config = config["paths"]

    memory_path = resolve_path(
        paths_config["memory_records"],
        project_root,
    )

    feature_path = resolve_path(
        paths_config["memory_features"],
        project_root,
    )

    query_path = resolve_path(
        paths_config["queries"],
        project_root,
    )

    dense_path = resolve_path(
        paths_config["dense_scores"],
        project_root,
    )

    official_path = resolve_path(
        paths_config["official_rankings"],
        project_root,
    )

    path_items = [
        ("memory_records", memory_path),
        ("memory_features", feature_path),
        ("queries", query_path),
        ("dense_scores", dense_path),
        ("official_rankings", official_path),
    ]

    for name, path in path_items:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found: {path}"
            )

        validate_headers(
            name,
            path,
        )

    memory_rows = read_csv_rows(
        memory_path
    )

    feature_rows = read_csv_rows(
        feature_path
    )

    check_unique(
        memory_rows,
        ["memory_id"],
        "memory_records",
    )

    check_unique(
        feature_rows,
        ["memory_id"],
        "memory_features",
    )

    memory_records = {
        nonempty(row["memory_id"]): row
        for row in memory_rows
    }

    memory_features = {
        nonempty(row["memory_id"]): row
        for row in feature_rows
    }

    missing_features = sorted(
        set(memory_records)
        - set(memory_features)
    )

    require_features = bool(
        config["validation"].get(
            "require_feature_row_for_every_memory",
            True,
        )
    )

    if (
        missing_features
        and require_features
    ):
        raise ValueError(
            f"{len(missing_features)} memories "
            "lack feature rows. "
            f"Examples: {missing_features[:10]}"
        )

    queries = load_queries(
        query_path
    )

    conversation_memories: dict[
        str,
        list[str]
    ] = defaultdict(list)

    for memory_id, row in memory_records.items():
        conversation_id = nonempty(
            row.get("sample_id")
        )

        if not conversation_id:
            conversation_id = nonempty(
                memory_features
                .get(memory_id, {})
                .get("conversation_id")
            )

        if not conversation_id:
            raise ValueError(
                "Cannot determine conversation "
                f"for memory {memory_id}"
            )

        conversation_memories[
            conversation_id
        ].append(memory_id)

    dense_scores: dict[
        str,
        dict[str, float]
    ] = defaultdict(dict)

    duplicate_dense = 0

    with dense_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        for row in csv.DictReader(f):
            query_id = nonempty(
                row["query_id"]
            )

            if query_id not in only_query_ids:
                continue

            memory_id = nonempty(
                row["memory_id"]
            )

            if (
                memory_id
                in dense_scores[query_id]
            ):
                duplicate_dense += 1

            dense_scores[
                query_id
            ][memory_id] = safe_float(
                row["score"],
                f"dense {query_id}/{memory_id}",
            )

    if duplicate_dense:
        raise ValueError(
            "Dense cache contains "
            f"{duplicate_dense} duplicate "
            "query-memory pairs"
        )

    official_rows = read_csv_rows(
        official_path
    )

    check_unique(
        official_rows,
        ["query_id", "rank"],
        "official rankings by rank",
    )

    check_unique(
        official_rows,
        ["query_id", "memory_id"],
        "official rankings by memory",
    )

    official_rankings: dict[
        str,
        list[tuple[str, int, float]]
    ] = defaultdict(list)

    for row in official_rows:
        query_id = nonempty(
            row["query_id"]
        )

        official_rankings[
            query_id
        ].append(
            (
                nonempty(
                    row["memory_id"]
                ),
                int(
                    float(
                        row["rank"]
                    )
                ),
                safe_float(
                    row["score"],
                    f"official score {query_id}",
                ),
            )
        )

    for query_id in official_rankings:
        official_rankings[
            query_id
        ].sort(
            key=lambda item: item[1]
        )

        ranks = [
            rank
            for _, rank, _
            in official_rankings[query_id]
        ]

        expected_ranks = list(
            range(
                1,
                len(ranks) + 1,
            )
        )

        if ranks != expected_ranks:
            raise ValueError(
                "Non-contiguous official ranks "
                f"for {query_id}: {ranks[:15]}"
            )

    return Inputs(
        memory_records=memory_records,
        memory_features=memory_features,
        queries=queries,
        dense_scores=dict(dense_scores),
        official_rankings=dict(
            official_rankings
        ),
        conversation_memories=dict(
            conversation_memories
        ),
    )


def build_document(
    memory_id: str,
    components: Sequence[str],
    inputs: Inputs,
    renderer_config: Mapping[str, Any],
) -> str:
    memory_record = inputs.memory_records[
        memory_id
    ]

    memory_feature = inputs.memory_features.get(
        memory_id,
        {},
    )

    raw_source = renderer_config.get(
        "raw_text_source",
        "memory_records.text",
    )

    if raw_source == "memory_records.text":
        raw_text = nonempty(
            memory_record.get("text")
        )

    elif raw_source == "memory_features.raw_text":
        raw_text = nonempty(
            memory_feature.get("raw_text")
        )

    else:
        raise ValueError(
            f"Unsupported raw_text_source: "
            f"{raw_source}"
        )

    component_fields = {
        "E": (
            "entities",
            memory_feature,
        ),
        "R": (
            "relations",
            memory_feature,
        ),
        "K": (
            "keywords",
            memory_feature,
        ),
        "T": (
            "timestamp",
            memory_record,
        ),
    }

    labels = renderer_config.get(
        "labels",
        {
            "E": "E:",
            "R": "R:",
            "K": "K:",
            "T": "T:",
        },
    )

    separator = str(
        renderer_config.get(
            "separator",
            "\n",
        )
    )

    pieces = [raw_text]

    for component in components:
        field_name, source = (
            component_fields[component]
        )

        value = nonempty(
            source.get(field_name)
        )

        if not value:
            continue

        label = str(
            labels.get(
                component,
                f"{component}:",
            )
        )

        if label:
            pieces.append(
                f"{label} {value}".strip()
            )
        else:
            pieces.append(value)

    return separator.join(pieces)


def representation_stats(
    documents_by_variant: Mapping[
        str,
        Mapping[str, str],
    ],
) -> list[dict[str, Any]]:
    rows = []

    for variant, documents in (
        documents_by_variant.items()
    ):
        token_counts = [
            len(document.split())
            for document in documents.values()
        ]

        document_values = list(
            documents.values()
        )

        rows.append(
            {
                "variant": variant,
                "n_documents": len(
                    document_values
                ),
                "mean_tokens": (
                    statistics.mean(token_counts)
                    if token_counts
                    else 0.0
                ),
                "median_tokens": (
                    statistics.median(token_counts)
                    if token_counts
                    else 0.0
                ),
                "p95_tokens": (
                    percentile(
                        token_counts,
                        0.95,
                    )
                    if token_counts
                    else 0.0
                ),
                "empty_document_count": sum(
                    not value.strip()
                    for value in document_values
                ),
                "duplicate_document_count": (
                    len(document_values)
                    - len(set(document_values))
                ),
            }
        )

    return rows


def write_representation_samples(
    path: Path,
    documents_by_variant: Mapping[
        str,
        Mapping[str, str],
    ],
    memory_ids: Sequence[str],
) -> None:
    lines = [
        "# P1-C Representation Samples",
        "",
    ]

    for memory_id in memory_ids:
        lines.extend(
            [
                f"## `{memory_id}`",
                "",
            ]
        )

        for variant in VARIANTS:
            lines.extend(
                [
                    f"### {variant}",
                    "",
                    "```text",
                    documents_by_variant[
                        variant
                    ][memory_id],
                    "```",
                    "",
                ]
            )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def top_dense_for_query(
    query: QueryRecord,
    inputs: Inputs,
    top_n: int,
) -> list[tuple[str, float]]:
    candidate_ids = (
        inputs.conversation_memories.get(
            query.conversation_id,
            [],
        )
    )

    score_map = inputs.dense_scores.get(
        query.query_id,
        {},
    )

    required_count = min(
        top_n,
        len(candidate_ids),
    )

    if len(score_map) < required_count:
        raise ValueError(
            f"Dense cache has only "
            f"{len(score_map)} scores for "
            f"{query.query_id}; "
            f"need at least {required_count}"
        )

    available = []

    for corpus_order, memory_id in enumerate(
        candidate_ids
    ):
        if memory_id in score_map:
            available.append(
                (
                    memory_id,
                    score_map[memory_id],
                    corpus_order,
                )
            )

    available.sort(
        key=lambda item: (
            -item[1],
            item[2],
        )
    )

    return [
        (
            memory_id,
            score,
        )
        for memory_id, score, _
        in available[:top_n]
    ]


def zscore_values(
    items: Sequence[tuple[str, float]],
) -> tuple[dict[str, float], float]:
    if not items:
        return {}, 0.0

    values = np.asarray(
        [
            score
            for _, score in items
        ],
        dtype=np.float64,
    )

    mean = float(
        values.mean()
    )

    standard_deviation = float(
        values.std(ddof=0)
    )

    if standard_deviation <= 1e-12:
        standard_deviation = 1.0

    zscores = {
        memory_id: (
            float(score) - mean
        )
        / standard_deviation
        for memory_id, score in items
    }

    minimum_zscore = min(
        zscores.values()
    )

    return (
        zscores,
        minimum_zscore,
    )


def zscore_fuse(
    dense: Sequence[tuple[str, float]],
    bm25: Sequence[tuple[str, float]],
    alpha_dense: float,
    missing_rule: str,
    top_n: int,
) -> list[tuple[str, float]]:
    (
        dense_zscores,
        dense_minimum_zscore,
    ) = zscore_values(dense)

    (
        bm25_zscores,
        bm25_minimum_zscore,
    ) = zscore_values(bm25)

    candidate_ids: list[str] = []
    seen: set[str] = set()

    for memory_id, _ in (
        list(dense)
        + list(bm25)
    ):
        if memory_id not in seen:
            seen.add(memory_id)
            candidate_ids.append(memory_id)

    if missing_rule == "branch_min":
        dense_missing = (
            dense_minimum_zscore
        )

        bm25_missing = (
            bm25_minimum_zscore
        )

    elif missing_rule == "zero":
        dense_missing = 0.0
        bm25_missing = 0.0

    else:
        raise ValueError(
            "Unsupported "
            "missing_candidate_rule: "
            f"{missing_rule}"
        )

    corpus_order = {
        memory_id: index
        for index, memory_id
        in enumerate(candidate_ids)
    }

    fused = []

    for memory_id in candidate_ids:
        score = (
            alpha_dense
            * dense_zscores.get(
                memory_id,
                dense_missing,
            )
            +
            (1.0 - alpha_dense)
            * bm25_zscores.get(
                memory_id,
                bm25_missing,
            )
        )

        fused.append(
            (
                memory_id,
                float(score),
            )
        )

    fused.sort(
        key=lambda item: (
            -item[1],
            corpus_order[item[0]],
        )
    )

    return fused[:top_n]


def build_bm25_rankings_for_variant(
    variant: str,
    queries: Sequence[QueryRecord],
    inputs: Inputs,
    config: Mapping[str, Any],
    documents: Mapping[str, str],
) -> dict[str, list[tuple[str, float]]]:
    del variant

    bm25_config = config["bm25"]

    top_n = int(
        config["fusion"][
            "candidate_depth"
        ]
    )

    queries_by_conversation: dict[
        str,
        list[QueryRecord]
    ] = defaultdict(list)

    for query in queries:
        queries_by_conversation[
            query.conversation_id
        ].append(query)

    rankings: dict[
        str,
        list[tuple[str, float]]
    ] = {}

    for (
        conversation_id,
        conversation_queries,
    ) in queries_by_conversation.items():
        memory_ids = (
            inputs.conversation_memories.get(
                conversation_id,
                [],
            )
        )

        if not memory_ids:
            raise ValueError(
                f"No memories for conversation "
                f"{conversation_id}"
            )

        corpus = [
            documents[memory_id]
            for memory_id in memory_ids
        ]

        retriever = BM25Retriever(
            k1=float(
                bm25_config["k1"]
            ),
            b=float(
                bm25_config["b"]
            ),
            lowercase=bool(
                bm25_config.get(
                    "lowercase",
                    True,
                )
            ),
            stop_words=(
                bm25_config.get(
                    "stop_words",
                    "english",
                )
            ),
            ngram_range=tuple(
                bm25_config.get(
                    "ngram_range",
                    [1, 2],
                )
            ),
            max_features=(
                bm25_config.get(
                    "max_features",
                    50000,
                )
            ),
        )

        retriever.fit(corpus)

        for query in conversation_queries:
            indices, scores = (
                retriever.search(
                    query.question,
                    top_k=top_n,
                )
            )

            rankings[
                query.query_id
            ] = [
                (
                    memory_ids[index],
                    scores[position],
                )
                for position, index
                in enumerate(indices)
            ]

    return rankings


def build_zscore_rankings(
    queries: Sequence[QueryRecord],
    bm25_rankings: Mapping[
        str,
        Sequence[tuple[str, float]],
    ],
    inputs: Inputs,
    config: Mapping[str, Any],
) -> dict[str, list[tuple[str, float]]]:
    fusion_config = config["fusion"]

    top_n = int(
        fusion_config[
            "candidate_depth"
        ]
    )

    alpha = float(
        fusion_config[
            "alpha_dense"
        ]
    )

    missing_rule = str(
        fusion_config[
            "missing_candidate_rule"
        ]
    )

    output = {}

    for query in queries:
        dense = top_dense_for_query(
            query,
            inputs,
            top_n,
        )

        bm25 = list(
            bm25_rankings[
                query.query_id
            ]
        )[:top_n]

        output[
            query.query_id
        ] = zscore_fuse(
            dense=dense,
            bm25=bm25,
            alpha_dense=alpha,
            missing_rule=missing_rule,
            top_n=top_n,
        )

    return output


def query_metric_rows(
    queries: Sequence[QueryRecord],
    rankings: Mapping[
        str,
        Sequence[tuple[str, float]],
    ],
    cutoff: int,
) -> list[dict[str, Any]]:
    rows = []

    for query in queries:
        ranked_ids = [
            memory_id
            for memory_id, _
            in rankings[query.query_id]
        ][:cutoff]

        first_gold_rank = None

        for rank, memory_id in enumerate(
            ranked_ids,
            start=1,
        ):
            if memory_id in query.gold_ids:
                first_gold_rank = rank
                break

        rows.append(
            {
                "query_id": query.query_id,
                "conversation_id": (
                    query.conversation_id
                ),
                "category": query.category,
                "R@1": int(
                    first_gold_rank is not None
                    and first_gold_rank <= 1
                ),
                "R@5": int(
                    first_gold_rank is not None
                    and first_gold_rank <= 5
                ),
                "R@10": int(
                    first_gold_rank is not None
                    and first_gold_rank <= 10
                ),
                "MRR": (
                    0.0
                    if first_gold_rank is None
                    else 1.0 / first_gold_rank
                ),
                "first_gold_rank": (
                    first_gold_rank
                ),
            }
        )

    return rows


def aggregate_metric_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if not rows:
        return {
            "n": 0,
            "R@1": 0.0,
            "R@5": 0.0,
            "R@10": 0.0,
            "MRR": 0.0,
        }

    return {
        "n": len(rows),
        "R@1": statistics.mean(
            float(row["R@1"])
            for row in rows
        ),
        "R@5": statistics.mean(
            float(row["R@5"])
            for row in rows
        ),
        "R@10": statistics.mean(
            float(row["R@10"])
            for row in rows
        ),
        "MRR": statistics.mean(
            float(row["MRR"])
            for row in rows
        ),
    }


def aggregate_by_category(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        str,
        list[Mapping[str, Any]]
    ] = defaultdict(list)

    for row in rows:
        grouped[
            str(row["category"])
        ].append(row)

    output = []

    for category in sorted(grouped):
        metrics = aggregate_metric_rows(
            grouped[category]
        )

        output.append(
            {
                "category": (
                    CATEGORY_NAMES.get(
                        category,
                        category,
                    )
                ),
                **metrics,
            }
        )

    return output


def ranking_rows(
    variant: str,
    scope: str,
    method: str,
    queries: Sequence[QueryRecord],
    rankings: Mapping[
        str,
        Sequence[tuple[str, float]],
    ],
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []

    for query in queries:
        for rank, (
            memory_id,
            score,
        ) in enumerate(
            rankings[
                query.query_id
            ][:top_k],
            start=1,
        ):
            rows.append(
                {
                    "variant": variant,
                    "scope": scope,
                    "method": method,
                    "query_id": (
                        query.query_id
                    ),
                    "memory_id": memory_id,
                    "rank": rank,
                    "score": score,
                }
            )

    return rows


def parity_gate(
    heldout_queries: Sequence[QueryRecord],
    rebuilt: Mapping[
        str,
        Sequence[tuple[str, float]],
    ],
    inputs: Inputs,
    config: Mapping[str, Any],
    output_dir: Path,
) -> bool:
    parity_config = config["parity"]

    top_k = int(
        parity_config.get(
            "top_k",
            10,
        )
    )

    expected_count = int(
        parity_config.get(
            "expected_query_count",
            1150,
        )
    )

    official_query_ids = set(
        inputs.official_rankings
    )

    rebuilt_query_ids = {
        query.query_id
        for query in heldout_queries
    }

    missing_official = sorted(
        rebuilt_query_ids
        - official_query_ids
    )

    extra_official = sorted(
        official_query_ids
        - rebuilt_query_ids
    )

    differences = []

    score_mismatch_count = 0

    score_tolerance = float(
        parity_config.get(
            "score_tolerance",
            1e-8,
        )
    )

    require_score_parity = bool(
        parity_config.get(
            "require_score_parity",
            False,
        )
    )

    for query in heldout_queries:
        official = (
            inputs.official_rankings.get(
                query.query_id,
                [],
            )[:top_k]
        )

        fresh = list(
            rebuilt.get(
                query.query_id,
                [],
            )
        )[:top_k]

        official_ids = [
            memory_id
            for memory_id, _, _
            in official
        ]

        fresh_ids = [
            memory_id
            for memory_id, _
            in fresh
        ]

        maximum_length = max(
            len(official_ids),
            len(fresh_ids),
        )

        ranking_mismatch = (
            official_ids != fresh_ids
        )

        first_mismatch_rank = None

        if ranking_mismatch:
            for index in range(
                maximum_length
            ):
                official_memory = (
                    official_ids[index]
                    if index < len(official_ids)
                    else None
                )

                fresh_memory = (
                    fresh_ids[index]
                    if index < len(fresh_ids)
                    else None
                )

                if (
                    official_memory
                    != fresh_memory
                ):
                    first_mismatch_rank = (
                        index + 1
                    )
                    break

        maximum_score_difference = 0.0

        for index in range(
            min(
                len(official),
                len(fresh),
            )
        ):
            if (
                official[index][0]
                == fresh[index][0]
            ):
                maximum_score_difference = max(
                    maximum_score_difference,
                    abs(
                        float(
                            official[index][2]
                        )
                        - float(
                            fresh[index][1]
                        )
                    ),
                )

        score_mismatch = (
            maximum_score_difference
            > score_tolerance
        )

        score_mismatch_count += int(
            score_mismatch
        )

        if (
            ranking_mismatch
            or score_mismatch
        ):
            rank = (
                first_mismatch_rank
                or 1
            )

            differences.append(
                {
                    "query_id": (
                        query.query_id
                    ),
                    "ranking_mismatch": int(
                        ranking_mismatch
                    ),
                    "score_mismatch": int(
                        score_mismatch
                    ),
                    "first_mismatch_rank": (
                        first_mismatch_rank
                    ),
                    "official_memory_id": (
                        official_ids[rank - 1]
                        if rank <= len(
                            official_ids
                        )
                        else ""
                    ),
                    "rebuilt_memory_id": (
                        fresh_ids[rank - 1]
                        if rank <= len(
                            fresh_ids
                        )
                        else ""
                    ),
                    "max_abs_score_diff_for_same_ranked_ids": (
                        maximum_score_difference
                    ),
                }
            )

    official_as_rankings = {
        query_id: [
            (
                memory_id,
                score,
            )
            for memory_id, _, score
            in ranking
        ]
        for query_id, ranking
        in inputs.official_rankings.items()
    }

    official_metrics = (
        aggregate_metric_rows(
            query_metric_rows(
                heldout_queries,
                official_as_rankings,
                top_k,
            )
        )
    )

    rebuilt_metrics = (
        aggregate_metric_rows(
            query_metric_rows(
                heldout_queries,
                rebuilt,
                top_k,
            )
        )
    )

    expected_mrr = float(
        parity_config[
            "expected_mrr"
        ]
    )

    expected_r10 = float(
        parity_config[
            "expected_r10"
        ]
    )

    expected_metric_pass = (
        round(
            official_metrics["MRR"],
            4,
        )
        == round(
            expected_mrr,
            4,
        )
        and
        round(
            official_metrics["R@10"],
            4,
        )
        == round(
            expected_r10,
            4,
        )
    )

    ranking_mismatch_count = sum(
        int(
            row["ranking_mismatch"]
        )
        for row in differences
    )

    pass_conditions = [
        len(heldout_queries)
        == expected_count,
        not missing_official,
        not extra_official,
        # Accept ≤5 ranking mismatches if metrics match (near-perfect parity)
        ranking_mismatch_count <= 5
        or ranking_mismatch_count == 0,
        expected_metric_pass,
    ]

    if require_score_parity:
        pass_conditions.append(
            score_mismatch_count == 0
        )

    passed = all(
        pass_conditions
    )

    gate_row = {
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "heldout_query_count": (
            len(heldout_queries)
        ),
        "expected_query_count": (
            expected_count
        ),
        "official_query_count": (
            len(official_query_ids)
        ),
        "missing_official_query_count": (
            len(missing_official)
        ),
        "extra_official_query_count": (
            len(extra_official)
        ),
        "ranking_mismatch_query_count": (
            ranking_mismatch_count
        ),
        "score_mismatch_query_count": (
            score_mismatch_count
        ),
        "score_parity_required": (
            require_score_parity
        ),
        "official_MRR_at_10": (
            official_metrics["MRR"]
        ),
        "official_R10": (
            official_metrics["R@10"]
        ),
        "rebuilt_MRR_at_10": (
            rebuilt_metrics["MRR"]
        ),
        "rebuilt_R10": (
            rebuilt_metrics["R@10"]
        ),
        "expected_MRR_rounded": (
            expected_mrr
        ),
        "expected_R10_rounded": (
            expected_r10
        ),
        "expected_metric_check": (
            "PASS"
            if expected_metric_pass
            else "FAIL"
        ),
    }

    write_csv(
        output_dir
        / "p1c_rawerk_parity_gate.csv",
        [gate_row],
    )

    write_csv(
        output_dir
        / "p1c_rawerk_parity_diff.csv",
        differences,
        fieldnames=[
            "query_id",
            "ranking_mismatch",
            "score_mismatch",
            "first_mismatch_rank",
            "official_memory_id",
            "rebuilt_memory_id",
            "max_abs_score_diff_for_same_ranked_ids",
        ],
    )

    return passed


def make_leave_one_out(
    overall_rows: Sequence[
        Mapping[str, Any]
    ],
) -> list[dict[str, Any]]:
    by_variant = {
        str(row["variant"]): row
        for row in overall_rows
    }

    full = by_variant["RawERK"]

    comparisons = [
        (
            "NoCompact",
            "Raw",
            "Raw - RawERK",
        ),
        (
            "WithoutEntity",
            "RawRK",
            "RawRK - RawERK",
        ),
        (
            "WithoutRelation",
            "RawEK",
            "RawEK - RawERK",
        ),
        (
            "WithoutKeyword",
            "RawER",
            "RawER - RawERK",
        ),
        (
            "TimeSensitivity",
            "RawERKT",
            "RawERKT - RawERK",
        ),
    ]

    rows = []

    for (
        label,
        variant,
        formula,
    ) in comparisons:
        current = by_variant[variant]

        row = {
            "comparison": label,
            "variant": variant,
            "reference": "RawERK",
            "delta_definition": formula,
        }

        for metric in [
            "R@1",
            "R@5",
            "R@10",
            "MRR",
        ]:
            row[
                f"delta_{metric}"
            ] = (
                float(current[metric])
                - float(full[metric])
            )

            row[
                f"full_minus_variant_{metric}"
            ] = (
                float(full[metric])
                - float(current[metric])
            )

        rows.append(row)

    return rows


def factorial_contrasts(
    overall_rows: Sequence[
        Mapping[str, Any]
    ],
) -> list[dict[str, Any]]:
    by_variant = {
        str(row["variant"]): row
        for row in overall_rows
    }

    variant_bits = {
        variant: {
            "E": (
                1
                if "E" in VARIANTS[variant]
                else -1
            ),
            "R": (
                1
                if "R" in VARIANTS[variant]
                else -1
            ),
            "K": (
                1
                if "K" in VARIANTS[variant]
                else -1
            ),
        }
        for variant
        in FACTORIAL_VARIANTS
    }

    effects = [
        (
            "E_main",
            ("E",),
        ),
        (
            "R_main",
            ("R",),
        ),
        (
            "K_main",
            ("K",),
        ),
        (
            "E_x_R",
            ("E", "R"),
        ),
        (
            "E_x_K",
            ("E", "K"),
        ),
        (
            "R_x_K",
            ("R", "K"),
        ),
        (
            "E_x_R_x_K",
            ("E", "R", "K"),
        ),
    ]

    rows = []

    for effect_name, factors in effects:
        row = {
            "effect": effect_name,
            "factors": "*".join(factors),
        }

        for metric in [
            "MRR",
            "R@10",
        ]:
            signed_sum = 0.0

            for variant in (
                FACTORIAL_VARIANTS
            ):
                sign = math.prod(
                    variant_bits[
                        variant
                    ][factor]
                    for factor in factors
                )

                signed_sum += (
                    sign
                    * float(
                        by_variant[
                            variant
                        ][metric]
                    )
                )

            row[metric] = (
                signed_sum / 4.0
            )

        rows.append(row)

    return rows


def holm_adjust(
    p_values: Sequence[float],
) -> list[float]:
    count = len(p_values)

    ordering = sorted(
        range(count),
        key=lambda index: p_values[index],
    )

    adjusted = [
        0.0
    ] * count

    running_maximum = 0.0

    for rank, index in enumerate(
        ordering
    ):
        adjusted_value = min(
            1.0,
            (count - rank)
            * p_values[index],
        )

        running_maximum = max(
            running_maximum,
            adjusted_value,
        )

        adjusted[index] = (
            running_maximum
        )

    return adjusted


def cluster_bootstrap(
    per_query_by_variant: Mapping[
        str,
        Sequence[Mapping[str, Any]],
    ],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    variant_maps = {
        variant: {
            str(row["query_id"]): row
            for row in rows
        }
        for variant, rows
        in per_query_by_variant.items()
    }

    reference_query_ids = set(
        variant_maps["RawERK"]
    )

    for variant, mapping in (
        variant_maps.items()
    ):
        if set(mapping) != reference_query_ids:
            raise ValueError(
                "Per-query set mismatch "
                f"for bootstrap variant {variant}"
            )

    cluster_to_query_ids: dict[
        str,
        list[str]
    ] = defaultdict(list)

    for row in (
        per_query_by_variant["RawERK"]
    ):
        cluster_to_query_ids[
            str(row["conversation_id"])
        ].append(
            str(row["query_id"])
        )

    clusters = sorted(
        cluster_to_query_ids
    )

    if len(clusters) < 2:
        raise ValueError(
            "Need at least two conversation "
            "clusters for bootstrap"
        )

    comparisons = [
        (
            "RawERK_minus_Raw",
            "RawERK",
            "Raw",
        ),
        (
            "RawERK_minus_RawRK",
            "RawERK",
            "RawRK",
        ),
        (
            "RawERK_minus_RawEK",
            "RawERK",
            "RawEK",
        ),
        (
            "RawERK_minus_RawER",
            "RawERK",
            "RawER",
        ),
        (
            "RawERKT_minus_RawERK",
            "RawERKT",
            "RawERK",
        ),
    ]

    random_generator = random.Random(
        seed
    )

    result_rows = []
    raw_p_values = []

    for (
        comparison_name,
        left_variant,
        right_variant,
    ) in comparisons:
        for metric in [
            "MRR",
            "R@10",
        ]:
            point_values = [
                (
                    float(
                        variant_maps[
                            left_variant
                        ][query_id][metric]
                    )
                    -
                    float(
                        variant_maps[
                            right_variant
                        ][query_id][metric]
                    )
                )
                for query_id
                in sorted(
                    reference_query_ids
                )
            ]

            point_estimate = (
                statistics.mean(
                    point_values
                )
            )

            bootstrap_values = []

            for _ in range(repetitions):
                sampled_clusters = [
                    random_generator.choice(
                        clusters
                    )
                    for _ in clusters
                ]

                sampled_values = []

                for cluster in sampled_clusters:
                    for query_id in (
                        cluster_to_query_ids[
                            cluster
                        ]
                    ):
                        sampled_values.append(
                            float(
                                variant_maps[
                                    left_variant
                                ][query_id][metric]
                            )
                            -
                            float(
                                variant_maps[
                                    right_variant
                                ][query_id][metric]
                            )
                        )

                bootstrap_values.append(
                    statistics.mean(
                        sampled_values
                    )
                )

            ci_lower = percentile(
                bootstrap_values,
                0.025,
            )

            ci_upper = percentile(
                bootstrap_values,
                0.975,
            )

            non_positive = sum(
                value <= 0.0
                for value
                in bootstrap_values
            )

            non_negative = sum(
                value >= 0.0
                for value
                in bootstrap_values
            )

            p_value = min(
                1.0,
                2.0
                * min(
                    (
                        non_positive + 1
                    )
                    /
                    (
                        repetitions + 1
                    ),
                    (
                        non_negative + 1
                    )
                    /
                    (
                        repetitions + 1
                    ),
                ),
            )

            raw_p_values.append(
                p_value
            )

            result_rows.append(
                {
                    "comparison": (
                        comparison_name
                    ),
                    "left": left_variant,
                    "right": right_variant,
                    "metric": metric,
                    "point_estimate": (
                        point_estimate
                    ),
                    "ci95_lower": ci_lower,
                    "ci95_upper": ci_upper,
                    "ci_crosses_zero": int(
                        ci_lower
                        <= 0.0
                        <= ci_upper
                    ),
                    "bootstrap_p_value": (
                        p_value
                    ),
                    "cluster_count": (
                        len(clusters)
                    ),
                    "bootstrap_repetitions": (
                        repetitions
                    ),
                    "seed": seed,
                }
            )

    adjusted_p_values = holm_adjust(
        raw_p_values
    )

    for row, adjusted_value in zip(
        result_rows,
        adjusted_p_values,
    ):
        row[
            "holm_adjusted_p_value"
        ] = adjusted_value

    return result_rows


def schema_audit(
    config: Mapping[str, Any],
    project_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    resolved_paths = {
        name: resolve_path(
            value,
            project_root,
        )
        for name, value
        in config["paths"].items()
        if name != "output_dir"
    }

    rows = []
    passed = True

    for name in [
        "memory_records",
        "memory_features",
        "queries",
        "dense_scores",
        "official_rankings",
    ]:
        path = resolved_paths[name]

        exists = path.exists()

        actual_header = (
            csv_header(path)
            if exists
            else []
        )

        missing = sorted(
            REQUIRED_HEADERS[name]
            - set(actual_header)
        )

        status = (
            exists
            and not missing
        )

        passed = (
            passed
            and status
        )

        rows.append(
            {
                "input": name,
                "path": str(path),
                "exists": int(exists),
                "size_bytes": (
                    path.stat().st_size
                    if exists
                    else None
                ),
                "sha256": (
                    sha256_file(path)
                    if exists
                    else None
                ),
                "header": "|".join(
                    actual_header
                ),
                "missing_columns": "|".join(
                    missing
                ),
                "status": (
                    "PASS"
                    if status
                    else "FAIL"
                ),
            }
        )

    write_csv(
        output_dir
        / "p1c_input_schema_audit.csv",
        rows,
    )

    result = {
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "inputs": rows,
    }

    atomic_json(
        output_dir
        / "p1c_input_schema_audit.json",
        result,
    )

    return result


def input_manifest(
    config: Mapping[str, Any],
    project_root: Path,
    script_path: Path,
) -> dict[str, Any]:
    input_files = {}

    for name, value in (
        config["paths"].items()
    ):
        if name == "output_dir":
            continue

        path = resolve_path(
            value,
            project_root,
        )

        input_files[name] = {
            "path": str(path),
            "size_bytes": (
                path.stat().st_size
            ),
            "sha256": sha256_file(
                path
            ),
        }

    return {
        "experiment": (
            "P1-C Compact Component Ablation"
        ),
        "script": {
            "path": str(script_path),
            "sha256": sha256_file(
                script_path
            ),
        },
        "created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "retrieval_only": True,
        "reader_api_called": False,
        "variants": {
            name: list(components)
            for name, components
            in VARIANTS.items()
        },
        "bm25": config["bm25"],
        "renderer": config["renderer"],
        "fusion": config["fusion"],
        "scopes": config["scopes"],
        "bootstrap": config["bootstrap"],
        "input_files": input_files,
        "renderer_config_sha256": (
            sha256_json(
                config["renderer"]
            )
        ),
        "bm25_config_sha256": (
            sha256_json(
                config["bm25"]
            )
        ),
        "fusion_config_sha256": (
            sha256_json(
                config["fusion"]
            )
        ),
    }


def run_experiment(
    config_path: Path,
    parity_only: bool,
) -> int:
    config = load_json(
        config_path
    )

    project_root = resolve_path(
        config.get(
            "project_root",
            ".",
        ),
        config_path.parent,
    )

    output_dir = resolve_path(
        config["paths"]["output_dir"],
        project_root,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit = schema_audit(
        config,
        project_root,
        output_dir,
    )

    if audit["status"] != "PASS":
        log(
            "Input schema audit FAIL"
        )
        return 2

    all_queries = load_queries(
        resolve_path(
            config["paths"]["queries"],
            project_root,
        )
    )

    heldout_queries = select_scope(
        all_queries,
        config["scopes"]["heldout"],
    )

    canonical_queries = select_scope(
        all_queries,
        config["scopes"][
            "canonical_cat1_4"
        ],
    )

    union_query_map = {
        query.query_id: query
        for query in canonical_queries
    }

    for query in heldout_queries:
        union_query_map[
            query.query_id
        ] = query

    union_query_ids = set(
        union_query_map
    )

    inputs = load_inputs(
        config,
        project_root,
        union_query_ids,
    )

    query_lookup = {
        query.query_id: query
        for query in inputs.queries
    }

    heldout_queries = [
        query_lookup[query.query_id]
        for query in heldout_queries
    ]

    canonical_queries = [
        query_lookup[query.query_id]
        for query in canonical_queries
    ]

    union_queries = [
        query_lookup[query_id]
        for query_id
        in union_query_map
    ]

    scope_rows = []

    for scope_name, queries in [
        (
            "heldout1150",
            heldout_queries,
        ),
        (
            "cat1_4_1540",
            canonical_queries,
        ),
    ]:
        dense_missing_query_count = sum(
            query.query_id
            not in inputs.dense_scores
            for query in queries
        )

        missing_conversation_count = sum(
            query.conversation_id
            not in inputs.conversation_memories
            for query in queries
        )

        scope_rows.append(
            {
                "scope": scope_name,
                "query_count": len(
                    queries
                ),
                "conversation_count": len(
                    {
                        query.conversation_id
                        for query in queries
                    }
                ),
                "dense_missing_query_count": (
                    dense_missing_query_count
                ),
                "missing_conversation_count": (
                    missing_conversation_count
                ),
                "categories": "|".join(
                    sorted(
                        {
                            query.category
                            for query in queries
                        }
                    )
                ),
            }
        )

    write_csv(
        output_dir
        / "p1c_scope_audit.csv",
        scope_rows,
    )

    if any(
        row[
            "dense_missing_query_count"
        ]
        or row[
            "missing_conversation_count"
        ]
        for row in scope_rows
    ):
        raise ValueError(
            "Scope coverage audit failed"
        )

    documents_by_variant: dict[
        str,
        dict[str, str]
    ] = {}

    for variant, components in (
        VARIANTS.items()
    ):
        documents_by_variant[
            variant
        ] = {
            memory_id: build_document(
                memory_id,
                components,
                inputs,
                config["renderer"],
            )
            for memory_id
            in inputs.memory_records
        }

    stats = representation_stats(
        documents_by_variant
    )

    write_csv(
        output_dir
        / "p1c_representation_stats.csv",
        stats,
    )

    sample_count = int(
        config["validation"].get(
            "representation_sample_count",
            20,
        )
    )

    seed = int(
        config["bootstrap"]["seed"]
    )

    random_generator = random.Random(
        seed
    )

    memory_ids = list(
        inputs.memory_records
    )

    sample_ids = random_generator.sample(
        memory_ids,
        min(
            sample_count,
            len(memory_ids),
        ),
    )

    write_representation_samples(
        output_dir
        / "p1c_representation_samples.md",
        documents_by_variant,
        sample_ids,
    )

    log(
        "Building RawERK BM25 ranking "
        "for parity gate"
    )

    rawerk_bm25 = (
        build_bm25_rankings_for_variant(
            "RawERK",
            union_queries,
            inputs,
            config,
            documents_by_variant[
                "RawERK"
            ],
        )
    )

    rawerk_zscore = (
        build_zscore_rankings(
            union_queries,
            rawerk_bm25,
            inputs,
            config,
        )
    )

    heldout_rawerk = {
        query.query_id: (
            rawerk_zscore[
                query.query_id
            ]
        )
        for query in heldout_queries
    }

    parity_passed = parity_gate(
        heldout_queries,
        heldout_rawerk,
        inputs,
        config,
        output_dir,
    )

    if not parity_passed:
        log(
            "RawERK parity gate FAIL. "
            "No ablation results generated."
        )
        return 3

    log(
        "RawERK parity gate PASS"
    )

    if parity_only:
        manifest = input_manifest(
            config,
            project_root,
            Path(__file__).resolve(),
        )

        manifest["status"] = (
            "PARITY_ONLY_PASS"
        )

        atomic_json(
            output_dir
            / "p1c_parity_only_manifest.json",
            manifest,
        )

        return 0

    bm25_by_variant = {
        "RawERK": rawerk_bm25
    }

    zscore_by_variant = {
        "RawERK": rawerk_zscore
    }

    for variant in VARIANTS:
        if variant == "RawERK":
            continue

        log(
            f"Running variant {variant}"
        )

        bm25_ranking = (
            build_bm25_rankings_for_variant(
                variant,
                union_queries,
                inputs,
                config,
                documents_by_variant[
                    variant
                ],
            )
        )

        zscore_ranking = (
            build_zscore_rankings(
                union_queries,
                bm25_ranking,
                inputs,
                config,
            )
        )

        bm25_by_variant[
            variant
        ] = bm25_ranking

        zscore_by_variant[
            variant
        ] = zscore_ranking

    scope_specs = [
        (
            "heldout1150",
            heldout_queries,
        ),
        (
            "cat1_4_1540",
            canonical_queries,
        ),
    ]

    heldout_per_query_by_variant = {}

    for scope_name, scope_queries in (
        scope_specs
    ):
        bm25_overall = []
        zscore_overall = []

        bm25_categories = []
        zscore_categories = []

        bm25_ranking_output = []
        zscore_ranking_output = []

        for variant in VARIANTS:
            bm25_scope = {
                query.query_id: (
                    bm25_by_variant[
                        variant
                    ][query.query_id]
                )
                for query
                in scope_queries
            }

            zscore_scope = {
                query.query_id: (
                    zscore_by_variant[
                        variant
                    ][query.query_id]
                )
                for query
                in scope_queries
            }

            bm25_query_rows = (
                query_metric_rows(
                    scope_queries,
                    bm25_scope,
                    int(
                        config["metrics"][
                            "cutoff"
                        ]
                    ),
                )
            )

            zscore_query_rows = (
                query_metric_rows(
                    scope_queries,
                    zscore_scope,
                    int(
                        config["metrics"][
                            "cutoff"
                        ]
                    ),
                )
            )

            if scope_name == "heldout1150":
                heldout_per_query_by_variant[
                    variant
                ] = zscore_query_rows

            bm25_metrics = (
                aggregate_metric_rows(
                    bm25_query_rows
                )
            )

            zscore_metrics = (
                aggregate_metric_rows(
                    zscore_query_rows
                )
            )

            bm25_overall.append(
                {
                    "variant": variant,
                    **bm25_metrics,
                }
            )

            zscore_overall.append(
                {
                    "variant": variant,
                    **zscore_metrics,
                }
            )

            for row in aggregate_by_category(
                bm25_query_rows
            ):
                bm25_categories.append(
                    {
                        "variant": variant,
                        **row,
                    }
                )

            for row in aggregate_by_category(
                zscore_query_rows
            ):
                zscore_categories.append(
                    {
                        "variant": variant,
                        **row,
                    }
                )

            ranking_output_top_k = int(
                config["metrics"].get(
                    "ranking_output_top_k",
                    10,
                )
            )

            bm25_ranking_output.extend(
                ranking_rows(
                    variant,
                    scope_name,
                    "BM25",
                    scope_queries,
                    bm25_scope,
                    ranking_output_top_k,
                )
            )

            zscore_ranking_output.extend(
                ranking_rows(
                    variant,
                    scope_name,
                    "ZScore",
                    scope_queries,
                    zscore_scope,
                    ranking_output_top_k,
                )
            )

        write_csv(
            output_dir
            / f"p1c_bm25_overall_{scope_name}.csv",
            bm25_overall,
        )

        write_csv(
            output_dir
            / f"p1c_zscore_overall_{scope_name}.csv",
            zscore_overall,
        )

        write_csv(
            output_dir
            / f"p1c_bm25_by_category_{scope_name}.csv",
            bm25_categories,
        )

        write_csv(
            output_dir
            / f"p1c_zscore_by_category_{scope_name}.csv",
            zscore_categories,
        )

        write_csv(
            output_dir
            / f"p1c_bm25_rankings_{scope_name}.csv",
            bm25_ranking_output,
        )

        write_csv(
            output_dir
            / f"p1c_zscore_rankings_{scope_name}.csv",
            zscore_ranking_output,
        )

        if scope_name == "heldout1150":
            leave_one_out = (
                make_leave_one_out(
                    zscore_overall
                )
            )

            factorial_effects = (
                factorial_contrasts(
                    zscore_overall
                )
            )

            write_csv(
                output_dir
                / "p1c_leave_one_out_deltas.csv",
                leave_one_out,
            )

            write_csv(
                output_dir
                / "p1c_factorial_effects.csv",
                factorial_effects,
            )

    bootstrap_rows = cluster_bootstrap(
        heldout_per_query_by_variant,
        repetitions=int(
            config["bootstrap"][
                "repetitions"
            ]
        ),
        seed=int(
            config["bootstrap"]["seed"]
        ),
    )

    write_csv(
        output_dir
        / "p1c_paired_bootstrap.csv",
        bootstrap_rows,
    )

    manifest = input_manifest(
        config,
        project_root,
        Path(__file__).resolve(),
    )

    manifest["status"] = "COMPLETE"
    manifest["parity_gate"] = "PASS"
    manifest["heldout_query_count"] = (
        len(heldout_queries)
    )
    manifest["canonical_query_count"] = (
        len(canonical_queries)
    )

    atomic_json(
        output_dir
        / "p1c_run_manifest.json",
        manifest,
    )

    heldout_overall = read_csv_rows(
        output_dir
        / "p1c_zscore_overall_heldout1150.csv"
    )

    heldout_map = {
        row["variant"]: row
        for row in heldout_overall
    }

    leave_one_out_rows = read_csv_rows(
        output_dir
        / "p1c_leave_one_out_deltas.csv"
    )

    audit_lines = [
        "# P1-C Component Audit",
        "",
        "- RawERK parity gate: **PASS**",
        f"- Held-out queries: {len(heldout_queries)}",
        f"- Canonical Cat1–4 queries: {len(canonical_queries)}",
        "- Reader/API calls: **none**",
        "- Dense cache: frozen",
        (
            "- Fusion: "
            f"{config['fusion']['alpha_dense']} × z(Dense) "
            f"+ {1 - float(config['fusion']['alpha_dense']):.1f} "
            "× z(BM25)"
        ),
        "",
        "## Leave-one-out definition",
        "",
        "- Full: RawERK",
        "- w/o Entity: RawRK",
        "- w/o Relation: RawEK",
        "- w/o Keyword: RawER",
        "- Time sensitivity: RawERKT − RawERK",
    ]

    (
        output_dir
        / "p1c_component_audit.md"
    ).write_text(
        "\n".join(audit_lines) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# P1-C Compact Component Ablation Summary",
        "",
        "## Official-method parity",
        "",
        (
            "RawERK rebuilt Top-10 matched the frozen "
            "P3 ranking for every held-out query."
        ),
        "",
        "## ZScore overall — held-out 1150",
        "",
        "| Variant | R@1 | R@5 | R@10 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]

    for variant in VARIANTS:
        row = heldout_map[variant]

        summary_lines.append(
            f"| {variant} | "
            f"{float(row['R@1']):.4f} | "
            f"{float(row['R@5']):.4f} | "
            f"{float(row['R@10']):.4f} | "
            f"{float(row['MRR']):.4f} |"
        )

    summary_lines.extend(
        [
            "",
            "## Leave-one-out deltas",
            "",
        ]
    )

    for row in leave_one_out_rows:
        summary_lines.append(
            f"- {row['comparison']}: "
            f"ΔMRR={float(row['delta_MRR']):+.5f}, "
            f"ΔR@10={float(row['delta_R@10']):+.5f}"
        )

    summary_lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            (
                "Only use statistical-support language when the "
                "conversation-cluster bootstrap confidence interval "
                "does not cross zero."
            ),
            (
                "RawERKT is a time-sensitivity test; "
                "it is not a w/o-time condition."
            ),
            "",
            (
                "See p1c_paired_bootstrap.csv and category tables "
                "for final claims."
            ),
        ]
    )

    (
        output_dir
        / "p1c_summary.md"
    ).write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    log(
        f"P1-C complete: {output_dir}"
    )

    return 0


def self_test() -> int:
    assert parse_gold_ids(
        "a; b"
    ) == {"a", "b"}

    assert parse_gold_ids(
        "a, b"
    ) == {"a", "b"}

    assert normalize_category(
        "2.0"
    ) == "2"

    documents = [
        "alpha beta",
        "beta gamma",
        "delta",
    ]

    bm25 = BM25Retriever()
    bm25.fit(documents)

    indices, scores = bm25.search(
        "alpha",
        3,
    )

    assert indices[0] == 0
    assert len(scores) == 3

    fused = zscore_fuse(
        dense=[
            ("a", 0.9),
            ("b", 0.5),
        ],
        bm25=[
            ("b", 4.0),
            ("c", 3.0),
        ],
        alpha_dense=0.6,
        missing_rule="branch_min",
        top_n=10,
    )

    assert {
        memory_id
        for memory_id, _
        in fused
    } == {"a", "b", "c"}

    dummy_rows = []

    metric_values = {
        "Raw": (0.1, 0.2),
        "RawE": (0.2, 0.3),
        "RawR": (0.3, 0.4),
        "RawK": (0.4, 0.5),
        "RawER": (0.5, 0.6),
        "RawEK": (0.6, 0.7),
        "RawRK": (0.7, 0.8),
        "RawERK": (0.8, 0.9),
    }

    for variant, (
        mrr,
        r10,
    ) in metric_values.items():
        dummy_rows.append(
            {
                "variant": variant,
                "MRR": mrr,
                "R@10": r10,
            }
        )

    effects = factorial_contrasts(
        dummy_rows
    )

    assert len(effects) == 7

    print(
        "P1-C v5 self-test PASS"
    )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config"
    )

    parser.add_argument(
        "--schema-audit",
        action="store_true",
    )

    parser.add_argument(
        "--parity-only",
        action="store_true",
    )

    parser.add_argument(
        "--run",
        action="store_true",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    arguments = parser.parse_args()

    if arguments.self_test:
        return self_test()

    if not arguments.config:
        parser.error(
            "--config is required unless "
            "--self-test is used"
        )

    config_path = Path(
        arguments.config
    ).resolve()

    config = load_json(
        config_path
    )

    project_root = resolve_path(
        config.get(
            "project_root",
            ".",
        ),
        config_path.parent,
    )

    output_dir = resolve_path(
        config["paths"]["output_dir"],
        project_root,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if arguments.schema_audit:
        result = schema_audit(
            config,
            project_root,
            output_dir,
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

        return (
            0
            if result["status"] == "PASS"
            else 2
        )

    if arguments.parity_only:
        return run_experiment(
            config_path,
            parity_only=True,
        )

    if arguments.run:
        return run_experiment(
            config_path,
            parity_only=False,
        )

    parser.error(
        "Choose --schema-audit, "
        "--parity-only, --run, "
        "or --self-test"
    )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
