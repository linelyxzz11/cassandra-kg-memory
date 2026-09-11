"""Backend-neutral protocol primitives for P5-1 v3.

The module intentionally contains no database imports so its event and ranking
contract can be unit-tested without live services.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
PROTOCOL_VERSION = "p5-1-logical-event-v3.1"
RETRIEVER_ID = "shared-rawerk-bm25-fixed-cohort-top10-v1"


@dataclass(frozen=True)
class LogicalEvent:
    """One backend-neutral memory materialization event.

    Observable logical effects are identical on every backend:
    one memory document, two entity records, and one directed KG edge.
    `probe_token` is unique to this memory and is part of RawERK text only so
    final retrieval visibility can be tested deterministically.
    """

    update_id: str
    run_scope: str
    memory_id: str
    version: int
    raw_text: str
    raw_erk_text: str
    src_id: str
    dst_id: str
    relation: str
    edge_id: str
    probe_token: str

    @property
    def probe_query(self) -> str:
        return f"{self.probe_token} {self.src_id} {self.relation}"

    def canonical_view(self) -> dict:
        return asdict(self)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def bm25_rank(
    query: str,
    documents: Sequence[tuple[str, str]],
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[str, float]]:
    """Deterministic BM25 with memory_id ascending as the tie-break."""

    if top_k <= 0 or not documents:
        return []
    tokenized = [(memory_id, tokenize(text)) for memory_id, text in documents]
    lengths = [len(tokens) for _, tokens in tokenized]
    avgdl = sum(lengths) / len(lengths) if lengths else 1.0
    avgdl = max(avgdl, 1.0)
    query_terms = list(dict.fromkeys(tokenize(query)))
    document_frequency = {
        term: sum(1 for _, tokens in tokenized if term in set(tokens))
        for term in query_terms
    }
    scores: list[tuple[str, float]] = []
    n_docs = len(tokenized)
    for memory_id, tokens in tokenized:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        score = 0.0
        doc_len = len(tokens)
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = document_frequency[term]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * doc_len / avgdl)
            score += idf * (tf * (k1 + 1.0)) / denom
        scores.append((memory_id, score))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores[: min(top_k, len(scores))]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"update_id", "memory_id", "version", "raw_text", "raw_erk_text", "src_id", "dst_id", "relation"}
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {missing}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No source events in {path}")
    return rows


def materialize_event(row: dict, run_scope: str, role: str, ordinal: int) -> LogicalEvent:
    """Turn a frozen source row into a collision-free v3 logical event."""

    source_id = str(row["update_id"])
    safe_source = re.sub(r"[^A-Za-z0-9_]", "_", source_id)
    probe_token = f"p5v3_{role}_{ordinal:08d}_{safe_source}"
    memory_id = f"p5v3_{role}_{ordinal:08d}_{row['memory_id']}"
    edge_id = f"p5v3_{role}_{ordinal:08d}_{safe_source}"
    raw_erk = f"{row['raw_erk_text']}\nP: {probe_token}"
    return LogicalEvent(
        update_id=f"{role}_{ordinal:08d}_{source_id}",
        run_scope=run_scope,
        memory_id=memory_id,
        version=int(row["version"]),
        raw_text=str(row["raw_text"]),
        raw_erk_text=raw_erk,
        src_id=str(row["src_id"]),
        dst_id=str(row["dst_id"]),
        relation=str(row["relation"]),
        edge_id=edge_id,
        probe_token=probe_token,
    )


def logical_event_digest(events: Iterable[LogicalEvent]) -> str:
    digest = hashlib.sha256()
    for event in events:
        payload = json.dumps(event.canonical_view(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
