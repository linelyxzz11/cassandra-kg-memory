"""Frozen graph-aware logical event semantics for live-cell v2.

This module is backend-independent. It converts the frozen feature strings into
deterministic mention and relation-edge records used by all four storage cells.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from online_retrieval import render_rawerk


def canonical_token(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def parse_csv_entities(value: str) -> tuple[str, ...]:
    return tuple(sorted({canonical_token(x) for x in str(value or "").split(",") if canonical_token(x)}))


def parse_triples(value: str) -> tuple[tuple[str, str, str], ...]:
    triples: list[tuple[str, str, str]] = []
    for raw in str(value or "").split(";"):
        item = raw.strip()
        if not item:
            continue
        if item.startswith("(") and item.endswith(")"):
            item = item[1:-1]
        parts = [canonical_token(x) for x in item.split(",", 2)]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Invalid frozen triple: {raw!r}")
        triples.append((parts[0], parts[1], parts[2]))
    return tuple(triples)


def digest(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GraphRecord:
    scope_id: str
    memory_id: str
    version: int
    raw_text: str
    entities: str
    relations: str
    keywords: str
    triples: str
    embedding_sha256: str

    @property
    def rawerk(self) -> str:
        return render_rawerk(self.raw_text, self.entities, self.relations, self.keywords)

    @property
    def mentions(self) -> tuple[str, ...]:
        return parse_csv_entities(self.entities)

    @property
    def edges(self) -> tuple[dict, ...]:
        return tuple(
            {"ordinal": ordinal, "src": src, "relation": relation, "dst": dst}
            for ordinal, (src, relation, dst) in enumerate(parse_triples(self.triples))
        )

    @property
    def graph_entities(self) -> tuple[str, ...]:
        values = set(self.mentions)
        for edge in self.edges:
            values.add(edge["src"])
            values.add(edge["dst"])
        return tuple(sorted(values))

    def memory_projection(self) -> dict:
        return {
            "scope_id": self.scope_id,
            "memory_id": self.memory_id,
            "version": self.version,
            "raw_text": self.raw_text,
            "entities": self.entities,
            "relations": self.relations,
            "keywords": self.keywords,
            "triples": self.triples,
            "rawerk": self.rawerk,
            "embedding_sha256": self.embedding_sha256,
        }

    def graph_projection(self) -> dict:
        return {
            "memory": self.memory_projection(),
            "mentions": list(self.mentions),
            "graph_entities": list(self.graph_entities),
            "edges": list(self.edges),
        }

    def expected_mutations(self, cell: str) -> int:
        entity_count = len(self.graph_entities)
        mention_count = len(self.mentions)
        edge_count = len(self.edges)
        if cell == "cassandra-base":
            return 2 + entity_count + mention_count + 3 * edge_count
        if cell == "cassandra-materialized":
            return 1 + entity_count + mention_count + 2 * edge_count
        if cell == "neo4j-native":
            return 3 + entity_count + mention_count + edge_count  # memory, feature, HAS_FEATURE
        if cell == "neo4j-materialized":
            return 1 + entity_count + mention_count + edge_count + len({edge["relation"] for edge in self.edges})
        raise ValueError(cell)
