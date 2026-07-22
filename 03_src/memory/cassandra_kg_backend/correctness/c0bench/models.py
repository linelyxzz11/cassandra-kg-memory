from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def logical_edge_id(graph_id: str, src_id: str, relation: str, dst_id: str, source: str = "") -> str:
    """Portable identity for a logical triple.

    Existing Cassandra rows use server-generated `timeuuid` values. Those values cannot
    be compared with Neo4j relationship IDs or with a CSV lacking edge IDs. C0 therefore
    compares a deterministic logical ID derived from the actual triple payload.
    """
    payload = json.dumps([graph_id, src_id, relation, dst_id, source], ensure_ascii=False, separators=(",", ":"))
    return "L" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True, order=True)
class Edge:
    graph_id: str
    src_id: str
    relation: str
    dst_id: str
    source: str = ""

    @property
    def logical_id(self) -> str:
        return logical_edge_id(self.graph_id, self.src_id, self.relation, self.dst_id, self.source)

    @property
    def sort_key(self) -> tuple[str, str, str]:
        # Source is a stable, backend-portable tie breaker. Exact duplicate logical edges
        # are deduplicated during canonicalisation because the existing timeuuid schema
        # cannot distinguish equivalent triples across systems.
        return (self.relation, self.dst_id, self.source)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "Edge":
        return cls(
            graph_id=_text(row["graph_id"]),
            src_id=_text(row["src_id"]),
            relation=_text(row["relation"]),
            dst_id=_text(row["dst_id"]),
            source=_text(row.get("source", "")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "graph_id": self.graph_id,
            "src_id": self.src_id,
            "relation": self.relation,
            "dst_id": self.dst_id,
            "source": self.source,
            "logical_edge_id": self.logical_id,
        }


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    graph_id: str
    seed_id: str
    relation_path: tuple[str, ...]
    hop: int
    fanout: int
    op_type: str = "read"
    random_seed: int = 0
    cycle_policy: str = "path"
    write_edge: Edge | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], default_graph_id: str | None = None,
                     default_cycle_policy: str = "path") -> "QuerySpec":
        op_type = _text(row.get("op_type", "read"))
        hop = int(row.get("hop", 1))
        relation_path = tuple(_text(v) for v in row.get("relation_path", []))
        if not relation_path:
            relation_path = tuple("*" for _ in range(hop))
        if len(relation_path) != hop:
            raise ValueError(
                f"query_id={row.get('query_id')} has hop={hop}, but relation_path has "
                f"{len(relation_path)} entries. Use '*' explicitly for wildcard hops."
            )
        if hop < 1 or int(row.get("fanout", 0)) < 1:
            raise ValueError("hop and fanout must both be >= 1")
        write_edge = Edge.from_mapping(row["write_edge"]) if row.get("write_edge") else None
        if op_type == "write" and write_edge is None:
            raise ValueError("write event requires write_edge")
        return cls(
            query_id=_text(row["query_id"]),
            graph_id=_text(row.get("graph_id") or default_graph_id or ""),
            seed_id=_text(row.get("seed_id", "")),
            relation_path=relation_path,
            hop=hop,
            fanout=int(row["fanout"]),
            op_type=op_type,
            random_seed=int(row.get("random_seed", 0)),
            cycle_policy=_text(row.get("cycle_policy", default_cycle_policy)),
            write_edge=write_edge,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "query_id": self.query_id,
            "graph_id": self.graph_id,
            "seed_id": self.seed_id,
            "relation_path": list(self.relation_path),
            "hop": self.hop,
            "fanout": self.fanout,
            "op_type": self.op_type,
            "random_seed": self.random_seed,
            "cycle_policy": self.cycle_policy,
        }
        if self.write_edge:
            result["write_edge"] = self.write_edge.to_dict()
        return result


@dataclass(frozen=True)
class TraversalResult:
    """Canonical output: every path is a tuple of portable logical-edge IDs."""
    edge_paths: tuple[tuple[str, ...], ...]
    raw_edges_read: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized_paths(self) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted(set(self.edge_paths)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_paths": [list(path) for path in self.normalized_paths()],
            "raw_edges_read": self.raw_edges_read,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "metadata": dict(self.metadata),
        }
