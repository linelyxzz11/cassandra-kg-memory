from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

from .models import Edge, QuerySpec, TraversalResult

FetchMany = Callable[[list[str], str | None], tuple[dict[str, list[Edge]], dict[str, int]]]


def _eligible(edges: Iterable[Edge], relation: str) -> list[Edge]:
    return sorted((edge for edge in edges if relation == "*" or edge.relation == relation), key=lambda edge: edge.sort_key)


def strict_frontier_traversal(query: QuerySpec, fetch_many: FetchMany) -> TraversalResult:
    """The single C0 traversal definition used by all systems and the reference oracle."""
    if query.op_type != "read":
        raise ValueError("strict_frontier_traversal only accepts read events")

    # (current_node, node_path, logical_edge_path)
    frontier: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [(query.seed_id, (query.seed_id,), tuple())]
    raw_edges_read = cache_hits = cache_misses = 0
    for relation in query.relation_path:
        sources = sorted({state[0] for state in frontier})
        relation_filter = None if relation == "*" else relation
        by_source, metrics = fetch_many(sources, relation_filter)
        raw_edges_read += int(metrics.get("raw_edges_read", 0))
        cache_hits += int(metrics.get("cache_hits", 0))
        cache_misses += int(metrics.get("cache_misses", 0))

        next_frontier: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        for source, node_path, edge_path in frontier:
            candidates = _eligible(by_source.get(source, []), relation)
            if query.cycle_policy == "path":
                candidates = [edge for edge in candidates if edge.dst_id not in node_path]
            elif query.cycle_policy != "none":
                raise ValueError(f"Unsupported cycle_policy={query.cycle_policy}")
            for edge in candidates[:query.fanout]:
                next_frontier.append((edge.dst_id, node_path + (edge.dst_id,), edge_path + (edge.logical_id,)))
        frontier = next_frontier
        if not frontier:
            break

    return TraversalResult(
        edge_paths=tuple(sorted({state[2] for state in frontier})),
        raw_edges_read=raw_edges_read,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        metadata={"semantic": "strict_frontier_v2_logical_edge_identity"},
    )


class ReferenceGraph:
    """Canonical in-memory oracle; backend output is never used as ground truth."""

    def __init__(self, edges: Iterable[Edge]):
        self._by_src: dict[tuple[str, str], list[Edge]] = defaultdict(list)
        for edge in edges:
            self._by_src[(edge.graph_id, edge.src_id)].append(edge)
        for values in self._by_src.values():
            values.sort(key=lambda edge: edge.sort_key)

    def execute(self, query: QuerySpec) -> TraversalResult:
        def fetch_many(sources: list[str], relation: str | None):
            output: dict[str, list[Edge]] = {}
            raw = 0
            for source in sources:
                rows = list(self._by_src.get((query.graph_id, source), []))
                output[source] = rows
                raw += len(rows)
            return output, {"raw_edges_read": raw, "cache_hits": 0, "cache_misses": 0}
        return strict_frontier_traversal(query, fetch_many)
