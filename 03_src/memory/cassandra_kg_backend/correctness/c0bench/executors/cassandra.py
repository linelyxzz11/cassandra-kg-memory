from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..models import Edge, QuerySpec, TraversalResult
from ..semantics import strict_frontier_traversal
from .base import BackendExecutor

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsafe Cassandra identifier in config: {value!r}")
    return value


def _bucket(value: str, count: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16) % count


class DegreeAwareOneHopCache:
    """Bounded application cache. It does not change traversal semantics."""

    def __init__(self, capacity: int, degree_threshold: int):
        self.capacity = max(0, int(capacity))
        self.degree_threshold = max(0, int(degree_threshold))
        self._entries: OrderedDict[tuple[str, str, str | None], list[Edge]] = OrderedDict()

    def get(self, key: tuple[str, str, str | None]) -> list[Edge] | None:
        rows = self._entries.get(key)
        if rows is not None:
            self._entries.move_to_end(key)
            return list(rows)
        return None

    def put(self, key: tuple[str, str, str | None], rows: list[Edge]) -> None:
        if self.capacity == 0 or len(rows) < self.degree_threshold:
            return
        self._entries[key] = list(rows)
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()


class CassandraExecutor(BackendExecutor):
    """Adapter for the project’s existing Cassandra schema.

    Default three-table profile:
      kg_edges_by_src, kg_edges_by_dst, kg_edges_by_relation_bucket.
    `kg_edges_by_src_relation` is optional. When absent, Cassandra-opt remains a fair
    parallel + high-degree-cache comparison, but it does *not* claim index acceleration.
    """

    def __init__(self, config: dict[str, Any], mode: str):
        if mode not in {"naive", "opt"}:
            raise ValueError("mode must be 'naive' or 'opt'")
        self.mode = mode
        self.system_name = f"cassandra_{mode}"
        self.config = config
        self.columns = config.get("columns", {})
        self.tables = config.get("tables", {})
        self.base_table = _ident(self.tables["by_src"])
        self.dst_table = self.tables.get("by_dst")
        self.relation_bucket_table = self.tables.get("by_relation_bucket")
        self.src_relation_table = self.tables.get("by_src_relation")
        self.index_enabled = bool(config.get("relation_index", {}).get("enabled", False))
        if self.index_enabled and not self.src_relation_table:
            raise ValueError("relation_index.enabled=true but tables.by_src_relation is empty")
        if self.src_relation_table:
            self.src_relation_table = _ident(self.src_relation_table)
        if self.dst_table:
            self.dst_table = _ident(self.dst_table)
        if self.relation_bucket_table:
            self.relation_bucket_table = _ident(self.relation_bucket_table)
        required_columns = ("graph_id", "src_id", "relation", "dst_id", "source")
        missing = [col for col in required_columns if not self.columns.get(col)]
        if missing:
            raise ValueError(f"Cassandra columns config missing: {missing}")
        for value in self.columns.values():
            _ident(value)

        opt = config.get("opt", {})
        self.workers = 1 if mode == "naive" else max(1, int(opt.get("workers", 1)))
        self.cache: DegreeAwareOneHopCache | None = None
        if mode == "opt" and opt.get("cache_policy") == "high_degree":
            self.cache = DegreeAwareOneHopCache(int(opt.get("cache_capacity", 0)), int(opt.get("degree_threshold", 0)))

        try:
            from cassandra.auth import PlainTextAuthProvider
            from cassandra.cluster import Cluster
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install cassandra-driver to use CassandraExecutor") from exc
        auth = None
        if config.get("username"):
            auth = PlainTextAuthProvider(config["username"], config.get("password") or "")
        self.cluster = Cluster(config.get("hosts", ["127.0.0.1"]), port=int(config.get("port", 9042)), auth_provider=auth)
        self.session = self.cluster.connect(_ident(config["keyspace"]))

    def _row_to_edge(self, row: Any) -> Edge:
        c = self.columns
        return Edge(
            graph_id=str(getattr(row, c["graph_id"])),
            src_id=str(getattr(row, c["src_id"])),
            relation=str(getattr(row, c["relation"])),
            dst_id=str(getattr(row, c["dst_id"])),
            source="" if getattr(row, c["source"], None) is None else str(getattr(row, c["source"])),
        )

    def _select_columns(self) -> str:
        c = self.columns
        return ", ".join(c[k] for k in ("graph_id", "src_id", "relation", "dst_id", "source"))

    def _query_rows(self, graph_id: str, source: str, relation: str | None) -> list[Edge]:
        c = self.columns
        selected = self._select_columns()
        if self.mode == "opt" and self.index_enabled and relation is not None:
            cql = (
                f"SELECT {selected} FROM {self.src_relation_table} "
                f"WHERE {c['graph_id']}=%s AND {c['src_id']}=%s AND {c['relation']}=%s"
            )
            rows = self.session.execute(cql, (graph_id, source, relation))
        else:
            cql = (
                f"SELECT {selected} FROM {self.base_table} "
                f"WHERE {c['graph_id']}=%s AND {c['src_id']}=%s"
            )
            rows = self.session.execute(cql, (graph_id, source))
        return [self._row_to_edge(row) for row in rows]

    def _fetch_many_for_query(self, query: QuerySpec, sources: list[str], relation: str | None):
        output: dict[str, list[Edge]] = {}
        raw_edges = hits = misses = 0
        pending: list[str] = []
        for source in sources:
            key = (query.graph_id, source, relation)
            cached = self.cache.get(key) if self.cache else None
            if cached is not None:
                output[source] = cached
                hits += 1
            else:
                pending.append(source)
                misses += 1
        if self.workers == 1:
            for source in pending:
                rows = self._query_rows(query.graph_id, source, relation)
                output[source] = rows
                raw_edges += len(rows)
                if self.cache:
                    self.cache.put((query.graph_id, source, relation), rows)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self._query_rows, query.graph_id, source, relation): source for source in pending}
                for future in as_completed(futures):
                    source = futures[future]
                    rows = future.result()
                    output[source] = rows
                    raw_edges += len(rows)
                    if self.cache:
                        self.cache.put((query.graph_id, source, relation), rows)
        return output, {"raw_edges_read": raw_edges, "cache_hits": hits, "cache_misses": misses}

    def execute(self, query: QuerySpec) -> TraversalResult:
        return strict_frontier_traversal(query, lambda sources, relation: self._fetch_many_for_query(query, sources, relation))

    def _write_base(self, table: str, edge: Edge) -> None:
        c = self.columns
        cql = (
            f"INSERT INTO {table} ({c['graph_id']}, {c['src_id']}, {c['relation']}, {c['dst_id']}, edge_id, {c['source']}, created_at) "
            "VALUES (%s, %s, %s, %s, now(), %s, toTimestamp(now()))"
        )
        self.session.execute(cql, (edge.graph_id, edge.src_id, edge.relation, edge.dst_id, edge.source))

    def _write_dst(self, table: str, edge: Edge) -> None:
        c = self.columns
        cql = (
            f"INSERT INTO {table} ({c['graph_id']}, {c['dst_id']}, {c['relation']}, {c['src_id']}, edge_id, {c['source']}, created_at) "
            "VALUES (%s, %s, %s, %s, now(), %s, toTimestamp(now()))"
        )
        self.session.execute(cql, (edge.graph_id, edge.dst_id, edge.relation, edge.src_id, edge.source))

    def _write_relation_bucket(self, table: str, edge: Edge) -> None:
        c = self.columns
        policy = self.config.get("write_policy", {})
        count = int(policy.get("bucket_count", 64))
        by = str(policy.get("bucket_by", "dst"))
        if count < 1 or by not in {"src", "dst"}:
            raise ValueError("write_policy.bucket_count must be >=1 and bucket_by must be src or dst")
        value = edge.src_id if by == "src" else edge.dst_id
        bucket = _bucket(value, count)
        cql = (
            f"INSERT INTO {table} ({c['graph_id']}, {c['relation']}, bucket, {c['src_id']}, {c['dst_id']}, edge_id, {c['source']}, created_at) "
            "VALUES (%s, %s, %s, %s, %s, now(), %s, toTimestamp(now()))"
        )
        self.session.execute(cql, (edge.graph_id, edge.relation, bucket, edge.src_id, edge.dst_id, edge.source))

    def apply_write(self, query: QuerySpec) -> None:
        if query.op_type != "write" or query.write_edge is None:
            raise ValueError("apply_write requires a write event with write_edge")
        edge = query.write_edge
        # C2's fair write cost preserves all materialized tables that exist in the active
        # schema. The optional relation-index table is maintained only when explicitly enabled.
        self._write_base(self.base_table, edge)
        if self.dst_table:
            self._write_dst(self.dst_table, edge)
        if self.relation_bucket_table:
            self._write_relation_bucket(self.relation_bucket_table, edge)
        if self.index_enabled and self.src_relation_table:
            self._write_base(self.src_relation_table, edge)
        self.clear_application_cache()

    def clear_application_cache(self) -> None:
        if self.cache:
            self.cache.clear()

    def close(self) -> None:
        self.cluster.shutdown()
