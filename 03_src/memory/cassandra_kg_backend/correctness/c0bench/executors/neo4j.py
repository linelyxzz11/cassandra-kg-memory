from __future__ import annotations

from typing import Any

from ..models import QuerySpec, TraversalResult, logical_edge_id
from .base import BackendExecutor


class Neo4jExecutor(BackendExecutor):
    """Native Cypher executor with the same per-frontier-node fanout semantics as Cassandra."""

    system_name = "neo4j"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install neo4j Python driver to use Neo4jExecutor") from exc
        username = config.get("username", "neo4j")
        password = config.get("password")
        auth = (username, password) if password is not None else None
        self.driver = GraphDatabase.driver(config["uri"], auth=auth)
        self.database = config.get("database") or None

    @staticmethod
    def _quote(identifier: str) -> str:
        return "`" + identifier.replace("`", "``") + "`"

    def _cypher(self, query: QuerySpec) -> tuple[str, dict[str, Any]]:
        c = self.config
        q = self._quote
        label, rtype = q(c["node_label"]), q(c["relationship_type"])
        ng, nid = q(c["node_graph_property"]), q(c["node_id_property"])
        rg, rrel = q(c["rel_graph_property"]), q(c["rel_relation_property"])
        rsource = q(c["rel_source_property"])

        clauses = [f"MATCH (n0:{label} {{{ng}: $graph_id, {nid}: $seed_id}})"]
        carry = ["n0"]
        for hop, relation in enumerate(query.relation_path):
            src, dst, rel = f"n{hop}", f"n{hop + 1}", f"r{hop}"
            where = []
            if relation != "*":
                where.append(f"{rel}.{rrel} = $rel_{hop}")
            if query.cycle_policy == "path":
                seen = ", ".join(f"n{i}.{nid}" for i in range(hop + 1))
                where.append(f"NOT {dst}.{nid} IN [{seen}]")
            elif query.cycle_policy != "none":
                raise ValueError(f"Unsupported cycle_policy={query.cycle_policy}")
            where_clause = " WHERE " + " AND ".join(where) if where else ""
            clauses.append(
                "CALL {\n"
                f"  WITH {', '.join(carry)}\n"
                f"  MATCH ({src}:{label})-[{rel}:{rtype} {{{rg}: $graph_id}}]->({dst}:{label})"
                f"{where_clause}\n"
                f"  WITH {rel}, {dst}\n"
                f"  ORDER BY {rel}.{rrel} ASC, {dst}.{nid} ASC, coalesce({rel}.{rsource}, '') ASC\n"
                "  LIMIT $fanout\n"
                f"  RETURN {rel}, {dst}\n"
                "}"
            )
            carry.extend([rel, dst])
        tokens = ", ".join(
            f"[{f'n{i}'}.{nid}, {f'r{i}'}.{rrel}, {f'n{i + 1}'}.{nid}, coalesce({f'r{i}'}.{rsource}, '')]"
            for i in range(query.hop)
        )
        clauses.append(f"RETURN [{tokens}] AS edge_tokens")
        params: dict[str, Any] = {"graph_id": query.graph_id, "seed_id": query.seed_id, "fanout": query.fanout}
        for hop, relation in enumerate(query.relation_path):
            if relation != "*":
                params[f"rel_{hop}"] = relation
        return "\n".join(clauses), params

    def execute(self, query: QuerySpec) -> TraversalResult:
        if query.op_type != "read":
            raise ValueError("Neo4jExecutor.execute only accepts read events")
        cypher, params = self._cypher(query)
        with self.driver.session(database=self.database) as session:
            rows = list(session.run(cypher, params))
        paths = []
        for row in rows:
            ids = tuple(logical_edge_id(query.graph_id, *[str(v) for v in token]) for token in row["edge_tokens"])
            paths.append(ids)
        return TraversalResult(tuple(sorted(set(paths))), metadata={"executor": "neo4j_native_cypher", "cypher": cypher})

    def apply_write(self, query: QuerySpec) -> None:
        if query.op_type != "write" or query.write_edge is None:
            raise ValueError("apply_write requires a write event with write_edge")
        edge = query.write_edge
        c, q = self.config, self._quote
        label, rtype = q(c["node_label"]), q(c["relationship_type"])
        ng, nid = q(c["node_graph_property"]), q(c["node_id_property"])
        rg, rrel, rsource = q(c["rel_graph_property"]), q(c["rel_relation_property"]), q(c["rel_source_property"])
        cypher = f"""
        MERGE (s:{label} {{{ng}: $graph_id, {nid}: $src_id}})
        MERGE (d:{label} {{{ng}: $graph_id, {nid}: $dst_id}})
        MERGE (s)-[r:{rtype} {{{rg}: $graph_id, logical_edge_id: $logical_edge_id}}]->(d)
        SET r.{rrel} = $relation, r.{rsource} = $source
        """
        params = edge.to_dict()
        with self.driver.session(database=self.database) as session:
            session.run(cypher, params).consume()

    def close(self) -> None:
        self.driver.close()
