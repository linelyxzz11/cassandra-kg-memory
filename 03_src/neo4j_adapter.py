#!/usr/bin/env python3
"""
Neo4j Backend Adapter — uses LocoMoMemory / LocoMoEntity labels.
Existing benchmark data (C0KGNode, C1KGNode, etc.) untouched.
"""
from neo4j import GraphDatabase
from collections import defaultdict

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "REDACTED_NEO4J_PASSWORD"
NEO4J_DATABASE = "neo4j"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def list_memories(scope_id: str) -> list[str]:
    query = "MATCH (m:LocoMoMemory {scope_id: $sid}) RETURN m.memory_id AS memory_id ORDER BY m.memory_id"
    with driver.session(database=NEO4J_DATABASE) as s:
        return [r["memory_id"] for r in s.run(query, sid=scope_id)]


def get_raw_records(scope_id: str) -> dict[str, str]:
    query = "MATCH (m:LocoMoMemory {scope_id: $sid}) RETURN m.memory_id AS memory_id, m.raw_text AS raw_text"
    with driver.session(database=NEO4J_DATABASE) as s:
        return {r["memory_id"]: r["raw_text"] for r in s.run(query, sid=scope_id)
                if r["raw_text"]}


def get_erk_records(scope_id: str) -> dict[str, str]:
    query = """
    MATCH (m:LocoMoMemory {scope_id: $sid})
    RETURN m.memory_id AS memory_id, m.raw_text AS raw_text,
           m.entities AS entities, m.relations AS relations, m.keywords AS keywords
    """
    with driver.session(database=NEO4J_DATABASE) as s:
        result = {}
        for r in s.run(query, sid=scope_id):
            parts = [r.get("raw_text", "") or ""]
            for field in ("entities", "relations", "keywords"):
                s_val = r.get(field) or ""
                parts.extend(s_val.split(";"))
            result[r["memory_id"]] = " ".join(filter(None, parts))
        return result


def get_triples(scope_id: str) -> dict[str, list[tuple[str, str, str]]]:
    """Get entity triples for KG expansion — (src_entity, relation, tgt_entity)"""
    query = """
    MATCH (a:LocoMoEntity)-[r:LOCORELATES]->(b:LocoMoEntity)
    RETURN a.name AS src_entity, r.type AS relation, b.name AS tgt_entity
    """
    result = defaultdict(list)
    with driver.session(database=NEO4J_DATABASE) as s:
        for rec in s.run(query):
            result[rec["src_entity"]].append(
                (rec["src_entity"], rec["relation"], rec["tgt_entity"])
            )
    return dict(result)


def get_edges_by_src(scope_id: str, src_entity: str) -> list[tuple[str, str, str]]:
    query = """
    MATCH (a:LocoMoEntity {name: $src})-[r:LOCORELATES]->(b:LocoMoEntity)
    RETURN a.name AS src, r.type AS rel, b.name AS tgt
    """
    with driver.session(database=NEO4J_DATABASE) as s:
        return [(rec["src"], rec["rel"], rec["tgt"]) for rec in s.run(query, src=src_entity)]


def get_edges_by_src_relation(scope_id: str, src_entity: str, relation: str) -> list[tuple[str, str, str]]:
    query = """
    MATCH (a:LocoMoEntity {name: $src})-[r:LOCORELATES {type: $rel}]->(b:LocoMoEntity)
    RETURN a.name AS src, r.type AS rel, b.name AS tgt
    """
    with driver.session(database=NEO4J_DATABASE) as s:
        return [(rec["src"], rec["rel"], rec["tgt"]) for rec in
                s.run(query, src=src_entity, rel=relation)]


def get_memories_by_entity(scope_id: str, entity_name: str) -> list[str]:
    """Get all memory_ids connected to a given entity."""
    query = """
    MATCH (m:LocoMoMemory {scope_id: $sid})-[:HAS_ENTITY]->(e:LocoMoEntity {name: $en})
    RETURN m.memory_id AS memory_id ORDER BY m.memory_id
    """
    with driver.session(database=NEO4J_DATABASE) as s:
        return [r["memory_id"] for r in s.run(query, sid=scope_id, en=entity_name)]


def close():
    driver.close()
