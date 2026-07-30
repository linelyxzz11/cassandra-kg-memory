#!/usr/bin/env python3
"""
Cassandra Backend Adapter — Cassandra runs in Docker on localhost:9042.
Configured based on running container (image: cassandra).
"""
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.query import SimpleStatement, dict_factory

# === Auto-detected from running Docker container ===
CASSANDRA_CONTACT_POINTS = ["127.0.0.1"]
CASSANDRA_PORT = 9042
CASSANDRA_KEYSPACE = "kg_memory"
# =========================================================

profile = ExecutionProfile(row_factory=dict_factory)
cluster = Cluster(
    contact_points=CASSANDRA_CONTACT_POINTS,
    port=CASSANDRA_PORT,
    execution_profiles={EXEC_PROFILE_DEFAULT: profile},
)
session = cluster.connect(CASSANDRA_KEYSPACE)


def list_memories(scope_id: str) -> list[str]:
    rows = session.execute(
        SimpleStatement(
            "SELECT memory_id FROM kg_memory_table WHERE conversation_id = %s ALLOW FILTERING"
        ),
        (scope_id,),
    )
    return [r["memory_id"] for r in rows]


def get_raw_records(scope_id: str) -> dict[str, str]:
    rows = session.execute(
        SimpleStatement(
            "SELECT memory_id, raw_text FROM kg_memory_table "
            "WHERE conversation_id = %s ALLOW FILTERING"
        ),
        (scope_id,),
    )
    return {r["memory_id"]: r["raw_text"] for r in rows}


def get_erk_records(scope_id: str) -> dict[str, str]:
    rows = session.execute(
        SimpleStatement(
            "SELECT memory_id, raw_text, entities, relations, keywords "
            "FROM kg_memory_table WHERE conversation_id = %s ALLOW FILTERING"
        ),
        (scope_id,),
    )
    result = {}
    for r in rows:
        parts = [r.get("raw_text", "")]
        for field in ("entities", "relations", "keywords"):
            s = r.get(field, "")
            if s:
                parts.extend(s.split(";"))
        result[r["memory_id"]] = " ".join(filter(None, parts))
    return result


def get_triples(scope_id: str) -> dict[str, list[tuple[str, str, str]]]:
    from collections import defaultdict
    rows = session.execute(
        SimpleStatement(
            "SELECT src_memory_id, src_entity, relation, tgt_memory_id "
            "FROM kg_triples WHERE scope_id = %s ALLOW FILTERING"
        ),
        (scope_id,),
    )
    result = defaultdict(list)
    for r in rows:
        result[r["src_memory_id"]].append(
            (r["src_entity"], r["relation"], r["tgt_memory_id"])
        )
    return dict(result)


def get_connections(scope_id: str, src_memory_id: str) -> list[str]:
    rows = session.execute(
        SimpleStatement(
            "SELECT tgt_memory_id FROM kg_triples "
            "WHERE scope_id = %s AND src_memory_id = %s ALLOW FILTERING"
        ),
        (scope_id, src_memory_id),
    )
    return sorted(set(r["tgt_memory_id"] for r in rows))


def close():
    cluster.shutdown()


if __name__ == "__main__":
    try:
        rows = list(session.execute("SELECT now() FROM system.local"))
        print(f"Cassandra OK | system.local now() = {rows[0][0]}")
    except Exception as e:
        print(f"Cassandra ERROR: {e}")
