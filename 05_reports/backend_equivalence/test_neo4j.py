#!/usr/bin/env python3
"""
Test Neo4j connection and verify imported data.
"""
import os

from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "")))

with driver.session() as s:
    r = s.run("RETURN 1 AS ok").single()
    print(f"Neo4j connection OK | result = {r['ok']}")

    mem_cnt = s.run("MATCH (m:Memory) RETURN count(m) AS c").single()["c"]
    rel_cnt = s.run("MATCH ()-[r:RELATES]->() RETURN count(r) AS c").single()["c"]
    ent_cnt = s.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
    
    print(f"Memory nodes: {mem_cnt}")
    print(f"Entity nodes: {ent_cnt}")
    print(f"Relationships (RELATES): {rel_cnt}")
    
    if mem_cnt > 0:
        sample = s.run("MATCH (m:Memory) RETURN m.memory_id, m.raw_text LIMIT 2")
        for r in sample:
            print(f"  Sample: memory_id={r['m.memory_id']}, raw_text[:60]={str(r['m.raw_text'])[:60]}")
    
    print("Neo4j data OK" if mem_cnt > 0 else "Neo4j: WARNING - no data found")

driver.close()
