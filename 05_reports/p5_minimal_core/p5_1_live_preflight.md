# P5-1 Live Preflight Audit

Time: 2026-07-20 12:24:11

## 2. Cassandra

- Keyspace: EXISTS
- Table memory_by_id: EXISTS
- Table kg_edges_by_src: EXISTS
- Table raw_erk_view: EXISTS
- CQL write+read: PASS
- Cassandra edges: 3875

## 3. Neo4j

- Bolt: PASS
- Constraints: 6
- Indexes: 8
- Cypher write+read: PASS
- Neo4j edges: 0

## 4. 1M Graph Data

- Path: D:\memorytable\cassandra-kg-memory\results\c3_source_scale_1M.csv
- SHA-256: e28b1e82766819469936646a408102555d0a24f950b6be659889f5948521e5ea
- Rows: 1000000
- Scale guard (>=900K): PASS (C: 3875, N: 0)

## 5. Events

- Path: D:\memorytable\cassandra-kg-memory\reports\p5_minimal_core\artifacts\frozen_update_events.jsonl
- SHA-256: 829974fea87dd97147cd661f59d667f95957e0f5a18dcd613346f455fe720a20
- Count: 55000 (50K warmup, 5K formal)
- Duplicates: 0, Version violations: 0
- Operations: {'insert_memory': 55000}

## 6. Progress

- No data yet (initializing)

## 7. Verdict

- [PASS] Docker containers
- [PASS] Cassandra real CQL
- [PASS] Neo4j real Bolt
- [PASS] 1M scale guard
- [PASS] Event artifact valid

**Verdict: ALL PASS - Continue P5-1**