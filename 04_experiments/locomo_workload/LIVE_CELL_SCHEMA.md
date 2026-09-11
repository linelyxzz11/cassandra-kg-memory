# Live-cell schema contract

All names use the `lw_v1` prefix and are isolated from existing P5/P7 artifacts.

## Canonical projection

Every adapter must return exactly:

`scope_id, memory_id, version, raw_text, entities, relations, keywords, triples, rawerk, embedding_sha256`

The projection is serialized as canonical sorted-key UTF-8 JSON and SHA-256 hashed. A canonical run passes only when all 5,882 memory IDs have the same digest in all four cells.

## Cassandra-base

- `lw_v1_base_memory_by_scope`: `PRIMARY KEY ((scope_id), memory_id)`; raw memory/version/embedding.
- `lw_v1_base_features_by_memory`: `PRIMARY KEY ((scope_id, memory_id))`; E/R/K/triples stored separately.
- `lw_v1_base_edges_by_src`: `PRIMARY KEY ((scope_id, src_id), relation, dst_id, memory_id)`.
- Read path: partition-key memory read + feature read + bounded edge read when requested; RawERK assembled after normalized reads.

## Cassandra-materialized

- `lw_v1_mat_memory_by_scope`: `PRIMARY KEY ((scope_id), memory_id)`; complete canonical projection and RawERK.
- `lw_v1_mat_by_scope_relation`: `PRIMARY KEY ((scope_id, relation), memory_id, src_id, dst_id)`.
- Read path: one partition-local materialized projection read.

## Neo4j-native

- `(:LWV1NativeMemory {scope_id,memory_id,...})` with composite uniqueness constraint.
- `(:LWV1Feature {scope_id,memory_id,entities,relations,keywords,triples})` linked by `[:HAS_FEATURE]`.
- `(:LWV1Entity {scope_id,entity_id})` with composite uniqueness constraint.
- `(:LWV1NativeMemory)-[:MENTIONS]->(:LWV1Entity)` and scope-local `[:LW_REL {relation,memory_id,ordinal}]->` relationships.
- Read path: indexed memory lookup plus bounded `HAS_FEATURE`/scope-local graph expansion; RawERK assembled from returned components.

## Neo4j-materialized

- `(:LWV1MatMemory {scope_id,memory_id,version,rawerk,...})` with composite uniqueness constraint.
- Optional relationship-index projection retained only for the same declared query workload.
- Read path: indexed materialized property read.

## Fairness requirements

- Same precomputed raw text, E/R/K/triples and BGE vector enter all cells.
- No LLM extraction or embedding computation is timed.
- Cassandra consistency level, Neo4j transaction mode, index/constraint DDL, pool size, warm-up and Neo4j `PROFILE` plans are recorded in each run manifest.
- Timed success requires the full canonical projection version, not merely an edge or node becoming visible.
