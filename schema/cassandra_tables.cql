CREATE KEYSPACE IF NOT EXISTS ai_memory
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 1
};

USE ai_memory;

CREATE TABLE IF NOT EXISTS kg_edges_by_src (
    graph_id text,
    src_id text,
    relation text,
    dst_id text,
    edge_id timeuuid,
    src_type text,
    dst_type text,
    confidence double,
    source text,
    created_at timestamp,
    PRIMARY KEY ((graph_id, src_id), relation, dst_id, edge_id)
) WITH CLUSTERING ORDER BY (relation ASC, dst_id ASC, edge_id ASC);

CREATE TABLE IF NOT EXISTS kg_edges_by_dst (
    graph_id text,
    dst_id text,
    relation text,
    src_id text,
    edge_id timeuuid,
    src_type text,
    dst_type text,
    confidence double,
    source text,
    created_at timestamp,
    PRIMARY KEY ((graph_id, dst_id), relation, src_id, edge_id)
) WITH CLUSTERING ORDER BY (relation ASC, src_id ASC, edge_id ASC);

CREATE TABLE IF NOT EXISTS kg_edges_by_relation_bucket (
    graph_id text,
    relation text,
    bucket int,
    src_id text,
    dst_id text,
    edge_id timeuuid,
    src_type text,
    dst_type text,
    confidence double,
    source text,
    created_at timestamp,
    PRIMARY KEY ((graph_id, relation, bucket), src_id, dst_id, edge_id)
) WITH CLUSTERING ORDER BY (src_id ASC, dst_id ASC, edge_id ASC);