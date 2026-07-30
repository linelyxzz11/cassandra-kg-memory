// Run once before the C0 Neo4j import. This does not modify Cassandra.
CREATE CONSTRAINT kg_node_identity IF NOT EXISTS
FOR (n:KGNode) REQUIRE (n.graph_id, n.node_id) IS UNIQUE;

// Optional but useful for graph-scoped relation filtering.
CREATE INDEX kg_edge_graph_relation IF NOT EXISTS
FOR ()-[r:KG_EDGE]-() ON (r.graph_id, r.relation);
