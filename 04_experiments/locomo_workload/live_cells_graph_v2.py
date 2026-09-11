"""Four isolated graph-aware live storage cells (v2)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from graph_event_v2 import GraphRecord


class CassandraGraphCells:
    names = ("cassandra-base", "cassandra-materialized")

    def __init__(self, host="127.0.0.1", keyspace="locomo_workload_v2_graph"):
        from cassandra.cluster import Cluster
        self.cluster = Cluster([host], protocol_version=4)
        self.s = self.cluster.connect()
        self.s.execute(f"CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication={{'class':'SimpleStrategy','replication_factor':1}}")
        self.s.set_keyspace(keyspace)
        self._schema()

    def _schema(self):
        ddl = (
            "CREATE TABLE IF NOT EXISTS g_base_memory_by_scope (scope_id text,memory_id text,version int,raw_text text,embedding_sha256 text,PRIMARY KEY ((scope_id),memory_id))",
            "CREATE TABLE IF NOT EXISTS g_base_features_by_memory (scope_id text,memory_id text,entities text,relations text,keywords text,triples text,PRIMARY KEY ((scope_id,memory_id)))",
            "CREATE TABLE IF NOT EXISTS g_base_entities_by_scope (scope_id text,entity_id text,PRIMARY KEY ((scope_id),entity_id))",
            "CREATE TABLE IF NOT EXISTS g_base_mentions_by_memory (scope_id text,memory_id text,entity_id text,PRIMARY KEY ((scope_id,memory_id),entity_id))",
            "CREATE TABLE IF NOT EXISTS g_base_edges_by_memory (scope_id text,memory_id text,ordinal int,src text,relation text,dst text,PRIMARY KEY ((scope_id,memory_id),ordinal))",
            "CREATE TABLE IF NOT EXISTS g_base_edges_by_src (scope_id text,src text,relation text,dst text,memory_id text,ordinal int,PRIMARY KEY ((scope_id,src),relation,dst,memory_id,ordinal))",
            "CREATE TABLE IF NOT EXISTS g_base_edges_by_scope (scope_id text,memory_id text,ordinal int,src text,relation text,dst text,PRIMARY KEY ((scope_id),memory_id,ordinal))",
            "CREATE TABLE IF NOT EXISTS g_mat_memory_by_scope (scope_id text,memory_id text,version int,raw_text text,entities text,relations text,keywords text,triples text,rawerk text,embedding_sha256 text,PRIMARY KEY ((scope_id),memory_id))",
            "CREATE TABLE IF NOT EXISTS g_mat_entities_by_scope (scope_id text,entity_id text,PRIMARY KEY ((scope_id),entity_id))",
            "CREATE TABLE IF NOT EXISTS g_mat_mentions_by_memory (scope_id text,memory_id text,entity_id text,PRIMARY KEY ((scope_id,memory_id),entity_id))",
            "CREATE TABLE IF NOT EXISTS g_mat_edges_by_memory (scope_id text,memory_id text,ordinal int,src text,relation text,dst text,PRIMARY KEY ((scope_id,memory_id),ordinal))",
            "CREATE TABLE IF NOT EXISTS g_mat_by_scope_relation (scope_id text,relation text,memory_id text,ordinal int,src text,dst text,PRIMARY KEY ((scope_id,relation),memory_id,ordinal,src,dst))",
        )
        for statement in ddl:
            self.s.execute(statement)
        self.p = {name: self.s.prepare(query) for name, query in {
            "bm": "INSERT INTO g_base_memory_by_scope (scope_id,memory_id,version,raw_text,embedding_sha256) VALUES (?,?,?,?,?)",
            "bf": "INSERT INTO g_base_features_by_memory (scope_id,memory_id,entities,relations,keywords,triples) VALUES (?,?,?,?,?,?)",
            "be": "INSERT INTO g_base_entities_by_scope (scope_id,entity_id) VALUES (?,?)",
            "bmention": "INSERT INTO g_base_mentions_by_memory (scope_id,memory_id,entity_id) VALUES (?,?,?)",
            "bedgem": "INSERT INTO g_base_edges_by_memory (scope_id,memory_id,ordinal,src,relation,dst) VALUES (?,?,?,?,?,?)",
            "bedges": "INSERT INTO g_base_edges_by_src (scope_id,src,relation,dst,memory_id,ordinal) VALUES (?,?,?,?,?,?)",
            "bedgescope": "INSERT INTO g_base_edges_by_scope (scope_id,memory_id,ordinal,src,relation,dst) VALUES (?,?,?,?,?,?)",
            "mm": "INSERT INTO g_mat_memory_by_scope (scope_id,memory_id,version,raw_text,entities,relations,keywords,triples,rawerk,embedding_sha256) VALUES (?,?,?,?,?,?,?,?,?,?)",
            "mmraw": "INSERT INTO g_mat_memory_by_scope (scope_id,memory_id,version,raw_text,embedding_sha256) VALUES (?,?,?,?,?)",
            "me": "INSERT INTO g_mat_entities_by_scope (scope_id,entity_id) VALUES (?,?)",
            "mmention": "INSERT INTO g_mat_mentions_by_memory (scope_id,memory_id,entity_id) VALUES (?,?,?)",
            "medgem": "INSERT INTO g_mat_edges_by_memory (scope_id,memory_id,ordinal,src,relation,dst) VALUES (?,?,?,?,?,?)",
            "mrel": "INSERT INTO g_mat_by_scope_relation (scope_id,relation,memory_id,ordinal,src,dst) VALUES (?,?,?,?,?,?)",
        }.items()}

    @property
    def tables(self):
        return (
            "g_base_memory_by_scope", "g_base_features_by_memory", "g_base_entities_by_scope",
            "g_base_mentions_by_memory", "g_base_edges_by_memory", "g_base_edges_by_src", "g_base_edges_by_scope",
            "g_mat_memory_by_scope", "g_mat_entities_by_scope", "g_mat_mentions_by_memory",
            "g_mat_edges_by_memory", "g_mat_by_scope_relation",
        )

    def reset(self):
        for table in self.tables:
            self.s.execute(f"TRUNCATE {table}")

    def insert(self, cell: str, r: GraphRecord):
        from cassandra.query import BatchStatement, BatchType
        batch = BatchStatement(batch_type=BatchType.LOGGED)
        if cell == "cassandra-base":
            batch.add(self.p["bm"], (r.scope_id, r.memory_id, r.version, r.raw_text, r.embedding_sha256))
            batch.add(self.p["bf"], (r.scope_id, r.memory_id, r.entities, r.relations, r.keywords, r.triples))
            for entity in r.graph_entities:
                batch.add(self.p["be"], (r.scope_id, entity))
            for entity in r.mentions:
                batch.add(self.p["bmention"], (r.scope_id, r.memory_id, entity))
            for edge in r.edges:
                batch.add(self.p["bedgem"], (r.scope_id, r.memory_id, edge["ordinal"], edge["src"], edge["relation"], edge["dst"]))
                batch.add(self.p["bedges"], (r.scope_id, edge["src"], edge["relation"], edge["dst"], r.memory_id, edge["ordinal"]))
                batch.add(self.p["bedgescope"], (r.scope_id, r.memory_id, edge["ordinal"], edge["src"], edge["relation"], edge["dst"]))
        else:
            batch.add(self.p["mm"], (r.scope_id, r.memory_id, r.version, r.raw_text, r.entities, r.relations, r.keywords, r.triples, r.rawerk, r.embedding_sha256))
            for entity in r.graph_entities:
                batch.add(self.p["me"], (r.scope_id, entity))
            for entity in r.mentions:
                batch.add(self.p["mmention"], (r.scope_id, r.memory_id, entity))
            for edge in r.edges:
                batch.add(self.p["medgem"], (r.scope_id, r.memory_id, edge["ordinal"], edge["src"], edge["relation"], edge["dst"]))
                batch.add(self.p["mrel"], (r.scope_id, edge["relation"], r.memory_id, edge["ordinal"], edge["src"], edge["dst"]))
        self.s.execute(batch)

    def commit_raw(self, cell: str, r: GraphRecord):
        """Commit only the raw memory row; used to isolate t_commit."""
        if cell == "cassandra-base":
            self.s.execute(self.p["bm"], (r.scope_id, r.memory_id, r.version, r.raw_text, r.embedding_sha256))
        else:
            self.s.execute(self.p["mmraw"], (r.scope_id, r.memory_id, r.version, r.raw_text, r.embedding_sha256))

    def write_structured(self, cell: str, r: GraphRecord):
        """Complete the graph-aware logical event after the raw commit."""
        from cassandra.query import BatchStatement, BatchType
        batch = BatchStatement(batch_type=BatchType.LOGGED)
        if cell == "cassandra-base":
            batch.add(self.p["bf"], (r.scope_id, r.memory_id, r.entities, r.relations, r.keywords, r.triples))
            for entity in r.graph_entities:
                batch.add(self.p["be"], (r.scope_id, entity))
            for entity in r.mentions:
                batch.add(self.p["bmention"], (r.scope_id, r.memory_id, entity))
            for edge in r.edges:
                batch.add(self.p["bedgem"], (r.scope_id, r.memory_id, edge["ordinal"], edge["src"], edge["relation"], edge["dst"]))
                batch.add(self.p["bedges"], (r.scope_id, edge["src"], edge["relation"], edge["dst"], r.memory_id, edge["ordinal"]))
                batch.add(self.p["bedgescope"], (r.scope_id, r.memory_id, edge["ordinal"], edge["src"], edge["relation"], edge["dst"]))
        else:
            batch.add(self.p["mm"], (r.scope_id, r.memory_id, r.version, r.raw_text, r.entities, r.relations, r.keywords, r.triples, r.rawerk, r.embedding_sha256))
            for entity in r.graph_entities:
                batch.add(self.p["me"], (r.scope_id, entity))
            for entity in r.mentions:
                batch.add(self.p["mmention"], (r.scope_id, r.memory_id, entity))
            for edge in r.edges:
                batch.add(self.p["medgem"], (r.scope_id, r.memory_id, edge["ordinal"], edge["src"], edge["relation"], edge["dst"]))
                batch.add(self.p["mrel"], (r.scope_id, edge["relation"], r.memory_id, edge["ordinal"], edge["src"], edge["dst"]))
        self.s.execute(batch)

    def delete(self, cell: str, r: GraphRecord):
        if cell == "cassandra-base":
            for edge in r.edges:
                self.s.execute("DELETE FROM g_base_edges_by_src WHERE scope_id=%s AND src=%s AND relation=%s AND dst=%s AND memory_id=%s AND ordinal=%s", (r.scope_id, edge["src"], edge["relation"], edge["dst"], r.memory_id, edge["ordinal"]))
                self.s.execute("DELETE FROM g_base_edges_by_scope WHERE scope_id=%s AND memory_id=%s AND ordinal=%s", (r.scope_id, r.memory_id, edge["ordinal"]))
            for table in ("g_base_mentions_by_memory", "g_base_edges_by_memory", "g_base_features_by_memory"):
                self.s.execute(f"DELETE FROM {table} WHERE scope_id=%s AND memory_id=%s", (r.scope_id, r.memory_id))
            self.s.execute("DELETE FROM g_base_memory_by_scope WHERE scope_id=%s AND memory_id=%s", (r.scope_id, r.memory_id))
        else:
            for edge in r.edges:
                self.s.execute("DELETE FROM g_mat_by_scope_relation WHERE scope_id=%s AND relation=%s AND memory_id=%s AND ordinal=%s AND src=%s AND dst=%s", (r.scope_id, edge["relation"], r.memory_id, edge["ordinal"], edge["src"], edge["dst"]))
            for table in ("g_mat_mentions_by_memory", "g_mat_edges_by_memory"):
                self.s.execute(f"DELETE FROM {table} WHERE scope_id=%s AND memory_id=%s", (r.scope_id, r.memory_id))
            self.s.execute("DELETE FROM g_mat_memory_by_scope WHERE scope_id=%s AND memory_id=%s", (r.scope_id, r.memory_id))

    def fetch_graph(self, cell: str, scope: str, memory_id: str):
        if cell == "cassandra-base":
            m = self.s.execute("SELECT * FROM g_base_memory_by_scope WHERE scope_id=%s AND memory_id=%s", (scope, memory_id)).one()
            f = self.s.execute("SELECT * FROM g_base_features_by_memory WHERE scope_id=%s AND memory_id=%s", (scope, memory_id)).one()
            if not (m and f):
                return None
            record = GraphRecord(scope, memory_id, int(m.version), m.raw_text, f.entities or "", f.relations or "", f.keywords or "", f.triples or "", m.embedding_sha256)
            mentions = sorted(x.entity_id for x in self.s.execute("SELECT entity_id FROM g_base_mentions_by_memory WHERE scope_id=%s AND memory_id=%s", (scope, memory_id)))
            edge_rows = self.s.execute("SELECT ordinal,src,relation,dst FROM g_base_edges_by_memory WHERE scope_id=%s AND memory_id=%s", (scope, memory_id))
        else:
            m = self.s.execute("SELECT * FROM g_mat_memory_by_scope WHERE scope_id=%s AND memory_id=%s", (scope, memory_id)).one()
            if not m:
                return None
            record = GraphRecord(scope, memory_id, int(m.version), m.raw_text, m.entities or "", m.relations or "", m.keywords or "", m.triples or "", m.embedding_sha256)
            mentions = sorted(x.entity_id for x in self.s.execute("SELECT entity_id FROM g_mat_mentions_by_memory WHERE scope_id=%s AND memory_id=%s", (scope, memory_id)))
            edge_rows = self.s.execute("SELECT ordinal,src,relation,dst FROM g_mat_edges_by_memory WHERE scope_id=%s AND memory_id=%s", (scope, memory_id))
        edges = sorted(({"ordinal": int(x.ordinal), "src": x.src, "relation": x.relation, "dst": x.dst} for x in edge_rows), key=lambda x: x["ordinal"])
        entities = sorted(set(mentions) | {v for edge in edges for v in (edge["src"], edge["dst"])})
        return {"memory": record.memory_projection(), "mentions": mentions, "graph_entities": entities, "edges": edges}

    def ids(self, cell: str, scope: str):
        table = "g_base_memory_by_scope" if cell == "cassandra-base" else "g_mat_memory_by_scope"
        return sorted(x.memory_id for x in self.s.execute(f"SELECT memory_id FROM {table} WHERE scope_id=%s", (scope,)))

    def ids_by_relation(self, cell: str, scope: str, relation: str):
        if cell == "cassandra-materialized":
            return sorted({x.memory_id for x in self.s.execute("SELECT memory_id FROM g_mat_by_scope_relation WHERE scope_id=%s AND relation=%s", (scope, relation))})
        rows = self.s.execute("SELECT memory_id,relation FROM g_base_edges_by_scope WHERE scope_id=%s", (scope,))
        return sorted({x.memory_id for x in rows if x.relation == relation})

    def candidate_projection(self, cell: str, scope: str, memory_id: str):
        """Return the retrieval projection used by the online sparse index."""
        if cell == "cassandra-materialized":
            row = self.s.execute(
                "SELECT rawerk FROM g_mat_memory_by_scope WHERE scope_id=%s AND memory_id=%s",
                (scope, memory_id),
            ).one()
            return None if row is None else row.rawerk
        memory = self.s.execute(
            "SELECT raw_text FROM g_base_memory_by_scope WHERE scope_id=%s AND memory_id=%s",
            (scope, memory_id),
        ).one()
        features = self.s.execute(
            "SELECT entities,relations,keywords FROM g_base_features_by_memory WHERE scope_id=%s AND memory_id=%s",
            (scope, memory_id),
        ).one()
        if not (memory and features):
            return None
        from online_retrieval import render_rawerk
        return render_rawerk(memory.raw_text, features.entities or "", features.relations or "", features.keywords or "")

    def table_counts(self):
        return {table: int(self.s.execute(f"SELECT COUNT(*) AS n FROM {table}").one().n) for table in self.tables}

    def close(self):
        self.s.shutdown()
        self.cluster.shutdown()


class Neo4jGraphCells:
    names = ("neo4j-native", "neo4j-materialized")

    def __init__(self, uri, user, password, database="neo4j"):
        from neo4j import GraphDatabase
        self.d = GraphDatabase.driver(uri, auth=(user, password), max_connection_pool_size=128)
        self.d.verify_connectivity()
        self.database = database
        self._schema()

    def run(self, query, **params):
        with self.d.session(database=self.database) as session:
            return list(session.run(query, **params))

    def _schema(self):
        statements = (
            "CREATE CONSTRAINT lwv2_native_memory IF NOT EXISTS FOR (m:LWV2NativeMemory) REQUIRE (m.scope_id,m.memory_id) IS UNIQUE",
            "CREATE CONSTRAINT lwv2_native_feature IF NOT EXISTS FOR (f:LWV2NativeFeature) REQUIRE (f.scope_id,f.memory_id) IS UNIQUE",
            "CREATE CONSTRAINT lwv2_native_entity IF NOT EXISTS FOR (e:LWV2NativeEntity) REQUIRE (e.scope_id,e.entity_id) IS UNIQUE",
            "CREATE CONSTRAINT lwv2_mat_memory IF NOT EXISTS FOR (m:LWV2MatMemory) REQUIRE (m.scope_id,m.memory_id) IS UNIQUE",
            "CREATE CONSTRAINT lwv2_mat_entity IF NOT EXISTS FOR (e:LWV2MatEntity) REQUIRE (e.scope_id,e.entity_id) IS UNIQUE",
            "CREATE CONSTRAINT lwv2_mat_candidate IF NOT EXISTS FOR (c:LWV2MatCandidate) REQUIRE (c.scope_id,c.relation,c.memory_id) IS UNIQUE",
            "CREATE INDEX lwv2_native_scope IF NOT EXISTS FOR (m:LWV2NativeMemory) ON (m.scope_id)",
            "CREATE INDEX lwv2_mat_scope IF NOT EXISTS FOR (m:LWV2MatMemory) ON (m.scope_id)",
        )
        for statement in statements:
            self.run(statement)
        self.run("CALL db.awaitIndexes(300)")

    def reset(self):
        self.run("MATCH (n) WHERE n:LWV2NativeMemory OR n:LWV2NativeFeature OR n:LWV2NativeEntity OR n:LWV2MatMemory OR n:LWV2MatEntity OR n:LWV2MatCandidate DETACH DELETE n")

    @staticmethod
    def payload(r: GraphRecord):
        return {**r.memory_projection(), "mentions": list(r.mentions), "graph_entities": list(r.graph_entities), "edges": list(r.edges)}

    def insert(self, cell: str, r: GraphRecord):
        p = self.payload(r)
        if cell == "neo4j-native":
            query = """
            MERGE (m:LWV2NativeMemory {scope_id:$p.scope_id,memory_id:$p.memory_id})
            SET m.version=$p.version,m.raw_text=$p.raw_text,m.embedding_sha256=$p.embedding_sha256
            MERGE (f:LWV2NativeFeature {scope_id:$p.scope_id,memory_id:$p.memory_id})
            SET f.entities=$p.entities,f.relations=$p.relations,f.keywords=$p.keywords,f.triples=$p.triples
            MERGE (m)-[:LWV2_HAS_FEATURE]->(f)
            WITH m,$p AS p
            CALL (p) { UNWIND p.graph_entities AS entity MERGE (:LWV2NativeEntity {scope_id:p.scope_id,entity_id:entity}) RETURN count(*) AS n1 }
            CALL (m,p) { UNWIND p.mentions AS entity MATCH (e:LWV2NativeEntity {scope_id:p.scope_id,entity_id:entity}) MERGE (m)-[:LWV2_NATIVE_MENTIONS]->(e) RETURN count(*) AS n2 }
            CALL (p) { UNWIND p.edges AS edge MATCH (src:LWV2NativeEntity {scope_id:p.scope_id,entity_id:edge.src}) MATCH (dst:LWV2NativeEntity {scope_id:p.scope_id,entity_id:edge.dst}) MERGE (src)-[rel:LWV2_NATIVE_REL {scope_id:p.scope_id,memory_id:p.memory_id,ordinal:edge.ordinal}]->(dst) SET rel.relation=edge.relation RETURN count(*) AS n3 }
            RETURN n1,n2,n3
            """
        else:
            query = """
            MERGE (m:LWV2MatMemory {scope_id:$p.scope_id,memory_id:$p.memory_id})
            SET m.version=$p.version,m.raw_text=$p.raw_text,m.entities=$p.entities,m.relations=$p.relations,m.keywords=$p.keywords,m.triples=$p.triples,m.rawerk=$p.rawerk,m.embedding_sha256=$p.embedding_sha256
            WITH m,$p AS p
            CALL (p) { UNWIND p.graph_entities AS entity MERGE (:LWV2MatEntity {scope_id:p.scope_id,entity_id:entity}) RETURN count(*) AS n1 }
            CALL (m,p) { UNWIND p.mentions AS entity MATCH (e:LWV2MatEntity {scope_id:p.scope_id,entity_id:entity}) MERGE (m)-[:LWV2_MAT_MENTIONS]->(e) RETURN count(*) AS n2 }
            CALL (p) { UNWIND p.edges AS edge MATCH (src:LWV2MatEntity {scope_id:p.scope_id,entity_id:edge.src}) MATCH (dst:LWV2MatEntity {scope_id:p.scope_id,entity_id:edge.dst}) MERGE (src)-[rel:LWV2_MAT_REL {scope_id:p.scope_id,memory_id:p.memory_id,ordinal:edge.ordinal}]->(dst) SET rel.relation=edge.relation MERGE (:LWV2MatCandidate {scope_id:p.scope_id,relation:edge.relation,memory_id:p.memory_id}) RETURN count(*) AS n3 }
            RETURN n1,n2,n3
            """
        self.run(query, p=p)

    def commit_raw(self, cell: str, r: GraphRecord):
        """Commit only the raw memory node; used to isolate t_commit."""
        label = "LWV2NativeMemory" if cell == "neo4j-native" else "LWV2MatMemory"
        self.run(
            f"MERGE (m:{label} {{scope_id:$s,memory_id:$id}}) "
            "SET m.version=$version,m.raw_text=$raw,m.embedding_sha256=$embedding",
            s=r.scope_id, id=r.memory_id, version=r.version, raw=r.raw_text,
            embedding=r.embedding_sha256,
        )

    def write_structured(self, cell: str, r: GraphRecord):
        """Complete the graph-aware logical event after the raw commit."""
        p = self.payload(r)
        if cell == "neo4j-native":
            query = """
            MATCH (m:LWV2NativeMemory {scope_id:$p.scope_id,memory_id:$p.memory_id})
            MERGE (f:LWV2NativeFeature {scope_id:$p.scope_id,memory_id:$p.memory_id})
            SET f.entities=$p.entities,f.relations=$p.relations,f.keywords=$p.keywords,f.triples=$p.triples
            MERGE (m)-[:LWV2_HAS_FEATURE]->(f)
            WITH m,$p AS p
            CALL (p) { UNWIND p.graph_entities AS entity MERGE (:LWV2NativeEntity {scope_id:p.scope_id,entity_id:entity}) RETURN count(*) AS n1 }
            CALL (m,p) { UNWIND p.mentions AS entity MATCH (e:LWV2NativeEntity {scope_id:p.scope_id,entity_id:entity}) MERGE (m)-[:LWV2_NATIVE_MENTIONS]->(e) RETURN count(*) AS n2 }
            CALL (p) { UNWIND p.edges AS edge MATCH (src:LWV2NativeEntity {scope_id:p.scope_id,entity_id:edge.src}) MATCH (dst:LWV2NativeEntity {scope_id:p.scope_id,entity_id:edge.dst}) MERGE (src)-[rel:LWV2_NATIVE_REL {scope_id:p.scope_id,memory_id:p.memory_id,ordinal:edge.ordinal}]->(dst) SET rel.relation=edge.relation RETURN count(*) AS n3 }
            RETURN n1,n2,n3
            """
        else:
            query = """
            MATCH (m:LWV2MatMemory {scope_id:$p.scope_id,memory_id:$p.memory_id})
            SET m.entities=$p.entities,m.relations=$p.relations,m.keywords=$p.keywords,m.triples=$p.triples,m.rawerk=$p.rawerk
            WITH m,$p AS p
            CALL (p) { UNWIND p.graph_entities AS entity MERGE (:LWV2MatEntity {scope_id:p.scope_id,entity_id:entity}) RETURN count(*) AS n1 }
            CALL (m,p) { UNWIND p.mentions AS entity MATCH (e:LWV2MatEntity {scope_id:p.scope_id,entity_id:entity}) MERGE (m)-[:LWV2_MAT_MENTIONS]->(e) RETURN count(*) AS n2 }
            CALL (p) { UNWIND p.edges AS edge MATCH (src:LWV2MatEntity {scope_id:p.scope_id,entity_id:edge.src}) MATCH (dst:LWV2MatEntity {scope_id:p.scope_id,entity_id:edge.dst}) MERGE (src)-[rel:LWV2_MAT_REL {scope_id:p.scope_id,memory_id:p.memory_id,ordinal:edge.ordinal}]->(dst) SET rel.relation=edge.relation MERGE (:LWV2MatCandidate {scope_id:p.scope_id,relation:edge.relation,memory_id:p.memory_id}) RETURN count(*) AS n3 }
            RETURN n1,n2,n3
            """
        self.run(query, p=p)

    def insert_many(self, cell: str, records, batch=250, workers=1):
        chunks = [records[i:i + batch] for i in range(0, len(records), batch)]
        def load(chunk):
            for record in chunk:
                self.insert(cell, record)
        if workers <= 1:
            for chunk in chunks:
                load(chunk)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(load, chunks))

    def delete(self, cell: str, r: GraphRecord):
        if cell == "neo4j-native":
            self.run("MATCH ()-[rel:LWV2_NATIVE_REL {scope_id:$s,memory_id:$m}]->() DELETE rel", s=r.scope_id, m=r.memory_id)
            self.run("MATCH (m:LWV2NativeMemory {scope_id:$s,memory_id:$id}) OPTIONAL MATCH (m)-[:LWV2_HAS_FEATURE]->(f:LWV2NativeFeature) DETACH DELETE m,f", s=r.scope_id, id=r.memory_id)
        else:
            self.run("MATCH ()-[rel:LWV2_MAT_REL {scope_id:$s,memory_id:$m}]->() DELETE rel", s=r.scope_id, m=r.memory_id)
            self.run("MATCH (c:LWV2MatCandidate {scope_id:$s,memory_id:$m}) DELETE c", s=r.scope_id, m=r.memory_id)
            self.run("MATCH (m:LWV2MatMemory {scope_id:$s,memory_id:$id}) DETACH DELETE m", s=r.scope_id, id=r.memory_id)

    def fetch_graph(self, cell: str, scope: str, memory_id: str):
        if cell == "neo4j-native":
            rows = self.run("MATCH (m:LWV2NativeMemory {scope_id:$s,memory_id:$id})-[:LWV2_HAS_FEATURE]->(f:LWV2NativeFeature) RETURN m.version AS version,m.raw_text AS raw_text,m.embedding_sha256 AS embedding_sha256,f.entities AS entities,f.relations AS relations,f.keywords AS keywords,f.triples AS triples", s=scope, id=memory_id)
            if not rows:
                return None
            x = dict(rows[0])
            record = GraphRecord(scope, memory_id, int(x["version"]), x["raw_text"], x["entities"] or "", x["relations"] or "", x["keywords"] or "", x["triples"] or "", x["embedding_sha256"])
            mentions = sorted(x["entity"] for x in self.run("MATCH (:LWV2NativeMemory {scope_id:$s,memory_id:$id})-[:LWV2_NATIVE_MENTIONS]->(e) RETURN e.entity_id AS entity", s=scope, id=memory_id))
            edge_rows = self.run("MATCH ()-[rel:LWV2_NATIVE_REL {scope_id:$s,memory_id:$id}]->() RETURN rel.ordinal AS ordinal,startNode(rel).entity_id AS src,rel.relation AS relation,endNode(rel).entity_id AS dst", s=scope, id=memory_id)
        else:
            rows = self.run("MATCH (m:LWV2MatMemory {scope_id:$s,memory_id:$id}) RETURN properties(m) AS p", s=scope, id=memory_id)
            if not rows:
                return None
            p = dict(rows[0]["p"])
            record = GraphRecord(scope, memory_id, int(p["version"]), p["raw_text"], p.get("entities", ""), p.get("relations", ""), p.get("keywords", ""), p.get("triples", ""), p["embedding_sha256"])
            mentions = sorted(x["entity"] for x in self.run("MATCH (:LWV2MatMemory {scope_id:$s,memory_id:$id})-[:LWV2_MAT_MENTIONS]->(e) RETURN e.entity_id AS entity", s=scope, id=memory_id))
            edge_rows = self.run("MATCH ()-[rel:LWV2_MAT_REL {scope_id:$s,memory_id:$id}]->() RETURN rel.ordinal AS ordinal,startNode(rel).entity_id AS src,rel.relation AS relation,endNode(rel).entity_id AS dst", s=scope, id=memory_id)
        edges = sorted(({"ordinal": int(x["ordinal"]), "src": x["src"], "relation": x["relation"], "dst": x["dst"]} for x in edge_rows), key=lambda x: x["ordinal"])
        entities = sorted(set(mentions) | {v for edge in edges for v in (edge["src"], edge["dst"])})
        return {"memory": record.memory_projection(), "mentions": mentions, "graph_entities": entities, "edges": edges}

    def ids(self, cell: str, scope: str):
        label = "LWV2NativeMemory" if cell == "neo4j-native" else "LWV2MatMemory"
        return sorted(x["id"] for x in self.run(f"MATCH (m:{label} {{scope_id:$s}}) RETURN m.memory_id AS id", s=scope))

    def ids_by_relation(self, cell: str, scope: str, relation: str):
        if cell == "neo4j-materialized":
            return sorted(x["id"] for x in self.run("MATCH (c:LWV2MatCandidate {scope_id:$s,relation:$r}) RETURN c.memory_id AS id", s=scope, r=relation))
        return sorted({x["id"] for x in self.run("MATCH ()-[rel:LWV2_NATIVE_REL {scope_id:$s,relation:$r}]->() RETURN rel.memory_id AS id", s=scope, r=relation)})

    def candidate_projection(self, cell: str, scope: str, memory_id: str):
        """Return the retrieval projection used by the online sparse index."""
        if cell == "neo4j-materialized":
            rows = self.run(
                "MATCH (m:LWV2MatMemory {scope_id:$s,memory_id:$id}) RETURN m.rawerk AS rawerk",
                s=scope, id=memory_id,
            )
            return None if not rows else rows[0]["rawerk"]
        rows = self.run(
            "MATCH (m:LWV2NativeMemory {scope_id:$s,memory_id:$id})-[:LWV2_HAS_FEATURE]->(f:LWV2NativeFeature) "
            "RETURN m.raw_text AS raw_text,f.entities AS entities,f.relations AS relations,f.keywords AS keywords",
            s=scope, id=memory_id,
        )
        if not rows:
            return None
        from online_retrieval import render_rawerk
        row = rows[0]
        return render_rawerk(row["raw_text"], row["entities"] or "", row["relations"] or "", row["keywords"] or "")

    def graph_counts(self):
        labels = ("LWV2NativeMemory", "LWV2NativeFeature", "LWV2NativeEntity", "LWV2MatMemory", "LWV2MatEntity", "LWV2MatCandidate")
        rels = ("LWV2_HAS_FEATURE", "LWV2_NATIVE_MENTIONS", "LWV2_MAT_MENTIONS", "LWV2_NATIVE_REL", "LWV2_MAT_REL")
        out = {label: int(self.run(f"MATCH (n:{label}) RETURN count(n) AS n")[0]["n"]) for label in labels}
        out.update({rel: int(self.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n")[0]["n"]) for rel in rels})
        return out

    def close(self):
        self.d.close()
