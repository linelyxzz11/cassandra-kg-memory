"""Four isolated live storage cells for the canonical workload gate."""
from __future__ import annotations
import hashlib, json, os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

from online_retrieval import render_rawerk

def digest_projection(p: dict) -> str:
    return hashlib.sha256(json.dumps(p,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

@dataclass(frozen=True)
class Record:
    scope_id:str; memory_id:str; version:int; raw_text:str; entities:str; relations:str; keywords:str; triples:str; embedding_sha256:str
    @property
    def rawerk(self): return render_rawerk(self.raw_text,self.entities,self.relations,self.keywords)
    def projection(self):
        return {"scope_id":self.scope_id,"memory_id":self.memory_id,"version":self.version,"raw_text":self.raw_text,"entities":self.entities,"relations":self.relations,"keywords":self.keywords,"triples":self.triples,"rawerk":self.rawerk,"embedding_sha256":self.embedding_sha256}

class CassandraCells:
    names=("cassandra-base","cassandra-materialized")
    def __init__(self,host="127.0.0.1",keyspace="locomo_workload_v1"):
        from cassandra.cluster import Cluster
        self.cluster=Cluster([host],protocol_version=4); self.s=self.cluster.connect()
        self.s.execute(f"CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication={{'class':'SimpleStrategy','replication_factor':1}}")
        self.s.set_keyspace(keyspace); self._schema()
    def _schema(self):
        self.s.execute("CREATE TABLE IF NOT EXISTS base_memory_by_scope (scope_id text,memory_id text,version int,raw_text text,embedding_sha256 text,PRIMARY KEY ((scope_id),memory_id))")
        self.s.execute("CREATE TABLE IF NOT EXISTS base_feature_by_memory (scope_id text,memory_id text,entities text,relations text,keywords text,triples text,PRIMARY KEY ((scope_id,memory_id)))")
        self.s.execute("CREATE TABLE IF NOT EXISTS mat_memory_by_scope (scope_id text,memory_id text,version int,raw_text text,entities text,relations text,keywords text,triples text,rawerk text,embedding_sha256 text,PRIMARY KEY ((scope_id),memory_id))")
        self.pb1=self.s.prepare("INSERT INTO base_memory_by_scope (scope_id,memory_id,version,raw_text,embedding_sha256) VALUES (?,?,?,?,?)")
        self.pb2=self.s.prepare("INSERT INTO base_feature_by_memory (scope_id,memory_id,entities,relations,keywords,triples) VALUES (?,?,?,?,?,?)")
        self.pm=self.s.prepare("INSERT INTO mat_memory_by_scope (scope_id,memory_id,version,raw_text,entities,relations,keywords,triples,rawerk,embedding_sha256) VALUES (?,?,?,?,?,?,?,?,?,?)")
    def reset(self):
        for t in ("base_memory_by_scope","base_feature_by_memory","mat_memory_by_scope"): self.s.execute(f"TRUNCATE {t}")
    def insert(self,cell,r):
        if cell=="cassandra-base":
            from cassandra.query import BatchStatement,BatchType
            b=BatchStatement(batch_type=BatchType.LOGGED); b.add(self.pb1,(r.scope_id,r.memory_id,r.version,r.raw_text,r.embedding_sha256)); b.add(self.pb2,(r.scope_id,r.memory_id,r.entities,r.relations,r.keywords,r.triples)); self.s.execute(b)
        else:self.s.execute(self.pm,(r.scope_id,r.memory_id,r.version,r.raw_text,r.entities,r.relations,r.keywords,r.triples,r.rawerk,r.embedding_sha256))
    def delete(self,cell,r):
        if cell=="cassandra-base":
            self.s.execute("DELETE FROM base_memory_by_scope WHERE scope_id=%s AND memory_id=%s",(r.scope_id,r.memory_id)); self.s.execute("DELETE FROM base_feature_by_memory WHERE scope_id=%s AND memory_id=%s",(r.scope_id,r.memory_id))
        else:self.s.execute("DELETE FROM mat_memory_by_scope WHERE scope_id=%s AND memory_id=%s",(r.scope_id,r.memory_id))
    def fetch(self,cell,scope,mid):
        if cell=="cassandra-base":
            m=self.s.execute("SELECT * FROM base_memory_by_scope WHERE scope_id=%s AND memory_id=%s",(scope,mid)).one(); f=self.s.execute("SELECT * FROM base_feature_by_memory WHERE scope_id=%s AND memory_id=%s",(scope,mid)).one()
            if not(m and f): return None
            r=Record(scope,mid,int(m.version),m.raw_text,f.entities or "",f.relations or "",f.keywords or "",f.triples or "",m.embedding_sha256)
        else:
            m=self.s.execute("SELECT * FROM mat_memory_by_scope WHERE scope_id=%s AND memory_id=%s",(scope,mid)).one()
            if not m:return None
            r=Record(scope,mid,int(m.version),m.raw_text,m.entities or "",m.relations or "",m.keywords or "",m.triples or "",m.embedding_sha256)
        return r.projection()
    def ids(self,cell,scope):
        table="base_memory_by_scope" if cell=="cassandra-base" else "mat_memory_by_scope"
        return sorted(x.memory_id for x in self.s.execute(f"SELECT memory_id FROM {table} WHERE scope_id=%s",(scope,)))
    def counts(self):
        return {c:sum(len(self.ids(c,s)) for s in self.scopes(c)) for c in self.names}
    def scopes(self,cell):
        table="base_memory_by_scope" if cell=="cassandra-base" else "mat_memory_by_scope"
        return sorted({x.scope_id for x in self.s.execute(f"SELECT scope_id FROM {table}")})
    def close(self):self.s.shutdown();self.cluster.shutdown()

class Neo4jCells:
    names=("neo4j-native","neo4j-materialized")
    def __init__(self,uri,user,password,database="neo4j"):
        from neo4j import GraphDatabase
        self.d=GraphDatabase.driver(uri,auth=(user,password),max_connection_pool_size=128); self.d.verify_connectivity(); self.database=database; self._schema()
    def run(self,q,**p):
        with self.d.session(database=self.database) as s:return list(s.run(q,**p))
    def _schema(self):
        self.run("CREATE CONSTRAINT lwv1_native IF NOT EXISTS FOR (m:LWV1NativeMemory) REQUIRE (m.scope_id,m.memory_id) IS UNIQUE")
        self.run("CREATE CONSTRAINT lwv1_feature IF NOT EXISTS FOR (f:LWV1Feature) REQUIRE (f.scope_id,f.memory_id) IS UNIQUE")
        self.run("CREATE CONSTRAINT lwv1_mat IF NOT EXISTS FOR (m:LWV1MatMemory) REQUIRE (m.scope_id,m.memory_id) IS UNIQUE")
        self.run("CREATE INDEX lwv1_native_scope IF NOT EXISTS FOR (m:LWV1NativeMemory) ON (m.scope_id)")
        self.run("CREATE INDEX lwv1_mat_scope IF NOT EXISTS FOR (m:LWV1MatMemory) ON (m.scope_id)")
        self.run("CALL db.awaitIndexes(300)")
    def reset(self):
        self.run("MATCH (n) WHERE n:LWV1NativeMemory OR n:LWV1Feature OR n:LWV1MatMemory DETACH DELETE n")
    def insert(self,cell,r):
        p=r.projection()
        if cell=="neo4j-native":
            self.run("MERGE (m:LWV1NativeMemory {scope_id:$scope_id,memory_id:$memory_id}) SET m.version=$version,m.raw_text=$raw_text,m.embedding_sha256=$embedding_sha256 MERGE (f:LWV1Feature {scope_id:$scope_id,memory_id:$memory_id}) SET f.entities=$entities,f.relations=$relations,f.keywords=$keywords,f.triples=$triples MERGE (m)-[:HAS_FEATURE]->(f)",**p)
        else:self.run("MERGE (m:LWV1MatMemory {scope_id:$scope_id,memory_id:$memory_id}) SET m += $p",p=p,scope_id=r.scope_id,memory_id=r.memory_id)
    def insert_many(self,cell,records,batch=250,workers=1):
        rows=[r.projection() for r in records]
        q=("UNWIND $rows AS p MERGE (m:LWV1NativeMemory {scope_id:p.scope_id,memory_id:p.memory_id}) SET m.version=p.version,m.raw_text=p.raw_text,m.embedding_sha256=p.embedding_sha256 MERGE (f:LWV1Feature {scope_id:p.scope_id,memory_id:p.memory_id}) SET f.entities=p.entities,f.relations=p.relations,f.keywords=p.keywords,f.triples=p.triples MERGE (m)-[:HAS_FEATURE]->(f)" if cell=="neo4j-native" else "UNWIND $rows AS p MERGE (m:LWV1MatMemory {scope_id:p.scope_id,memory_id:p.memory_id}) SET m += p")
        chunks=[rows[i:i+batch] for i in range(0,len(rows),batch)]
        if workers <= 1:
            for chunk in chunks:self.run(q,rows=chunk)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(lambda chunk:self.run(q,rows=chunk),chunks))
    def delete(self,cell,r):
        if cell=="neo4j-native":self.run("MATCH (m:LWV1NativeMemory {scope_id:$s,memory_id:$m}) OPTIONAL MATCH (m)-[:HAS_FEATURE]->(f:LWV1Feature) DETACH DELETE m,f",s=r.scope_id,m=r.memory_id)
        else:self.run("MATCH (m:LWV1MatMemory {scope_id:$s,memory_id:$m}) DELETE m",s=r.scope_id,m=r.memory_id)
    def fetch(self,cell,scope,mid):
        if cell=="neo4j-native":
            rows=self.run("MATCH (m:LWV1NativeMemory {scope_id:$s,memory_id:$id})-[:HAS_FEATURE]->(f:LWV1Feature) RETURN m.version AS version,m.raw_text AS raw_text,m.embedding_sha256 AS embedding_sha256,f.entities AS entities,f.relations AS relations,f.keywords AS keywords,f.triples AS triples",s=scope,id=mid)
            if not rows:return None
            x=dict(rows[0]); r=Record(scope,mid,int(x['version']),x['raw_text'],x['entities'] or '',x['relations'] or '',x['keywords'] or '',x['triples'] or '',x['embedding_sha256'])
            return r.projection()
        rows=self.run("MATCH (m:LWV1MatMemory {scope_id:$s,memory_id:$id}) RETURN properties(m) AS p",s=scope,id=mid)
        return dict(rows[0]['p']) if rows else None
    def ids(self,cell,scope):
        label="LWV1NativeMemory" if cell=="neo4j-native" else "LWV1MatMemory"
        return sorted(x['id'] for x in self.run(f"MATCH (m:{label} {{scope_id:$s}}) RETURN m.memory_id AS id",s=scope))
    def counts(self):
        out={}
        for c,label in (("neo4j-native","LWV1NativeMemory"),("neo4j-materialized","LWV1MatMemory")):
            out[c]=self.run(f"MATCH (m:{label}) RETURN count(m) AS n")[0]['n']
        return out
    def close(self):self.d.close()
