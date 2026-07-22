#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P5-3C restart-only runner. No LLM/API calls.

Producer persists frozen events into a durable SQLite queue and commits raw/KG
state to Cassandra or Neo4j. A separate materializer process consumes committed
events, writes RawERK views, updates a persistent SQLite FTS5 BM25 index,
probes top-10 visibility, and advances an atomic checkpoint.

At t=60s the materializer process is terminated for 30s while the producer and
backend writes continue. It is then restarted from the durable checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import sqlite3
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping


def log(msg: str) -> None:
    print(time.strftime('%Y-%m-%d %H:%M:%S'), msg, flush=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve(value: str, base: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(value)))
    return p if p.is_absolute() else (base / p).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f'Bad JSONL {path}:{i}: {exc}') from exc
    return rows


def count_csv_rows(path: Path, has_header: bool) -> int:
    with path.open('rb') as f:
        n = sum(1 for _ in f)
    return n - 1 if has_header else n


def pct(values: list[float], q: float) -> float:
    if not values:
        return float('nan')
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ----------------------------- Frozen events -----------------------------

NAMES = ['Avery','Bailey','Casey','Devon','Emery','Finley','Harper','Jordan','Kai','Logan','Morgan','Parker']
PROJECTS = ['Atlas','Beacon','Cedar','Delta','Echo','Falcon','Gemini','Harbor','Iris','Juniper','Keystone','Lumen']
RELS = ['works_on','met_at','owns','visited','supports','reported_to']


def generate_events(path: Path, cfg: Mapping[str, Any]) -> None:
    exp = cfg['experiment']
    total = int(exp['event_count'])
    graph_id = str(exp['graph_id'])
    rng = random.Random(int(exp.get('seed', 20260722)))
    if path.exists() and not exp.get('overwrite_events', False):
        raise FileExistsError(f'{path} exists; set overwrite_events=true to replace')
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[str] = []
    versions: dict[str, int] = {}
    edges: dict[str, str] = {}
    next_mem = 0
    counts: dict[str, int] = {}
    with path.open('w', encoding='utf-8') as f:
        for seq in range(1, total + 1):
            slot = (seq - 1) % 10
            if slot < 6 or not existing:
                op = 'insert_memory'
                mid = f'p53c-memory-{next_mem:07d}'
                next_mem += 1
                existing.append(mid)
                versions[mid] = 1
                edges[mid] = f'edge-{mid}'
            else:
                mid = existing[rng.randrange(max(0, len(existing)-5000), len(existing))]
                versions[mid] += 1
                op = 'update_relation' if slot < 9 else 'replace_relation'
            version = versions[mid]
            num = int(mid.rsplit('-', 1)[1])
            person = f'{NAMES[num % len(NAMES)]}{num:07d}'
            project = f'{PROJECTS[(num // len(NAMES)) % len(PROJECTS)]}{num:07d}'
            rel = RELS[(num + version + (2 if op == 'replace_relation' else 0)) % len(RELS)]
            raw = f'{person} discussed project {project}. Current relationship: {rel}; version {version}.'
            entities = [
                {'entity_id': f'person:{person}', 'name': person},
                {'entity_id': f'project:{project}', 'name': project},
            ]
            relations = [{
                'edge_id': edges[mid], 'src_id': f'person:{person}',
                'dst_id': f'project:{project}', 'relation': rel,
            }]
            keywords = [person, project, rel, f'version{version}']
            raw_erk = raw + '\nEntities: ' + ', '.join(x['name'] for x in entities)
            raw_erk += '\nRelations: ' + rel + '\nKeywords: ' + ', '.join(keywords)
            event = {
                'sequence': seq, 'update_id': f'p53c-update-{seq:08d}',
                'graph_id': graph_id, 'memory_id': mid, 'version': version,
                'operation': op, 'raw_text': raw, 'entities': entities,
                'relations': relations, 'keywords': keywords,
                'raw_erk_text': raw_erk, 'probe_query': f'{person} {project}',
            }
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
            counts[op] = counts.get(op, 0) + 1
    atomic_json(path.with_name('p5_3c_event_manifest.json'), {
        'path': str(path), 'sha256': sha256_file(path), 'event_count': total,
        'graph_id': graph_id, 'operation_counts': counts,
    })
    log(f'Generated {total} events: {path}')


# ----------------------------- Durable queue -----------------------------

def connect_runtime(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60, isolation_level=None)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=FULL')
    conn.execute('PRAGMA busy_timeout=60000')
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS events(
      sequence INTEGER PRIMARY KEY,
      update_id TEXT UNIQUE NOT NULL,
      event_json TEXT NOT NULL,
      generated_ns INTEGER NOT NULL,
      persisted_ns INTEGER NOT NULL,
      raw_committed INTEGER NOT NULL DEFAULT 0,
      backend_committed_ns INTEGER,
      backend_error TEXT
    );
    CREATE TABLE IF NOT EXISTS checkpoint(
      singleton INTEGER PRIMARY KEY CHECK(singleton=1),
      sequence INTEGER NOT NULL,
      update_id TEXT,
      version INTEGER,
      updated_ns INTEGER NOT NULL
    );
    INSERT OR IGNORE INTO checkpoint(singleton,sequence,updated_ns) VALUES(1,0,0);
    CREATE TABLE IF NOT EXISTS materialized(
      sequence INTEGER PRIMARY KEY,
      update_id TEXT NOT NULL,
      memory_id TEXT NOT NULL,
      version INTEGER NOT NULL,
      operation TEXT NOT NULL,
      processing_started_ns INTEGER NOT NULL,
      rawerk_committed_ns INTEGER NOT NULL,
      bm25_committed_ns INTEGER NOT NULL,
      searchable_ns INTEGER,
      retrieved_rank INTEGER,
      queue_wait_ms REAL,
      backend_commit_ms REAL,
      view_update_ms REAL,
      index_update_ms REAL,
      visibility_probe_ms REAL,
      update_to_searchable_ms REAL,
      status TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS docs(
      rowid INTEGER PRIMARY KEY AUTOINCREMENT,
      memory_id TEXT UNIQUE NOT NULL,
      version INTEGER NOT NULL,
      text TEXT NOT NULL
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
      text, content='docs', content_rowid='rowid'
    );
    CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
      INSERT INTO docs_fts(rowid,text) VALUES(new.rowid,new.text);
    END;
    CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
      INSERT INTO docs_fts(docs_fts,rowid,text) VALUES('delete',old.rowid,old.text);
    END;
    CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
      INSERT INTO docs_fts(docs_fts,rowid,text) VALUES('delete',old.rowid,old.text);
      INSERT INTO docs_fts(rowid,text) VALUES(new.rowid,new.text);
    END;
    CREATE TABLE IF NOT EXISTS timeline(
      ts REAL, elapsed REAL, phase TEXT, persisted INTEGER, committed INTEGER,
      checkpoint_seq INTEGER, event_log_backlog INTEGER,
      materialization_backlog INTEGER, searchability_backlog INTEGER,
      oldest_event_age_ms REAL, materializer_status TEXT
    );
    ''')
    return conn


def fts_query(text: str) -> str:
    toks = re.findall(r'[A-Za-z0-9_]+|[\u4e00-\u9fff]+', text)
    toks = [t for t in toks if len(t) > 1][:6]
    return ' OR '.join('"' + t.replace('"','""') + '"' for t in toks) or '""'


# ----------------------------- Backends -----------------------------

class Backend:
    def commit_raw_kg(self, event: Mapping[str, Any]) -> None: raise NotImplementedError
    def materialize_view(self, event: Mapping[str, Any]) -> None: raise NotImplementedError
    def memory_version(self, graph_id: str, memory_id: str) -> int | None: raise NotImplementedError
    def view_version(self, graph_id: str, memory_id: str) -> int | None: raise NotImplementedError
    def close(self) -> None: pass


class CassandraBackend(Backend):
    def __init__(self, cfg: Mapping[str, Any]):
        from cassandra.cluster import Cluster
        c = cfg['cassandra']
        self.keyspace = c.get('keyspace', 'ai_memory')
        self.cluster = Cluster(c.get('contact_points', ['127.0.0.1']), port=int(c.get('port', 9042)))
        self.session = self.cluster.connect()
        self.session.execute(f"CREATE KEYSPACE IF NOT EXISTS {self.keyspace} WITH replication={{'class':'SimpleStrategy','replication_factor':1}}")
        self.session.set_keyspace(self.keyspace)
        self.session.execute('CREATE TABLE IF NOT EXISTS p5_memory_by_id(graph_id text,memory_id text,raw_text text,version int,PRIMARY KEY((graph_id,memory_id)))')
        self.session.execute('CREATE TABLE IF NOT EXISTS p5_edges_by_memory(graph_id text,memory_id text,edge_id text,src_id text,dst_id text,relation text,version int,PRIMARY KEY((graph_id,memory_id),edge_id))')
        self.session.execute('CREATE TABLE IF NOT EXISTS p5_raw_erk_view_by_memory(graph_id text,memory_id text,text text,version int,PRIMARY KEY((graph_id,memory_id)))')
        self.mem_i = self.session.prepare('INSERT INTO p5_memory_by_id(graph_id,memory_id,raw_text,version) VALUES(?,?,?,?) IF NOT EXISTS')
        self.mem_u = self.session.prepare('UPDATE p5_memory_by_id SET raw_text=?,version=? WHERE graph_id=? AND memory_id=? IF version < ?')
        self.edge_i = self.session.prepare('INSERT INTO p5_edges_by_memory(graph_id,memory_id,edge_id,src_id,dst_id,relation,version) VALUES(?,?,?,?,?,?,?) IF NOT EXISTS')
        self.edge_u = self.session.prepare('UPDATE p5_edges_by_memory SET src_id=?,dst_id=?,relation=?,version=? WHERE graph_id=? AND memory_id=? AND edge_id=? IF version < ?')
        self.view_i = self.session.prepare('INSERT INTO p5_raw_erk_view_by_memory(graph_id,memory_id,text,version) VALUES(?,?,?,?) IF NOT EXISTS')
        self.view_u = self.session.prepare('UPDATE p5_raw_erk_view_by_memory SET text=?,version=? WHERE graph_id=? AND memory_id=? IF version < ?')
        self.mem_r = self.session.prepare('SELECT version FROM p5_memory_by_id WHERE graph_id=? AND memory_id=?')
        self.view_r = self.session.prepare('SELECT version FROM p5_raw_erk_view_by_memory WHERE graph_id=? AND memory_id=?')
    @staticmethod
    def applied(result: Any) -> bool:
        row = result.one()
        if row is None: return True
        return bool(getattr(row, 'applied', row[0]))
    def commit_raw_kg(self, e: Mapping[str, Any]) -> None:
        g,m,v = str(e['graph_id']),str(e['memory_id']),int(e['version'])
        if not self.applied(self.session.execute(self.mem_i,(g,m,str(e['raw_text']),v))):
            self.session.execute(self.mem_u,(str(e['raw_text']),v,g,m,v))
        for r in e.get('relations',[]):
            vals=(g,m,str(r['edge_id']),str(r['src_id']),str(r['dst_id']),str(r['relation']),v)
            if not self.applied(self.session.execute(self.edge_i,vals)):
                self.session.execute(self.edge_u,(str(r['src_id']),str(r['dst_id']),str(r['relation']),v,g,m,str(r['edge_id']),v))
    def materialize_view(self, e: Mapping[str, Any]) -> None:
        g,m,v,t=str(e['graph_id']),str(e['memory_id']),int(e['version']),str(e['raw_erk_text'])
        if not self.applied(self.session.execute(self.view_i,(g,m,t,v))):
            self.session.execute(self.view_u,(t,v,g,m,v))
    def memory_version(self,g:str,m:str)->int|None:
        row=self.session.execute(self.mem_r,(g,m)).one(); return None if row is None else int(row.version)
    def view_version(self,g:str,m:str)->int|None:
        row=self.session.execute(self.view_r,(g,m)).one(); return None if row is None else int(row.version)
    def close(self)->None: self.cluster.shutdown()


class Neo4jBackend(Backend):
    def __init__(self,cfg:Mapping[str,Any]):
        from neo4j import GraphDatabase
        n=cfg['neo4j']
        uri=os.environ.get(n.get('uri_env','NEO4J_URI'),n.get('uri','bolt://127.0.0.1:7687'))
        user=os.environ.get(n.get('user_env','NEO4J_USER'),n.get('user','neo4j'))
        pwd=os.environ.get(n.get('password_env','NEO4J_PASSWORD'),n.get('password',''))
        self.database=os.environ.get(n.get('database_env','NEO4J_DATABASE'),n.get('database','neo4j'))
        self.driver=GraphDatabase.driver(uri,auth=(user,pwd),max_connection_pool_size=int(n.get('max_connection_pool_size',128)),connection_acquisition_timeout=60,max_transaction_retry_time=30)
        self.driver.verify_connectivity()
        for q in [
          'CREATE CONSTRAINT p5_memory_key IF NOT EXISTS FOR (m:P5Memory) REQUIRE (m.graph_id,m.memory_id) IS UNIQUE',
          'CREATE CONSTRAINT p5_entity_key IF NOT EXISTS FOR (e:P5Entity) REQUIRE (e.graph_id,e.entity_id) IS UNIQUE',
          'CREATE CONSTRAINT p5_view_key IF NOT EXISTS FOR (v:P5RawERKView) REQUIRE (v.graph_id,v.memory_id) IS UNIQUE']:
            self.driver.execute_query(q,database_=self.database)
    @staticmethod
    def raw_tx(tx:Any,e:Mapping[str,Any])->None:
        tx.run('''MERGE (m:P5Memory {graph_id:$graph_id,memory_id:$memory_id})
        ON CREATE SET m.raw_text=$raw_text,m.version=$version
        ON MATCH SET m.raw_text=CASE WHEN coalesce(m.version,-1)<=$version THEN $raw_text ELSE m.raw_text END,
        m.version=CASE WHEN coalesce(m.version,-1)<=$version THEN $version ELSE m.version END''',**dict(e)).consume()
        tx.run('''UNWIND $entities AS x MERGE (n:P5Entity {graph_id:$graph_id,entity_id:x.entity_id})
        ON CREATE SET n.name=x.name,n.version=$version
        ON MATCH SET n.name=CASE WHEN coalesce(n.version,-1)<=$version THEN x.name ELSE n.name END,
        n.version=CASE WHEN coalesce(n.version,-1)<=$version THEN $version ELSE n.version END
        WITH n MATCH (m:P5Memory {graph_id:$graph_id,memory_id:$memory_id}) MERGE (m)-[:P5_MENTIONS]->(n)''',**dict(e)).consume()
        tx.run('''UNWIND $relations AS x MATCH (s:P5Entity {graph_id:$graph_id,entity_id:x.src_id})
        MATCH (d:P5Entity {graph_id:$graph_id,entity_id:x.dst_id})
        MERGE (s)-[r:P5_KG_EDGE {graph_id:$graph_id,edge_id:x.edge_id}]->(d)
        ON CREATE SET r.relation=x.relation,r.source_memory_id=$memory_id,r.version=$version
        ON MATCH SET r.relation=CASE WHEN coalesce(r.version,-1)<=$version THEN x.relation ELSE r.relation END,
        r.source_memory_id=CASE WHEN coalesce(r.version,-1)<=$version THEN $memory_id ELSE r.source_memory_id END,
        r.version=CASE WHEN coalesce(r.version,-1)<=$version THEN $version ELSE r.version END''',**dict(e)).consume()
    @staticmethod
    def view_tx(tx:Any,e:Mapping[str,Any])->None:
        tx.run('''MERGE (v:P5RawERKView {graph_id:$graph_id,memory_id:$memory_id})
        ON CREATE SET v.text=$raw_erk_text,v.version=$version
        ON MATCH SET v.text=CASE WHEN coalesce(v.version,-1)<=$version THEN $raw_erk_text ELSE v.text END,
        v.version=CASE WHEN coalesce(v.version,-1)<=$version THEN $version ELSE v.version END''',**dict(e)).consume()
    def commit_raw_kg(self,e:Mapping[str,Any])->None:
        with self.driver.session(database=self.database) as s: s.execute_write(self.raw_tx,e)
    def materialize_view(self,e:Mapping[str,Any])->None:
        with self.driver.session(database=self.database) as s: s.execute_write(self.view_tx,e)
    def memory_version(self,g:str,m:str)->int|None:
        rec,_,_=self.driver.execute_query('MATCH (x:P5Memory {graph_id:$g,memory_id:$m}) RETURN x.version AS v',g=g,m=m,database_=self.database)
        return None if not rec else int(rec[0]['v'])
    def view_version(self,g:str,m:str)->int|None:
        rec,_,_=self.driver.execute_query('MATCH (x:P5RawERKView {graph_id:$g,memory_id:$m}) RETURN x.version AS v',g=g,m=m,database_=self.database)
        return None if not rec else int(rec[0]['v'])
    def close(self)->None: self.driver.close()


def make_backend(name:str,cfg:Mapping[str,Any])->Backend:
    return CassandraBackend(cfg) if name=='cassandra' else Neo4jBackend(cfg)


# ----------------------------- Materializer process -----------------------------

def materializer_main(config_path:str,backend_name:str,runtime_db:str,stop_flag:str)->None:
    cfg=load_json(Path(config_path)); db=connect_runtime(Path(runtime_db)); backend=make_backend(backend_name,cfg)
    stop_path=Path(stop_flag)
    probe_interval=float(cfg['experiment'].get('probe_interval_seconds',0.05))
    probe_timeout=float(cfg['experiment'].get('probe_timeout_seconds',30))
    try:
        while not stop_path.exists():
            cp=int(db.execute('SELECT sequence FROM checkpoint WHERE singleton=1').fetchone()[0])
            row=db.execute('SELECT event_json,generated_ns,persisted_ns,backend_committed_ns FROM events WHERE sequence=? AND raw_committed=1',(cp+1,)).fetchone()
            if row is None:
                time.sleep(0.03); continue
            e=json.loads(row[0]); generated,persisted,backend_done=int(row[1]),int(row[2]),int(row[3])
            started=time.perf_counter_ns(); backend.materialize_view(e); view_done=time.perf_counter_ns()
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT version FROM docs WHERE memory_id=?',(e['memory_id'],)).fetchone()
            if old is None or int(old[0])<=int(e['version']):
                db.execute('''INSERT INTO docs(memory_id,version,text) VALUES(?,?,?)
                ON CONFLICT(memory_id) DO UPDATE SET version=excluded.version,text=excluded.text
                WHERE excluded.version>=docs.version''',(e['memory_id'],int(e['version']),e['raw_erk_text']))
            db.execute('COMMIT'); index_done=time.perf_counter_ns()
            deadline=time.monotonic()+probe_timeout; rank=None
            while time.monotonic()<deadline:
                rows=db.execute('''SELECT d.memory_id,d.version FROM docs_fts f JOIN docs d ON d.rowid=f.rowid
                WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts) LIMIT 10''',(fts_query(e['probe_query']),)).fetchall()
                for i,(mid,ver) in enumerate(rows,1):
                    if mid==e['memory_id'] and int(ver)==int(e['version']): rank=i; break
                if rank is not None: break
                time.sleep(probe_interval)
            searchable=time.perf_counter_ns(); status='success' if rank is not None else 'searchability_timeout'
            db.execute('BEGIN IMMEDIATE')
            db.execute('''INSERT OR REPLACE INTO materialized (
  sequence,update_id,memory_id,version,operation,
  processing_started_ns,rawerk_committed_ns,bm25_committed_ns,searchable_ns,retrieved_rank,
  queue_wait_ms,backend_commit_ms,view_update_ms,index_update_ms,visibility_probe_ms,
  update_to_searchable_ms,status
) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(
                int(e['sequence']),e['update_id'],e['memory_id'],int(e['version']),e['operation'],
                started,view_done,index_done,
                searchable if rank is not None else None,rank,
                (started-persisted)/1e6,(backend_done-persisted)/1e6,
                (view_done-started)/1e6,(index_done-view_done)/1e6,(searchable-index_done)/1e6,
                (searchable-generated)/1e6 if rank is not None else None,status))
            if status=='success':
                db.execute('UPDATE checkpoint SET sequence=?,update_id=?,version=?,updated_ns=? WHERE singleton=1',(int(e['sequence']),e['update_id'],int(e['version']),time.perf_counter_ns()))
            db.execute('COMMIT')
            if status!='success': time.sleep(0.2)
    finally:
        backend.close(); db.close()


# ----------------------------- Scale guard -----------------------------

def scale_guard(cfg:Mapping[str,Any],base:Path,out:Path)->dict[str,Any]:
    sg=cfg['scale_guard']; src=resolve(sg['source_artifact'],base)
    result={'source_artifact':str(src),'status':'FAIL'}
    if not src.exists(): result['error']='missing source'; atomic_json(out,result); return result
    rows=count_csv_rows(src,bool(sg.get('has_header',True))); sha=sha256_file(src)
    result.update({'actual_rows':rows,'actual_sha256':sha,'expected_rows':int(sg.get('expected_rows',1000000)),'expected_sha256':sg.get('expected_sha256')})
    ok=rows==int(sg.get('expected_rows',1000000))
    if sg.get('expected_sha256'): ok=ok and sha.lower()==str(sg['expected_sha256']).lower()
    manifest=resolve(sg['validated_manifest'],base)
    ok=ok and manifest.exists(); result['validated_manifest']=str(manifest)
    result['status']='PASS' if ok else 'FAIL'; atomic_json(out,result); return result


# ----------------------------- Run orchestration -----------------------------

def run_one(config_path:Path,backend_name:str)->int:
    cfg=load_json(config_path); base=config_path.parent; exp=cfg['experiment']; out=resolve(cfg['output_dir'],base); out.mkdir(parents=True,exist_ok=True)
    guard=scale_guard(cfg,base,out/f'p5_3c_{backend_name}_scale_guard.json')
    if guard['status']!='PASS': raise RuntimeError(f'Scale guard failed: {guard}')
    events_path=resolve(cfg['files']['events'],base); frozen=read_jsonl(events_path); total=int(exp['event_count']); frozen=frozen[:total]
    if len(frozen)<total: raise RuntimeError('Not enough frozen events')
    runtime=out/'runtime'/backend_name
    if exp.get('reset_runtime',True) and runtime.exists(): shutil.rmtree(runtime)
    runtime.mkdir(parents=True,exist_ok=True); db_path=runtime/'runtime.sqlite'; stop_flag=runtime/'worker.stop'
    db=connect_runtime(db_path); backend=make_backend(backend_name,cfg)
    rate=float(exp.get('input_rate_per_second',100)); duration=float(exp.get('producer_duration_seconds',180)); stop_at=float(exp.get('materializer_stop_at_seconds',60)); outage=float(exp.get('materializer_outage_seconds',30)); restart_at=stop_at+outage
    stable_verify=float(exp.get('stable_verify_seconds',60)); concurrency=int(exp.get('concurrency',32)); drain_timeout=float(exp.get('drain_timeout_seconds',180))
    needed=math.ceil(rate*duration)
    if total<needed: raise RuntimeError(f'event_count {total} < needed {needed}')
    manifest={'backend':backend_name,'events_sha256':sha256_file(events_path),'event_count':total,'graph_id':exp['graph_id'],'rate':rate,'duration':duration,'stop_at':stop_at,'outage':outage,'concurrency':concurrency,'scale_guard':guard,'config_sha256':sha256_file(config_path)}
    atomic_json(out/f'p5_3c_{backend_name}_run_manifest.json',manifest)
    ctx=mp.get_context('spawn'); proc=ctx.Process(target=materializer_main,args=(str(config_path),backend_name,str(db_path),str(stop_flag)),daemon=False); proc.start()
    start_ns=time.perf_counter_ns(); next_emit=time.perf_counter(); produced=0; futures=[]; stop_done=False; restart_done=False; cp_before_stop=0; stop_ns=restart_ns=None
    lock=mp.Lock()
    def commit_task(e:dict[str,Any])->None:
        try:
            backend.commit_raw_kg(e); done=time.perf_counter_ns()
            c=connect_runtime(db_path); c.execute('UPDATE events SET raw_committed=1,backend_committed_ns=? WHERE sequence=?',(done,int(e['sequence']))); c.close()
        except Exception as exc:
            c=connect_runtime(db_path); c.execute('UPDATE events SET backend_error=? WHERE sequence=?',(repr(exc),int(e['sequence']))); c.close()
    executor=ThreadPoolExecutor(max_workers=concurrency)
    timeline_next=time.perf_counter()
    try:
        while True:
            elapsed=(time.perf_counter_ns()-start_ns)/1e9
            if elapsed<duration:
                now=time.perf_counter()
                while now>=next_emit and produced<total:
                    e=dict(frozen[produced]); produced+=1; gen=time.perf_counter_ns(); persisted=time.perf_counter_ns()
                    db.execute('INSERT INTO events(sequence,update_id,event_json,generated_ns,persisted_ns) VALUES(?,?,?,?,?)',(int(e['sequence']),e['update_id'],json.dumps(e,ensure_ascii=False),gen,persisted))
                    futures.append(executor.submit(commit_task,e)); next_emit+=1/rate; now=time.perf_counter()
            if not stop_done and elapsed>=stop_at:
                cp_before_stop=int(db.execute('SELECT sequence FROM checkpoint WHERE singleton=1').fetchone()[0]); stop_ns=time.perf_counter_ns(); proc.terminate(); proc.join(timeout=10); stop_done=True; log(f'{backend_name}: materializer stopped checkpoint={cp_before_stop}')
            if stop_done and not restart_done and elapsed>=restart_at:
                if stop_flag.exists(): stop_flag.unlink()
                restart_ns=time.perf_counter_ns(); proc=ctx.Process(target=materializer_main,args=(str(config_path),backend_name,str(db_path),str(stop_flag)),daemon=False); proc.start(); restart_done=True; log(f'{backend_name}: materializer restarted')
            if time.perf_counter()>=timeline_next:
                cp=int(db.execute('SELECT sequence FROM checkpoint WHERE singleton=1').fetchone()[0]); committed=int(db.execute('SELECT COUNT(*) FROM events WHERE raw_committed=1').fetchone()[0])
                oldest=db.execute('SELECT persisted_ns FROM events WHERE sequence=?',(cp+1,)).fetchone(); age=0 if oldest is None else (time.perf_counter_ns()-int(oldest[0]))/1e6
                phase='stable_before_outage' if elapsed<stop_at else 'outage' if elapsed<restart_at else 'recovery_with_input' if elapsed<duration else 'drain'
                db.execute('INSERT INTO timeline VALUES(?,?,?,?,?,?,?,?,?,?,?)',(time.time(),elapsed,phase,produced,committed,cp,produced-cp,max(0,committed-cp),produced-cp,age,'running' if proc.is_alive() else 'stopped'))
                timeline_next+=1
            if elapsed>=duration: break
            time.sleep(0.001)
        executor.shutdown(wait=True)
        drain_start=time.perf_counter_ns(); final_cp=0
        while (time.perf_counter_ns()-drain_start)/1e9<drain_timeout:
            final_cp=int(db.execute('SELECT sequence FROM checkpoint WHERE singleton=1').fetchone()[0])
            if final_cp>=produced: break
            time.sleep(0.2)
        backlog_zero_ns=time.perf_counter_ns() if final_cp>=produced else None
        if backlog_zero_ns: time.sleep(stable_verify)
        stop_flag.write_text('stop\n',encoding='utf-8'); proc.join(timeout=15)
        if proc.is_alive(): proc.terminate(); proc.join(timeout=5)
    finally:
        executor.shutdown(wait=False,cancel_futures=False); backend.close()
    # export
    cols=[d[1] for d in db.execute('PRAGMA table_info(materialized)').fetchall()]
    mrows=[dict(zip(cols,row)) for row in db.execute('SELECT * FROM materialized ORDER BY sequence').fetchall()]
    ecols=[d[1] for d in db.execute('PRAGMA table_info(events)').fetchall()]
    erows=[dict(zip(ecols,row)) for row in db.execute('SELECT * FROM events ORDER BY sequence').fetchall()]
    tcols=[d[1] for d in db.execute('PRAGMA table_info(timeline)').fetchall()]
    trows=[dict(zip(tcols,row)) for row in db.execute('SELECT * FROM timeline ORDER BY ts').fetchall()]
    materialized={int(r['sequence']):r for r in mrows}
    per=[]
    for r in erows:
        e=json.loads(r['event_json']); m=materialized.get(int(r['sequence']),{})
        elapsed=(int(r['generated_ns'])-start_ns)/1e9; phase='stable_before_outage' if elapsed<stop_at else 'outage' if elapsed<restart_at else 'recovery_with_input'
        per.append({'sequence':r['sequence'],'update_id':r['update_id'],'memory_id':e['memory_id'],'version':e['version'],'operation':e['operation'],'backend':backend_name,'phase':phase,'raw_committed':r['raw_committed'],'backend_error':r['backend_error'],**{k:v for k,v in m.items() if k not in {'sequence','update_id','memory_id','version','operation'}}})
    write_csv(out/f'p5_3c_{backend_name}_per_event.csv',per); write_csv(out/f'p5_3c_{backend_name}_timeline.csv',trows)
    phase_rows=[]
    for ph in ['stable_before_outage','outage','recovery_with_input']:
        vals=[float(r['update_to_searchable_ms']) for r in per if r['phase']==ph and r.get('status')=='success' and r.get('update_to_searchable_ms') is not None]
        phase_rows.append({'backend':backend_name,'phase':ph,'n':len(vals),'p50_ms':pct(vals,.5),'p95_ms':pct(vals,.95),'p99_ms':pct(vals,.99),'mean_ms':statistics.mean(vals) if vals else float('nan')})
    write_csv(out/f'p5_3c_{backend_name}_phase_summary.csv',phase_rows)
    latest={}
    for r in erows:
        e=json.loads(r['event_json']); mid=e['memory_id']
        if mid not in latest or int(e['version'])>=int(latest[mid]['version']): latest[mid]=e
    audit_backend=make_backend(backend_name,cfg); audit=[]
    for mid,e in latest.items():
        mem=audit_backend.memory_version(e['graph_id'],mid); view=audit_backend.view_version(e['graph_id'],mid); idx=db.execute('SELECT version FROM docs WHERE memory_id=?',(mid,)).fetchone(); idxv=None if idx is None else int(idx[0]); ok=mem==view==idxv==int(e['version'])
        audit.append({'backend':backend_name,'memory_id':mid,'expected_version':e['version'],'memory_version':mem,'rawerk_version':view,'bm25_version':idxv,'status':'PASS' if ok else 'FAIL'})
    audit_backend.close(); write_csv(out/f'p5_3c_{backend_name}_final_state_audit.csv',audit)
    final_cp=int(db.execute('SELECT sequence FROM checkpoint WHERE singleton=1').fetchone()[0]); max_backlog=max((int(r['searchability_backlog']) for r in trows),default=0); max_lag=max((float(r['oldest_event_age_ms']) for r in trows),default=0)
    after=[r for r in mrows if restart_ns and int(r['processing_started_ns'])>=restart_ns and r['status']=='success']; first_replay=min((int(r['processing_started_ns']) for r in after),default=None); first_search=min((int(r['searchable_ns']) for r in after if r['searchable_ns'] is not None),default=None)
    summary={'backend':backend_name,'produced_events':produced,'searchable_events':len([r for r in mrows if r['status']=='success']),'checkpoint_before_stop':cp_before_stop,'final_checkpoint':final_cp,'maximum_searchability_backlog':max_backlog,'maximum_searchable_lag_ms':max_lag,'restart_to_first_replay_ms':(first_replay-restart_ns)/1e6 if first_replay and restart_ns else None,'restart_to_first_searchable_ms':(first_search-restart_ns)/1e6 if first_search and restart_ns else None,'backlog_drain_ms_after_restart':(backlog_zero_ns-restart_ns)/1e6 if backlog_zero_ns and restart_ns else None}
    atomic_json(out/f'p5_3c_{backend_name}_recovery_summary.json',summary)
    outage_cp={int(r['checkpoint_seq']) for r in trows if r['phase']=='outage'}
    gate={'backend':backend_name,'scale_guard':guard['status'],'materializer_stopped':stop_done,'materializer_restarted':restart_done,'checkpoint_fixed_during_outage':len(outage_cp)<=1 and len(outage_cp)>0,'backlog_formed':max_backlog>0,'final_backlog_zero':final_cp==produced,'backend_commit_failures':sum(1 for r in erows if r['backend_error']),'missed_updates':produced-len([r for r in mrows if r['status']=='success']),'searchability_timeouts':sum(1 for r in mrows if r['status']=='searchability_timeout'),'final_state_failures':sum(1 for r in audit if r['status']=='FAIL')}
    gate['status']='PASS' if all([gate['scale_guard']=='PASS',gate['materializer_stopped'],gate['materializer_restarted'],gate['checkpoint_fixed_during_outage'],gate['backlog_formed'],gate['final_backlog_zero'],gate['backend_commit_failures']==0,gate['missed_updates']==0,gate['searchability_timeouts']==0,gate['final_state_failures']==0]) else 'FAIL'
    atomic_json(out/f'p5_3c_{backend_name}_gate.json',gate)
    (out/f'p5_3c_{backend_name}_summary.md').write_text(f"# P5-3C {backend_name}\n\n- Gate: **{gate['status']}**\n- Searchable: {summary['searchable_events']}/{produced}\n- Maximum backlog: {max_backlog}\n- Maximum lag: {max_lag:.1f} ms\n- Restart→first searchable: {summary['restart_to_first_searchable_ms']} ms\n- Drain after restart: {summary['backlog_drain_ms_after_restart']} ms\n",encoding='utf-8')
    db.close(); log(f'{backend_name}: gate={gate["status"]}'); return 0 if gate['status']=='PASS' else 2


def compare(config_path:Path)->int:
    cfg=load_json(config_path); out=resolve(cfg['output_dir'],config_path.parent); rows=[]
    for b in ['cassandra','neo4j']:
        s=load_json(out/f'p5_3c_{b}_recovery_summary.json'); g=load_json(out/f'p5_3c_{b}_gate.json')
        rows.append({'backend':b,'gate':g['status'],'maximum_searchability_backlog':s['maximum_searchability_backlog'],'maximum_searchable_lag_ms':s['maximum_searchable_lag_ms'],'restart_to_first_searchable_ms':s['restart_to_first_searchable_ms'],'backlog_drain_ms_after_restart':s['backlog_drain_ms_after_restart']})
    write_csv(out/'p5_3c_cross_backend_comparison.csv',rows)
    both=all(r['gate']=='PASS' for r in rows); atomic_json(out/'p5_3c_cross_backend_gate.json',{'both_pass':both})
    lines=['# P5-3C Cross-Backend Comparison','','| Backend | Gate | Max backlog | Max lag ms | Restart→searchable ms | Drain ms |','|---|---|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['backend']} | {r['gate']} | {r['maximum_searchability_backlog']} | {r['maximum_searchable_lag_ms']:.1f} | {r['restart_to_first_searchable_ms']} | {r['backlog_drain_ms_after_restart']} |")
    (out/'p5_3c_cross_backend_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); return 0 if both else 2


def main()->int:
    mp.freeze_support(); ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--backend',choices=['cassandra','neo4j']); ap.add_argument('--generate-events',action='store_true'); ap.add_argument('--run',action='store_true'); ap.add_argument('--compare',action='store_true'); a=ap.parse_args(); cp=Path(a.config).resolve(); cfg=load_json(cp)
    if a.generate_events: generate_events(resolve(cfg['files']['events'],cp.parent),cfg); return 0
    if a.run:
        if not a.backend: ap.error('--run requires --backend')
        return run_one(cp,a.backend)
    if a.compare: return compare(cp)
    ap.error('choose --generate-events, --run, or --compare'); return 2


if __name__=='__main__': raise SystemExit(main())
