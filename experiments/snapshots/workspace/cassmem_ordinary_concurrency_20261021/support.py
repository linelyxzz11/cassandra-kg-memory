"""Isolated ordinary-load experiment; imports frozen production components."""
from __future__ import annotations

import os
import sys
from pathlib import Path
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
HERE = Path(__file__).resolve().parent
ROOT = Path('D:/memorytable/cassandra-kg-memory')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments/serving_100k'),
               'D:/memorytable/representation_serving_cost',
               'D:/memorytable/cassmem_e0_preflight_20261015']
import collections
import dataclasses
import hashlib
import json
import subprocess
import time
import numpy as np
from common import build_graph_records, vectors, rows, write_csv
from run_graph_100k_load_gate_v2 import expand
from cassmem.representation.graph_event import GraphRecord, digest
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex, _bm25_search, _dense_search, zscore_fuse
from cassmem.serving.serialized_updates import SerializedUpdateService
from instrumentation import TimedLock, LOCAL

CELL = 'cassandra-materialized'
CONTAINER = 'cassmem-e0-preflight-20261019222905-cassandra'
PORT = 49042
RUN = HERE/'run_v1'
RECEIVER = Path('D:/memorytable/cassmem_e0_ingress_20261018/receiver.py')

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def hash_array(v):
    return hashlib.sha256(np.asarray(v,dtype=np.float32).tobytes()).hexdigest()

def normalized_hash(v):
    # Match actual dense snapshot's row-wise float32 normalization exactly.
    a = np.asarray(v,dtype=np.float32)[None,:]
    return hash_array((a / np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12))[0])

def verify_manifest(out=RUN):
    for p,h in load(out/'source_manifest.json').items():
        if sha(p) != h:
            raise RuntimeError('Frozen source drift: '+p)

def docker(*args):
    return subprocess.check_output(['docker',*args],text=True,stderr=subprocess.PIPE,timeout=60).strip()

def environment_gate():
    names = docker('ps','--format','{{.Names}}').splitlines()
    if names != [CONTAINER]:
        raise RuntimeError('Exclusive experiment requires only the selected container: '+repr(names))
    x = json.loads(docker('inspect',CONTAINER))[0]
    cfg = x['HostConfig']
    if (cfg['Memory'],cfg['NanoCpus'],cfg['MemorySwap']) != (12*1024**3,8*10**9,12*1024**3):
        raise RuntimeError('Resource budget differs from frozen E0 budget')
    env = dict(v.split('=',1) for v in x['Config']['Env'])
    if env.get('MAX_HEAP_SIZE') != '6G' or env.get('HEAP_NEWSIZE') != '1G':
        raise RuntimeError('Heap budget drift')
    version = docker('exec',CONTAINER,'cqlsh','-e','SELECT release_version FROM system.local;')
    return dict(container=CONTAINER,id=x['Id'],image=x['Image'],memory=cfg['Memory'],nano_cpus=cfg['NanoCpus'],
                memory_swap=cfg['MemorySwap'],heap='6G',young_heap='1G',port=PORT,version=version,
                consistency='LOCAL_ONE',replication_factor=1,protocol_version=4,
                reuse='Existing container, NEW isolated keyspace; old keyspaces remain untouched')

def connect(keyspace):
    if not keyspace.startswith('oc20261021_'):
        raise ValueError('Refusing a non-experiment keyspace')
    os.environ['CASSANDRA_PORT'] = str(PORT)
    from cassmem.backend.live_cells_graph import CassandraGraphCells
    from cassandra import ConsistencyLevel
    b = CassandraGraphCells(keyspace=keyspace)
    b.s.default_timeout = 30
    b.s.default_consistency_level = ConsistencyLevel.LOCAL_ONE
    for p in b.p.values():
        p.consistency_level = ConsistencyLevel.LOCAL_ONE
    return b

def data():
    source = build_graph_records()
    initial = expand(source)
    assert len(initial)==100000 and len({r.scope_id for r in initial})==171
    return source, initial, vectors('memory'), vectors('qa'), rows(ROOT/'data/frozen_retrieval/p3_queries.csv')

def resolve(trace, initial, source, mv):
    original = {r.memory_id:r for r in initial}
    src = {r.memory_id:r for r in source}
    records = {}
    for t in trace:
        key = (t['scope'],t['memory_id'])
        r0 = original[t['memory_id']]
        records.setdefault(key,{1:(r0,mv[r0.memory_id.split('::')[-1]])})
        r = dataclasses.replace(src[t['source_memory_id']],scope_id=t['scope'],memory_id=t['memory_id'],version=t['version'])
        assert hash_array(mv[t['source_memory_id']])==r.embedding_sha256
        records[key][r.version]=(r,mv[t['source_memory_id']])
    return records

def expected_payload(r, v):
    return dict(version=r.version,graph_hash=digest(r.graph_projection()),projection_hash=hashlib.sha256(r.rawerk.encode()).hexdigest(),
                dense_hash=normalized_hash(v),relations=sorted({e['relation'] for e in r.edges}))

def capture(b,index,key,relations):
    s,mid=key
    start=time.perf_counter_ns()
    with index._scope_locks[s]:
        sp,de=index._sparse[s],index._dense[s]
        sv=index._sparse_versions[(s,mid)];dv=index._dense_versions[(s,mid)]
        doc=sp.documents[sp.memory_ids.index(mid)]
        vec=de.normalized_embeddings[de.memory_ids.index(mid)]
        snap=time.perf_counter_ns()
    graph=b.fetch_graph(CELL,s,mid)
    projection=b.candidate_projection(CELL,s,mid)
    memberships={rel:mid in b.ids_by_relation(CELL,s,rel) for rel in sorted(relations)}
    ids=b.ids(CELL,s)
    return dict(start_ns=start,snapshot_ns=snap,end_ns=time.perf_counter_ns(),scope=s,memory_id=mid,
                graph=graph,projection=projection,sparse_version=sv,dense_version=dv,sparse_document=doc,
                dense_vector_sha256=hash_array(vec),relation_memberships=memberships,candidate_present=mid in ids)

def capture_matches(obs, expected):
    return (obs['graph']==expected.graph_projection() and obs['projection']==expected.rawerk
            and obs['sparse_document']==expected.rawerk and obs['sparse_version']==expected.version
            and obs['dense_version']==expected.version and obs['candidate_present']
            and all(present==(rel in {e['relation'] for e in expected.edges}) for rel,present in obs['relation_memberships'].items()))

class UncoordinatedService:
    """E3 baseline pipeline, matched target checks; no per-key admission or lock.

    Per-channel version guards and canonical checks remain enabled. The short
    registry mutex protects diagnostic relation history, not the write pipeline.
    """
    def __init__(self,b,index):
        import threading
        self.b,self.index=b,index
        self.guard=threading.Lock()
        self.known={}
        self.verifier=SerializedUpdateService(b,CELL,index)

    def adopt(self,r,v):
        self.verifier.adopt(r,v)
        self.known[(r.scope_id,r.memory_id)]={e['relation'] for e in r.edges}

    def apply(self,r,v,hook=None):
        v,_=SerializedUpdateService._payload(r,v)
        emit=hook or (lambda *a:None)
        with self.index._scope_locks[r.scope_id]:
            if v.shape != (self.index._dense[r.scope_id].normalized_embeddings.shape[1],):
                raise ValueError('embedding dimension differs from active index')
        with self.guard:
            known=self.known[(r.scope_id,r.memory_id)]
            known.update(e['relation'] for e in r.edges)
            known=set(known)
        def canonical():
            if self.b.fetch_graph(CELL,r.scope_id,r.memory_id)!=r.graph_projection():
                raise AssertionError('canonical content mismatch')
            if self.b.candidate_projection(CELL,r.scope_id,r.memory_id)!=r.rawerk:
                raise AssertionError('projection content mismatch')
        def ready():
            if not self.verifier.verify(r,v,known):
                raise AssertionError('target readiness failed')
        operations=(('raw',lambda:self.b.commit_raw(CELL,r)),('structured',lambda:self.b.write_structured(CELL,r)),
                    ('canonical_check',canonical),('sparse',lambda:self.index.upsert_sparse(r.scope_id,r.memory_id,r.rawerk,r.version)),
                    ('dense',lambda:self.index.upsert_dense(r.scope_id,r.memory_id,v,r.version)),('ready',ready))
        for stage,fn in operations:
            emit('before',stage,time.perf_counter_ns())
            fn()
            emit('after',stage,time.perf_counter_ns())
        return dict(status='applied',version=r.version,lock_wait_ms=None,lock_hold_ms=None)
