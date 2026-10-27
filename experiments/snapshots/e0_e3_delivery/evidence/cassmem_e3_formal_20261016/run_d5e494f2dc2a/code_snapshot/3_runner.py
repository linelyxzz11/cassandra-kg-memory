"""External hooks around unchanged v2 APIs. Append-only observations."""
import os,sys,json,time,hashlib,subprocess,secrets,threading,dataclasses,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.path.insert(0,'D:/memorytable/cassandra-kg-memory/src')
import numpy as np
from cassmem.backend.live_cells_graph import CassandraGraphCells,Neo4jGraphCells
from cassmem.representation.graph_event import GraphRecord
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex,_bm25_search,_dense_search,zscore_fuse
sys.path.insert(0,'D:/memorytable/concurrent_update_consistency')
from preflight import observation

def sh(*args):return subprocess.check_output(['docker',*args],text=True,stderr=subprocess.PIPE).strip()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Case:
    def __init__(self,b,c,f,out,protocol):
        self.b,self.c,self.f,self.out,self.p=b,c,f,out,protocol;self.lock=threading.RLock();self.seq=0
        self.scope='e3_'+out.name+'_'+c['case_id'];self.mid=f['memory_id'];self.cell=c['cell']
        self.records={i+1:GraphRecord(**{**v['record'],'scope_id':self.scope}) for i,v in enumerate(f['versions'])}
        self.vec={i+1:np.asarray(v['vector'],dtype=np.float32) for i,v in enumerate(f['versions'])}
        self.index=ScopedOnlineRetrievalIndex();self.latest=1;self.reached=threading.Event();self.release=threading.Event();self.stop=threading.Event()
    def emit(self,stream,kind,**kw):
        with self.lock:
            row=dict(case_id=self.c['case_id'],seq=self.seq,ns=time.perf_counter_ns(),kind=kind,**kw);self.seq+=1
            with (self.out/stream).open('a',encoding='utf-8') as h:h.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        return row
    def submit(self,v):
        with self.lock:
            self.latest=v
            self.emit('timeline.jsonl','submit',version=v)
    def read(self,tag):
        with self.lock:start=time.perf_counter_ns();target=self.latest
        try:
            ids=self.b.ids(self.cell,self.scope)
            with self.index._scope_locks[self.scope]:
                sp,de=self.index._sparse[self.scope],self.index._dense[self.scope]
                sv=self.index._sparse_versions[(self.scope,self.mid)];dv=self.index._dense_versions[(self.scope,self.mid)]
                snap=time.perf_counter_ns()
            sr=_bm25_search(sp,self.records[3].raw_text,50);dr=_dense_search(de,self.vec[3],50)
            sr=[x for x in sr if x[0] in ids];dr=[x for x in dr if x[0] in ids];top=zscore_fuse(dr,sr)
            query_end=time.perf_counter_ns();doc=sp.documents[sp.memory_ids.index(self.mid)];vec=de.normalized_embeddings[de.memory_ids.index(self.mid)]
            g=self.b.fetch_graph(self.cell,self.scope,self.mid);projection=self.b.candidate_projection(self.cell,self.scope,self.mid)
            allrel=sorted({r for v in self.f['versions'] for r in v['required_relations']})
            memberships={r:self.mid in self.b.ids_by_relation(self.cell,self.scope,r) for r in allrel}
            r=self.records[target];expected_rel={e['relation'] for e in r.edges}
            expected_vec=self.vec[target]/max(float(np.linalg.norm(self.vec[target])),1e-12)
            strict=g==r.graph_projection() and projection==r.rawerk and sv==target and dv==target and doc==r.rawerk and np.allclose(vec,expected_vec,rtol=1e-6,atol=1e-7) and self.mid in ids and all(v==(rel in expected_rel) for rel,v in memberships.items())
            row=dict(start_ns=start,end_ns=time.perf_counter_ns(),target_at_start=target,tag=tag,scope=self.scope,memory_id=self.mid,
                snapshot_ns=snap,query_end_ns=query_end,sparse_version=sv,dense_version=dv,sparse_document=doc,dense_vector_sha256=hashlib.sha256(vec.tobytes()).hexdigest(),
                graph=g,projection=projection,relation_memberships=memberships,ids=ids,top10=top,sparse_candidates=sr,dense_candidates=dr,strict_observed=bool(strict))
            self.emit('reads.jsonl','read',**row)
            self.emit('component_states.jsonl','snapshot',read_start_ns=start,sparse_version=sv,dense_version=dv,raw_version=g['memory']['version'] if g else -1)
            return bool(strict)
        except Exception:
            self.emit('reads.jsonl','read_error',start_ns=start,end_ns=time.perf_counter_ns(),tag=tag,error=traceback.format_exc());raise
    def update(self,v):
        r=self.records[v]
        try:
            for stage,fn in [('raw',lambda:self.b.commit_raw(self.cell,r)),('structured',lambda:self.b.write_structured(self.cell,r)),
               ('canonical_check',lambda:self.check_canonical(r)),('sparse',lambda:self.index.upsert_sparse(self.scope,self.mid,r.rawerk,v)),('dense',lambda:self.index.upsert_dense(self.scope,self.mid,self.vec[v],v))]:
                if v==2 and stage==self.c['blocked_stage']:
                    self.emit('timeline.jsonl','hook_reached',version=v,stage=stage);self.reached.set()
                    if not self.release.wait(self.p['barrier_timeout_s']):raise TimeoutError('barrier release timeout')
                    self.emit('timeline.jsonl','hook_released',version=v,stage=stage)
                self.emit('timeline.jsonl','stage_start',version=v,stage=stage);result=fn()
                self.emit('timeline.jsonl','stage_end',version=v,stage=stage,result=result)
            # Preserve original target-version check, not merely latest-version visibility.
            checked=observation(self.b,self.cell,self.records[v-1],r,self.index,self.vec[v])
            self.emit('timeline.jsonl','original_ready_check',version=v,check=checked)
            if not checked['strict_ready']:raise AssertionError('strict target check failed')
            self.emit('timeline.jsonl','ready',version=v)
        except AssertionError as exc:
            self.emit('timeline.jsonl','guard_rejection',version=v,error=str(exc))
        except Exception:
            self.emit('timeline.jsonl','execution_error',version=v,error=traceback.format_exc());raise
        finally:self.emit('timeline.jsonl','update_end',version=v)
    def check_canonical(self,r):
        assert self.b.fetch_graph(self.cell,self.scope,self.mid)==r.graph_projection(),'canonical mismatch before index update'
        assert self.b.candidate_projection(self.cell,self.scope,self.mid)==r.rawerk,'projection mismatch before index update'
    def run(self):
        self.b.insert(self.cell,self.records[1]);docs={self.mid:self.records[1].rawerk};embeds={self.mid:self.vec[1]}
        fixtures=json.loads((self.out/'fixtures.json').read_text(encoding='utf-8'))
        for fixture in fixtures[:5]:
            if fixture['memory_id']==self.mid:continue
            r=GraphRecord(**{**fixture['versions'][0]['record'],'scope_id':self.scope})
            self.b.insert(self.cell,r);docs[r.memory_id]=r.rawerk;embeds[r.memory_id]=fixture['versions'][0]['vector']
        self.index.load_scope(self.scope,docs,embeds)
        assert self.read('initial')
        def reader():
            while not self.stop.is_set():
                self.read('concurrent');self.stop.wait(self.p['reader_interval_ms']/1000)
        with ThreadPoolExecutor(3) as pool:
            rf=pool.submit(reader)
            try:
                self.submit(2);v2=pool.submit(self.update,2)
                if self.c['blocked_stage']:
                    if not self.reached.wait(self.p['barrier_timeout_s']):raise TimeoutError('v2 hook not reached')
                    self.read('v2_blocked');self.submit(3);v3=pool.submit(self.update,3);v3.result(timeout=self.p['worker_timeout_s'])
                    assert self.read('v3_before_release'),'v3 not strictly ready before release'
                    self.emit('timeline.jsonl','release_authorized',version=2);self.release.set();v2.result(timeout=self.p['worker_timeout_s'])
                else:
                    v2.result(timeout=self.p['worker_timeout_s']);self.submit(3);pool.submit(self.update,3).result(timeout=self.p['worker_timeout_s'])
            finally:self.release.set();self.stop.set();rf.result(timeout=60)
        for i in range(self.p['post_final_probes']):self.read('final_'+str(i))
        self.emit('timeline.jsonl','case_complete')

def main(out):
    p=json.loads((out/'protocol.json').read_text(encoding='utf-8'));manifest=json.loads((out/'source_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(k)==v for k,v in manifest.items()),'frozen source drift'
    assert not (out/'timeline.jsonl').exists(),'Refuse overwrite or implicit resume'
    fixtures=json.loads((out/'fixtures.json').read_text(encoding='utf-8'));cases=json.loads((out/'schedules.json').read_text(encoding='utf-8'))
    runid=time.strftime('%Y%m%d%H%M%S');original=sh('ps','--format','{{.Names}}').splitlines();save(out/'containers_before.json',original)
    for engine in ('cassandra','neo4j'):
        name='cassmem-e3-smoke-'+runid+'-'+engine;port=39042 if engine=='cassandra' else 37687
        image='2c6576f9b8e0' if engine=='cassandra' else '0b5d3ab6ec1b';password=secrets.token_urlsafe(24)
        args=['run','-d','--name',name,'--memory','3g','--cpus','2','-p',f'127.0.0.1:{port}:'+('9042' if engine=='cassandra' else '7687'),'-v',name+'-data:'+('/var/lib/cassandra' if engine=='cassandra' else '/data')]
        args+=['-e','MAX_HEAP_SIZE=1G','-e','HEAP_NEWSIZE=256M'] if engine=='cassandra' else ['-e','NEO4J_AUTH=neo4j/'+password,'-e','NEO4J_server_memory_heap_initial__size=1G','-e','NEO4J_server_memory_heap_max__size=1G','-e','NEO4J_server_memory_pagecache_size=512M']
        sh(*args,image);print('database_starting',engine,name,flush=True);b=None
        try:
            for attempt in range(90):
                try:
                    if engine=='cassandra':
                        # Probe first, avoid leaking driver threads during startup.
                        sh('exec',name,'cqlsh','-e','SELECT release_version FROM system.local;')
                        os.environ['CASSANDRA_PORT']=str(port);b=CassandraGraphCells(keyspace='e3_smoke_'+runid)
                        from cassandra import ConsistencyLevel
                        b.s.default_timeout=30;b.s.default_consistency_level=ConsistencyLevel.LOCAL_ONE
                        for ps in b.p.values():ps.consistency_level=ConsistencyLevel.LOCAL_ONE
                    else:
                        from neo4j import GraphDatabase
                        with GraphDatabase.driver(f'bolt://127.0.0.1:{port}',auth=('neo4j',password),connection_timeout=3) as d:d.verify_connectivity()
                        b=Neo4jGraphCells(f'bolt://127.0.0.1:{port}','neo4j',password)
                    break
                except Exception as exc:
                    if attempt==89:raise
                    time.sleep(2)
            info=json.loads(sh('inspect',name))[0]
            version=sh('exec',name,'cqlsh','-e','SELECT release_version FROM system.local;') if engine=='cassandra' else [dict(r) for r in b.run('CALL dbms.components() YIELD name,versions RETURN name,versions')]
            save(out/(engine+'_environment.json'),dict(name=name,image=info['Image'],memory=info['HostConfig']['Memory'],cpus=info['HostConfig']['NanoCpus'],version=version,port=port))
            for c in cases:
                if not c['cell'].startswith(engine):continue
                Case(b,c,fixtures[c['fixture_id']],out,p).run();print('case_complete',c['case_id'],flush=True)
        finally:
            if b:b.close()
            sh('stop','-t','30',name)
    assert all(sha(k)==v for k,v in manifest.items()),'source drift after execution'
    save(out/'complete.json',dict(cases=len(cases),validation_pending=True,original_containers_still_running=all(n in sh('ps','--format','{{.Names}}').splitlines() for n in original)))
if __name__=='__main__':
    out=Path(sys.argv[1])
    try:main(out)
    except Exception:
        save(out/'failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
