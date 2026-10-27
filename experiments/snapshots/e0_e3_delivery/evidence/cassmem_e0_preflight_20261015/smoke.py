"""Open-loop E0 smoke only. Shared workers, fixed absolute deadline, full logs."""
import os,sys,json,time,threading,hashlib,subprocess,traceback,gc,datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from environment import start,stop,suspend_old,restore,save,docker
from instrumentation import TimedLock,LOCAL,classify
import numpy as np
import psutil
from cassmem.representation.graph_event import GraphRecord
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex,_bm25_search,_dense_search,zscore_fuse
from cassmem.serving.serialized_updates import SerializedUpdateService
EMITTER='D:/memorytable/representation_serving_cost/arrival_emitter.py'
def logrows(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')

class Trial:
    def __init__(self,b,cell,fixtures,out,protocol):
        self.b,self.cell,self.fixtures,self.out,self.p=b,cell,fixtures,out,protocol;self.scope='e0_smoke_scope'
        self.records=[];self.vectors=[]
        for f in fixtures:
            self.records.append({i+1:GraphRecord(**{**v['record'],'scope_id':self.scope}) for i,v in enumerate(f['versions'])})
            self.vectors.append({i+1:np.asarray(v['vector'],dtype=np.float32) for i,v in enumerate(f['versions'])})
        for versions in self.records:b.insert(cell,versions[1])
        self.index=ScopedOnlineRetrievalIndex();self.index.load_scope(self.scope,{v[1].memory_id:v[1].rawerk for v in self.records},{v[1].memory_id:self.vectors[i][1] for i,v in enumerate(self.records)})
        self.index._scope_locks[self.scope]=TimedLock(self.index._scope_locks[self.scope],self.scope)
        self.service=SerializedUpdateService(b,cell,self.index)
        for i,vs in enumerate(self.records):self.service.adopt(vs[1],self.vectors[i][1],known_relations={r for v in fixtures[i]['versions'] for r in v['required_relations']})
    def capture(self,i):
        r=self.records[i][2];mid=r.memory_id
        with self.index._scope_locks[self.scope]:
            sp,de=self.index._sparse[self.scope],self.index._dense[self.scope]
            sv=self.index._sparse_versions[(self.scope,mid)];dv=self.index._dense_versions[(self.scope,mid)]
            doc=sp.documents[sp.memory_ids.index(mid)];vec=de.normalized_embeddings[de.memory_ids.index(mid)]
        return dict(scope=self.scope,memory_id=mid,graph=self.b.fetch_graph(self.cell,self.scope,mid),projection=self.b.candidate_projection(self.cell,self.scope,mid),
            sparse_version=sv,dense_version=dv,sparse_document=doc,dense_vector_sha256=hashlib.sha256(vec.tobytes()).hexdigest(),
            relation_memberships={rel:mid in self.b.ids_by_relation(self.cell,self.scope,rel) for rel in sorted({x for v in self.fixtures[i]['versions'] for x in v['required_relations']})},
            ids=self.b.ids(self.cell,self.scope))
    def phase(self,phase,duration,offset):
        rate=self.p['smoke']['rate'];count=int(duration*rate)*20;rows=[];subs=[];guard=threading.Lock();stages={};done_flags={}
        planned_start=time.perf_counter_ns()+2_000_000_000;injection_end=planned_start+int(duration*1e9);deadline=injection_end+int(self.p['smoke']['drain_s']*1e9)
        phaseout=self.out/phase;phaseout.mkdir()
        context=dict(phase=phase,planned_start_ns=planned_start,injection_end_ns=injection_end,absolute_deadline_ns=deadline,operations=count,updates=count//20,reads=count-count//20,
            utc_started=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=32,update_rate=rate)
        save(phaseout/'context.json',context)
        def execute(row):
            i=row['fixture_id'];r=self.records[i][2];mid=r.memory_id
            with guard:row['worker_start_ns']=time.perf_counter_ns();stages[row['event_id']]='started'
            LOCAL.row=row
            try:
                if row['op']=='update':
                    row['service_hooks']=[]
                    def hook(event,stage,ns):
                        row['service_hooks'].append(dict(event=event,stage=stage,ns=ns))
                        if stage=='lock':
                            row[{'waiting':'id_lock_wait_start_ns','acquired':'id_lock_acquired_ns','releasing':'id_lock_released_ns'}[event]]=ns
                        else:
                            row[stage+('_start_ns' if event=='before' else '_done_ns')]=ns
                    result=self.service.apply(r,self.vectors[i][2],hook=hook)
                    if result['status']!='applied':raise AssertionError('Unexpected revision result')
                    row.update(raw_call_start_ns=row['raw_start_ns'],raw_ack_ns=row['raw_done_ns'],ready_check_start_ns=row['ready_start_ns'],ready_check_done_ns=row['ready_done_ns'],
                        relation_cleanup_start_ns=None,relation_cleanup_done_ns=None,structured_maintenance_start_ns=row['structured_start_ns'],structured_maintenance_done_ns=row['structured_done_ns'])
                    row['post_ready_audit_start_ns']=time.perf_counter_ns();row['audit']=self.capture(i);row['post_ready_audit_done_ns']=time.perf_counter_ns()
                    query=self.fixtures[i]['query'];row['post_update_query_start_ns']=time.perf_counter_ns()
                    ranked=self.index.search(self.scope,query['question'],query['vector'],row['audit']['ids'])
                    row['post_update_query_done_ns']=time.perf_counter_ns();row['post_update_top10']=[m for m,_ in ranked.top10];row['post_update_target_present']=mid in row['post_update_top10']
                    row['ready']=True
                else:
                    query=self.fixtures[i]['query'];ids=self.b.ids(self.cell,self.scope);row['candidate_ids_done_ns']=time.perf_counter_ns()
                    with self.index._scope_locks[self.scope]:
                        sp,de=self.index._sparse[self.scope],self.index._dense[self.scope]
                        sv={m:self.index._sparse_versions[(self.scope,m)] for m in sp.memory_ids};dv={m:self.index._dense_versions[(self.scope,m)] for m in de.memory_ids}
                        row['snapshot_ns']=time.perf_counter_ns()
                    t=time.perf_counter_ns();sr=_bm25_search(sp,query['question'],50);row['sparse_scoring_ns']=time.perf_counter_ns()-t
                    t=time.perf_counter_ns();dr=_dense_search(de,query['vector'],50);row['dense_scoring_ns']=time.perf_counter_ns()-t
                    t=time.perf_counter_ns();ranking=zscore_fuse([x for x in dr if x[0] in ids],[x for x in sr if x[0] in ids]);row['fusion_ns']=time.perf_counter_ns()-t
                    row.update(query_id=query['query_id'],query_finish_ns=time.perf_counter_ns(),returned_memory_ids=[m for m,_ in ranking],
                        returned_versions=[dict(memory_id=m,sparse=sv[m],dense=dv[m]) for m,_ in ranking],target_present=mid in {m for m,_ in ranking})
                    row['audit']=self.capture(i)
                row['final_status']='ok'
            except Exception as exc:row.update(final_status='error',exception_type=type(exc).__name__,exception_message=str(exc),traceback=traceback.format_exc())
            finally:
                row['driver_pool_wait_ms']=None
                row['raw_version']=row.get('audit',{}).get('graph',{}).get('memory',{}).get('version',-1)
                row['relation_version']=None  # No persisted relation version exists in this adapter.
                if row.get('scope_locks'):
                    row['scope_lock_wait_start_ns']=row['scope_locks'][0]['wait_start_ns'];row['scope_lock_acquired_ns']=row['scope_locks'][0]['acquired_ns'];row['scope_lock_released_ns']=row['scope_locks'][-1]['released_ns']
                with guard:row['response_finish_ns']=time.perf_counter_ns();stages[row['event_id']]='finished';done_flags[row['event_id']]=row['response_finish_ns']
                LOCAL.row=None
            return row
        def deadline_watch():
            delay=max(0,(deadline-time.perf_counter_ns())/1e9);time.sleep(delay)
            with guard:snapshot=dict(stages);finishes=dict(done_flags)
            save(phaseout/'deadline.json',dict(absolute_deadline_ns=deadline,observed_ns=time.perf_counter_ns(),states=snapshot,finish_times=finishes))
        watcher=threading.Thread(target=deadline_watch);watcher.start()
        with ThreadPoolExecutor(32) as pool:
            emitter=subprocess.Popen([sys.executable,'-B',EMITTER,'--start-ns',str(planned_start),'--count',str(count),'--ops-rate',str(rate*20),'--log',str(phaseout/'emitter.jsonl')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            futures=[]
            try:
                for line in emitter.stdout:
                    ordinal,scheduled,emitted=map(int,line.strip().split(','));assert ordinal==len(rows)
                    enqueued=time.perf_counter_ns();i=offset+ordinal//20
                    row=dict(run_id=self.cell,phase=phase,event_id=ordinal,associated_update_id=(ordinal//20)*20,op='update' if ordinal%20==0 else 'read',fixture_id=i,
                        target_memory_id=self.records[i][2].memory_id,target_version=2,query_scope=self.scope,planned_arrival_ns=scheduled,emit_ns=emitted,actual_enqueue_ns=enqueued,
                        worker_start_ns=None,response_finish_ns=None,absolute_deadline_ns=deadline,ready=False,exception_type=None,exception_message=None)
                    with guard:stages[ordinal]='queued'
                    rows.append(row);subs.append(dict(event_id=ordinal,planned_arrival_ns=scheduled,emit_ns=emitted,actual_enqueue_ns=enqueued));futures.append(pool.submit(execute,row))
                if emitter.wait(timeout=15):raise RuntimeError(emitter.stderr.read())
                watcher.join();wait(futures)
                for f in futures:f.result()
            finally:
                if emitter.poll() is None:emitter.terminate();emitter.wait(timeout=10)
                # The finite diagnostic deadline watcher preserves already-offered states on failure.
                watcher.join()
                for row in rows:row['deadline_class']=classify(row,deadline)
                logrows(phaseout/'events.jsonl',[r for r in rows if r['op']=='update']);logrows(phaseout/'reads.jsonl',[r for r in rows if r['op']=='read']);logrows(phaseout/'submissions.jsonl',subs)
        return rows

def main(out):
    p=json.loads((out/'protocol.json').read_text(encoding='utf-8'));manifest=json.loads((out/'source_manifest.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in manifest.items())
    fixtures=json.loads((out/'fixtures.json').read_text(encoding='utf-8'));state=suspend_old(out)
    try:
        for engine in ('cassandra','neo4j'):
            b,name=start(engine,out)
            try:
                for cell in [c for c in p['cells'] if c.startswith(engine)]:
                    dest=out/cell;dest.mkdir(exist_ok=False);trial=Trial(b,cell,fixtures,dest,p)
                    samples=[];gc_events=[];gc_start={};halt=threading.Event();proc=psutil.Process();proc.cpu_percent();psutil.cpu_percent()
                    def gccallback(phase,info):
                        key=threading.get_ident()
                        if phase=='start':gc_start[key]=time.perf_counter_ns()
                        else:gc_events.append(dict(start_ns=gc_start.pop(key,None),end_ns=time.perf_counter_ns(),generation=info['generation'],collected=info['collected']))
                    def sample():
                        while not halt.is_set():
                            t=time.perf_counter_ns()
                            try:container=json.loads(docker('stats','--no-stream','--format','{{json .}}',name))
                            except Exception:container=None
                            samples.append(dict(start_ns=t,end_ns=time.perf_counter_ns(),host_cpu_percent=psutil.cpu_percent(),host_ram_available=psutil.virtual_memory().available,client_cpu_percent=proc.cpu_percent(),client_rss=proc.memory_info().rss,container=container))
                            halt.wait(1)
                    monitor=threading.Thread(target=sample);monitor.start();gc.callbacks.append(gccallback)
                    try:
                        trial.phase('warmup',p['smoke']['warmup_s'],0)
                        trial.phase('measured',p['smoke']['injection_s'],2)
                        checkpoint=[dict(fixture_id=i,audit=trial.capture(i)) for i in range(8)];save(dest/'checkpoint.json',checkpoint)
                    finally:
                        gc.callbacks.remove(gccallback);halt.set();monitor.join();logrows(dest/'resource_usage.jsonl',samples);logrows(dest/'gc.jsonl',gc_events)
                    print('cell_complete',cell,flush=True)
            finally:stop(b,name)
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in manifest.items())
        save(out/'complete.json',dict(cells=p['cells'],formal_started=False,validation_pending=True))
    finally:restore(state,out)
if __name__=='__main__':
    out=Path(sys.argv[1])
    try:main(out)
    except Exception:save(out/'failure.json',dict(error=traceback.format_exc()));raise
