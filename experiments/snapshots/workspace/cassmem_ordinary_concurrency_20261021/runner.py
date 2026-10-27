"""Finite open-loop runner. No forced interleaving and no automatic retries."""
import support as S
import argparse
import datetime
import gc
import os
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
import psutil

def status(stage,**kw):
    value=dict(stage=stage,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
    S.save(S.RUN/'progress.json',value)
    print(stage,kw,flush=True)

class Journal:
    """Single append-only writer; large logs never overwrite a previous attempt."""
    def __init__(self,path):
        self.path=path;self.q=queue.Queue();self.error=None;self.n=0
        self.f=path.open('x',encoding='utf-8',buffering=1)
        self.thread=threading.Thread(target=self.run);self.thread.start()
    def run(self):
        try:
            while True:
                row=self.q.get()
                if row is None:break
                self.f.write(S.json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.n+=1
        except BaseException:self.error=traceback.format_exc()
        finally:self.f.close()
    def put(self,row):self.q.put(row)
    def close(self):
        self.q.put(None);self.thread.join()
        if self.error:raise RuntimeError(self.error)

def all_targets(traces,initial,source,mv):
    merged={}
    for trace in traces.values():
        for key,vs in S.resolve(trace,initial,source,mv).items():
            entry=merged.setdefault(key,dict(record=vs[1][0],relations=set()))
            for r,_ in vs.values():entry['relations'].update(e['relation'] for e in r.edges)
    return merged

def restore_targets(b,targets):
    # A failed uncoordinated write may leave relation rows no longer reachable
    # via edges_by_memory. Clean all frozen possible keys, not just current edges.
    def reset(item):
        (scope,mid),entry=item
        for rel in sorted(entry['relations']):
            b.s.execute('DELETE FROM g_mat_by_scope_relation WHERE scope_id=%s AND relation=%s AND memory_id=%s',(scope,rel,mid))
        for table in ('g_mat_mentions_by_memory','g_mat_edges_by_memory'):
            b.s.execute(f'DELETE FROM {table} WHERE scope_id=%s AND memory_id=%s',(scope,mid))
        b.insert(S.CELL,entry['record'])
        if b.fetch_graph(S.CELL,scope,mid)!=entry['record'].graph_projection():raise RuntimeError('baseline restore failed')
    with ThreadPoolExecutor(12) as pool:list(pool.map(reset,targets.items()))

class Trial:
    def __init__(self,b,point,trace,initial,source,mv,qv,queries,dest):
        self.b,self.point,self.trace,self.dest=b,point,trace,dest
        self.records=S.resolve(trace,initial,source,mv)
        self.qv=qv;self.queries={q['query_id']:q for q in queries}
        self.index=S.ScopedOnlineRetrievalIndex()
        groups=S.collections.defaultdict(list)
        for r in initial:groups[r.scope_id].append(r)
        for scope,rs in groups.items():
            self.index.load_scope(scope,{r.memory_id:r.rawerk for r in rs},{r.memory_id:mv[r.memory_id.split('::')[-1]] for r in rs})
            self.index._scope_locks[scope]=S.TimedLock(self.index._scope_locks[scope],scope)
        self.service=(S.SerializedUpdateService(b,S.CELL,self.index) if point['arm']=='serialized' else S.UncoordinatedService(b,self.index))
        for vs in self.records.values():self.service.adopt(*vs[1])

    def execute(self,row):
        row['worker_start_ns']=time.perf_counter_ns();S.LOCAL.row=row
        t=self.trace[row['update_index']];key=(t['scope'],t['memory_id'])
        r,v=self.records[key][t['version']]
        try:
            if row['op']=='update':
                row['service_start_ns']=time.perf_counter_ns();row['service_hooks']=[]
                def hook(event,stage,ns):
                    row['service_hooks'].append(dict(event=event,stage=stage,ns=ns))
                    if stage=='lock':
                        row[{'waiting':'id_lock_wait_start_ns','acquired':'id_lock_acquired_ns','releasing':'id_lock_released_ns'}[event]]=ns
                    else:row[stage+('_start_ns' if event=='before' else '_done_ns')]=ns
                try:
                    result=self.service.apply(r,v,hook=hook)
                    row['outcome']=result['status'];row['service_result']=result
                    row['ready']=result['status']=='applied'
                except AssertionError as e:
                    row['outcome']='guard_rejected';row['guard_message']=str(e)
                finally:row['service_finish_ns']=time.perf_counter_ns()
                row['final_status']='ok'
            else:
                query=self.queries[t['query_id']]
                row['candidate_start_ns']=time.perf_counter_ns()
                ids=self.b.ids(S.CELL,t['scope']);row['candidate_done_ns']=time.perf_counter_ns()
                with self.index._scope_locks[t['scope']]:
                    sp,de=self.index._sparse[t['scope']],self.index._dense[t['scope']]
                    sv={m:self.index._sparse_versions[(t['scope'],m)] for m in sp.memory_ids}
                    dv={m:self.index._dense_versions[(t['scope'],m)] for m in de.memory_ids}
                    row['snapshot_ns']=time.perf_counter_ns()
                # All memory IDs are stable (updates, not insert/delete); filter
                # before Top-L to preserve the production candidate contract.
                allowed=set(ids)
                begin=time.perf_counter_ns()
                sparse=[x for x in S._bm25_search(sp,query['question'],len(sp.memory_ids)) if x[0] in allowed][:50]
                row['sparse_scoring_ns']=time.perf_counter_ns()-begin
                begin=time.perf_counter_ns()
                dense=[x for x in S._dense_search(de,self.qv[t['query_id']],len(de.memory_ids)) if x[0] in allowed][:50]
                row['dense_scoring_ns']=time.perf_counter_ns()-begin
                begin=time.perf_counter_ns();ranked=S.zscore_fuse(dense,sparse)
                row['fusion_ns']=time.perf_counter_ns()-begin;row['query_finish_ns']=time.perf_counter_ns()
                mid=t['memory_id'];doc=sp.documents[sp.memory_ids.index(mid)];vec=de.normalized_embeddings[de.memory_ids.index(mid)]
                row.update(query_id=t['query_id'],returned_memory_ids=[m for m,_ in ranked],
                           dense_candidates=[m for m,_ in dense],sparse_candidates=[m for m,_ in sparse],
                           returned_versions=[dict(memory_id=m,sparse=sv[m],dense=dv[m]) for m,_ in ranked],
                           target_present=mid in {m for m,_ in ranked},target_sparse_version=sv[mid],target_dense_version=dv[mid],
                           target_document_hash=S.hashlib.sha256(doc.encode()).hexdigest(),target_vector_hash=S.hash_array(vec),
                           candidate_count=len(set(m for m,_ in sparse+dense)),candidate_pool_size=len(ids),read_backend_audit=False,
                           outcome='returned',final_status='ok')
        except Exception as e:
            row.update(final_status='error',outcome='execution_error',exception_type=type(e).__name__,exception_message=str(e),traceback=traceback.format_exc())
        finally:
            row['response_finish_ns']=time.perf_counter_ns()
            row['completed_by_deadline']=row['response_finish_ns']<=row['absolute_deadline_ns']
            S.LOCAL.row=None
        return row

    def phase(self,name,offset,updates,drain):
        out=self.dest/name;out.mkdir(exist_ok=False)
        duration=updates;count=updates*20
        begin=time.perf_counter_ns()+2_000_000_000;end=begin+duration*1_000_000_000;deadline=end+drain*1_000_000_000
        ctx=dict(phase=name,operations=count,updates=updates,reads=count-updates,update_offset=offset,workers=32,ops_rate=20,
                 planned_start_ns=begin,injection_end_ns=end,absolute_deadline_ns=deadline)
        S.save(out/'context.json',ctx)
        journal=Journal(out/'requests.jsonl');submissions=Journal(out/'submissions.jsonl')
        active={};guard=threading.Lock();all_rows=[];futures=[]
        def task(row):
            with guard:active[row['event_id']]='running'
            result=self.execute(row)
            with guard:active[row['event_id']]='finished'
            journal.put(result)
            return result
        def at_deadline():
            time.sleep(max(0,(deadline-time.perf_counter_ns())/1e9))
            with guard:state=dict(active)
            S.save(out/'deadline.json',dict(deadline_ns=deadline,observed_ns=time.perf_counter_ns(),states=state))
        watcher=threading.Thread(target=at_deadline);watcher.start()
        receiver=S.subprocess.Popen([S.sys.executable,'-B',str(S.RECEIVER),'--start-ns',str(begin),'--count',str(count),
                       '--ops-rate','20','--log',str(out/'emitter.jsonl')],stdout=S.subprocess.PIPE,stderr=S.subprocess.PIPE,text=True)
        try:
            with ThreadPoolExecutor(32) as pool:
                for line in receiver.stdout:
                    ordinal,planned,emitted,enqueued=map(int,line.strip().split(','))
                    assert ordinal==len(all_rows)
                    dispatch=time.perf_counter_ns();i=offset+ordinal//20;t=self.trace[i]
                    row=dict(event_id=ordinal,phase=name,point_id=self.point['id'],op='update' if ordinal%20==0 else 'read',update_index=i,
                             scope=t['scope'],memory_id=t['memory_id'],version=t['version'],query_id=t['query_id'],
                             planned_arrival_ns=planned,emit_ns=emitted,actual_enqueue_ns=enqueued,execution_dispatch_ns=dispatch,
                             worker_start_ns=None,response_finish_ns=None,absolute_deadline_ns=deadline,ready=False)
                    all_rows.append(row)
                    submissions.put(dict(event_id=ordinal,planned_arrival_ns=planned,emit_ns=emitted,actual_enqueue_ns=enqueued,execution_dispatch_ns=dispatch))
                    with guard:active[ordinal]='queued'
                    futures.append(pool.submit(task,row))
                if receiver.wait(timeout=30):raise RuntimeError(receiver.stderr.read())
                assert len(all_rows)==count
                watcher.join()
                # Preserve finite late completions after the absolute deadline.
                # Parent process has a separate 1200 s safety timeout; censored
                # and failed points are never silently rerun.
                wait(futures)
                for f in futures:f.result()
        finally:
            if receiver.poll() is None:receiver.terminate();receiver.wait(timeout=15)
            receiver.stdout.close();receiver.stderr.close()
            watcher.join();journal.close();submissions.close()
        S.save(out/'finished.json',dict(count=journal.n,finished_ns=time.perf_counter_ns()))

    def final(self):
        final=[]
        for key,vs in sorted(self.records.items()):
            r,v=vs[max(vs)]
            relations={e['relation'] for rr,_ in vs.values() for e in rr.edges}
            obs=S.capture(self.b,self.index,key,relations)
            final.append(dict(scope=key[0],memory_id=key[1],expected_version=r.version,observation=obs))
        S.save(self.dest/'checkpoint.json',final)

def run_point(point_id,smoke):
    S.verify_manifest();S.environment_gate()
    protocol=S.load(S.RUN/'protocol.json');matrix=S.load(S.RUN/'formal_matrix.json')
    point=next(x for x in matrix if x['id']==point_id)
    trace=S.load(S.RUN/'schedules.json')[point['trace']]
    if smoke:trace=trace[:8]
    parent=S.RUN/('smoke' if smoke else 'points');parent.mkdir(exist_ok=True)
    dest=parent/point_id;dest.mkdir(exist_ok=False)
    S.save(dest/'point.json',dict(point,smoke=smoke));S.save(dest/'trace.json',trace)
    b=S.connect(protocol['keyspace'])
    try:
        status('point_initialization',point=point_id,smoke=smoke)
        source,initial,mv,qv,queries=S.data()
        restore_targets(b,all_targets(S.load(S.RUN/'schedules.json'),initial,source,mv))
        trial=Trial(b,point,trace,initial,source,mv,qv,queries,dest)
        samples=Journal(dest/'resources.jsonl');gcs=Journal(dest/'gc.jsonl');halt=threading.Event();gstarts={}
        proc=psutil.Process();proc.cpu_percent();psutil.cpu_percent()
        def sample():
            while not halt.is_set():
                start=time.perf_counter_ns()
                try:container=S.json.loads(S.docker('stats','--no-stream','--format','{{json .}}',S.CONTAINER))
                except Exception:container=None
                samples.put(dict(start_ns=start,end_ns=time.perf_counter_ns(),client_cpu=proc.cpu_percent(),client_rss=proc.memory_info().rss,
                                 host_cpu=psutil.cpu_percent(),host_ram_available=psutil.virtual_memory().available,container=container))
                halt.wait(2)
        def gccb(phase,info):
            tid=threading.get_ident()
            if phase=='start':gstarts[tid]=time.perf_counter_ns()
            else:gcs.put(dict(start_ns=gstarts.pop(tid,None),end_ns=time.perf_counter_ns(),generation=info['generation']))
        monitor=threading.Thread(target=sample);monitor.start();gc.callbacks.append(gccb)
        try:
            if not smoke:
                status('warmup',point=point_id,smoke=False);trial.phase('warmup',0,60,0)
            status('measured',point=point_id,smoke=smoke)
            trial.phase('measured',0 if smoke else 60,8 if smoke else 120,2 if smoke else 120)
            status('checkpoint',point=point_id,smoke=smoke);trial.final()
        finally:
            gc.callbacks.remove(gccb);halt.set();monitor.join();samples.close();gcs.close()
        from validate import validate_point
        result=validate_point(dest);S.save(dest/'validate.json',result)
        S.save(dest/'complete.json',dict(point_id=point_id,measurement_complete=True,validity=result['status'],smoke=smoke))
        if result['status']!='PASS':raise RuntimeError('Protocol/transport validation failed; preserve point and stop')
        if result['execution_errors']:raise RuntimeError('Infrastructure execution errors; preserve point and stop')
        status('point_complete',point=point_id,smoke=smoke,ready=result['ready_updates'],updates=result['updates'],final_mismatches=result['final_mismatches'])
    except BaseException:
        S.save(dest/'failure.json',dict(error=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));raise
    finally:b.close()

def load_database():
    S.verify_manifest();S.environment_gate()
    p=S.load(S.RUN/'protocol.json');b=S.connect(p['keyspace'])
    try:
        source,initial,mv,qv,queries=S.data()
        if S.hashlib.sha256(''.join(S.digest(r.graph_projection()) for r in initial).encode()).hexdigest()!=S.load(S.RUN/'initial_manifest.json')['records_sha256']:
            raise RuntimeError('Initial record hash drift')
        if (S.RUN/'load_complete.json').exists():raise RuntimeError('Refuse implicit repeated load')
        if b.s.execute('SELECT memory_id FROM g_mat_memory_by_scope LIMIT 1').one():raise RuntimeError('Experiment keyspace is not empty')
        status('loading_100k',loaded=0)
        with ThreadPoolExecutor(32) as pool:
            for n,_ in enumerate(pool.map(lambda r:b.insert(S.CELL,r),initial),1):
                if n%5000==0:status('loading_100k',loaded=n)
        groups=S.collections.defaultdict(list)
        for r in initial:groups[r.scope_id].append(r.memory_id)
        checks=[]
        for scope,ids in groups.items():
            observed=b.ids(S.CELL,scope)
            if sorted(ids)!=observed:raise RuntimeError('Full scope ID mismatch: '+scope)
            checks.append(dict(scope=scope,count=len(ids),ids_sha256=S.hashlib.sha256('\n'.join(observed).encode()).hexdigest()))
        for i in range(1000):
            r=initial[(i*1009)%len(initial)]
            if b.fetch_graph(S.CELL,r.scope_id,r.memory_id)!=r.graph_projection():raise RuntimeError('Initial graph sample mismatch')
        S.save(S.RUN/'load_complete.json',dict(memories=100000,scopes=171,scope_checks=checks,graph_sample=1000,keyspace=p['keyspace']))
        status('load_complete',memories=100000,scopes=171)
    finally:b.close()

def child(point_id,smoke):
    log=S.RUN/(('smoke_' if smoke else 'point_')+point_id+'.log')
    with log.open('x',encoding='utf-8') as h:
        cmd=[S.sys.executable,'-u',str(S.HERE/'runner.py'),'point','--id',point_id]
        if smoke:cmd.append('--smoke')
        p=S.subprocess.Popen(cmd,cwd=S.HERE,stdout=h,stderr=S.subprocess.STDOUT)
        S.save(S.RUN/'active_child.json',dict(pid=p.pid,point=point_id,smoke=smoke,started_ns=time.perf_counter_ns()))
        try:code=p.wait(timeout=1200)
        except S.subprocess.TimeoutExpired:
            p.terminate();p.wait(timeout=30)
            S.save(S.RUN/('timeout_'+point_id+'.json'),dict(pid=p.pid,smoke=smoke,timeout_s=1200,partial_evidence_retained=True))
            raise RuntimeError('Owned point process exceeded watchdog; no automatic retry')
        if code:raise RuntimeError(f'Point {point_id} exited {code}; inspect {log}')

def campaign():
    import validate
    S.verify_manifest();S.environment_gate()
    lock=S.RUN/'runner.lock'
    with lock.open('x',encoding='utf-8') as h:h.write(S.json.dumps(dict(pid=os.getpid(),created=time.time())))
    try:
        if not S.load(S.RUN/'unit_tests.json')['pass']:raise RuntimeError('Unit gates absent')
        if not S.load(S.RUN/'ingress_tests.json')['pass']:raise RuntimeError('Ingress gates absent')
        load_database()
        matrix=S.load(S.RUN/'formal_matrix.json')
        for point in matrix[:6]:child(point['id'],True)
        S.save(S.RUN/'smoke_complete.json',dict(points=6,validated=True,excluded_from_formal=True))
        status('formal_started',points=18)
        S.save(S.RUN/'formal_started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),points=18))
        for point in matrix:
            S.verify_manifest();S.environment_gate();child(point['id'],False)
            validate.summarize()
        S.verify_manifest();validate.summarize()
        S.save(S.RUN/'complete.json',dict(points=18,formal=True,all_points_validated=True))
        status('complete',points=18)
    except BaseException:
        S.save(S.RUN/'failure.json',dict(error=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        status('stopped_error');raise
    finally:
        S.save(S.RUN/'runner_exit.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['campaign','point','load']);ap.add_argument('--id');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.action=='campaign':campaign()
    elif a.action=='load':load_database()
    else:run_point(a.id,a.smoke)
