"""Frozen-trace E0 campaign. Reuses audited phase clocks, not smoke data."""
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRE=Path('D:/memorytable/cassmem_e0_preflight_20261015')
FROZEN=PRE/'e0_0246002cc853'
sys.path.insert(0,str(PRE))
import smoke as S
import json,time,hashlib,inspect,textwrap,dataclasses,collections,threading,gc,traceback,datetime,shutil
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'D:/memorytable/representation_serving_cost')
from common import build_graph_records,vectors,rows,ROOT,write_csv
from run_graph_100k_load_gate_v2 import expand
from environment import save,start,stop,suspend_old,restore,docker
from instrumentation import TimedLock
from cassmem.representation.graph_event import GraphRecord
from cassmem.serving.serialized_updates import SerializedUpdateService
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex
import numpy as np
import psutil

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(m):
    for p,h in m.items():
        if sha(p)!=h:raise RuntimeError('Frozen source drift: '+p)

def phase_source():
    # The substitutions only adapt fixture scope/rate/deadline/run ID. Core clocks stay unchanged.
    s=textwrap.dedent(inspect.getsource(S.Trial.phase))
    s=s.replace("self.p['smoke']","self.p['active']")
    s=s.replace('self.scope','self.records[i][2].scope_id')
    s=s.replace('run_id=self.cell','run_id=self.run_id,layout=self.cell')
    s=s.replace("row['driver_pool_wait_ms']=None", "row['driver_pool_wait_ms']=None\n            row['target_content_match']=row.get('audit',{}).get('graph')==r.graph_projection()")
    s=s.replace("for row in rows:row['deadline_class']", "emitter.stdout.close();emitter.stderr.close()\n            for row in rows:row['deadline_class']")
    return s

scope=dict(vars(S));exec(compile(phase_source(),'<frozen-formal-phase>','exec'),scope)

def version(r,vector):
    v=np.asarray(vector,dtype=np.float32);n=v/max(float(np.linalg.norm(v)),1e-12)
    return dict(record=dataclasses.asdict(r),graph=r.graph_projection(),rawerk=r.rawerk,
                vector=v.tolist(),normalized_vector_sha256=hashlib.sha256(n.tobytes()).hexdigest(),
                required_relations=sorted({e['relation'] for e in r.edges}))

class FormalTrial:
    phase=scope['phase']
    def __init__(self,b,cell,initial,mv,fixtures,out,p,run_id):
        self.b,self.cell,self.fixtures,self.out,self.p,self.run_id=b,cell,fixtures,out,p,run_id
        self.records=[{i+1:GraphRecord(**v['record']) for i,v in enumerate(f['versions'])} for f in fixtures]
        self.vectors=[{i+1:np.asarray(v['vector'],dtype=np.float32) for i,v in enumerate(f['versions'])} for f in fixtures]
        groups=collections.defaultdict(list)
        for r in initial:groups[r.scope_id].append(r)
        self.index=ScopedOnlineRetrievalIndex()
        for scope_id,rs in groups.items():
            self.index.load_scope(scope_id,{r.memory_id:r.rawerk for r in rs},{r.memory_id:mv[r.memory_id.split('::')[-1]] for r in rs})
            self.index._scope_locks[scope_id]=TimedLock(self.index._scope_locks[scope_id],scope_id)
        self.service=SerializedUpdateService(b,cell,self.index)
        for i,rs in enumerate(self.records):
            self.service.adopt(rs[1],self.vectors[i][1],known_relations={r for v in fixtures[i]['versions'] for r in v['required_relations']})
    def capture(self,i):
        r=self.records[i][2];mid=r.memory_id;s=r.scope_id
        with self.index._scope_locks[s]:
            sp,de=self.index._sparse[s],self.index._dense[s]
            sv=self.index._sparse_versions[(s,mid)];dv=self.index._dense_versions[(s,mid)]
            doc=sp.documents[sp.memory_ids.index(mid)];vec=de.normalized_embeddings[de.memory_ids.index(mid)]
        return dict(scope=s,memory_id=mid,graph=self.b.fetch_graph(self.cell,s,mid),projection=self.b.candidate_projection(self.cell,s,mid),
            sparse_version=sv,dense_version=dv,sparse_document=doc,dense_vector_sha256=hashlib.sha256(vec.tobytes()).hexdigest(),
            relation_memberships={rel:mid in self.b.ids_by_relation(self.cell,s,rel) for rel in sorted({x for v in self.fixtures[i]['versions'] for x in v['required_relations']})},ids=self.b.ids(self.cell,s))

def freeze():
    verify(load(FROZEN/'source_manifest.json'))
    p=load(FROZEN/'protocol.json');p.pop('no_formal_launch');p.pop('code_hashes');p.pop('fixture_sha');p.pop('smoke')
    p['experiment_id']='E0-formal-serialized-20261015'
    p['preflight']=str(FROZEN)
    p['formal_clarifications']={
        'readiness':'Exact target raw/graph/projection/sparse/dense and relation content, plus scope candidate membership. Natural-query Top10 presence separately reported, not required for readiness.',
        'point_reset':'Load 100K once per layout into isolated engine; before next point restore every previously touched record raw+structure to frozen v1 and rebuild all index scopes. Verify restored graph/projection and all target adoption.',
        'audit_overhead':'Reads include post-query component audit; response_finish includes audit. Query_finish also retained separately. Updates readiness excludes post-ready audit and ranking.',
        'phase_reuse':'Mechanically adapt audited smoke phase to per-record scope and formal timing. Generated source is saved and hashed.',
        'failure_policy':'Preserve FAILED/INVALID points. No automatic replacement or deletion; infrastructure exception stops campaign.',
        'concurrency':'E0 exclusive same-host DB experiment; E3 not simultaneous.'}
    p['formal_clarifications']['warmup_drain']='Warmup has zero extra fixed drain; await its finite requests before measured phase. Measured absolute drain remains 120 seconds.'
    code={str(x):sha(x) for x in HERE.glob('*.py')}
    p['runner_hashes']=code
    ident=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:12]
    out=HERE/('run_'+ident);out.mkdir(exist_ok=False)
    save(out/'protocol.json',p)
    for name in ('formal_matrix.json','formal_update_traces.json','environment.json'):shutil.copy2(FROZEN/name,out/name)
    (out/'phase_generated.py').write_text(phase_source(),encoding='utf-8')
    manifest={**load(FROZEN/'source_manifest.json'),**code}
    for name in ('protocol.json','formal_matrix.json','formal_update_traces.json','phase_generated.py'):manifest[str(out/name)]=sha(out/name)
    manifest[str(Path('D:/memorytable/representation_serving_cost/common.py'))]=sha('D:/memorytable/representation_serving_cost/common.py')
    save(out/'source_manifest.json',manifest)
    (out/'code_snapshot').mkdir()
    for x in HERE.glob('*.py'):shutil.copy2(x,out/'code_snapshot'/x.name)
    print(out)

def campaign(out):
    manifest=load(out/'source_manifest.json');verify(manifest)
    p=load(out/'protocol.json');matrix=load(out/'formal_matrix.json');traces=load(out/'formal_update_traces.json')
    assert phase_source()==(out/'phase_generated.py').read_text(encoding='utf-8')
    def status(stage,**kw):
        save(out/'progress.json',dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),**kw));print(stage,kw,flush=True)
    source=build_graph_records();initial=expand(source);assert len(initial)==100000
    assert all(r.version==1 for r in initial)
    original={r.memory_id:r for r in initial};sourceby={r.memory_id:r for r in source};mv=vectors('memory');qv=vectors('qa');qs={q['query_id']:q for q in rows(ROOT/'data/frozen_retrieval/p3_queries.csv')}
    fixtures={}
    for key,trace in traces.items():
        fs=[]
        for t in trace:
            old=original[t['memory_id']];new=dataclasses.replace(sourceby[t['source_memory_id']],memory_id=old.memory_id,scope_id=old.scope_id,version=2)
            assert old.scope_id==t['scope'];q=qs[t['query_id']]
            fs.append(dict(memory_id=old.memory_id,scope=old.scope_id,versions=[version(old,mv[old.memory_id.split('::')[-1]]),version(new,mv[t['source_memory_id']])],query=dict(query_id=t['query_id'],question=q['question'],vector=qv[t['query_id']].tolist())))
        fixtures[key]=fs
    save(out/'resolved_fixtures.json',fixtures);save(out/'resolved_fixture_sha.json',{'sha256':sha(out/'resolved_fixtures.json')})
    state=suspend_old(out)
    try:
        for engine in ('cassandra','neo4j'):
            b,name=start(engine,out)
            try:
                for cell in [c for c in p['cells'] if c.startswith(engine)]:
                    # The only reset target is this newly created isolated container.
                    b.reset();status('loading_100k',layout=cell,loaded=0)
                    if engine=='cassandra':
                        with ThreadPoolExecutor(32) as pool:
                            for n,_ in enumerate(pool.map(lambda r:b.insert(cell,r),initial),1):
                                if n%5000==0:status('loading_100k',layout=cell,loaded=n)
                    else:
                        for offset in range(0,len(initial),5000):
                            b.insert_many(cell,initial[offset:offset+5000],batch=100,workers=12);status('loading_100k',layout=cell,loaded=offset+5000)
                    grouped=collections.defaultdict(list)
                    for r in initial:grouped[r.scope_id].append(r.memory_id)
                    assert all(sorted(b.ids(cell,s))==sorted(ids) for s,ids in grouped.items())
                    save(out/(cell+'_load_gate.json'),dict(memories=100000,scopes=len(grouped),all_scope_ids_match=True))
                    # Formal adapter integration gate at full loaded scope sizes; excluded from 36 points.
                    gate=out/(cell+'_integration_smoke');gate.mkdir()
                    gatefs=fixtures['0.25_20261015'][:2]
                    gtrial=FormalTrial(b,cell,initial,mv,gatefs,gate,{'active':{'rate':1,'drain_s':10}},cell+'-integration')
                    save(gate/'fixtures.json',gatefs);status('integration_smoke',layout=cell)
                    gtrial.phase('measured',2,0)
                    from validator import audit_phase,lines
                    gd=gate/'measured';gcheck=audit_phase(load(gd/'context.json'),lines(gd/'events.jsonl')+lines(gd/'reads.jsonl'),lines(gd/'submissions.jsonl'),gatefs)
                    save(gate/'validate.json',gcheck)
                    if gcheck['errors'] or gcheck['execution_errors'] or not gcheck['all_updates_eventually_ready']:raise RuntimeError('Formal adapter integration smoke failed; evidence retained')
                    for f in gatefs:
                        r=original[f['memory_id']];b.commit_raw(cell,r);b.write_structured(cell,r)
                        assert b.fetch_graph(cell,r.scope_id,r.memory_id)==r.graph_projection()
                    del gtrial;gc.collect()
                    touched=[]
                    for point in [x for x in matrix if x['layout']==cell]:
                        verify(manifest)
                        for r in touched:
                            b.commit_raw(cell,r);b.write_structured(cell,r)
                            assert b.fetch_graph(cell,r.scope_id,r.memory_id)==r.graph_projection()
                            assert b.candidate_projection(cell,r.scope_id,r.memory_id)==r.rawerk
                        run_id=f"{cell}_r{point['rate']}_s{point['seed']}";dest=out/run_id;dest.mkdir(exist_ok=False)
                        active={**p,'active':dict(rate=point['rate'],drain_s=120)};fs=fixtures[point['trace']]
                        save(dest/'point.json',point);save(dest/'fixtures.json',fs)
                        status('initializing_point',run_id=run_id)
                        trial=FormalTrial(b,cell,initial,mv,fs,dest,active,run_id)
                        touched=[original[f['memory_id']] for f in fs]
                        samples=[];gcs=[];gstart={};halt=threading.Event();proc=psutil.Process();proc.cpu_percent();psutil.cpu_percent()
                        def gccb(phase,info):
                            key=threading.get_ident()
                            if phase=='start':gstart[key]=time.perf_counter_ns()
                            else:gcs.append(dict(start_ns=gstart.pop(key,None),end_ns=time.perf_counter_ns(),generation=info['generation'],collected=info['collected']))
                        def sample():
                            while not halt.is_set():
                                t=time.perf_counter_ns()
                                try:container=json.loads(docker('stats','--no-stream','--format','{{json .}}',name))
                                except Exception:container=None
                                samples.append(dict(start_ns=t,end_ns=time.perf_counter_ns(),host_cpu_percent=psutil.cpu_percent(),host_ram_available=psutil.virtual_memory().available,client_cpu_percent=proc.cpu_percent(),client_rss=proc.memory_info().rss,container=container));halt.wait(1)
                        monitor=threading.Thread(target=sample);monitor.start();gc.callbacks.append(gccb)
                        try:
                            active['active']['drain_s']=0
                            status('warmup',run_id=run_id);trial.phase('warmup',60,0)
                            active['active']['drain_s']=120
                            status('measured',run_id=run_id);trial.phase('measured',120,int(60*point['rate']))
                            save(dest/'checkpoint.json',[dict(fixture_id=i,audit=trial.capture(i)) for i in range(len(fs))])
                        finally:
                            gc.callbacks.remove(gccb);halt.set();monitor.join();S.logrows(dest/'resource_usage.jsonl',samples);S.logrows(dest/'gc.jsonl',gcs)
                        from validator import validate_point
                        result=validate_point(dest);save(dest/'validate.json',result);status('point_complete',run_id=run_id,result=result['status'])
                        del trial;gc.collect()
            finally:stop(b,name)
        verify(manifest)
        from validator import summarize
        summarize(out);save(out/'complete.json',dict(matrix_executed=True,formal=True,points=36))
        status('complete')
    finally:restore(state,out)

if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:
        out=Path(sys.argv[1])
        try:campaign(out)
        except Exception:save(out/'failure.json',dict(error=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));raise
