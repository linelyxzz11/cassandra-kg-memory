"""Isolated Cassandra access-path campaign; no QA/gold/model calls.

prepare is offline. run owns only containers created via the E0 environment helper.
Original data, source, and all previous measurements are read-only.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, json, time, hashlib, random, dataclasses, collections, traceback, platform, gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).resolve().parent
ROOT = Path('D:/memorytable/cassandra-kg-memory')
sys.path[:0] = ['D:/memorytable/representation_serving_cost',
               'D:/memorytable/cassmem_e0_preflight_20261015', str(ROOT/'src')]
from common import build_graph_records
from run_graph_100k_load_gate_v2 import expand
from cassmem.representation.graph_event import GraphRecord
sys.path.insert(0,'D:/memorytable/cassmem_e0_preflight_20261015')
from environment import start, stop, suspend_old, restore, docker
import numpy as np
import psutil

SEEDS = [20261015, 20261016, 20261017]
CELLS = ['cassandra-base','cassandra-materialized']
KINDS = ['keywords','replace','append','delete']
OUT = HERE/'run_v1'

def save(p, value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def status(stage, **kw):
    value=dict(stage=stage,pid=os.getpid(),unix_time=time.time(),**kw)
    save(OUT/'status.json',value); print(json.dumps(value),flush=True)
def lines(p, rows):
    with Path(p).open('a',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def distribution(xs):
    a=np.asarray(xs,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)))

def resources():
    return dict(unix_time=time.time(),host_available_bytes=psutil.virtual_memory().available,
                process_rss_bytes=psutil.Process().memory_info().rss,gc_counts=list(gc.get_count()))

def eligible(r):
    es=r.edges
    if not es or sum(e['relation']==es[-1]['relation'] for e in es)!=1: return False
    remaining=set(r.mentions)|{v for e in es[:-1] for v in (e['src'],e['dst'])}
    return remaining==set(r.graph_entities)

def mutate(r, kind):
    es=[dict(e) for e in r.edges]
    unique='cap_relation_'+hashlib.sha256(r.memory_id.encode()).hexdigest()[:16]
    if kind=='keywords': return dataclasses.replace(r,version=2,keywords=r.keywords+', cap_keyword')
    if kind=='replace': es[-1]['relation']=unique
    elif kind=='append': es.append(dict(es[-1],ordinal=len(es),relation=unique))
    elif kind=='delete': es=es[:-1]
    else: raise ValueError(kind)
    triples='; '.join('('+e['src']+', '+e['relation']+', '+e['dst']+')' for e in es)
    nr=dataclasses.replace(r,version=2,triples=triples,relations=', '.join(sorted({e['relation'] for e in es})))
    assert nr.graph_entities==r.graph_entities
    assert nr.raw_text==r.raw_text and nr.embedding_sha256==r.embedding_sha256
    assert len(nr.edges)==len(es)
    return nr

def relation_map(records):
    result=collections.defaultdict(set); edge_count=collections.Counter(); scope_edges=collections.Counter()
    for r in records:
        for e in r.edges:
            key=(r.scope_id,e['relation']); result[key].add(r.memory_id)
            edge_count[key]+=1; scope_edges[r.scope_id]+=1
    return {k:sorted(v) for k,v in result.items()},edge_count,scope_edges

def prepare():
    if OUT.exists(): raise RuntimeError('Existing campaign retained; refuse overwrite')
    rs=expand(build_graph_records()); assert len(rs)==100000
    first=[r for r in rs if r.memory_id.startswith('lr000000::')]
    pool=[r for r in first if eligible(r)]
    # Match classes within groups of four adjacent complexity-sorted source records.
    pool.sort(key=lambda r:(len(r.edges),len(r.rawerk),r.scope_id,r.memory_id))
    available_groups=len(pool)//4
    warm=min(50,max(1,available_groups//5))
    measured=min(200,available_groups-warm)
    if measured<50: raise RuntimeError('Insufficient diverse eligible source records; user decision required')
    OUT.mkdir()
    save(OUT/'protocol.json',dict(experiment='Cassandra access-path breakdown',seeds=SEEDS,cells=CELLS,
         source_records=len(first),memories=100000,eligible_source_records=len(pool),
         update_warmup=warm,update_measured=measured,read_measured=5000,
         points=36,application_workers=1,load_workers=32,cache='request-pool warmup, no OS flush',
         resources=dict(cpus=8,memory_gib=12,heap='6G',image='2c6576f9b8e0',RF=1,consistency='LOCAL_ONE'),
         timing='client-observed method entry to complete result; structured includes old-state reads and batch construction',
         overhead_gate='read paired blocks, median relative overhead <=5%; otherwise stop before formal results',
         mutation='same-ID v1->v2 once; original text, vector hash, graph entity set unchanged',
         interpretation='actual two layouts, not single-factor materialization effect; micro does not measure retrieval-ready',
         failures='preserve all records and stop; no automatic replacements',
         relation_pages='null unless directly observed; logical CQL calls do not equal network requests',
         cohort_policy='historical E0 main/audit/ingress never pooled'))
    # Independent literal oracle, without using backend retrieval to generate expectations.
    rels,ec,se=relation_map(first)
    byscope=collections.defaultdict(list)
    for r in first: byscope[r.scope_id].append(r)
    manifests={}
    for seed in SEEDS:
        rng=random.Random(seed)
        groups=[pool[i:i+4] for i in range(0,len(pool)-3,4)]
        rng.shuffle(groups); groups=groups[:warm+measured]
        updates={k:[] for k in KINDS}
        for gi,group in enumerate(groups):
            labels=KINDS.copy(); rng.shuffle(labels)
            for r,k in zip(group,labels):
                nr=mutate(r,k)
                updates[k].append(dict(old=dataclasses.asdict(r),new=dataclasses.asdict(nr),
                    phase='warmup' if gi<warm else 'measured',source_id=r.memory_id.split('::',1)[1],
                    source_scope=r.scope_id.split('::',1)[1],edges_before=len(r.edges),rawerk_chars_before=len(r.rawerk)))
        targets=[f['old']['memory_id'] for fs in updates.values() for f in fs]
        assert len(targets)==len(set(targets))
        projections=[]; relations=[]
        for scope,records in sorted(byscope.items()):
            sample=rng.sample(records,min(100,len(records)))
            projections += [dict(scope=scope,memory_id=r.memory_id,expected=r.rawerk) for r in sample]
            keys=sorted([k for k in rels if k[0]==scope],key=lambda k:(ec[k]/max(se[scope],1),k[1]))
            selected=[]
            for label,fraction in [('low',0.0),('medium',0.5),('high',1.0)]:
                if keys:
                    key=keys[round((len(keys)-1)*fraction)]
                    if key not in selected:
                        selected.append(key)
                        relations.append(dict(scope=scope,relation=key[1],stratum=label,expected=rels[key],
                            matching_edges=ec[key],scope_edges=se[scope],scope_memories=len(records)))
            relations.append(dict(scope=scope,relation='cap_absent_relation',stratum='absent',expected=[],
                              matching_edges=0,scope_edges=se[scope],scope_memories=len(records)))
        rng.shuffle(projections); rng.shuffle(relations)
        manifests[str(seed)]=dict(projection=projections,relation=relations,updates=updates)
    save(OUT/'fixtures.json',manifests)
    save(OUT/'corpus_identity.json',dict(n=len(rs),sha256=digest([dataclasses.asdict(r) for r in rs]),
                                        scopes=len({r.scope_id for r in rs})))
    sourcefiles=[ROOT/'data/locomo_memory_records.csv',ROOT/'data/frozen_retrieval/p3_memory_features.csv',
        ROOT/'data/locomo_memory_ids_bge.txt',ROOT/'data/locomo_memory_bge_large.npy',
        ROOT/'src/cassmem/backend/live_cells_graph.py',ROOT/'src/cassmem/representation/graph_event.py',
        ROOT/'src/cassmem/retrieval/online.py',ROOT/'experiments/serving_100k/run_graph_canonical_gate_v2.py',
        ROOT/'experiments/serving_100k/run_graph_100k_load_gate_v2.py',
        Path('D:/memorytable/representation_serving_cost/common.py'),
        Path('D:/memorytable/cassmem_e0_preflight_20261015/environment.py')]
    save(OUT/'source_manifest.json',{str(p):sha(p) for p in sourcefiles})
    status('fixtures_prepared',eligible=len(pool),warmup=warm,measured=measured)

def verify_sources():
    for p,h in load(OUT/'source_manifest.json').items():
        if sha(p)!=h: raise RuntimeError('Frozen source mismatch: '+p)
    if (OUT/'runner_manifest.json').exists():
        for p,h in load(OUT/'runner_manifest.json').items():
            if sha(p)!=h: raise RuntimeError('Frozen runner mismatch: '+p)

class ResultProbe:
    def __init__(self,result,entry): self.result,self.entry=result,entry
    def one(self):
        row=self.result.one()
        if row is not None: self.entry['rows'].append(row)
        return row
    def __iter__(self):
        for row in self.result:
            self.entry['rows'].append(row)
            yield row

class SessionProbe:
    def __init__(self,session): self.session=session; self.calls=[]
    def execute(self,query,*args,**kw):
        t=time.perf_counter_ns(); result=self.session.execute(query,*args,**kw); end=time.perf_counter_ns()
        entry=dict(start_ns=t,end_ns=end,query=query,args=args,rows=[])
        self.calls.append(entry)
        return ResultProbe(result,entry)
    def summary(self):
        calls=[]
        for e in self.calls:
            q=e['query']; batch=getattr(q,'_statements_and_parameters',None)
            deletes=sum(not x[0] and str(x[1]).lstrip().upper().startswith('DELETE') for x in batch) if batch else 0
            calls.append(dict(start_ns=e['start_ns'],end_ns=e['end_ns'],rows=len(e['rows']),
                type='batch' if batch is not None else 'select_or_raw',
                batch_statements=len(batch) if batch is not None else 0,delete_statements=deletes,
                insert_statements=len(batch)-deletes if batch is not None else 0,
                decoded_rows_json_bytes=len(json.dumps([list(x) for x in e['rows']],ensure_ascii=False,default=str).encode()),
                parameter_proxy_bytes=len(repr(batch if batch is not None else e['args']).encode()),
                page_requests=None))
        return dict(cql_calls=len(calls),returned_rows=sum(x['rows'] for x in calls),calls=calls,
                    execute_wait_ms=sum(x['end_ns']-x['start_ns'] for x in calls)/1e6)

def timed(b,fn,instrument=True):
    original=b.s; probe=SessionProbe(original)
    if instrument: b.s=probe
    started=time.perf_counter_ns()
    try: result=fn()
    finally:
        ended=time.perf_counter_ns(); b.s=original
    info=probe.summary() if instrument else None
    return result,dict(start_ns=started,end_ns=ended,latency_ms=(ended-started)/1e6,probe=info)

def read_call(b,cell,op,req):
    if op=='projection': return b.candidate_projection(cell,req['scope'],req['memory_id'])
    return b.ids_by_relation(cell,req['scope'],req['relation'])

def require_equal(actual,expected,label):
    if actual!=expected: raise AssertionError(label)

def validate_update(b,cell,old,new):
    require_equal(b.fetch_graph(cell,new.scope_id,new.memory_id),new.graph_projection(),'graph mismatch')
    require_equal(b.candidate_projection(cell,new.scope_id,new.memory_id),new.rawerk,'projection mismatch')
    wanted={e['relation'] for e in new.edges}
    for rel in {e['relation'] for e in old.edges}|wanted:
        require_equal(new.memory_id in b.ids_by_relation(cell,new.scope_id,rel),rel in wanted,'relation membership mismatch')
    return True

def expected_batch(old,new,cell):
    wanted={(e['ordinal'],e['src'],e['relation'],e['dst']) for e in new.edges}
    ords={e['ordinal'] for e in new.edges}; deletes=0
    for e in old.edges:
        if (e['ordinal'],e['src'],e['relation'],e['dst']) not in wanted:
            deletes+=1
            if e['ordinal'] not in ords: deletes+=2 if cell=='cassandra-base' else 1
    inserts=1+len(new.graph_entities)+len(new.mentions)+(3 if cell=='cassandra-base' else 2)*len(new.edges)
    return dict(inserts=inserts,deletes=deletes,total=inserts+deletes)

def do_update(b,cell,fixture):
    old,new=(GraphRecord(**fixture[k]) for k in ('old','new'))
    _,raw=timed(b,lambda:b.commit_raw(cell,new))
    _,structured=timed(b,lambda:b.write_structured(cell,new))
    expected=expected_batch(old,new,cell)
    calls=structured['probe']['calls']; batches=[c for c in calls if c['type']=='batch']
    assert len(calls)==3 and len(batches)==1
    require_equal(batches[0]['batch_statements'],expected['total'],'batch count')
    require_equal(batches[0]['delete_statements'],expected['deletes'],'delete count')
    # Residual includes Python result consumption/construction and probe bookkeeping.
    structured['client_processing_residual_ms']=structured['latency_ms']-structured['probe']['execute_wait_ms']
    check_start=time.perf_counter_ns(); validate_update(b,cell,old,new)
    return dict(memory_id=new.memory_id,source_id=fixture.get('source_id'),source_scope=fixture.get('source_scope'),
        phase=fixture['phase'],raw=raw,structured=structured,expected_batch=expected,
        validation_ms=(time.perf_counter_ns()-check_start)/1e6,valid=True)

def overhead(b,cell,op,requests,dest):
    rows=[]
    for req in requests: require_equal(read_call(b,cell,op,req),req['expected'],'overhead warmup')
    for block in range(10):
        reqs=[requests[(block*20+i)%len(requests)] for i in range(20)]
        order=[False,True] if block%2==0 else [True,False]
        totals={}
        for flag in order:
            times=[]
            for req in reqs:
                result,event=timed(b,lambda:read_call(b,cell,op,req),flag)
                require_equal(result,req['expected'],'overhead parity');times.append(event['latency_ms'])
            totals[str(flag)]=float(np.mean(times))
        rows.append(dict(block=block,means=totals,relative=totals['True']/totals['False']-1))
    ratio=float(np.median([r['relative'] for r in rows]))
    save(dest,dict(blocks=rows,median_relative_overhead=ratio,passed=ratio<=0.05))
    if ratio>0.05: raise RuntimeError('Instrumentation overhead exceeds preregistered 5% gate; measurements not started')

def smoke(b,cell,fixtures,dest):
    dest.mkdir(exist_ok=False)
    # Different scope/IDs from formal corpus; no restore/reuse of formal update targets.
    checks=[]; records=[]
    for kind in KINDS:
        f=fixtures['updates'][kind][0]
        old=GraphRecord(**f['old'])
        old=dataclasses.replace(old,scope_id='cap_smoke_'+cell,memory_id='smoke_'+kind)
        new=mutate(old,kind); b.insert(cell,old); records.append(old)
        require_equal(b.candidate_projection(cell,old.scope_id,old.memory_id),old.rawerk,'smoke initial')
        row=do_update(b,cell,dict(old=dataclasses.asdict(old),new=dataclasses.asdict(new),phase='smoke'))
        checks.append(dict(kind=kind,**row))
    # Negative tests prove exact validator rejects changed/missing/extra/reordered values.
    rejected=0
    for actual,wanted in [('wrong','right'),([],['x']),(['x','y'],['x']),(['b','a'],['a','b'])]:
        try: require_equal(actual,wanted,'negative')
        except AssertionError: rejected+=1
    assert rejected==4
    save(dest/'checks.json',dict(passed=True,updates=checks,negative_checks=rejected))

def read_point(b,cell,seed,op,reqs,dest):
    verify_sources()
    dest.mkdir(exist_ok=False)
    save(dest/'resources_before.json',resources())
    for req in reqs: require_equal(read_call(b,cell,op,req),req['expected'],'read warmup')
    rng=random.Random(seed+int(op=='relation')); schedule=[]
    while len(schedule)<5000:
        order=list(range(len(reqs)));rng.shuffle(order);schedule.extend(order)
    events=[]; started=time.time()
    try:
        for i,j in enumerate(schedule[:5000]):
            req=reqs[j]; result,event=timed(b,lambda:read_call(b,cell,op,req))
            require_equal(result,req['expected'],'formal read mismatch')
            events.append(dict(event_id=i,request_index=j,scope=req['scope'],stratum=req.get('stratum'),
                               result_sha256=digest(result),valid=True,**event))
            if (i+1)%1000==0: status('measuring_read',layout=cell,seed=seed,operation=op,completed=i+1,total=5000)
    finally: lines(dest/'events.jsonl',events)
    summary=dict(layout=cell,seed=seed,operation=op,requests=len(events),unique_requests=len(reqs),
                 elapsed_s=time.time()-started,latency_ms=distribution([e['latency_ms'] for e in events]),valid=True)
    save(dest/'summary.json',summary); save(dest/'complete.json',dict(n=len(events),sha256=sha(dest/'events.jsonl')))
    save(dest/'resources_after.json',resources())
    return summary

def update_point(b,cell,seed,kind,fixtures,dest):
    verify_sources()
    dest.mkdir(exist_ok=False); events=[]; started=time.time()
    save(dest/'resources_before.json',resources())
    try:
        for i,f in enumerate(fixtures):
            old=GraphRecord(**f['old'])
            require_equal(b.fetch_graph(cell,old.scope_id,old.memory_id),old.graph_projection(),'target initial state')
            events.append(dict(event_id=i,kind=kind,**do_update(b,cell,f)))
            if (i+1)%50==0: status('measuring_update',layout=cell,seed=seed,operation=kind,completed=i+1,total=len(fixtures))
    finally: lines(dest/'events.jsonl',events)
    formal=[e for e in events if e['phase']=='measured']
    summary=dict(layout=cell,seed=seed,operation=kind,requests=len(formal),elapsed_s=time.time()-started,
        raw_ms=distribution([e['raw']['latency_ms'] for e in formal]),
        structured_ms=distribution([e['structured']['latency_ms'] for e in formal]),valid=True,
        unique_source_records=len({e['source_id'] for e in formal}))
    save(dest/'summary.json',summary);save(dest/'complete.json',dict(n=len(formal),sha256=sha(dest/'events.jsonl')))
    save(dest/'resources_after.json',resources())
    return summary

def historical():
    sources={'E0_main':Path('D:/memorytable/cassmem_e0_formal_20261015/run_b9da27ebdd6e'),
             'E0_audit':Path('D:/memorytable/cassmem_e0_audit_overhead_20261017/run_v1'),
             'E0_ingress':Path('D:/memorytable/cassmem_e0_ingress_20261018/run_v1')}
    output=[]
    for cohort,root in sources.items():
        for point in sorted(root.glob('cassandra-*_r*')):
            m=point/'measured';ev=m/'events.jsonl';reads=m/'reads.jsonl'
            if not ev.exists() or not reads.exists():continue
            es=[json.loads(x) for x in ev.read_text(encoding='utf-8').splitlines() if x]
            rr=[json.loads(x) for x in reads.read_text(encoding='utf-8').splitlines() if x]
            row=dict(cohort=cohort,run_id=point.name,updates=len(es),reads=len(rr),
                     source=str(point),event_sha256=sha(ev),read_sha256=sha(reads))
            assert len({x['event_id'] for x in es})==len(es)
            assert len({x['event_id'] for x in rr})==len(rr)
            for label,a,z in [('queue','actual_enqueue_ns','worker_start_ns'),('raw','raw_call_start_ns','raw_ack_ns'),
                ('structured','structured_start_ns','structured_done_ns'),('canonical','canonical_check_start_ns','canonical_check_done_ns'),
                ('sparse','sparse_start_ns','sparse_done_ns'),('dense','dense_start_ns','dense_done_ns'),
                ('ready','actual_enqueue_ns','ready_done_ns')]:
                values=[(e[z]-e[a])/1e6 for e in es if e.get(a) is not None and e.get(z) is not None]
                row[label+'_ms']=distribution(values) if values else None
            validation=point/'validate.json'
            row['original_validation']=load(validation) if validation.exists() else None
            corrected=Path('D:/memorytable/cassmem_e0_recovery_20261016/revalidation')/point.name/'validate_corrected.json'
            row['corrected_validation']=load(corrected) if cohort=='E0_main' and corrected.exists() else None
            output.append(row)
    save(OUT/'historical_cohorts.json',output)
    return len(output)

def run():
    verify_sources()
    lock=OUT/'runner.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    summaries=[]; state=None; b=None; name=None
    try:
        fixtures=load(OUT/'fixtures.json'); rs=expand(build_graph_records())
        assert digest([dataclasses.asdict(r) for r in rs])==load(OUT/'corpus_identity.json')['sha256']
        status('historical_extraction'); historical()
        import cassandra, sklearn
        save(OUT/'runtime.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),pid=os.getpid(),
             numpy=np.__version__,cassandra_driver=cassandra.__version__,sklearn=sklearn.__version__,
             logical_cpus=os.cpu_count(),host_ram=psutil.virtual_memory().total,
             docker_version=docker('version','--format','{{json .Server}}')))
        status('environment_guard');state=suspend_old(OUT)
        for si,seed in enumerate(SEEDS):
            order=CELLS if si%2==0 else list(reversed(CELLS))
            for cell in order:
                verify_sources(); trial=OUT/f'{cell}_s{seed}';trial.mkdir(exist_ok=False)
                status('starting_container',layout=cell,seed=seed);b,name=start('cassandra',trial)
                save(trial/'driver.json',dict(default_fetch_size=b.s.default_fetch_size,default_timeout=b.s.default_timeout,
                                             keyspace=b.s.keyspace))
                status('live_smoke',layout=cell,seed=seed);smoke(b,cell,fixtures[str(seed)],trial/'smoke')
                # Separate formal keyspace: smoke data never enter the measured corpus.
                b.close()
                from cassmem.backend.live_cells_graph import CassandraGraphCells
                from cassandra import ConsistencyLevel
                b=CassandraGraphCells(keyspace='cap_formal_'+str(seed)+'_'+cell.split('-')[-1])
                b.s.default_timeout=30;b.s.default_consistency_level=ConsistencyLevel.LOCAL_ONE
                for statement in b.p.values():statement.consistency_level=ConsistencyLevel.LOCAL_ONE
                save(trial/'formal_driver.json',dict(keyspace=b.s.keyspace,fetch_size=b.s.default_fetch_size,
                                                    timeout=b.s.default_timeout,consistency='LOCAL_ONE'))
                status('loading_100k',layout=cell,seed=seed,loaded=0)
                with ThreadPoolExecutor(32) as pool:
                    for n,_ in enumerate(pool.map(lambda r:b.insert(cell,r),rs),1):
                        if n%5000==0:status('loading_100k',layout=cell,seed=seed,loaded=n)
                grouped=collections.defaultdict(list)
                for r in rs:grouped[r.scope_id].append(r.memory_id)
                for scope,mids in grouped.items():require_equal(b.ids(cell,scope),sorted(mids),'load scope IDs')
                save(trial/'load_gate.json',dict(memories=100000,scopes=len(grouped),passed=True,smoke_scope_excluded=True))
                mf=fixtures[str(seed)]
                for op in ('projection','relation'):
                    status('instrumentation_gate',layout=cell,seed=seed,operation=op)
                    overhead(b,cell,op,mf[op],trial/f'overhead_{op}.json')
                for op in ('projection','relation'):
                    summaries.append(read_point(b,cell,seed,op,mf[op],trial/op));save(OUT/'summary.json',summaries)
                for kind in KINDS:
                    summaries.append(update_point(b,cell,seed,kind,mf['updates'][kind],trial/kind));save(OUT/'summary.json',summaries)
                status('layout_complete',layout=cell,seed=seed,points=len(summaries));stop(b,name);b=None;name=None
        verify_sources(); assert len(summaries)==36
        protocol=load(OUT/'protocol.json')
        assert all(s['requests']==(5000 if s['operation'] in ('projection','relation') else protocol['update_measured']) for s in summaries)
        save(OUT/'complete.json',dict(points=36,passed=True,summary_sha256=sha(OUT/'summary.json')))
        status('complete',points=36)
    except BaseException:
        save(OUT/'failure.json',dict(pid=os.getpid(),time=time.time(),traceback=traceback.format_exc()))
        status('failed',completed_points=len(summaries));raise
    finally:
        try:
            if b is not None and name is not None:stop(b,name)
        finally:
            if state is not None:restore(state,OUT)
            if lock.exists():lock.rename(OUT/f'runner.exited.{os.getpid()}.lock')

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='freeze':
        verify_sources()
        save(OUT/'runner_manifest.json',{str(p):sha(p) for p in [*HERE.glob('*.py'),OUT/'protocol.json',OUT/'fixtures.json',OUT/'corpus_identity.json']})
    elif sys.argv[1]=='run':run()
    else:raise SystemExit('prepare | freeze | run')
