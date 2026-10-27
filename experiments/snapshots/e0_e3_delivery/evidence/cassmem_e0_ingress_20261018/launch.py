import sys,os,json,inspect,shutil,traceback,datetime,ast,copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'D:/memorytable/cassmem_e0_audit_overhead_20261017')
import run as A
F=A.F;E=A.E;V=A.V;OLD=A.OUT;OUT=HERE/'run_v1'
RECEIVER=str(HERE/'receiver.py')

def phase_source():
    s=A.phase_source()
    edits=[('ordinal,scheduled,emitted=map(int,line.strip().split(\',\'))','ordinal,scheduled,emitted,enqueued=map(int,line.strip().split(\',\'))'),
       ('enqueued=time.perf_counter_ns();i=offset+ordinal//20','dispatch=time.perf_counter_ns();i=offset+ordinal//20'),
       ('actual_enqueue_ns=enqueued','actual_enqueue_ns=enqueued,execution_dispatch_ns=dispatch')]
    for old,new in edits:
        assert s.count(old)==(2 if old=='actual_enqueue_ns=enqueued' else 1);s=s.replace(old,new)
    return s
g=dict(vars(F.S));g['EMITTER']=RECEIVER;exec(compile(phase_source(),'<isolated-phase>','exec'),g)
class IngressTrial(A.AuditTrial):phase=g['phase']

def ingress_check(ctx,rs,subs,received,emitted):
    errors=[]
    def ck(ok,msg):
        if not ok:errors.append(msg)
    count=ctx['operations'];ck([r['event_id'] for r in received]==list(range(count)),'receiver coverage')
    ck([r['ordinal'] for r in emitted]==list(range(count)),'emitter coverage')
    rm={x['event_id']:x for x in received};em={x['ordinal']:x for x in emitted};sm={x['event_id']:x for x in subs}
    for r in rs:
        i=r['event_id'];x=rm.get(i,{});e=em.get(i,{});sub=sm.get(i,{})
        ck(all(r.get(k)==x.get(k) for k in ('planned_arrival_ns','emit_ns','actual_enqueue_ns')),'receiver raw clock mismatch')
        ck(x.get('emit_ns')==e.get('emit_ns') and x.get('planned_arrival_ns')==e.get('scheduled_ns'),'emitter raw clock mismatch')
        ts=[r.get(k) for k in ('planned_arrival_ns','emit_ns','actual_enqueue_ns','execution_dispatch_ns','worker_start_ns','response_finish_ns')]
        ck(all(isinstance(t,int) for t in ts) and ts==sorted(t for t in ts if isinstance(t,int)),'ingress dispatch clock order')
        ck(sub.get('execution_dispatch_ns')==r.get('execution_dispatch_ns'),'dispatch evidence mismatch')
    return errors

base_validate=V.validate_point
def validate_point(dest):
    dest=Path(dest);v=base_validate(dest);v['ingress']={}
    for phase in ('warmup','measured'):
        d=dest/phase;ctx=F.load(d/'context.json');rs=V.lines(d/'events.jsonl')+V.lines(d/'reads.jsonl');subs=V.lines(d/'submissions.jsonl')
        errors=ingress_check(ctx,rs,subs,V.lines(d/'receiver.jsonl'),V.lines(d/'emitter.jsonl'));v['errors']+=errors
        def pct(a,b):return {str(p):float(F.np.percentile([(r[b]-r[a])/1e6 for r in rs],p)) for p in (50,95,99)}
        v['ingress'][phase]=dict(errors=errors,planned_to_receiver_ms=pct('planned_arrival_ns','actual_enqueue_ns'),receiver_to_dispatch_ms=pct('actual_enqueue_ns','execution_dispatch_ns'),dispatch_to_worker_ms=pct('execution_dispatch_ns','worker_start_ns'),receiver_to_worker_ms=pct('actual_enqueue_ns','worker_start_ns'),queued_at_deadline=sum(r['actual_enqueue_ns']<=ctx['absolute_deadline_ns']<r['worker_start_ns'] for r in rs))
    if v['errors']:v['status']='FAIL'
    return v

def campaign_source():
    s=A.campaign_source().replace("for engine in ('cassandra','neo4j'):","for engine in ('cassandra',):").replace('points=72','points=12')
    # Independent ingress evidence must also pass each integration gate.
    s=s.replace("save(gate/'validate.json',gcheck)","gcheck['errors']+=ingress_check(load(gd/'context.json'),lines(gd/'events.jsonl')+lines(gd/'reads.jsonl'),lines(gd/'submissions.jsonl'),lines(gd/'receiver.jsonl'),lines(gd/'emitter.jsonl'))\n                    save(gate/'validate.json',gcheck)")
    s=s.replace("save(offgate/'validate.json',oc)","oc['errors']+=ingress_check(load(od/'context.json'),lines(od/'events.jsonl')+lines(od/'reads.jsonl'),lines(od/'submissions.jsonl'),lines(od/'receiver.jsonl'),lines(od/'emitter.jsonl'))\n                    save(offgate/'validate.json',oc)")
    return s

def summarize(out):
    A.summarize(out)
    F.save(out/'ingress_summary.json',[dict(point=d.name,**F.load(d/'validate.json')['ingress']) for d in out.iterdir() if (d/'point.json').exists()])

def prepare():
    F.verify(F.load(OLD/'source_manifest.json'))
    assert F.load(HERE/'loadgen_test/result.json')['pass']
    p=F.load(OLD/'protocol.json');p['experiment_id']='E0-ingress-separated-20261018';p['cells']=[x for x in p['cells'] if x.startswith('cassandra')]
    matrix=[x for x in F.load(OLD/'formal_matrix.json') if x['layout'].startswith('cassandra') and x['rate']==1]
    assert len(matrix)==12
    p['formal'].update(rates=[1.0],points=12,offered_updates=1440,offered_reads=27360)
    p['ingress_amendment']=dict(parent=str(OLD),arrival='actual_enqueue_ns is stamped at independent receiver queue insertion, NOT execution-process dispatch',queue='receiver queue + stdout transport + execution ThreadPool queue; all waits remain in receiver-to-worker and end-to-end latency',clocks=['planned_arrival_ns','emit_ns','actual_enqueue_ns','execution_dispatch_ns','worker_start_ns'],gc='unchanged default GC',validity='same 50ms p95 applied to receiver enqueue; dispatch lag reported separately, never discarded',comparison='only within 12 paired amended points; preserve previous invalid points; do not silently pool receiver-arrival with old executor-arrival metrics',receiver_resources='one additional lightweight standard-library process; same host, no affinity/priority change')
    OUT.mkdir(exist_ok=False);F.save(OUT/'protocol.json',p);F.save(OUT/'formal_matrix.json',matrix)
    for n in ('formal_update_traces.json','environment.json'):shutil.copy2(OLD/n,OUT/n)
    for n,s in [('phase_generated.py',phase_source()),('campaign_generated.py',campaign_source()),('engine_start_generated.py',A.R.start_source())]:
        compile(s,n,'exec');(OUT/n).write_text(s,encoding='utf-8')
    manifest=F.load(OLD/'source_manifest.json')
    for f in [*HERE.glob('*.py'),HERE/'loadgen_test/result.json',*OUT.glob('*.json'),*OUT.glob('*.py')]:manifest[str(f)]=F.sha(f)
    F.save(OUT/'source_manifest.json',manifest)
    print(OUT)

def run():
    log=(OUT/'runner.log').open('a',encoding='utf-8',buffering=1);sys.stdout=log;sys.stderr=log
    F.verify(F.load(OUT/'source_manifest.json'))
    eg=dict(vars(E));exec(compile(A.R.start_source(),'<start>','exec'),eg)
    V.validate_point=validate_point;V.summarize=summarize
    gl=dict(vars(F));gl.update(AuditTrial=IngressTrial,phase_source=phase_source,start=eg['start'],load_scope_safe=A.R.R.load_scope_safe,ingress_check=ingress_check)
    exec(compile(campaign_source(),'<campaign>','exec'),gl)
    try:gl['campaign'](OUT)
    except BaseException:
        F.save(OUT/'failure.json',dict(error=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));traceback.print_exc();raise
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:run()
