import sys,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'D:/memorytable/cassmem_e0_recovery_20261016')
import resume_v2 as R
F=R.F;E=R.E;V=R.R.V
import inspect,textwrap,json,hashlib,shutil,traceback,datetime
OLD=R.OLD
OUT=HERE/'run_v1'

def phase_source():
    s=F.phase_source()
    needle="                row['audit']=self.capture(i)\n"
    assert s.count(needle)==1
    return s.replace(needle,"                row['read_audit_enabled']=self.audit_arm=='full'\n                if self.audit_arm=='full':\n                    row['read_audit_start_ns']=time.perf_counter_ns();row['audit']=self.capture(i);row['read_audit_done_ns']=time.perf_counter_ns()\n")

g=dict(vars(F.S));exec(compile(phase_source(),'<read-audit-phase>','exec'),g)
class AuditTrial(F.FormalTrial):
    phase=g['phase']
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.audit_arm=self.run_id.rsplit('__',1)[-1] if '__' in self.run_id else 'full'
        assert self.audit_arm in ('full','read_audit_off')

def campaign_source():
    s=textwrap.dedent(inspect.getsource(F.campaign))
    s=s.replace('FormalTrial(', 'AuditTrial(')
    s=s.replace("b.insert_many(cell,initial[offset:offset+5000],batch=100,workers=12)","load_scope_safe(b,cell,initial[offset:offset+5000],out/(cell+'_loader_retries.jsonl'),workers=12)")
    s=s.replace('run_id=f"{cell}_r{point[\'rate\']}_s{point[\'seed\']}"','run_id=f"{cell}_r{point[\'rate\']}_s{point[\'seed\']}__{point[\'audit_arm\']}"')
    # Existing full-audit integration smoke is followed by a matched off-arm gate.
    needle='del gtrial;gc.collect()'
    extra="""del gtrial;gc.collect()
                    offgate=out/(cell+'_integration_smoke_read_audit_off');offgate.mkdir()
                    offtrial=AuditTrial(b,cell,initial,mv,gatefs,offgate,{'active':{'rate':1,'drain_s':10}},cell+'-integration__read_audit_off')
                    save(offgate/'fixtures.json',gatefs)
                    offtrial.phase('measured',2,0)
                    od=offgate/'measured';oc=audit_phase(load(od/'context.json'),lines(od/'events.jsonl')+lines(od/'reads.jsonl'),lines(od/'submissions.jsonl'),gatefs)
                    assert all(r.get('read_audit_enabled') is False and 'audit' not in r for r in lines(od/'reads.jsonl'))
                    save(offgate/'validate.json',oc)
                    if oc['errors'] or oc['execution_errors'] or not oc['all_updates_eventually_ready']:raise RuntimeError('Off-arm integration smoke failed')
                    for f in gatefs:
                        rr=original[f['memory_id']];b.commit_raw(cell,rr);b.write_structured(cell,rr)
                        assert b.fetch_graph(cell,rr.scope_id,rr.memory_id)==rr.graph_projection()
                    del offtrial;gc.collect()"""
    assert s.count(needle)==1;s=s.replace(needle,extra)
    s=s.replace('points=36','points=72')
    return s

def summarize(out):
    results=[]
    for pt in F.load(out/'formal_matrix.json'):
        d=out/f"{pt['layout']}_r{pt['rate']}_s{pt['seed']}__{pt['audit_arm']}"
        result=V.validate_point(d);F.save(d/'validate.json',result)
        rs=V.lines(d/'measured/reads.jsonl');us=V.lines(d/'measured/events.jsonl')
        assert all(r.get('read_audit_enabled')==(pt['audit_arm']=='full') for r in rs if r['final_status']=='ok')
        def pct(values,p):return float(F.np.percentile(values,p)) if values else None
        def ms(rows,a,b):return [(r[b]-r[a])/1e6 for r in rows if r.get(a) is not None and r.get(b) is not None]
        x={**pt,'status':result['status'],**result['phases']['measured']}
        for label,values in [('read_arrival_to_query',ms(rs,'actual_enqueue_ns','query_finish_ns')),('read_arrival_to_response',ms(rs,'actual_enqueue_ns','response_finish_ns')),('read_queue',ms(rs,'actual_enqueue_ns','worker_start_ns')),('update_queue',ms(us,'actual_enqueue_ns','worker_start_ns')),('read_extra_audit',ms(rs,'read_audit_start_ns','read_audit_done_ns'))]:
            for p in (50,95,99):x[f'{label}_p{p}_ms']=pct(values,p)
        ctx=F.load(d/'measured/context.json');duration=(ctx['injection_end_ns']-ctx['planned_start_ns'])/1e9
        x['completed_responses_during_injection_per_s']=sum(r.get('response_finish_ns',10**30)<=ctx['injection_end_ns'] for r in rs+us)/duration
        results.append(x)
    F.save(out/'summary.json',results)
    F.write_csv(out/'per-run-summary.csv',results)
    F.save(out/'validate.json',dict(points=len(results),status_counts=dict(F.collections.Counter(x['status'] for x in results))))

def prepare():
    F.verify(F.load(OLD/'source_manifest.json'))
    p=F.load(OLD/'protocol.json');p['experiment_id']='E0-read-audit-overhead-20261017'
    p['audit_control']=dict(arms=['full','read_audit_off'],boundary='Only ordinary-read post-query capture is disabled; update readiness, post-ready audit, post-update ranking, final checkpoint and all instrumentation remain unchanged.',ordering='For each original point, full/off for even indices and off/full for odd indices; frozen before running.',historical='Historical E0 not used as paired arm.',read_version_metadata='raw_version=-1 and target_content_match=false in off arm denote no capture, not a measured mismatch; use audit_enabled.')
    p['formal']['points']=72;p['formal']['offered_updates']*=2;p['formal']['offered_reads']*=2
    matrix=[]
    for i,pt in enumerate(F.load(OLD/'formal_matrix.json')):
        for arm in (['full','read_audit_off'] if i%2==0 else ['read_audit_off','full']):matrix.append(dict(pt,audit_arm=arm))
    assert len(matrix)==72
    OUT.mkdir(exist_ok=False);F.save(OUT/'protocol.json',p);F.save(OUT/'formal_matrix.json',matrix)
    for n in ('formal_update_traces.json','environment.json'):shutil.copy2(OLD/n,OUT/n)
    for n,s in [('phase_generated.py',phase_source()),('campaign_generated.py',campaign_source()),('engine_start_generated.py',R.start_source())]:
        compile(s,n,'exec');(OUT/n).write_text(s,encoding='utf-8')
    manifest=F.load(OLD/'source_manifest.json')
    for f in [Path(__file__),Path(R.__file__),Path(R.R.__file__),Path(V.__file__),Path(R.R.load_scope_safe.__code__.co_filename),*OUT.glob('*.json'),*OUT.glob('*.py')]:manifest[str(f)]=F.sha(f)
    F.save(OUT/'source_manifest.json',manifest)
    F.save(OUT/'prelaunch_tests.json',dict(original_hashes_verified=True,read_capture_replacement_count=1,update_phase_unchanged=True,points=72,compilation='PASS'))
    print(OUT,flush=True)

def run():
    log=(OUT/'runner.log').open('a',encoding='utf-8',buffering=1);sys.stdout=log;sys.stderr=log
    F.verify(F.load(OUT/'source_manifest.json'))
    eg=dict(vars(E));exec(compile(R.start_source(),'<frozen-start>','exec'),eg)
    V.summarize=summarize
    gl=dict(vars(F));gl.update(AuditTrial=AuditTrial,phase_source=phase_source,start=eg['start'],load_scope_safe=R.R.load_scope_safe)
    exec(compile(campaign_source(),'<audit-campaign>','exec'),gl)
    try:gl['campaign'](OUT)
    except BaseException:
        F.save(OUT/'failure.json',dict(error=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));traceback.print_exc();raise

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:run()
