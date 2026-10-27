"""Independent raw-evidence validator; ranking is separate from readiness."""
import json,hashlib,csv,sys
from pathlib import Path
import numpy as np
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def write_csv(p,rs):
    if not rs:return
    keys=list(dict.fromkeys(k for r in rs for k in r))
    with Path(p).open('w',newline='',encoding='utf-8-sig') as h:
        w=csv.DictWriter(h,keys);w.writeheader();w.writerows(rs)
def pct(v,p):return float(np.percentile(v,p)) if v else None
def content(a,f):
    v=f['versions'][1];rels={r for x in f['versions'] for r in x['required_relations']}
    return (a.get('graph')==v['graph'] and a.get('projection')==v['rawerk'] and a.get('sparse_document')==v['rawerk']
       and a.get('sparse_version')==a.get('dense_version')==2 and a.get('dense_vector_sha256')==v['normalized_vector_sha256']
       and f['memory_id'] in a.get('ids',[]) and set(a.get('relation_memberships',{}))==rels
       and all(value==(r in v['required_relations']) for r,value in a['relation_memberships'].items()))
def audit_phase(ctx,rs,subs,fs):
    errors=[]
    def ck(ok,msg):
        if not ok:errors.append(msg)
    ids=list(range(ctx['operations']))
    ck(sorted(r['event_id'] for r in rs)==ids,'event coverage/unique IDs')
    ck([r['event_id'] for r in subs]==ids,'submission coverage/unique IDs')
    ck(len(rs)==len(subs)==ctx['operations'],'offered count')
    u=[r for r in rs if r['op']=='update'];reads=[r for r in rs if r['op']=='read']
    ck(len(u)==ctx['updates'] and len(reads)==ctx['reads'],'operation mix')
    for r in rs:
        ts=[r.get(k) for k in ('planned_arrival_ns','emit_ns','actual_enqueue_ns','worker_start_ns','response_finish_ns')]
        ck(all(isinstance(t,int) for t in ts) and ts==sorted(t for t in ts if t is not None),'clock order')
        ck(r['absolute_deadline_ns']==ctx['absolute_deadline_ns'],'deadline boundary')
        ck(r['planned_arrival_ns']==ctx['planned_start_ns']+int(r['event_id']/(ctx['update_rate']*20)*1e9),'schedule')
        f=fs[r['fixture_id']];ck(r['target_memory_id']==f['memory_id'] and r['query_scope']==f['scope'],'fixture scope/ID')
        ck(r['driver_pool_wait_ms'] is None and r['relation_version'] is None,'unknown metrics')
        if r['final_status']!='ok':continue
        if r['op']=='update':
            keys=('worker_start_ns','raw_call_start_ns','raw_ack_ns','structured_start_ns','structured_done_ns','canonical_check_start_ns','canonical_check_done_ns','sparse_start_ns','sparse_done_ns','dense_start_ns','dense_done_ns','ready_check_start_ns','ready_check_done_ns','response_finish_ns')
            t=[r.get(k) for k in keys];ck(all(isinstance(x,int) for x in t) and t==sorted(x for x in t if x is not None),'update phase clocks')
            ck(content(r['audit'],f),'ready content')
            ck(r['post_update_target_present']==(f['memory_id'] in r['post_update_top10']),'ranking flag')
        else:
            ck(r['query_id']==f['query']['query_id'],'query source')
            ck(len(r['returned_memory_ids'])==len(set(r['returned_memory_ids'])),'duplicate results')
            ck([v['memory_id'] for v in r['returned_versions']]==r['returned_memory_ids'],'version coverage')
        for x in r.get('scope_locks',[]):ck(r['worker_start_ns']<=x['wait_start_ns']<=x['acquired_ns']<=x['released_ns']<=r['response_finish_ns'],'scope lock clock')
    lag=[(r['actual_enqueue_ns']-r['planned_arrival_ns'])/1e6 for r in subs];lag95=pct(lag,95)
    execution=sum(r['final_status']!='ok' for r in rs)
    ready=[r for r in u if r.get('ready') and content(r.get('audit',{}),fs[r['fixture_id']])]
    times=[(r['ready_check_done_ns']-r['actual_enqueue_ns'])/1e6 for r in ready]
    valid=not errors and execution==0 and lag95 is not None and lag95<=50 and max((r['actual_enqueue_ns'] for r in subs),default=10**30)<=ctx['injection_end_ns']+10**9
    return dict(errors=errors,offered_load_valid=valid,execution_errors=execution,updates=len(u),reads=len(reads),submission_lag_p95_ms=lag95,
       completed_before_absolute_drain=len(rs)==ctx['operations'] and all(r.get('response_finish_ns') is not None and r['response_finish_ns']<=ctx['absolute_deadline_ns'] for r in rs),
       all_updates_eventually_ready=len(ready)==ctx['updates'],ready_denominator=ctx['updates'],ready_before_deadline=sum(r['ready_check_done_ns']<=ctx['absolute_deadline_ns'] for r in ready),
       arrival_to_ready_p50_ms=pct(times,50),arrival_to_ready_p95_ms=pct(times,95),arrival_to_ready_p99_ms=pct(times,99),
       post_update_top10_hits=sum(r.get('post_update_target_present',False) for r in u),post_update_top10_denominator=ctx['updates'])
def validate_point(dest):
    dest=Path(dest);fs=load(dest/'fixtures.json');result={'phases':{},'errors':[]};details=[]
    for phase in ('warmup','measured'):
        d=dest/phase;ctx=load(d/'context.json');rs=lines(d/'events.jsonl')+lines(d/'reads.jsonl');subs=lines(d/'submissions.jsonl')
        a=audit_phase(ctx,rs,subs,fs);result['phases'][phase]=a;result['errors']+=a['errors']
        dl=load(d/'deadline.json')
        if dl['observed_ns']<ctx['absolute_deadline_ns']:result['errors'].append('early deadline snapshot')
        if len(lines(d/'emitter.jsonl'))!=ctx['operations']:result['errors'].append('emitter coverage')
        for r in rs:
            def elapsed(a,b):return (r[b]-r[a])/1e6 if r.get(a) is not None and r.get(b) is not None else None
            details.append(dict(phase=phase,event_id=r['event_id'],op=r['op'],app_queue_ms=elapsed('actual_enqueue_ns','worker_start_ns'),
              scope_lock_wait_ms=sum((x['acquired_ns']-x['wait_start_ns'])/1e6 for x in r.get('scope_locks',[])),scope_lock_hold_ms=sum((x['released_ns']-x['acquired_ns'])/1e6 for x in r.get('scope_locks',[])),
              driver_pool_wait_ms=None,backend_observed_call_ms=elapsed('raw_call_start_ns','raw_ack_ns'),structured_maintenance_ms=elapsed('structured_start_ns','structured_done_ns'),
              sparse_rebuild_ms=elapsed('sparse_start_ns','sparse_done_ns'),dense_rebuild_ms=elapsed('dense_start_ns','dense_done_ns'),ready_check_ms=elapsed('ready_check_start_ns','ready_check_done_ns'),
              scoring_fusion_ms=sum(r.get(x,0) for x in ('sparse_scoring_ns','dense_scoring_ns','fusion_ns'))/1e6 if r['op']=='read' else None))
    for x in load(dest/'checkpoint.json'):
        if not content(x['audit'],fs[x['fixture_id']]):result['errors'].append('final checkpoint')
    write_csv(dest/'phase-breakdown.csv',details)
    result['status']='FAIL' if result['errors'] or any(a['execution_errors'] for a in result['phases'].values()) else ('INVALID' if not all(a['offered_load_valid'] for a in result['phases'].values()) else 'PASS')
    return result
def summarize(out):
    out=Path(out);summary=[];resources=[]
    for pt in load(out/'formal_matrix.json'):
        d=out/f"{pt['layout']}_r{pt['rate']}_s{pt['seed']}";r=validate_point(d);save(d/'validate.json',r)
        summary.append({**pt,'status':r['status'],**r['phases']['measured']})
        resources.extend(dict(run_id=d.name,**x) for x in lines(d/'resource_usage.jsonl'))
    write_csv(out/'workload-validity.csv',summary);write_csv(out/'per-run-summary.csv',summary);write_csv(out/'resource-usage.csv',resources)
    save(out/'validate.json',dict(points=len(summary),updates=sum(x['updates'] for x in summary),reads=sum(x['reads'] for x in summary),statuses=summary))
if __name__=='__main__':summarize(Path(sys.argv[1]))
