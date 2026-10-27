import json,sys,hashlib,copy,csv
from pathlib import Path
import numpy as np
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def content_ok(a,f):
    v=f['versions'][1];g=copy.deepcopy(v['graph']);g['memory']['scope_id']=a['scope']
    expected=set(v['required_relations']);allrels={r for version in f['versions'] for r in version['required_relations']}
    return (a['graph']==g and a['projection']==v['rawerk'] and a['sparse_version']==2 and a['dense_version']==2 and a['sparse_document']==v['rawerk'] and a['dense_vector_sha256']==v['normalized_vector_sha256'] and f['memory_id'] in a['ids'] and set(a['relation_memberships'])==allrels and all(value==(r in expected) for r,value in a['relation_memberships'].items()))
def phase_check(ctx,rows,subs,fixtures):
    errors=[]
    def ck(ok,msg):
        if not ok:errors.append(msg)
    ck(len(rows)==len(subs)==ctx['operations'],'offered coverage')
    ck(sorted(r['event_id'] for r in rows)==list(range(ctx['operations'])),'unique complete IDs')
    ck([r['event_id'] for r in subs]==list(range(ctx['operations'])),'submission sequence')
    updates=[r for r in rows if r['op']=='update'];reads=[r for r in rows if r['op']=='read']
    ck(len(updates)==ctx['updates'] and len(reads)==ctx['reads'],'operation mix')
    for r in rows:
        ck(r['planned_arrival_ns']<=r['emit_ns']<=r['actual_enqueue_ns']<=r['worker_start_ns']<=r['response_finish_ns'],'clock ordering')
        ck(r['absolute_deadline_ns']==ctx['absolute_deadline_ns'],'absolute deadline')
        ck(r['final_status']=='ok','execution error')
        ck(r['driver_pool_wait_ms'] is None and r['relation_version'] is None,'unknown fields retained')
        for lock in r.get('scope_locks',[]):ck(r['worker_start_ns']<=lock['wait_start_ns']<=lock['acquired_ns']<=lock['released_ns']<=r['response_finish_ns'],'scope-lock clock')
        if r['op']=='update' and r['final_status']=='ok':
            times=[r[x] for x in ('worker_start_ns','id_lock_wait_start_ns','id_lock_acquired_ns','raw_call_start_ns','raw_ack_ns','structured_start_ns','structured_done_ns','canonical_check_start_ns','canonical_check_done_ns','sparse_start_ns','sparse_done_ns','dense_start_ns','dense_done_ns','ready_check_start_ns','ready_check_done_ns','id_lock_released_ns','response_finish_ns')]
            ck(times==sorted(times),'update phase ordering')
            ck(content_ok(r['audit'],fixtures[r['fixture_id']]),'ready content mismatch')
            ck(r['post_update_target_present'] and r['target_memory_id'] in r['post_update_top10'],'post-update retrieval presence')
        if r['op']=='read' and r['final_status']=='ok':
            ck(len(r['returned_memory_ids'])==len(set(r['returned_memory_ids'])),'duplicate candidates')
            ck(len(r['returned_versions'])==len(r['returned_memory_ids']),'returned version coverage')
    lag=float(np.percentile([(r['actual_enqueue_ns']-r['planned_arrival_ns'])/1e6 for r in subs],95))
    valid=(not errors and lag<=50 and max(r['actual_enqueue_ns'] for r in subs)<=ctx['injection_end_ns']+1_000_000_000)
    return errors,dict(offered_load_valid=valid,submission_lag_p95_ms=lag,updates=len(updates),reads=len(reads),execution_errors=sum(r['final_status']!='ok' for r in rows),
       completed_before_absolute_drain=all(r['response_finish_ns']<=ctx['absolute_deadline_ns'] for r in rows),
       all_updates_eventually_ready=all(r.get('ready',False) for r in updates),
       ready_before_deadline=sum(r.get('ready_check_done_ns',10**30)<=ctx['absolute_deadline_ns'] for r in updates),
       ready_denominator=ctx['updates'],
       arrival_to_ready_p95_ms=float(np.percentile([(r['ready_check_done_ns']-r['actual_enqueue_ns'])/1e6 for r in updates if r.get('ready')],95)))
def main(out):
    fixtures=load(out/'fixtures.json');p=load(out/'protocol.json');errors=[];summary=[];details=[];breakdown=[];negative=[]
    for path,h in load(out/'source_manifest.json').items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:errors.append('source drift '+path)
    for cell in p['cells']:
        for phase in ('warmup','measured'):
            dest=out/cell/phase;ctx=load(dest/'context.json');rows=lines(dest/'events.jsonl')+lines(dest/'reads.jsonl');subs=lines(dest/'submissions.jsonl')
            er,s=phase_check(ctx,rows,subs,fixtures);errors.extend(cell+'/'+phase+': '+e for e in er);summary.append(dict(cell=cell,phase=phase,**s))
            deadline=load(dest/'deadline.json')
            if deadline['observed_ns']<ctx['absolute_deadline_ns']:errors.append('deadline observed too early')
            emitted=lines(dest/'emitter.jsonl')
            if len(emitted)!=ctx['operations']:errors.append('emitter coverage')
            for r in rows:
                if r['planned_arrival_ns']!=ctx['planned_start_ns']+int(r['event_id']/(ctx['update_rate']*20)*1e9):errors.append('schedule drift')
                details.append(dict(cell=cell,phase=phase,event_id=r['event_id'],op=r['op'],app_queue_ms=(r['worker_start_ns']-r['actual_enqueue_ns'])/1e6,
                    scope_lock_wait_ms=sum((x['acquired_ns']-x['wait_start_ns'])/1e6 for x in r.get('scope_locks',[])),scope_lock_hold_ms=sum((x['released_ns']-x['acquired_ns'])/1e6 for x in r.get('scope_locks',[])),
                    driver_pool_wait_ms=None,backend_observed_raw_call_ms=(r['raw_ack_ns']-r['raw_call_start_ns'])/1e6 if r['op']=='update' else None,
                    id_lock_wait_ms=(r['id_lock_acquired_ns']-r['id_lock_wait_start_ns'])/1e6 if r['op']=='update' else None,
                    structured_maintenance_ms=(r['structured_done_ns']-r['structured_start_ns'])/1e6 if r['op']=='update' else None,
                    sparse_rebuild_ms=(r['sparse_done_ns']-r['sparse_start_ns'])/1e6 if r['op']=='update' else None,
                    dense_rebuild_ms=(r['dense_done_ns']-r['dense_start_ns'])/1e6 if r['op']=='update' else None,
                    ready_check_ms=(r['ready_check_done_ns']-r['ready_check_start_ns'])/1e6 if r['op']=='update' else None,
                    scoring_fusion_ms=sum(r.get(x,0) for x in ('sparse_scoring_ns','dense_scoring_ns','fusion_ns'))/1e6 if r['op']=='read' else None))
            # Independent negative tests only mutate in-memory copies.
            if phase=='measured':
                assert phase_check(ctx,rows[:-1],subs,fixtures)[0];negative.append(cell+': missing event')
                bad=copy.deepcopy(rows);bad[0]['raw_ack_ns']=bad[0]['raw_call_start_ns']-1;assert phase_check(ctx,bad,subs,fixtures)[0];negative.append(cell+': clock')
                bad=copy.deepcopy(rows);bad[0]['audit']['projection']='CORRUPTED';assert phase_check(ctx,bad,subs,fixtures)[0];negative.append(cell+': content')
                shifted=copy.deepcopy(subs)
                for r in shifted:r['actual_enqueue_ns']+=100_000_000
                assert not phase_check(ctx,rows,shifted,fixtures)[1]['offered_load_valid'];negative.append(cell+': arrival lag')
        for item in load(out/cell/'checkpoint.json'):
            if not content_ok(item['audit'],fixtures[item['fixture_id']]):errors.append(cell+': final checkpoint')
        if not lines(out/cell/'resource_usage.jsonl'):errors.append(cell+': missing resources')
    for name,data in [('workload-validity.csv',summary),('phase-breakdown.csv',details)]:
        with (out/name).open('w',newline='',encoding='utf-8-sig') as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    measured=[r for r in summary if r['phase']=='measured']
    result=dict(passed=not errors and all(r['offered_load_valid'] and r['completed_before_absolute_drain'] and r['all_updates_eventually_ready'] for r in summary),errors=errors,
       cells=4,measured_updates=sum(r['updates'] for r in measured),measured_reads=sum(r['reads'] for r in measured),negative_tests=negative,formal_started=False,summary=summary)
    (out/'validate.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
