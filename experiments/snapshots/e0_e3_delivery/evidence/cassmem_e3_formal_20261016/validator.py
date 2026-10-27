"""Independent case verifier; no live runner imports."""
import json,hashlib,copy,importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('old_assessment','D:/memorytable/cassmem_e3_preflight_20261015/validate.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]
def assess(r,f,target):
    ff=copy.deepcopy(f)
    for v in ff['versions']:
        a=np.asarray([v['vector']],dtype=np.float32);a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
        v['normalized_vector_sha256']=hashlib.sha256(a[0].tobytes()).hexdigest()
    return old.assess(r,ff,target)
def check(c,f,es,rs,ss):
    errors=[];schedule=[]
    def ck(ok,msg):
        if not ok:errors.append(msg)
    allrows=sorted(es+rs+ss,key=lambda e:e['seq'])
    ck(len({r['seq'] for r in allrows})==len(allrows),'duplicate event ID')
    ck([r['seq'] for r in allrows]==list(range(len(allrows))),'event sequence coverage')
    ck(all(a['ns']<=b['ns'] for a,b in zip(allrows,allrows[1:])),'monotonic clock')
    ck(sum(e['kind']=='case_complete' for e in es)==1,'case completion')
    ck(not any(e['kind']=='execution_error' for e in es),'execution error')
    ck(all(r['kind']=='read' for r in rs),'read error')
    submits=[e for e in es if e['kind']=='submit'];ck([e['version'] for e in submits]==[2,3],'offered versions')
    if c['implementation']=='baseline':ready=[e['ns'] for e in es if e['kind']=='ready' and e['version']==3]
    else:ready=[e['ns'] for e in es if e['kind']=='service_hook' and e['version']==3 and e['stage']=='ready' and e['event']=='after']
    ck(len(ready)==1,'v3 readiness evidence');ready_ns=ready[0] if ready else 10**30
    if c['blocked_stage']:
        if c['implementation']=='baseline':
            h=[e['ns'] for e in es if e['kind']=='hook_reached'];rel=[e['ns'] for e in es if e['kind']=='hook_released'];st=[e['ns'] for e in es if e['kind']=='stage_start' and e['version']==2 and e['stage']==c['blocked_stage']]
            if not (len(h)==len(rel)==len(st)==1 and h[0]<ready_ns<rel[0]<=st[0]):schedule.append('forced overtaking not proven')
        else:
            hs=[e for e in es if e['kind']=='service_hook'];rel=[e['ns'] for e in es if e['kind']=='release_authorized']
            wa=[e['ns'] for e in hs if e['version']==3 and e['event']=='waiting'];ac=[e['ns'] for e in hs if e['version']==3 and e['event']=='acquired']
            if not(len(rel)==len(wa)==len(ac)==1 and wa[0]<rel[0]<ac[0]):schedule.append('serialization not proven')
    waits=[]
    if c['implementation']=='serialized':
        active={};ws={}
        for e in [e for e in es if e['kind']=='service_hook']:
            mid=e['memory_id'];v=e['version'];ev=e['event']
            if ev=='waiting':ws[(mid,v)]=e['service_ns']
            elif ev=='acquired':
                ck(mid not in active,'overlapping same-ID writers');active[mid]=v;waits.append((e['service_ns']-ws.get((mid,v),e['service_ns']))/1e6)
            elif ev=='releasing':ck(active.pop(mid,None)==v,'unmatched unlock')
            else:ck(active.get(mid)==v,'operation outside ID lock')
        ck(not active,'lock retained')
        ck([e['status'] for e in es if e['kind']=='apply_result']==['applied','applied'],'apply outcomes')
    classified=[];query=[];extra=[];update_latencies=[];intervals=[]
    for sub in submits:
        end=[e['ns'] for e in es if e.get('version')==sub['version'] and e['kind'] in ('update_end','apply_result')]
        if end:
            intervals.append((sub['ns'],max(end)));update_latencies.append(dict(version=sub['version'],submit_to_end_ms=(max(end)-sub['ns'])/1e6))
    for r in rs:
        if r['kind']!='read':continue
        target=max([1]+[e['version'] for e in submits if e['ns']<=r['start_ns']])
        ck(target==r['target_at_start'],'target-at-read-start')
        ck(r['start_ns']<=r['snapshot_ns']<=r['query_end_ns']<=r['end_ns']<=r['ns'],'read clock')
        a=assess(r,f,target);ck(a['strict']==r['strict_observed'],'independent content disagreement')
        window='post-final-ready' if r['start_ns']>=ready_ns else 'overlapping' if any(r['start_ns']<end and r['end_ns']>start for start,end in intervals) else 'pre-ready'
        classified.append(dict(tag=r['tag'],start_ns=r['start_ns'],end_ns=r['end_ns'],window=window,post_ready=r['start_ns']>=ready_ns,target=target,**a))
        query.append((r['query_end_ns']-r['start_ns'])/1e6);extra.append((r['end_ns']-r['query_end_ns'])/1e6)
    finals=[r for r in classified if r['tag'].startswith('final_')];initial=[r for r in classified if r['tag']=='initial'];post=[r for r in classified if r['post_ready']]
    ck(len(finals)==3,'final probe count');ck(len(initial)==1 and initial[0]['strict'],'initial state')
    regression=any(not r['strict'] for r in finals)
    # Scientific violations retained as observations, including repaired-arm violations.
    submit3=next((e['ns'] for e in submits if e['version']==3),None)
    good=[r for r in classified if r['target']==3 and r['strict']]
    return dict(case_id=c['case_id'],implementation=c['implementation'],cell=c['cell'],scenario=c['scenario'],seed=c['seed'],fixture_id=c['fixture_id'],
      harness_status='FAIL' if errors else 'INVALID' if schedule else 'PASS',errors=errors,schedule_errors=schedule,reads=len(classified),post_ready_reads=len(post),
      mixed=sum(r['mixed'] for r in classified),unknown=sum(r['unknown'] for r in classified),stale_candidate=sum(r['stale_candidate'] for r in classified),stale_relation=sum(r['stale_relation'] for r in classified),
      post_ready_failures=sum(not r['strict'] for r in post),final_regression=regression,
      guard_rejections=sum(e['kind']=='guard_rejection' for e in es),
      first_latest_correct_ms=(min(r['end_ns'] for r in good)-submit3)/1e6 if good and submit3 else None,
      final_incorrect_right_censored=regression,id_lock_wait_ms=waits,update_latencies=update_latencies,query_ms=query,extra_audit_ms=extra,classified=classified)
def validate_case(d):
    d=Path(d);c=load(d/'case.json');f=load(d/'fixtures.json')[0]
    return check(c,f,lines(d/'timeline.jsonl'),lines(d/'reads.jsonl'),lines(d/'component_states.jsonl'))
def negative_tests(d):
    d=Path(d);c=load(d/'case.json');f=load(d/'fixtures.json')[0];es=lines(d/'timeline.jsonl');rs=lines(d/'reads.jsonl');ss=lines(d/'component_states.jsonl')
    assert check(c,f,es,rs,ss)['harness_status']=='PASS'
    assert check(c,f,es[:-1],rs,ss)['harness_status']=='FAIL'
    assert check(c,f,es+[es[0]],rs,ss)['harness_status']=='FAIL'
    bad=copy.deepcopy(rs);bad[0]['query_end_ns']=bad[0]['start_ns']-1;assert check(c,f,es,bad,ss)['harness_status']=='FAIL'
    bad=copy.deepcopy(rs);bad[0]['dense_vector_sha256']='0'*64;assert check(c,f,es,bad,ss)['harness_status']=='FAIL'
    bad=copy.deepcopy(rs);bad[0]['projection']='corrupt';assert check(c,f,es,bad,ss)['harness_status']=='FAIL'
    bad=[e for e in es if not(e['kind']=='submit' and e['version']==2)];assert check(c,f,bad,rs,ss)['harness_status']=='FAIL'
    if c['blocked_stage']:
        bad=copy.deepcopy(es)
        for e in bad:
            if e['kind']=='hook_released' or e['kind']=='release_authorized':e['ns']=0
        assert check(c,f,bad,rs,ss)['harness_status']!='PASS'
    return 7 if c['blocked_stage'] else 6
