import json,sys,hashlib,collections
from pathlib import Path
sys.path.insert(0,'D:/memorytable/cassmem_e3_preflight_20261015')
from validate import assess
def main(out):
    load=lambda n:json.loads((out/n).read_text(encoding='utf-8'))
    lines=lambda n:[json.loads(s) for s in (out/n).read_text(encoding='utf-8').splitlines()]
    cases=load('schedules.json');fixtures=load('fixtures.json');es=lines('timeline.jsonl');rs=lines('reads.jsonl');ss=lines('component_states.jsonl');errors=[];summary=[]
    def check(ok,why):
        if not ok:errors.append(why)
    check(len(cases)==45,'matrix size')
    for path,h in load('source_manifest.json').items():check(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,'hash '+path)
    check(len({(r['case_id'],r['seq']) for r in es+rs+ss})==len(es+rs+ss),'unique log IDs')
    for c in cases:
        cid=c['case_id'];ee=[r for r in es if r['case_id']==cid];rr=[r for r in rs if r['case_id']==cid];f=fixtures[c['fixture_id']]
        check(sum(e['kind']=='case_complete' for e in ee)==1,cid+' completion')
        check(all(r['kind']=='read' for r in rr),cid+' read errors')
        hooks=[e for e in ee if e['kind']=='service_hook'];results=[e for e in ee if e['kind']=='apply_result'];expected=[e for e in ee if e['kind']=='expected_exception']
        # Every apply lock wait must have a matching acquired + releasing pair.
        for event in ('acquired','releasing'):check(sum(e['event']==event for e in hooks)==sum(e['event']=='waiting' for e in hooks),cid+' lock coverage '+event)
        active={}
        for e in hooks:
            mid=e['memory_id']
            if e['event']=='acquired':check(mid not in active,cid+' interleaved writers');active[mid]=e['version']
            elif e['event']=='releasing':check(active.pop(mid,None)==e['version'],cid+' unmatched unlock')
            elif e['event'] in ('before','after'):check(active.get(mid)==e['version'],cid+' write without lock')
        check(not active,cid+' held lock')
        for r in rr:
            check(r['start_ns']<=r['snapshot_ns']<=r['query_end_ns']<=r['end_ns']<=r['ns'],cid+' clocks')
            a=assess(r,f,r['target_at_start']);check(a['strict']==r['strict_observed'],cid+' content disagreement')
        finals=[r for r in rr if r['tag'].startswith('final_')];check(len(finals)==3,cid+' final coverage')
        check(all(assess(r,f,3)['strict'] for r in finals),cid+' final regression')
        if c['blocked_stage']:
            release=next(e['ns'] for e in ee if e['kind']=='release_authorized')
            waiting=next(e['ns'] for e in hooks if e['version']==3 and e['event']=='waiting')
            acquire=next(e['ns'] for e in hooks if e['version']==3 and e['event']=='acquired')
            check(waiting<release<acquire,cid+' serialized barrier')
        if c['scenario']=='different_id_progress':
            other=next(e['ns'] for e in ee if e['kind']=='other_id_finished_while_target_blocked');check(other<release,cid+' other ID blocked')
        status=[e['status'] for e in results]
        if c['scenario']=='reverse_arrival':check(status==['applied','stale'],cid+' reverse')
        if c['scenario']=='duplicate':check(status==['applied','duplicate'],cid+' duplicate')
        if c['scenario']=='conflicting_duplicate':check(len(expected)==1 and expected[0]['error_type']=='VersionConflict',cid+' conflict')
        if c['scenario']=='retry_after_sparse':check(status==['stale','applied'] and len(expected)==1 and expected[0]['error_type']=='InjectedFailure',cid+' recovery')
        # Stale and duplicate calls must not mutate; check final two admissions via hook segments.
        if c['scenario'] in ('reverse_arrival','duplicate','conflicting_duplicate'):
            last_acquire=max(e['ns'] for e in hooks if e['event']=='acquired')
            check(not any(e['ns']>last_acquire and e['event']=='before' for e in hooks),cid+' rejected event wrote')
        summary.append(dict(case_id=cid,final_correct=all(assess(r,f,3)['strict'] for r in finals),reads=len(rr)))
    result=dict(passed=not errors,errors=errors,cases=len(cases),reads=len(rs),final_probes=sum(r['tag'].startswith('final_') for r in rs),
       final_regression_cases=sum(not r['final_correct'] for r in summary),same_id_serialized_cases=sum(bool(c['blocked_stage']) for c in cases),
       different_id_progress_cases=sum(c['scenario']=='different_id_progress' for c in cases),formal_run=False)
    (out/'validate.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(out/'case_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
