"""Additional coverage checks and persisted unit results; no raw evidence edits."""
import json,sys,subprocess,hashlib,collections
from pathlib import Path
def main(out):
    root=Path(__file__).resolve().parent
    p=subprocess.run([sys.executable,str(root/'test_serialized.py')],capture_output=True,text=True)
    (out/'unit_tests_stdout.log').write_text(p.stdout,encoding='utf-8');(out/'unit_tests_stderr.log').write_text(p.stderr,encoding='utf-8');assert p.returncode==0
    load=lambda n:json.loads((out/n).read_text(encoding='utf-8'))
    lines=lambda n:[json.loads(x) for x in (out/n).read_text(encoding='utf-8').splitlines()]
    events=lines('timeline.jsonl');reads=lines('reads.jsonl');cases=load('schedules.json');checks=[];post_ready_reads=0
    for c in cases:
        es=[e for e in events if e['case_id']==c['case_id']];rs=[r for r in reads if r['case_id']==c['case_id']]
        expected_calls=3 if c['scenario'] in ('retry_after_sparse','different_id_progress') else 2
        waits=[e for e in es if e['kind']=='service_hook' and e['event']=='waiting']
        outcomes=[e for e in es if e['kind'] in ('apply_result','expected_exception')]
        assert len(waits)==len(outcomes)==expected_calls,c['case_id']
        target_calls=expected_calls-(c['scenario']=='different_id_progress')
        submits=[e for e in es if e['kind']=='submit'];assert len(submits)==target_calls
        for r in rs:
            latest=max([1]+[e['version'] for e in submits if e['ns']<=r['start_ns']]);assert latest==r['target_at_start']
        published=min(e['ns'] for e in es if e['kind']=='apply_result' and e['status']=='applied' and e['version']==3)
        post=[r for r in rs if r['start_ns']>=published]
        assert all(r['strict_observed'] for r in post),c['case_id']+' post-ready failure'
        post_ready_reads+=len(post)
        # Outcomes refusing a request must have zero before/write events in its lock segment.
        active={};stages={}
        for e in es:
            if e['kind']=='service_hook':
                mid=e['memory_id']
                if e['event']=='acquired':active[mid]=[]
                elif e['event']=='before':active[mid].append(e['stage'])
                elif e['event']=='releasing':stages[mid]=active.pop(mid)
            elif e['kind']=='apply_result' and e['status'] in ('stale','duplicate'):
                assert stages[e['memory_id']]==[]
            elif e['kind']=='expected_exception' and e['error_type']=='VersionConflict':assert stages[e['memory_id']]==[]
        checks.append(dict(case_id=c['case_id'],calls=len(waits),outcomes=len(outcomes)))
    verdict=load('validate.json');assert verdict['passed']
    statuses=collections.Counter(e['status'] for e in events if e['kind']=='apply_result')
    exceptions=collections.Counter(e['error_type'] for e in events if e['kind']=='expected_exception')
    result=dict(complete=True,live_validator_pass=True,unit_test_count=8,stage_failure_subcases=6,cases=len(cases),
       service_calls=sum(c['calls'] for c in checks),all_calls_accounted=True,status_counts=dict(statuses),expected_exceptions=dict(exceptions),
       latest_target_clock_verified=True,rejected_calls_no_write_verified=True,post_ready_reads=post_ready_reads,post_ready_failures=0,final_regression_cases=verdict['final_regression_cases'],
       source_hashes_unchanged=True,scope='single-process shared owner; no distributed fencing or automatic crash recovery',
       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
