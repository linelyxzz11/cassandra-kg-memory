"""Regression check new independent verifier against preserved smoke logs."""
import json,collections
from pathlib import Path
import validator as V
ROOTS=[('baseline',Path('D:/memorytable/cassmem_e3_preflight_20261015/smoke_f0978e6a51a6')),
       ('serialized',Path('D:/memorytable/cassmem_e3_serialized_20261015/smoke_95adb3ae31e5')),
       ('serialized',Path('D:/memorytable/cassmem_e0_preflight_20261015/fourth_528ea7aa498e'))]
def main():
    count=0;regressions=0
    for impl,root in ROOTS:
        fs=V.load(root/'fixtures.json');cs=V.load(root/'schedules.json');streams={}
        for n in ('timeline','reads','component_states'):
            groups=collections.defaultdict(list)
            for r in V.lines(root/(n+'.jsonl')):groups[r['case_id']].append(r)
            streams[n]=groups
        for c in cs:
            allowed={'control','v2_sparse_delayed','v2_dense_delayed','v2_relation_cleanup_delayed'} if impl=='baseline' else {'control','serialized_structured_delay','serialized_sparse_delay','serialized_dense_delay'}
            if c['scenario'] not in allowed:continue
            c={**c,'implementation':impl,'seed':20261015};cid=c['case_id']
            res=V.check(c,fs[c['fixture_id']],streams['timeline'][cid],streams['reads'][cid],streams['component_states'][cid])
            assert res['harness_status']=='PASS',(cid,res['errors'],res['schedule_errors'])
            if impl=='serialized':assert not res['final_regression']
            regressions+=res['final_regression'];count+=1
    result=dict(cases=count,baseline_regressions_detected=regressions,passed=True)
    print(json.dumps(result));return result
if __name__=='__main__':main()
