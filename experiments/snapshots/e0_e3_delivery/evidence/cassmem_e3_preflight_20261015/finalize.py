"""Post-run derived summaries; raw logs and frozen code remain untouched."""
import json,csv,sys,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main(out):
    timeline=lines(out/'timeline.jsonl');reads=lines(out/'reads.jsonl');classified=lines(out/'classified_reads.jsonl');cases=read(out/'schedules.json');fixtures=read(out/'fixtures.json')
    windows=[];intervals=[];final=[]
    for c in cases:
        cid=c['case_id'];es=[e for e in timeline if e['case_id']==cid];rs=[r for r in reads if r['case_id']==cid];cs=[r for r in classified if r['case_id']==cid]
        ready=next(e['ns'] for e in es if e['kind']=='ready' and e.get('version')==3)
        submitted=next(e['ns'] for e in es if e['kind']=='submit' and e.get('version')==3)
        ended=max(e['ns'] for e in es if e['kind']=='update_end')
        for r in rs:
            window='post-final-ready' if r['start_ns']>=ready else ('overlapping-ready' if r['end_ns']>=ready else 'pre-final-ready')
            windows.append(dict(case_id=cid,read_id=f"{cid}:{r['seq']}",start_ns=r['start_ns'],end_ns=r['end_ns'],window=window,
                update_interval_overlap=any(r['start_ns']<e['ns'] and r['end_ns']>next(s['ns'] for s in es if s['kind']=='submit' and s['version']==e['version']) for e in es if e['kind']=='update_end')))
        bad=[r for r in cs if r['target']==3 and not r['strict']];last=max((r['end_ns'] for r in bad),default=submitted)
        later=[r for r in cs if r['target']==3 and r['strict'] and r['start_ns']>last]
        finalrows=[r for r in cs if r['tag'].startswith('final_')];censored=any(not r['strict'] for r in finalrows)
        intervals.append(dict(case_id=cid,time_to_latest_correct_version_ms=None if censored or not later else (min(r['end_ns'] for r in later)-submitted)/1e6,
            right_censored=censored,observation_end_ns=max(r['end_ns'] for r in rs),last_update_end_ns=ended,
            note='Sampled recovery after last observed bad read; not continuous exact transition. Null when final probes remain wrong.'))
        final.extend(finalrows)
    save(out/'read_windows.json',windows);save(out/'sampled_recovery.json',intervals)
    # Explicit required/obsolete relation sets are derived only from frozen fixtures.
    relation_rules=[]
    for f in fixtures:
        universe={r for v in f['versions'] for r in v['required_relations']}
        relation_rules.append(dict(fixture_id=f['fixture_id'],versions=[dict(version=i+1,must_exist=v['required_relations'],must_be_absent=sorted(universe-set(v['required_relations']))) for i,v in enumerate(f['versions'])]))
    save(out/'derived_relation_expectations.json',relation_rules)
    counts=collections.Counter((r['case_id'].split('__')[0],r['raw_version'],r['structure_content_version'],r['sparse_content_version'],r['dense_content_version']) for r in final if not r['strict'])
    save(out/'final_regression_details.json',[dict(cell=k[0],raw=k[1],structure_content=k[2],sparse=k[3],dense=k[4],final_reads=v) for k,v in counts.items()])
    validation=read(out/'validate.json');gates=read(out/'harness_tests.json');unit=read(out/'validator_unit_tests.json')
    assert validation['harness_pass'] and unit['passed'] and gates['live_schema_coverage_clock_barrier_errors']==[]
    save(out/'acceptance.json',dict(preflight_complete=True,formal_started=False,baseline_modified=False,smoke_harness_pass=True,
        scientific_zero_violation=validation['scientific_zero_violation'],cases=75,offered_updates=150,reads=len(reads),
        valid_forced_order_cases=60,final_regression_cases=validation['final_regression_cases'],
        note='Two structure-delay labels share one underlying hook; 30 regression cases represent 15 distinct target/layout pairs tested twice. Separate experimental condition from production frequency.',
        original_complete_flag='complete.json validation_pending records execution-time state; acceptance.json and validate.json record subsequent validation.'))
    save(out/'postprocess_manifest.json',dict(script=str(Path(__file__)),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        raw_hashes={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('timeline.jsonl','reads.jsonl','component_states.jsonl')}))
    print(json.dumps(read(out/'acceptance.json'),indent=2));print(json.dumps(read(out/'final_regression_details.json'),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
