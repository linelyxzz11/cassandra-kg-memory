import sys,os,json,inspect,shutil,hashlib,datetime,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=Path('D:/memorytable/cassmem_e3_formal_20261016/run_d5e494f2dc2a')
sys.path.insert(0,str(OLD.parent))
import campaign as C
OUT=ROOT/'continuation_v1'

def prepare():
    C.verify(C.load(OLD/'source_manifest.json'))
    rows=[json.loads(s) for s in (OLD/'case_summaries.jsonl').read_text().splitlines()]
    ids=[r['case_id'] for r in rows];assert len(ids)==len(set(ids))==1795
    assert C.load(OLD/'smoke_complete.json')['cases']==160
    evidence={}
    for row in rows:
        dest=OLD/row['phase']/row['case_id']
        res=C.V.validate_case(dest)
        compact={k:v for k,v in res.items() if k not in ('classified','query_ms','extra_audit_ms','id_lock_wait_ms')}
        assert dict(phase=row['phase'],**compact)==row, row['case_id']
        assert res['harness_status']=='PASS'
        for f in dest.iterdir():
            if f.is_file():evidence[str(f)]=C.sha(f)
    OUT.mkdir(exist_ok=False)
    for n in ('protocol.json','fixtures.json','schedules.json','engine_start_generated.py','serialized_run_generated.py','source_manifest.json','smoke_complete.json'):
        shutil.copy2(OLD/n,OUT/n)
    C.save(OUT/'prior_evidence_manifest.json',evidence)
    C.save(OUT/'recovery.json',dict(original=str(OLD),completed_formal=1635,smoke=160,remaining=3165,interrupted_case='formal_cassandra-base_baseline_control_20261018_44',policy='Original evidence immutable; fresh containers; unchanged protocol and case bodies; original summaries copied only as cumulative index.'))
    shutil.copy2(OLD/'case_summaries.jsonl',OUT/'case_summaries.jsonl')
    # Transform orchestration only; per-case execution, validator, and protocol stay frozen.
    source=inspect.getsource(C.main)
    source=source.replace("state=E.suspend_old(out);results=[]", "state=resume_suspend(out);results=prior_results.copy()")
    source=source.replace("for phase in ('smoke','formal'):","for phase in ('formal',):")
    source=source.replace("selected=[c for c in schedule if c['phase']==phase];count=0", "selected=[c for c in schedule if c['phase']==phase];count=len(done)")
    source=source.replace("dest=out/phase/c['case_id'];dest.mkdir", "if c['case_id'] in done:continue\n                        dest=out/phase/c['case_id'];dest.mkdir")
    compile(source,'<continuation>','exec')
    (OUT/'continuation_generated.py').write_text(source,encoding='utf-8')
    C.save(OUT/'recovery_source_manifest.json',{str(p):C.sha(p) for p in (Path(__file__),OUT/'continuation_generated.py')})
    print('Validated 1795 completed cases; prepared 3165 remaining.',flush=True)

def run():
    log=(OUT/'resume.log').open('a',encoding='utf-8',buffering=1)
    sys.stdout=log;sys.stderr=log
    C.verify(C.load(OUT/'recovery_source_manifest.json'))
    C.verify(C.load(OUT/'source_manifest.json'))
    C.verify(C.load(OUT/'prior_evidence_manifest.json'))
    rows=[json.loads(s) for s in (OUT/'case_summaries.jsonl').read_text().splitlines()]
    done={r['case_id'] for r in rows if r['phase']=='formal'}
    assert len(done)==1635 and not (OUT/'formal').exists()
    def resume_suspend(out):
        orphan='cassmem-e3-formal-20261016173919-cassandra'
        running=C.E.docker('ps','--format','{{.Names}}').splitlines()
        assert set(running)=={orphan},'Unexpected containers; stop for review'
        C.save(out/'orphan_state.json',json.loads(C.E.docker('inspect',orphan))[0]['State'])
        C.E.docker('stop','-t','60',orphan)
        state=C.load(OLD/'original_state.json')
        C.save(out/'original_state.json',state)
        return state
    g=dict(vars(C));g.update(resume_suspend=resume_suspend,prior_results=rows,done=done)
    exec(compile((OUT/'continuation_generated.py').read_text(),'<continuation>','exec'),g)
    try:g['main'](OUT)
    except BaseException:
        C.save(OUT/'failure.json',dict(error=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        traceback.print_exc();raise

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:run()
