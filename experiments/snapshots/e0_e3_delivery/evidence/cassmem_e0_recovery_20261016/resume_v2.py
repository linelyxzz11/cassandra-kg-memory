"""Bootstrap-only followup: ensure generated password is not a CLI option."""
import sys,inspect,textwrap,json,hashlib,shutil,traceback,datetime
from pathlib import Path
import resume as R
import environment as E
F=R.F;HERE=R.HERE;OLD=R.OLD
def start_source():
    s=textwrap.dedent(inspect.getsource(E.start))
    assert 'password=secrets.token_urlsafe(24)' in s
    return s.replace('password=secrets.token_urlsafe(24)',"password='E0_'+secrets.token_urlsafe(24)")
def prepare():
    F.verify(F.load(OLD/'source_manifest.json'))
    assert F.load(HERE/'dense_audit/result.json')['exact_row_hash_all_match']
    assert all(x['corrected_status']=='PASS' for x in F.load(HERE/'revalidation/summary.json')['points'])
    # Preserve first failed attempt and its credentials; no raw outputs copied over it.
    p=F.load(HERE/'resume_b0c23a1ffb45/protocol.json')
    p['recovery']['container_bootstrap']='Prefix randomly generated temporary password with E0_ to prevent leading hyphen interpreted as a CLI option. No secret recorded in report.'
    p['recovery']['previous_resume_attempt']=str(HERE/'resume_b0c23a1ffb45')
    p['recovery']['changes']={str(x):F.sha(x) for x in HERE.glob('*.py')}
    h=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:12];out=HERE/('resume_'+h);out.mkdir(exist_ok=False)
    F.save(out/'protocol.json',p);F.save(out/'formal_matrix.json',[x for x in F.load(OLD/'formal_matrix.json') if x['layout'].startswith('neo4j')])
    for name in ('formal_update_traces.json','environment.json','phase_generated.py'):shutil.copy2(OLD/name,out/name)
    (out/'campaign_generated.py').write_text(R.campaign_source(),encoding='utf-8');(out/'engine_start_generated.py').write_text(start_source(),encoding='utf-8')
    m={**F.load(OLD/'source_manifest.json'),**p['recovery']['changes']}
    for name in ('protocol.json','formal_matrix.json','formal_update_traces.json','phase_generated.py','campaign_generated.py','engine_start_generated.py'):m[str(out/name)]=F.sha(out/name)
    F.save(out/'source_manifest.json',m);(out/'code_snapshot').mkdir()
    for x in HERE.glob('*.py'):shutil.copy2(x,out/'code_snapshot'/x.name)
    print(out)
def run(out):
    assert start_source()==(out/'engine_start_generated.py').read_text(encoding='utf-8')
    g=dict(vars(E));exec(compile(start_source(),str(out/'engine_start_generated.py'),'exec'),g);F.start=g['start']
    R.run(out)
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        out=Path(sys.argv[1])
        try:run(out)
        except Exception:F.save(out/'failure.json',dict(error=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));raise
