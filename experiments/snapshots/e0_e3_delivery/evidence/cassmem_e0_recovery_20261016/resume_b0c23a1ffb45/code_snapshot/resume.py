"""New attempt, frozen amendment; no edits to original runner/raw files."""
import sys,json,inspect,textwrap,hashlib,shutil,traceback,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=Path('D:/memorytable/cassmem_e0_formal_20261015/run_b9da27ebdd6e')
sys.path.insert(0,str(OLD.parent))
import formal as F
import corrected_validator as V
from loader import load_scope_safe
sys.modules['validator']=V
def campaign_source():
    s=textwrap.dedent(inspect.getsource(F.campaign))
    s=s.replace("for engine in ('cassandra','neo4j'):","for engine in ('neo4j',):")
    s=s.replace("b.insert_many(cell,initial[offset:offset+5000],batch=100,workers=12)","load_scope_safe(b,cell,initial[offset:offset+5000],out/(cell+'_loader_retries.jsonl'),workers=12)")
    s=s.replace('points=36','points=18')
    return s
def prepare():
    F.verify(F.load(OLD/'source_manifest.json'))
    audit=F.load(HERE/'dense_audit/result.json');assert audit['exact_row_hash_all_match']
    results=F.load(HERE/'revalidation/summary.json');assert all(x['corrected_status']=='PASS' for x in results['points'])
    changes={str(x):F.sha(x) for x in HERE.glob('*.py')}
    p=F.load(OLD/'protocol.json');p['cells']=[c for c in p['cells'] if c.startswith('neo4j')]
    p['recovery']={'parent_run':str(OLD),'remaining_points':18,'original_full_matrix_points':36,'changes':changes,
      'dense_hash':'Validator uses exact float32 row normalization of frozen vector, same as index. No tolerance relaxation. Original FAIL verdicts preserved; corrected Cassandra analysis stored separately.',
      'loader':'Fresh isolated DB, groups by scope, max12 workers, each scope sequential; at most5 attempts only for Transaction.Outdated, .1/.2/.4/.8 seconds backoff; attempts logged. Pre-measurement only.',
      'measured_path':'Unchanged original phase source, update service, query trace, resource budget, durations and thresholds. No measured-request retries added.',
      'initial_data':'Reload Neo4j 100K into fresh container; preserve interrupted partial-load container. Validate complete ID coverage and per-target content before measurements.'}
    h=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:12];out=HERE/('resume_'+h);out.mkdir(exist_ok=False)
    F.save(out/'protocol.json',p)
    F.save(out/'formal_matrix.json',[x for x in F.load(OLD/'formal_matrix.json') if x['layout'].startswith('neo4j')])
    for name in ('formal_update_traces.json','environment.json','phase_generated.py'):shutil.copy2(OLD/name,out/name)
    (out/'campaign_generated.py').write_text(campaign_source(),encoding='utf-8')
    m={**F.load(OLD/'source_manifest.json'),**changes}
    for name in ('protocol.json','formal_matrix.json','formal_update_traces.json','phase_generated.py','campaign_generated.py'):m[str(out/name)]=F.sha(out/name)
    F.save(out/'source_manifest.json',m)
    (out/'code_snapshot').mkdir()
    for x in HERE.glob('*.py'):shutil.copy2(x,out/'code_snapshot'/x.name)
    print(out)
def run(out):
    F.verify(F.load(out/'source_manifest.json'))
    assert campaign_source()==(out/'campaign_generated.py').read_text(encoding='utf-8')
    gl=dict(vars(F));gl['load_scope_safe']=load_scope_safe
    exec(compile(campaign_source(),str(out/'campaign_generated.py'),'exec'),gl)
    gl['campaign'](out)
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        out=Path(sys.argv[1])
        try:run(out)
        except Exception:F.save(out/'failure.json',dict(error=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));raise
