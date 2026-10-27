import json,hashlib,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=Path('D:/memorytable/cassmem_e3_preflight_20261015/smoke_f0978e6a51a6')
MODULE=Path('D:/memorytable/cassandra-kg-memory/src/cassmem/serving/serialized_updates.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    scenarios={'control':None,'serialized_structured_delay':'structured','serialized_sparse_delay':'sparse','serialized_dense_delay':'dense',
        'reverse_arrival':None,'duplicate':None,'conflicting_duplicate':None,'retry_after_sparse':None,'different_id_progress':'structured'}
    p=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'))
    p.update(experiment='E3-serialized-update-minimal-repair',scenarios=scenarios,
        schedule='v2 paused WITH per-ID lock; v3 confirmed waiting; release v2 then v3 completes. Different-ID case advances another ID while v2 paused.',
        baseline='New opt-in SerializedUpdateService. Original backend/index and baseline files unchanged.',
        alias_boundary='Not the original forced v3-overtakes-v2 schedule; no generation-fencing claim.',
        semantics='Highest admitted version retained after failure. Same-version payload conflicts rejected. Retry same payload allowed. Single process/owner; explicit verified bootstrap required.',
        code_hashes={str(x):sha(x) for x in [MODULE,*HERE.glob('*.py')]})
    out=HERE/('smoke_'+hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:12]);out.mkdir(exist_ok=False)
    save(out/'protocol.json',p);shutil.copy2(BASE/'fixtures.json',out/'fixtures.json');shutil.copy2(BASE/'environment.json',out/'environment.json')
    save(out/'schedules.json',[dict(case_id=f'{cell}__{sc}__{i}',cell=cell,scenario=sc,fixture_id=i,blocked_stage=stage) for cell in p['cells'] for sc,stage in scenarios.items() for i in range(5)])
    paths=[MODULE,*HERE.glob('*.py'),Path('D:/memorytable/cassmem_e3_preflight_20261015/runner.py'),Path('D:/memorytable/cassmem_e3_preflight_20261015/validate.py')]
    baseline_manifest=json.loads((BASE/'source_manifest.json').read_text(encoding='utf-8'))
    paths += [Path(x) for x in baseline_manifest if '/src/' in x.replace('\\','/')]
    paths += [out/'protocol.json',out/'fixtures.json',out/'schedules.json']
    save(out/'source_manifest.json',{str(x):sha(x) for x in paths});(out/'code_snapshot').mkdir()
    for x in [MODULE,*HERE.glob('*.py')]:shutil.copy2(x,out/'code_snapshot'/x.name)
    print(out)
if __name__=='__main__':main()
