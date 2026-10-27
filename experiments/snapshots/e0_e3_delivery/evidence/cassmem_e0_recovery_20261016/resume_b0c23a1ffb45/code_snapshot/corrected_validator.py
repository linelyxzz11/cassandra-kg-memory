"""Exact expected-hash correction. Original logs and verdicts are immutable."""
import importlib.util,copy,json,hashlib
from pathlib import Path
from dense_audit import normalized,digest
HERE=Path(__file__).resolve().parent
OLD=Path('D:/memorytable/cassmem_e0_formal_20261015/run_b9da27ebdd6e')
spec=importlib.util.spec_from_file_location('frozen_validator_v1',OLD/'code_snapshot/validator.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
old_content=base.content
def content(a,f):
    fixed=dict(f);fixed['versions']=list(f['versions']);v=dict(f['versions'][1])
    v['normalized_vector_sha256']=digest(normalized(v['vector']));fixed['versions'][1]=v
    return old_content(a,fixed)
base.content=content
audit_phase=base.audit_phase
lines=base.lines
def validate_point(dest):
    dest=Path(dest);oldwrite=base.write_csv
    if dest.parent==OLD:
        target=HERE/'revalidation'/dest.name;target.mkdir(parents=True,exist_ok=True)
        base.write_csv=lambda p,rows:oldwrite(target/Path(p).name,rows)
    try:return base.validate_point(dest)
    finally:base.write_csv=oldwrite
def summarize(out):return base.summarize(out)
def revalidate():
    from dense_audit import sha,load
    evidence=load(HERE/'dense_audit/raw_manifest.json')
    assert all(sha(p)==h for p,h in evidence.items())
    results=[]
    for d in sorted(OLD.iterdir()):
        if not (d/'point.json').exists():continue
        v=validate_point(d);target=HERE/'revalidation'/d.name
        base.save(target/'validate_corrected.json',v)
        results.append(dict(point=d.name,original_status=load(d/'validate.json')['status'],corrected_status=v['status'],measured=v['phases']['measured'],errors=v['errors']))
    assert len(results)==18 and all(sha(p)==h for p,h in evidence.items())
    # Exact comparison still rejects damaged embeddings and projection.
    d=next(d for d in OLD.iterdir() if (d/'point.json').exists());fs=load(d/'fixtures.json');r=lines(d/'measured/events.jsonl')[0];f=fs[r['fixture_id']]
    assert content(r['audit'],f)
    bad=copy.deepcopy(r['audit']);bad['dense_vector_sha256']='0'*64;assert not content(bad,f)
    bad=copy.deepcopy(r['audit']);bad['projection']='damaged';assert not content(bad,f)
    wrong=copy.deepcopy(f);wrong['versions'][1]['vector'][0]+=0.01;assert not content(r['audit'],wrong)
    base.save(HERE/'revalidation/summary.json',dict(points=results,original_files_unchanged=True,negative_tests=3))
    print(json.dumps({'points':len(results),'statuses':[r['corrected_status'] for r in results],'negative_tests':3}))
if __name__=='__main__':revalidate()
