import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,collections
from pathlib import Path
import numpy as np
ROOT=Path('D:/memorytable/cassmem_e0_formal_20261015/run_b9da27ebdd6e')
HERE=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalized(v):
    m=np.asarray([v],dtype=np.float32)
    return (m/np.maximum(np.linalg.norm(m,axis=1,keepdims=True),1e-12))[0]
def digest(v):return hashlib.sha256(v.tobytes()).hexdigest()
def main():
    out=HERE/'dense_audit';out.mkdir(exist_ok=False)
    manifest=load(ROOT/'source_manifest.json')
    assert all(sha(p)==h for p,h in manifest.items())
    counts=collections.Counter();mismatch=[];rawhashes={};max_abs=0.;fixtures_checked=set()
    for d in sorted(ROOT.iterdir()):
        if not (d/'point.json').exists():continue
        fs=load(d/'fixtures.json');rawhashes[str(d/'fixtures.json')]=sha(d/'fixtures.json')
        expected={}
        for i,f in enumerate(fs):
            for ver,v in enumerate(f['versions'],1):
                vec=np.asarray(v['vector'],dtype=np.float32);scalar=vec/max(float(np.linalg.norm(vec)),1e-12);row=normalized(vec)
                assert digest(vec)==v['record']['embedding_sha256']
                assert digest(scalar)==v['normalized_vector_sha256']
                max_abs=max(max_abs,float(np.max(np.abs(scalar-row))))
                expected[(i,ver)]=(digest(scalar),digest(row))
                fixtures_checked.add((f['memory_id'],ver,v['record']['embedding_sha256']))
        paths=[d/ph/name for ph in ('warmup','measured') for name in ('events.jsonl','reads.jsonl')]+[d/'checkpoint.json']
        for path in paths:
            rawhashes[str(path)]=sha(path)
            rows=load(path) if path.suffix=='.json' else [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
            for r in rows:
                a=r.get('audit');counts['observations']+=1
                if not a:counts['missing_audit']+=1;continue
                old,new=expected[(r['fixture_id'],a['dense_version'])];actual=a['dense_vector_sha256']
                counts['old_hash_mismatch']+=actual!=old;counts['row_hash_mismatch']+=actual!=new
                if actual!=new:mismatch.append({'file':str(path),'fixture_id':r['fixture_id'],'version':a['dense_version'],'actual':actual,'expected':new})
    assert all(sha(p)==h for p,h in rawhashes.items())
    result=dict(counts=counts,unique_fixture_versions=len(fixtures_checked),max_scalar_vs_row_abs_difference=max_abs,unexplained=mismatch,exact_row_hash_all_match=not mismatch and counts['missing_audit']==0)
    (out/'raw_manifest.json').write_text(json.dumps(rawhashes,indent=2),encoding='utf-8')
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
