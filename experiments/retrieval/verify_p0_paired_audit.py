"""Independent verification of saved P0 outputs; no API calls."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

root=Path(__file__).resolve().parents[2]
out=root/'results/p0_audit_20261012'
spec=importlib.util.spec_from_file_location('p0',Path(__file__).with_name('run_p0_paired_audit.py'))
p0=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)
rows=json.loads((out/'reader_paired_per_query.json').read_text(encoding='utf-8'))
results=json.loads((out/'paired_bootstrap.json').read_text(encoding='utf-8'))
passed=[]
for result in results:
    if result['kind']!='reader':continue
    subset=sorted([r for r in rows if r['setting']==result['setting'] and (result['population']=='full1986' or r['heldout'])],key=lambda r:r['query_id'])
    key=result['metric'].lower()
    da=np.array([r['cassmem_'+key]-r['hipporag2_'+key] for r in subset])
    assert len(subset)==result['n']
    assert abs(da.mean()-result['delta'])<1e-12
    # Independent direct concatenation of sampled conversations (not group sums).
    labels=sorted({r['conversation'] for r in subset})
    groups=[np.array([i for i,r in enumerate(subset) if r['conversation']==s]) for s in labels]
    draws=np.random.default_rng(20261012).integers(0,len(labels),size=(10000,len(labels)))
    means=np.array([da[np.concatenate([groups[i] for i in draw])].mean() for draw in draws])
    assert np.max(np.abs(np.quantile(means,[.025,.975])-result['cluster_ci95']))<1e-12
    passed.append(result['population']+'/'+result['setting']+'/'+result['metric'])
# Basic invariants check weighting, sign and deterministic random draws.
groups=['a','a','a','b','c']
a=np.array([[.1],[.3],[.6],[.2],[.4]])
b=a.copy()
_,_,cluster,query=p0.intervals(a,b,groups)
assert np.all(cluster==0) and np.all(query==0)
_,_,cluster,query=p0.intervals(a+.2,b,groups)
assert np.allclose(cluster,.2) and np.allclose(query,.2)
_,_,c1,q1=p0.intervals(a,1-a,groups)
_,_,c2,q2=p0.intervals(1-a,a,groups)
assert np.allclose(c1,-c2[::-1]) and np.allclose(q1,-q2[::-1])
manifest=json.loads((out/'manifest.json').read_text())
assert all(p0.sha(path)==digest for path,digest in manifest['source_sha256'].items())
result=dict(status='PASS',independent_cluster_ci_checks=passed,invariant_checks=['zero_difference','constant_difference','sign_symmetry'],all_input_hashes_unchanged=True)
(out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
