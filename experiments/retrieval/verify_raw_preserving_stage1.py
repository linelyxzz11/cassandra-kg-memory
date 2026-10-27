"""Independent verification from saved rankings and canonical gold."""
import csv,json,math,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/retrieval/raw_preserving_stage1_20261012'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    gold={r['query_id']:set(r['gold_memory_ids'].split(';')) for r in rows(ROOT/'data/retrieval_gold/locomo_cat1_4_gold_memory.csv') if r['split']=='test' and r['gold_status']=='mapped' and r['gold_memory_ids']}
    rr=rows(OUT/'per_query.csv');groups=defaultdict(list);checks=0
    for r in rr:
        ids=r['top10_memory_ids'].split(';');g=gold[r['query_id']]
        hit=[i+1 for i,m in enumerate(ids) if m in g];first=min(hit) if hit else 999
        assert len(ids)==len(set(ids))==10
        actual={'MRR@10':1/first if first<=10 else 0,'Recall@10':len(g.intersection(ids))/len(g),'Hit@1':int(first==1),'Hit@5':int(first<=5),'Hit@10':int(first<=10),'nDCG@10':sum(1/math.log2(i+1) for i in hit)/sum(1/math.log2(i+2) for i in range(min(10,len(g))))}
        for k,v in actual.items():assert abs(v-float(r[k]))<1e-12;(checks:=checks+1)
        groups[r['method']].append(r)
    assert all(len(v)==len({r['query_id'] for r in v})==1146 for v in groups.values())
    previous=rows(ROOT/'results/representation_ablation/p1_compact_component_ablation/p1c_zscore_per_query_heldout1146_aligned.csv')
    now={(r['method'],r['query_id']):r for r in rr}
    legacy_checks=0
    for r in previous:
        if r['variant']=='RawERKT':continue
        current=now['H_FrozenRaw_'+r['variant'],r['query_id']]
        for k in actual:assert abs(float(current[k])-float(r[k]))<1e-12;legacy_checks+=1
    for r in rows(OUT/'overall.csv'):
        for k in actual:assert abs(np.mean([float(x[k]) for x in groups[r['method']]])-float(r[k]))<1e-12
    cis=json.loads((OUT/'paired_bootstrap.json').read_text())
    for r in cis:
        a=groups[r['a']];b=groups[r['b']];assert [x['query_id'] for x in a]==[x['query_id'] for x in b]
        d=np.array([float(x[r['metric']])-float(y[r['metric']]) for x,y in zip(a,b)])
        scopes=sorted({x['scope'] for x in a});ix=[np.array([i for i,x in enumerate(a) if x['scope']==s]) for s in scopes]
        rng=np.random.default_rng(20261012);vals=[]
        for _ in range(10000):vals.append(d[np.concatenate([ix[k] for k in rng.integers(0,len(scopes),size=len(scopes))])].mean())
        assert np.allclose(np.quantile(vals,[.025,.975]),r['cluster_ci95'],rtol=0,atol=1e-12)
        assert abs(float(d.mean())-r['delta'])<1e-12
    contract=json.loads((OUT/'contract.json').read_text());assert all(sha(Path(p))==h for p,h in contract['sources'].items())
    zero=np.zeros(1146);assert zero.mean()==0
    report={'status':'PASS','methods':len(groups),'per_query_rows':len(rr),'independent_metric_checks':checks,'legacy_eight_variants_metric_checks':legacy_checks,'independent_full_10000_cluster_CIs':len(cis),'all_source_hashes_unchanged':True}
    (OUT/'independent_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
