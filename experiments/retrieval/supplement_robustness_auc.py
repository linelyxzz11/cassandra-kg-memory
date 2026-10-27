"""Supplement existing results; full-range trapezoidal AUC, no retrieval rerun."""
import csv,json,hashlib,itertools
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[2]/'results/retrieval/representation_robustness_20261012'
metrics=['MRR@10','Recall@10','nDCG@10','Hit@1','Hit@10'];cols=[0,4,5,1,3]
rates=np.array([0,.1,.25,.5,.75,1.])
weights=np.array([.05,.125,.20,.25,.25,.125])
assert np.isclose(weights.sum(),1)
methods=json.loads((OUT/'methods.json').read_text(encoding='utf-8'))
qs=json.loads((OUT/'queries.json').read_text(encoding='utf-8'))
ix=np.array([i for i,q in enumerate(qs) if q['split']=='test'])
scopes=sorted({qs[i]['scope'] for i in ix})
groups=[np.array([j for j,i in enumerate(ix) if qs[i]['scope']==s]) for s in scopes]
sizes=np.array([len(g) for g in groups])
seeds=list(range(20261012,20261022))
curves=[];aucs=[];pairs=[];report={}
for mode in ['memory','field']:
    integrated=np.zeros((10,15,1146,5));means=[]
    for rate,w in zip(rates,weights):
        vv=[]
        for seed in (seeds[:1] if rate in [0,1] else seeds):
            name=f'{"memory" if rate==0 else mode}_r{int(rate*100):03d}_s{seed}'
            x=np.load(OUT/'conditions'/f'{name}.npz')['metrics'][:,ix][:,:,cols]
            vv.append(x)
        stack=np.stack(vv)
        integrated+=w*stack # endpoints identical for all seeds
        mean=stack.mean(axis=(0,2));means.append(mean)
        for j,m in enumerate(methods):curves.append({'mode':mode,'rate':rate,'method':m,'n':1146,'seeds':len(vv),**dict(zip(metrics,mean[j]))})
    point=integrated.mean(axis=(0,2))
    manual=sum((rates[j+1]-rates[j])*(means[j]+means[j+1])/2 for j in range(5))
    assert np.allclose(point,manual,atol=1e-12)
    sums=np.stack([integrated[:,:,g,:].sum(2) for g in groups],axis=1)
    rng=np.random.default_rng(20261023);boot=[]
    for _ in range(40):
        sd=rng.integers(0,10,(250,10));gd=rng.integers(0,8,(250,8))
        boot.append(sums[sd[:,:,None],gd[:,None,:]].sum(axis=(1,2))/(10*sizes[gd].sum(1))[:,None,None])
    boot=np.concatenate(boot)
    for j,m in enumerate(methods):
        for k,metric in enumerate(metrics):
            ci=np.quantile(boot[:,j,k],[.025,.975])
            aucs.append({'mode':mode,'method':m,'metric':metric,'auc_0_100':point[j,k],'ci_low':ci[0],'ci_high':ci[1]})
    for a,b in itertools.combinations(range(15),2):
        for k,metric in enumerate(metrics):
            ci=np.quantile(boot[:,a,k]-boot[:,b,k],[.025,.975])
            pairs.append({'mode':mode,'a':methods[a],'b':methods[b],'metric':metric,'delta_auc':point[a,k]-point[b,k],'ci_low':ci[0],'ci_high':ci[1]})
    report[mode]={'auc':[{ 'method':m,**{metric:round(float(point[j,k]),6) for k,metric in enumerate(metrics)}} for j,m in enumerate(methods)],'winners_by_rate':[{ 'rate':float(r),**{metric:methods[int(np.argmax(means[i][:,k]))] for k,metric in enumerate(metrics)}} for i,r in enumerate(rates)]}
for name,rows in [('supplement_metrics_by_rate.csv',curves),('robustness_auc.csv',aucs),('robustness_auc_paired.csv',pairs)]:
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
contract={'range':[0,1],'rates':rates.tolist(),'weights':weights.tolist(),'definition':'absolute normalized trapezoidal area under score-versus-missingness; not ROC-AUC; not relative to clean score','scope':'heldout1146','bootstrap':'10000 paired joint seed/conversation draws, same draws across methods and all rates; seed20261023; marginal95%CI, no multiplicity adjustment','verification':'per-seed integration then averaging equals trapezoid of mean curves to1e-12','source_overall_sha256':hashlib.sha256((OUT/'overall.csv').read_bytes()).hexdigest()}
(OUT/'auc_protocol.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
print(json.dumps({'clean':[r for r in curves if r['mode']=='memory' and r['rate']==0],'analysis':report},indent=2))
