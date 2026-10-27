"""Paired uncertainty, independent rank metric verification and complete report.

Does not modify the runner contract or historical files. --partial supports
baseline inspection; complete mode requires all83 unique conditions.
"""
import argparse, itertools, json, math
import numpy as np
import run_representation_robustness as run

OUT=run.OUT
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
    qs=read('queries.json');ids=read('memory_order.json');methods=read('methods.json')
    test=np.array([i for i,q in enumerate(qs) if q['split']=='test'])
    scopes=sorted({qs[i]['scope'] for i in test});groups=[np.array([j for j,i in enumerate(test) if qs[i]['scope']==s]) for s in scopes]
    sizes=np.array([len(g) for g in groups]);conditions={};checks=0
    for f in sorted((OUT/'conditions').glob('*.npz')):
        meta=read('conditions/'+f.stem+'.json');z=np.load(f)
        assert run.old.sha(f)==meta['sha256']
        scores=z['metrics'];tops=z['top10'];bits=z['field_bits']
        assert scores.shape==(15,1536,6) and tops.shape==(15,1536,10)
        # Independent metric calculation from saved ranks; no scorer helper.
        for k,q in enumerate(qs):
            gold=set(q['gold']);ranked=[[ids[j] for j in row] for row in tops[:,k]]
            rel=np.array([[m in gold for m in row] for row in ranked],dtype=float)
            rr=np.max(rel/np.arange(1,11),axis=1)
            denom=sum(1/math.log2(j+2) for j in range(min(10,len(gold))))
            expected=np.column_stack([rr,rel[:,0],rel[:,:5].max(1),rel.max(1),rel.sum(1)/len(gold),(rel/np.log2(np.arange(2,12))).sum(1)/denom])
            assert np.allclose(scores[:,k],expected,atol=1e-14,rtol=0),f.name
            checks+=expected.size
        conditions[(meta['mode'],meta['rate'],meta['seed'])]=(scores[:,test][:,:,[0,4]],tops[:,test],bits)
    assert conditions,'no completed conditions'
    clean=conditions[('memory',0.,run.SEEDS[0])]
    for key,(_,top,bits) in conditions.items():
        for method in ['Dense_Raw','BM25_Raw','H_Raw_Raw']:
            a=methods.index(method);assert np.array_equal(top[a],clean[1][a]),(key,method)
        if key[1]==1:
            assert not bits.any()
            for a,b in [('Dense_Raw','Dense_RawERK'),('BM25_Raw','BM25_RawERK'),('H_Raw_Raw','H_RawERK_RawERK')]:assert np.array_equal(top[methods.index(a)],top[methods.index(b)])
    comparisons=[];summary=[]
    for mode in ['memory','field']:
        for rate in run.RATES:
            seeds=run.SEEDS[:1] if rate in [0,1] else run.SEEDS
            keys=[('memory' if rate==0 else mode,rate,s) for s in seeds]
            if not all(k in conditions for k in keys):continue
            values=np.stack([conditions[k][0] for k in keys]) # seed,method,query,metric
            point=values.mean(axis=(0,2));sums=np.stack([values[:,:,g,:].sum(2) for g in groups],axis=1) # seed,group,method,metric
            samples=[];rng=np.random.default_rng(20261022)
            for start in range(0,10000,250):
                sd=rng.integers(0,len(seeds),(250,len(seeds)))
                gd=rng.integers(0,len(groups),(250,len(groups)))
                selected=sums[sd[:,:,None],gd[:,None,:]]
                samples.append(selected.sum(axis=(1,2))/(len(seeds)*sizes[gd].sum(1))[:,None,None])
            boot=np.concatenate(samples)
            for a,m in enumerate(methods):
                for j,metric in enumerate(['MRR@10','Recall@10']):
                    ci=np.quantile(boot[:,a,j],[.025,.975]);seedmeans=values[:,a,:,j].mean(1)
                    summary.append({'mode':mode,'rate':rate,'method':m,'metric':metric,'n_queries':1146,'n_seeds':len(seeds),'mean':point[a,j],'seed_sd':float(seedmeans.std(ddof=1)) if len(seeds)>1 else 0.,'ci_low':ci[0],'ci_high':ci[1],'delta_from_clean':point[a,j]-clean[0][a,:,j].mean()})
            for a,b in itertools.combinations(range(15),2):
                for j,metric in enumerate(['MRR@10','Recall@10']):
                    ci=np.quantile(boot[:,a,j]-boot[:,b,j],[.025,.975])
                    comparisons.append({'mode':mode,'rate':rate,'a':methods[a],'b':methods[b],'metric':metric,'delta':point[a,j]-point[b,j],'ci_low':ci[0],'ci_high':ci[1],'n_seeds':len(seeds),'n_queries':1146})
    run.csvout('paired_all_comparisons.csv',comparisons);run.csvout('robustness_summary.csv',summary)
    complete=len(conditions)==83
    run.save('verification.json',{'status':'PASS','conditions_verified':len(conditions),'independent_metric_checks':checks,'raw_invariance':True,'endpoint_checks':'checked where present','complete':complete})
    lines=['# 表示组合与结构缺失鲁棒性实验','','## 实验设置','','- 6 个单路 + 9 个混合组合，统一本地 BGE-large-en-v1.5、float32、CLS/L2、1500字符/512tokens，固定 query 向量。','- 主结果 held-out1146；dev390、全1536单列。所有memory保留，原始空结构不回填原文。','- 各分支Top50，最终Top10，alpha=.6，原BM25、branch-min和稳定排序。','- memory：整条结构字段共同缺失；field：E/R/K分别缺失。rate=0/.10/.25/.50/.75/1；10个固定种子20261012..20261021；端点一次确定性运行。','- mask以scope和seed生成，与query/gold无关，各率嵌套；报告实际剩余非空结构数。','- Dense结构输入按受损文本编码；相同文本按完整字符串复用相同编码。不同文本不复用原始完整结构向量。','- 10,000次配对conversation与seed联合重采样，query-weighted micro均值；全部105对方法、两项主指标给出边际95%区间，未做多重检验校正。仅8个held-out conversations。','- 本轮无Reader/Judge调用，不修改论文；历史冻结结果独立保留。','','## 0%缺失，held-out完整矩阵','','|Method|MRR@10|Recall@10|','|---|---:|---:|']
    for a,m in enumerate(methods):lines.append(f'|{m}|{clean[0][a,:,0].mean():.6f}|{clean[0][a,:,1].mean():.6f}|')
    lines+=['','## 缺失鲁棒性结果','','|Mode|Rate|Method|MRR@10|Recall@10|Seeds|','|---|---:|---|---:|---:|---:|']
    for r in summary:
        if r['metric']!='MRR@10':continue
        rr=next(x for x in summary if x['mode']==r['mode'] and x['rate']==r['rate'] and x['method']==r['method'] and x['metric']=='Recall@10')
        lines.append(f"|{r['mode']}|{r['rate']:.0%}|{r['method']}|{r['mean']:.6f}|{rr['mean']:.6f}|{r['n_seeds']}|")
    lines+=['','## 数据文件','','- overall.csv：每种子、每split全部六指标。','- by_category.csv：每条件每split每问题类型六指标。','- robustness_summary.csv：主指标均值、seed标准差、95%CI和相对自身clean下降。','- paired_all_comparisons.csv：所有方法两两配对区间，差值为A减B。','- conditions/*.npz：逐题六指标、Top10、共同缺失mask；索引对应queries.json/memory_order.json。','- contract.json：运行前设置与来源哈希；verification.json：独立度量核验。','','状态：'+('全部83个独立条件完成。' if complete else f'已核验{len(conditions)}个条件，其余仍在运行；不是完整最终结果。')]
    (OUT/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    if not args.partial:
        assert complete,f'only {len(conditions)}/83 conditions'
        contract=read('contract.json');assert all(run.old.sha(run.Path(p))==h for p,h in contract['sources'].items())
        run.save('completion.json',{'status':'complete','conditions':83,'methods':15,'primary_queries':1146,'all_queries':1536,'reader_calls':0,'judge_calls':0,'verification':True})
    print('Verified',len(conditions),'conditions;',checks,'independent metric checks',flush=True)

if __name__=='__main__':main()
