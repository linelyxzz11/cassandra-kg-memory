"""Offline candidate-budget audit. No network/API calls or source mutations."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import csv, json, math, time, sys, platform, traceback
from pathlib import Path
import importlib.util
import numpy as np

ROOT=Path('D:/memorytable/cassandra-kg-memory')
OUT=Path(__file__).resolve().parent
LS=(10,20,50,100)
KS=(1,5,10,20)
METHODS=('Dense-Raw','BM25-RawERK','Equal-RRF','Weighted-RRF','CassMem')
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def status(stage,**kw):
    save('status.json',dict(stage=stage,time=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),**kw))
    print(stage,kw,flush=True)
def writecsv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def loadcsv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def metrics(gold,rank,k):
    ranks=[i for i,x in enumerate(rank[:k],1) if x in gold]
    dcg=sum(1/math.log2(i+1) for i in ranks)
    ideal=sum(1/math.log2(i+1) for i in range(1,min(k,len(gold))+1))
    return dict(MRR=1/ranks[0] if ranks else 0.,Hit=float(bool(ranks)),Recall=len(ranks)/len(gold),nDCG=dcg/ideal)
def main():
    status('loading')
    spec=importlib.util.spec_from_file_location('aligned',ROOT/'experiments/retrieval/build_fusion_ablation_aligned_v2.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    p=a.load_p1_module();config=p.load_json(a.P1_CONFIG)
    goldrows=[r for r in loadcsv(a.GOLD_PATH) if r['gold_status']=='mapped' and r['gold_memory_ids']]
    assert len(goldrows)==1536 and sum(r['split']=='test' for r in goldrows)==1146
    gold={r['query_id']:set(r['gold_memory_ids'].split(';')) for r in goldrows}
    splits={r['query_id']:r['split'] for r in goldrows}
    queries=[q for q in p.load_queries(ROOT/config['paths']['queries']) if q.query_id in gold]
    assert len({q.query_id for q in queries})==1536
    inputs=p.load_inputs(config,ROOT,set(gold))
    # Full scope coverage, rather than count>=100, proves true Top100 coverage.
    for q in queries:
        assert set(inputs.conversation_memories[q.conversation_id]) <= set(inputs.dense_scores[q.query_id]),q.query_id
    paths=[a.P1_SCRIPT,a.P1_CONFIG,a.GOLD_PATH,Path(__file__),ROOT/'experiments/retrieval/build_fusion_ablation_aligned_v2.py']
    paths += [ROOT/config['paths'][k] for k in ('memory_records','memory_features','queries','dense_scores')]
    manifest=dict(L=LS,k=KS,methods=METHODS,alpha=.6,rrf_c=10,rrf_missing_rank=999,bootstrap=10000,seed=20261014,
        split_counts=dict(full=1536,test=1146,dev=390),config=config,python=sys.version,platform=platform.platform(),
        hashes={str(f):a.sha256(f) for f in paths},timing='cached-score sorting/selection and fusion only; no scoring, encoding, I/O, index build or database time')
    save('protocol.json',manifest)
    documents={mid:p.build_document(mid,('E','R','K'),inputs,config['renderer']) for mid in inputs.memory_records}
    full_depth=max(len(x) for x in inputs.conversation_memories.values())
    config['fusion']['candidate_depth']=full_depth
    status('building_full_scope_bm25',queries=len(queries),max_scope=full_depth)
    bm=p.build_bm25_rankings_for_variant('RawERK',queries,inputs,config,documents)
    dense={q.query_id:p.top_dense_for_query(q,inputs,full_depth) for q in queries}
    def rank(method,d,b):
        if method=='Dense-Raw':return d
        if method=='BM25-RawERK':return b
        if method=='CassMem':return p.zscore_fuse(d,b,.6,'branch_min',200)
        return a.rrf(d,b,.5 if method=='Equal-RRF' else .6)
    refs=loadcsv(a.PER_QUERY_PATH)
    qmap={q.query_id:q for q in queries}
    checked=0
    for r in refs:
        method={'ZScore':'CassMem','Equal-RRF':'Equal-RRF','Weighted-RRF':'Weighted-RRF'}.get(r['method'])
        if not method:continue
        qid=r['query_id'];ids=[x[0] for x in rank(method,dense[qid][:50],bm[qid][:50])[:10]]
        assert ids==r['top10_memory_ids'].split(';'),(method,qid,'ranking mismatch')
        m=metrics(gold[qid],ids,10)
        for key in m:assert abs(m[key]-float(r[key+'@10']))<1e-12,(method,qid,key)
        checked+=1
    assert checked==1146*3
    save('parity.json',dict(passed=True,checked_query_method_pairs=checked,full_dense_scope_coverage=True))
    status('parity_passed',pairs=checked)
    rows=[];ranksout=[];coverage=[]
    for L in LS:
        for q in queries:
            qid=q.query_id;g=gold[qid];d=dense[qid][:L];b=bm[qid][:L];ds={x[0] for x in d};bs={x[0] for x in b};u=ds|bs
            coverage.append(dict(query_id=qid,scope=q.conversation_id,split=splits[qid],category=q.category,L=L,
                dense_coverage=len(ds&g)/len(g),bm25_coverage=len(bs&g)/len(g),union_coverage=len(u&g)/len(g),
                union_hit=int(bool(u&g)),union_size=len(u),jaccard=len(ds&bs)/len(u)))
            for method in METHODS:
                ranked=[x[0] for x in rank(method,d,b)]
                assert len(ranked)==len(set(ranked))
                ranksout.append(dict(query_id=qid,L=L,method=method,ranking=ranked[:20]))
                cand=ds if method=='Dense-Raw' else bs if method=='BM25-RawERK' else u
                for k in KS:
                    if k>L and method in METHODS[:2]:continue
                    m=metrics(g,ranked,k);oracle=min(k,len(cand&g))/len(g)
                    rows.append(dict(query_id=qid,scope=q.conversation_id,split=splits[qid],category=q.category,L=L,k=k,method=method,
                        **m,oracle_recall=oracle,ranking_recall_gap=oracle-m['Recall']))
        status('quality_depth_complete',L=L)
    writecsv('per_query_metrics.csv',rows);writecsv('candidate_coverage.csv',coverage);save('rankings.json',ranksout)
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:
        for pop in ('full',r['split']):
            for cat in ('all',str(r['category'])):groups[(pop,cat,r['L'],r['k'],r['method'])].append(r)
    summary=[]
    for (pop,cat,L,k,method),rs in groups.items():
        summary.append(dict(population=pop,category=cat,L=L,k=k,method=method,n=len(rs),**{m:float(np.mean([r[m] for r in rs])) for m in ('MRR','Hit','Recall','nDCG','oracle_recall','ranking_recall_gap')}))
    writecsv('metrics_summary.csv',summary)
    status('bootstrap')
    lookup={(r['query_id'],r['L'],r['k'],r['method']):r for r in rows if r['split']=='test'}
    scopes=sorted({q.conversation_id for q in queries if splits[q.query_id]=='test'})
    rng=np.random.default_rng(20261014);draws=rng.integers(0,len(scopes),(10000,len(scopes)))
    contrasts=[]
    comparisons=[(L,'CassMem',L,m) for L in LS for m in ('Equal-RRF','Weighted-RRF')]
    comparisons += [(L,m,50,m) for L in LS if L!=50 for m in METHODS]
    for l1,m1,l0,m0 in comparisons:
        for metric in ('MRR','Hit','Recall','nDCG'):
            sums=[];counts=[]
            for s in scopes:
                qs=[q.query_id for q in queries if q.conversation_id==s and splits[q.query_id]=='test']
                diffs=[lookup[(q,l1,10,m1)][metric]-lookup[(q,l0,10,m0)][metric] for q in qs]
                sums.append(sum(diffs));counts.append(len(diffs))
            boot=np.array(sums)[draws].sum(1)/np.array(counts)[draws].sum(1)
            contrasts.append(dict(L=l1,method=m1,reference_L=l0,reference_method=m0,k=10,metric=metric,
                delta=sum(sums)/sum(counts),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),clusters=len(scopes),repetitions=10000))
    writecsv('paired_cluster_bootstrap.csv',contrasts)
    status('cached_postprocessing_timing')
    timing=[]
    testqs=[q for q in queries if splits[q.query_id]=='test']
    # Cached full-scope scores in original corpus order; warm up all configurations once.
    scorelists={}
    for q in testqs:
        order=inputs.conversation_memories[q.conversation_id];db=dict(dense[q.query_id]);bb=dict(bm[q.query_id])
        scorelists[q.query_id]=([(m,db[m]) for m in order],[(m,bb[m]) for m in order])
    for rep in range(6):
        for L in (LS if rep%2==0 else LS[::-1]):
            for q in testqs:
                dd,bb=scorelists[q.query_id]
                t=time.perf_counter_ns();d=sorted(dd,key=lambda x:-x[1])[:L];td=time.perf_counter_ns()-t
                t=time.perf_counter_ns();b=sorted(bb,key=lambda x:-x[1])[:L];tb=time.perf_counter_ns()-t
                for method in (METHODS if rep%2==0 else METHODS[::-1]):
                    t=time.perf_counter_ns();rank(method,d,b);tf=time.perf_counter_ns()-t
                    if rep:timing.append(dict(repetition=rep,query_id=q.query_id,L=L,method=method,
                        selection_us=(td if method=='Dense-Raw' else tb if method=='BM25-RawERK' else td+tb)/1000,
                        fusion_us=tf/1000 if method not in METHODS[:2] else 0.))
        status('timing_round_complete',round=rep)
    writecsv('cached_postprocessing_timing.csv',timing)
    for f,h in manifest['hashes'].items():assert a.sha256(Path(f))==h,f
    assert len(rows)==1536*(5*4*4-2)
    save('complete.json',dict(quality=True,bootstrap=True,cached_postprocessing_timing=True,online_scoring_timing=False,
        unique_queries=1536,heldout=1146,rows=len(rows),parity=True))
    status('complete')
if __name__=='__main__':
    try:main()
    except Exception:
        status('failed',error=traceback.format_exc());raise
