"""Independent BM25/fusion reconstruction; no original scoring functions imported."""
import csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/retrieval/raw_preserving_stage1_20261012'
OUT=RUN/'algorithm_audit'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,x): (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def clean(x):return '' if x is None or str(x).lower()=='nan' else str(x).strip()
def rank_metrics(ids,gold):
    flags=[m in gold for m in ids[:10]]
    rr=next((1/(i+1) for i,h in enumerate(flags) if h),0.)
    return rr,len(set(ids[:10])&gold)/len(gold)
def standardize(items):
    # Independent scalar math, not original numpy zscore implementation.
    mean=math.fsum(s for _,s in items)/len(items)
    sd=math.sqrt(math.fsum((s-mean)**2 for _,s in items)/len(items))
    if sd<=1e-12:sd=1.
    return {m:(s-mean)/sd for m,s in items}
def fuse(dense,lexical):
    dz=standardize(dense);bz=standardize(lexical)
    union=list(dict.fromkeys([m for m,_ in dense]+[m for m,_ in lexical]))
    scores={m:.6*dz.get(m,min(dz.values()))+.4*bz.get(m,min(bz.values())) for m in union}
    order={m:i for i,m in enumerate(union)}
    return sorted(union,key=lambda m:(-scores[m],order[m])),scores
def main():
    OUT.mkdir(exist_ok=True)
    memories=read(ROOT/'data/locomo_memory_records.csv');features={r['memory_id']:r for r in read(ROOT/'data/frozen_retrieval/p3_memory_features.csv')}
    saved_docs=js(RUN/'documents.json');scopeids=defaultdict(list);docs={'ERK':{},'RawERK':{}}
    for m in memories:
        mid=m['memory_id'];f=features[mid];scopeids[m['sample_id']].append(mid)
        parts=[label+': '+clean(f.get(field)) for label,field in [('E','entities'),('R','relations'),('K','keywords')] if clean(f.get(field))]
        docs['ERK'][mid]='\n'.join(parts);docs['RawERK'][mid]='\n'.join([clean(m['text'])]+parts)
        for v in docs:assert docs[v][mid]==saved_docs[v][mid],(v,mid)
    assert dict(scopeids)==js(RUN/'scope_memory_order.json')
    qsource={r['query_id']:r for r in read(ROOT/'data/frozen_retrieval/p3_queries.csv')}
    gold={r['query_id']:set(r['gold_memory_ids'].split(';')) for r in read(ROOT/'data/retrieval_gold/locomo_cat1_4_gold_memory.csv') if r['split']=='test' and r['gold_status']=='mapped' and r['gold_memory_ids']}
    savedq=js(RUN/'queries.json');qids=[r['query_id'] for r in savedq];assert set(qids)==set(gold) and len(qids)==1146
    for q in savedq:
        s=qsource[q['query_id']];assert q['question']==s['question'].strip() and q['scope']==s['conversation_id'] and q['category']==s['category']
        assert set(q['gold'])==gold[q['query_id']]
    denseall=defaultdict(dict)
    with (ROOT/'data/frozen_retrieval/frozen_dense_scores_long.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['query_id'] in gold:denseall[r['query_id']][r['memory_id']]=float(r['score'])
    dsave=js(RUN/'frozen_dense_top50.json');dense={}
    for qid in qids:
        ids=scopeids[qsource[qid]['conversation_id']];order={m:i for i,m in enumerate(ids)}
        dense[qid]=sorted([(m,denseall[qid][m]) for m in ids if m in denseall[qid]],key=lambda x:(-x[1],order[x[0]]))[:50]
        assert dense[qid]==[tuple(x) for x in dsave[qid]],qid
    saved_lex=js(RUN/'lexical_top50.json');lex={v:{} for v in docs};score_error=0.;top50_mismatch=[];termstats=[]
    # Share only scikit-learn tokenization/vocabulary; BM25 score formula independently reimplemented.
    for v in docs:
        for scope,ids in scopeids.items():
            scopeq=[q for q in qids if qsource[q]['conversation_id']==scope]
            if not scopeq:continue
            vec=CountVectorizer(lowercase=True,stop_words='english',ngram_range=(1,2),max_features=50000)
            tf=vec.fit_transform([docs[v][mid] for mid in ids]).tocsc()
            dl=np.asarray(tf.sum(1)).ravel();meanlen=float(dl.mean())
            df=np.asarray((tf>0).sum(0)).ravel()
            idf=np.log(1+(len(ids)-df+.5)/(df+.5))
            denominator=1.5*(.25+.75*dl/max(meanlen,1e-12))
            termstats.append({'variant':v,'scope':scope,'n_docs':len(ids),'vocab':len(vec.vocabulary_),'avgdl':meanlen,'empty_docs':sum(not docs[v][m] for m in ids)})
            for qid in scopeq:
                cols=sorted(vec.transform([qsource[qid]['question']]).indices.tolist())
                scores=np.zeros(len(ids),dtype=np.float64)
                for col in cols:
                    freq=tf[:,col].toarray().ravel()
                    scores+=idf[col]*(2.5*freq)/(freq+denominator+1e-9)
                rank=sorted(range(len(ids)),key=lambda i:(-float(scores[i]),i))[:50]
                result=[(ids[i],float(scores[i])) for i in rank];lex[v][qid]=result
                expected=saved_lex[v][qid]
                err=max(abs(float(s)-float(scores[ids.index(m)])) for m,s in expected)
                score_error=max(score_error,err)
                if [m for m,_ in result]!=[m for m,_ in expected]:
                    top50_mismatch.append({'variant':v,'query_id':qid,'same_candidate_set':set(m for m,_ in result)==set(m for m,_ in expected),'max_score_error':err,'changed_positions':[{'rank':i+1,'independent':a,'original':b} for i,(a,b) in enumerate(zip(result,expected)) if a[0]!=b[0]]})
        print('Independent BM25 completed',v,flush=True)
    original={ (r['method'],r['query_id']):r for r in read(RUN/'per_query.csv') if r['method'] in ['H_FrozenRaw_ERK','H_FrozenRaw_RawERK']}
    fused={v:{} for v in docs};fusedscore={v:{} for v in docs};mismatch=[];details=[];overlap=[]
    for qid in qids:
        for v in docs:
            ranking,scores=fuse(dense[qid],lex[v][qid]);fused[v][qid]=ranking;fusedscore[v][qid]=scores
            if ranking[:10]!=original['H_FrozenRaw_'+v,qid]['top10_memory_ids'].split(';'):mismatch.append({'variant':v,'query_id':qid})
        er,ec=rank_metrics(fused['ERK'][qid],gold[qid]);rr,rc=rank_metrics(fused['RawERK'][qid],gold[qid])
        details.append({'query_id':qid,'scope':qsource[qid]['conversation_id'],'category':qsource[qid]['category'],'question':qsource[qid]['question'],'erk_mrr':er,'rawerk_mrr':rr,'delta_erk_minus_rawerk':er-rr,'erk_recall':ec,'rawerk_recall':rc,'erk_top10':fused['ERK'][qid][:10],'rawerk_top10':fused['RawERK'][qid][:10],'gold_ids':sorted(gold[qid])})
        ds={m for m,_ in dense[qid]}
        overlap.append({'query_id':qid,**{v:len(ds&{m for m,_ in lex[v][qid]}) for v in docs}})
    summary={'status':'PASS' if not mismatch and not top50_mismatch and score_error<1e-10 else 'REVIEW_REQUIRED','rendered_text_checks':len(memories)*2,'query_gold_checks':len(qids),'frozen_dense_top50_checks':len(qids),'bm25_top50_cases':2*len(qids),'bm25_top50_mismatches':top50_mismatch,'bm25_max_absolute_score_error':score_error,'fusion_top10_mismatches':mismatch,'bm25_independence':'shares CountVectorizer preprocessing only; scores and sorting independently implemented','fusion_independence':'math.fsum scalar normalization; fresh union/scoring/tie-break implementation','erk_mrr':float(np.mean([r['erk_mrr'] for r in details])),'rawerk_mrr':float(np.mean([r['rawerk_mrr'] for r in details])),'erk_recall':float(np.mean([r['erk_recall'] for r in details])),'rawerk_recall':float(np.mean([r['rawerk_recall'] for r in details])),'mrr_erk_wins':sum(r['delta_erk_minus_rawerk']>1e-12 for r in details),'mrr_rawerk_wins':sum(r['delta_erk_minus_rawerk']< -1e-12 for r in details),'mrr_ties':sum(abs(r['delta_erk_minus_rawerk'])<=1e-12 for r in details),'positive_rr_sum':math.fsum(max(r['delta_erk_minus_rawerk'],0) for r in details),'negative_rr_sum':math.fsum(min(r['delta_erk_minus_rawerk'],0) for r in details),'mean_top50_intersection_with_dense':{v:float(np.mean([r[v] for r in overlap])) for v in docs}}
    if not mismatch and score_error<1e-10 and top50_mismatch and all(x['same_candidate_set'] and x['max_score_error']<1e-12 and all(abs(z['independent'][1]-z['original'][1])<1e-12 for z in x['changed_positions']) for x in top50_mismatch):
        summary['status']='FUSION_REPRODUCED_WITH_BM25_ROUNDING_NOTE'
        summary['rounding_note']='Independent algebraic operation order differs by floating-point roundoff. One near-tie swaps BM25 ranks 1/2; same Top50 set and all final fused Top10 unchanged. Original scores/order were not altered.'
    write('summary.json',summary);write('per_query_differences.json',details);write('scope_bm25_statistics.json',termstats);write('candidate_overlap.json',overlap)
    # Full candidate score decomposition for pre-defined largest wins and losses, not hand-picked examples.
    chosen=sorted(details,key=lambda r:(-abs(r['delta_erk_minus_rawerk']),r['query_id']))[:20];cases=[]
    for item in chosen:
        qid=item['query_id'];dz=standardize(dense[qid]);case={'query':item,'variants':{}}
        for v in docs:
            bz=standardize(lex[v][qid]);ld=dict(lex[v][qid]);dd=dict(dense[qid])
            case['variants'][v]=[{'memory_id':m,'is_gold':m in gold[qid],'dense_score':dd.get(m),'bm25_score':ld.get(m),'dense_z_or_imputed':dz.get(m,min(dz.values())),'bm25_z_or_imputed':bz.get(m,min(bz.values())),'fused_score':fusedscore[v][qid][m]} for m in fused[v][qid][:10]]
        cases.append(case)
    write('largest_absolute_changes_top20.json',cases)
    lines=['# 独立 BM25 与融合算法核验','','核验范围：原冻结 Dense(Raw) + BM25(ERK-only / RawERK)，同一 1,146 问。没有新编码、模型调用、调参或修改原结果。','','此前独立核验主要覆盖排名到指标；本次从原始字段和冻结 Dense 分数开始重建打分和排名。CountVectorizer 分词/词表仍共用既有依赖，BM25 打分和融合则独立实现，不导入原实验函数。','','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','','解释：候选重叠统计仅为描述，不足以证明性能差异的因果机制。所有逐题收益和损失见 per_query_differences.json；绝对差值最大的20题按固定规则输出候选分数分解，不只选择有利例子。','','范围：通过复算说明当前结果不是该轮新脚本的明显打分/排序实现错误；它不证明所有设计选择最优，也不替代对 tokenizer、gold 或上游抽取质量的外部验证。','']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
