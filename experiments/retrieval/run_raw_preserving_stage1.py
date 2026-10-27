"""Raw-preserving necessity: frozen reference plus matched-local 2x2.

Offline only. No Reader/Judge, no test tuning. Empty ERK has zero vector;
all corpus IDs retained. Original artifacts are read-only.
"""
from __future__ import annotations
import csv, json, hashlib, os, sys, time, argparse
from pathlib import Path
from collections import defaultdict
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import build_fusion_ablation_aligned_v2 as base
ROOT=base.ROOT
OUT=ROOT/'results/retrieval/raw_preserving_stage1_20261012'
MODEL=Path('C:/Users/PC/.cache/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09')
METRICS=list(base.METRICS)
def readcsv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def sha(p):return base.sha256(p)
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def csvout(name,rows):
    base.write_csv(OUT/name,rows,list(rows[0]))
def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    p=base.load_p1_module();config=p.load_json(base.P1_CONFIG)
    gold={r['query_id']:r for r in readcsv(base.GOLD_PATH) if r['split']=='test' and r['gold_status']=='mapped' and r['gold_memory_ids']}
    assert len(gold)==1146
    queries=[q for q in p.load_queries(ROOT/config['paths']['queries']) if q.query_id in gold]
    inputs=p.load_inputs(config,ROOT,set(gold))
    ids=list(inputs.memory_records)
    assert len(ids)==5882 and len(queries)==1146
    docs={v:{mid:p.build_document(mid,c,inputs,config['renderer']) for mid in ids} for v,c in p.VARIANTS.items() if v!='RawERKT'}
    docs['ERK']={}
    for mid in ids:
        f=inputs.memory_features[mid]
        docs['ERK'][mid]='\n'.join(f'{x}: {p.nonempty(f.get(k))}' for x,k in [('E','entities'),('R','relations'),('K','keywords')] if p.nonempty(f.get(k)))
        raw=docs['Raw'][mid];whole=docs['RawERK'][mid]
        assert whole==raw+('' if not docs['ERK'][mid] else '\n'+docs['ERK'][mid])
    dense={q.query_id:p.top_dense_for_query(q,inputs,50) for q in queries}
    log('Rebuilding RawERK parity gate')
    lex={'RawERK':p.build_bm25_rankings_for_variant('RawERK',queries,inputs,config,docs['RawERK'])}
    reference={r['query_id']:r for r in readcsv(base.PER_QUERY_PATH) if r['method']=='ZScore'}
    for q in queries:
        ranking=[m for m,_ in p.zscore_fuse(dense[q.query_id],lex['RawERK'][q.query_id],.6,'branch_min',50)[:10]]
        assert ranking==reference[q.query_id]['top10_memory_ids'].split(';'),q.query_id
        scored=base.score(gold[q.query_id]['gold_memory_ids'].split(';'),ranking)
        assert all(abs(scored[m]-float(reference[q.query_id][m]))<1e-12 for m in METRICS)
    save('parity_gate.json',{'status':'PASS','ordered_top10_matches':1146,'six_metric_matches':1146,'mrr':sum(float(r['MRR@10']) for r in reference.values())/1146})
    log('PASS: canonical ordered Top10 1146/1146')
    for v in docs:
        if v not in lex:
            log('BM25 '+v)
            lex[v]=p.build_bm25_rankings_for_variant(v,queries,inputs,config,docs[v])
    memory=[{'memory_id':mid,'scope':inputs.memory_records[mid].get('user_id'),'raw_dense':(inputs.memory_records[mid]['speaker']+': '+inputs.memory_records[mid]['text'])[:1500],'erk_dense':docs['ERK'][mid][:1500]} for mid in ids]
    qrows=[{'query_id':q.query_id,'scope':q.conversation_id,'category':q.category,'question':q.question,'gold':gold[q.query_id]['gold_memory_ids'].split(';')} for q in queries]
    save('documents.json',docs);save('memory_inputs.json',memory);save('queries.json',qrows)
    save('lexical_top50.json',lex);save('frozen_dense_top50.json',dense)
    save('scope_memory_order.json',inputs.conversation_memories)
    stats=[]
    for v,dd in docs.items():
        lengths=np.array([len(x) for x in dd.values()]);w=np.array([len(x.split()) for x in dd.values()])
        stats.append({'variant':v,'n':len(dd),'empty':int(sum(not x.strip() for x in dd.values())),'chars_mean':float(lengths.mean()),'chars_p95':float(np.quantile(lengths,.95)),'whitespace_tokens_mean':float(w.mean()),'dense_chars_over1500':int(sum(lengths>1500))})
    save('representation_stats.json',stats)
    empty={m for m,t in docs['ERK'].items() if not t.strip()}
    coverage=[]
    for q in qrows:
        ng=sum(m not in empty for m in q['gold'])
        coverage.append({'query_id':q['query_id'],'category':q['category'],'gold_count':len(q['gold']),'gold_with_erk':ng,'all_gold_have_erk':ng==len(q['gold']),'any_gold_has_erk':ng>0})
    save('gold_erk_coverage.json',coverage)
    diagnostics=[{'variant':v,'all_zero_top50_queries':sum(all(s==0 for _,s in ranks) for ranks in byq.values()),'zero_score_top50_slots':sum(s==0 for ranks in byq.values() for _,s in ranks)} for v,byq in lex.items()]
    save('lexical_diagnostics.json',diagnostics)
    files=[base.P1_SCRIPT,base.P1_CONFIG,base.GOLD_PATH,base.PER_QUERY_PATH,Path(__file__),ROOT/'data/locomo_memory_bge_large.npy',ROOT/'data/locomo_qa_bge_large.npy',ROOT/'data/locomo_memory_ids_bge.txt',ROOT/'data/locomo_qa_ids_bge.txt']+[ROOT/config['paths'][k] for k in ['memory_records','memory_features','queries','dense_scores']]
    save('contract.json',{'n':1146,'conversations':8,'seed':20261012,'bootstrap_resamples':10000,'alpha':.6,'depth':50,'bm25':config['bm25'],'raw_dense_input':'speaker: text truncated to 1500 chars','erk_dense_input':'labelled E/R/K only, truncated to 1500 chars; no speaker or raw fallback','empty_erk':'keep corpus record; dense zero vector; lexical zero-score stable order as original retriever','query_vectors':'same frozen question float32 arrays for all arms, normalized cosine','local_encoder':str(MODEL),'pooling':'CLS + L2 normalize','max_tokens':512,'device':'CPU float32','local_protocol_note':'raw and ERK both recomputed locally; frozen raw arms are separate historical reference','no_reader_or_judge':True,'sources':{str(f):sha(f) for f in files}})
    log('Preparation complete')
def encode():
    import torch
    from transformers import AutoTokenizer,AutoModel
    torch.set_num_threads(8)
    contract=load('contract.json');mem=load('memory_inputs.json')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModel.from_pretrained(MODEL,local_files_only=True).eval()
    detail={'model_files':{str(f):sha(f) for f in MODEL.iterdir() if f.is_file()},'torch':torch.__version__,'threads':8,'batch_size':16}
    for rep,key in [('raw','raw_dense'),('erk','erk_dense')]:
        dest=OUT/f'local_{rep}_vectors.npy'
        if dest.exists():
            assert (OUT/f'local_{rep}_complete.json').exists(),'incomplete encoding'
            assert np.load(dest).shape==(len(mem),1024)
            log('Reusing '+rep);continue
        texts=list(dict.fromkeys(r[key] for r in mem if r[key].strip()))
        texts.sort(key=lambda s:(len(s),s))
        result={};start=time.time();over512=0
        for offset in range(0,len(texts),16):
            tt=texts[offset:offset+16]
            over512+=sum(len(x)>512 for x in tok(tt,truncation=False)['input_ids'])
            batch=tok(tt,padding=True,truncation=True,max_length=512,return_tensors='pt')
            with torch.inference_mode():
                arr=torch.nn.functional.normalize(model(**batch).last_hidden_state[:,0],p=2,dim=1).numpy()
            result.update(zip(tt,arr))
            if offset%256==0:log(f'Encode {rep}: {min(offset+16,len(texts))}/{len(texts)} unique, {time.time()-start:.1f}s')
        vectors=np.stack([result[r[key]] if r[key].strip() else np.zeros(1024,dtype=np.float32) for r in mem]).astype(np.float32)
        np.save(dest,vectors)
        save(f'local_{rep}_complete.json',{'n':len(mem),'nonempty_unique':len(texts),'empty_records':sum(not r[key].strip() for r in mem),'unique_inputs_over512_tokens':over512,'seconds':time.time()-start,'sha256':sha(dest),'input_sha256':sha(OUT/'memory_inputs.json')})
    save('encoder_manifest.json',detail)
def bootstrap(d,scopes):
    unique=sorted(set(scopes)); groups=[np.flatnonzero(np.array(scopes)==s) for s in unique]
    sums=np.array([d[g].sum() for g in groups]);sizes=np.array([len(g) for g in groups])
    draw=np.random.default_rng(20261012).integers(0,len(groups),size=(10000,len(groups)))
    cluster=sums[draw].sum(1)/sizes[draw].sum(1)
    # Independent direct concatenation on the first 100 deterministic draws.
    assert np.allclose(cluster[:100],[d[np.concatenate([groups[i] for i in row])].mean() for row in draw[:100]],atol=1e-14)
    rng=np.random.default_rng(20261013);query=np.concatenate([d[rng.integers(0,len(d),size=(100,len(d)))].mean(1) for _ in range(100)])
    return {'delta':float(d.mean()),'cluster_ci95':np.quantile(cluster,[.025,.975]).tolist(),'query_ci95':np.quantile(query,[.025,.975]).tolist()}
def analyze():
    p=base.load_p1_module();qs=load('queries.json');mem=load('memory_inputs.json');lex=load('lexical_top50.json');frozen=load('frozen_dense_top50.json');scope_order=load('scope_memory_order.json')
    midx={r['memory_id']:i for i,r in enumerate(mem)}
    qids=(ROOT/'data/locomo_qa_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines();qidx={q:i for i,q in enumerate(qids)}
    qv=np.load(ROOT/'data/locomo_qa_bge_large.npy');qv=qv/np.linalg.norm(qv,axis=1,keepdims=True)
    dense={'FrozenRaw':frozen};localcheck={}
    for rep in ['raw','erk']:
        v=np.load(OUT/f'local_{rep}_vectors.npy');byq={}
        for q in qs:
            ids=scope_order[q['scope']];ix=[midx[m] for m in ids]
            scores=v[ix]@qv[qidx[q['query_id']]]
            rank=np.argsort(-scores,kind='stable')[:50]
            byq[q['query_id']]=[(ids[i],float(scores[i])) for i in rank]
        dense['LocalRaw' if rep=='raw' else 'LocalERK']=byq
    oldids=(ROOT/'data/locomo_memory_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines();old=np.load(ROOT/'data/locomo_memory_bge_large.npy');old=old/np.linalg.norm(old,axis=1,keepdims=True)
    oldlookup={m:i for i,m in enumerate(oldids)};new=np.load(OUT/'local_raw_vectors.npy')
    cos=(new*np.stack([old[oldlookup[r['memory_id']]] for r in mem])).sum(1)
    localcheck={'raw_vector_cosine_min':float(cos.min()),'raw_vector_cosine_mean':float(cos.mean()),'raw_vector_cosine_p01':float(np.quantile(cos,.01)),'top10_exact_matches':sum([m for m,_ in dense['LocalRaw'][q['query_id']][:10]]==[m for m,_ in frozen[q['query_id']][:10]] for q in qs),'n':len(qs)}
    save('local_vs_frozen_dense.json',localcheck);save('local_dense_top50.json',{k:v for k,v in dense.items() if k!='FrozenRaw'})
    methods={f'Dense_{k}':v for k,v in dense.items()}
    for v,b in lex.items():
        methods['BM25_'+v]=b
        methods['H_FrozenRaw_'+v]={q['query_id']:p.zscore_fuse(frozen[q['query_id']],b[q['query_id']],.6,'branch_min',50) for q in qs}
    for d in ['LocalRaw','LocalERK']:
        for v in (['ERK','RawERK','Raw'] if d=='LocalRaw' else ['ERK','RawERK']):
            methods[f'H_{d}_{v}']={q['query_id']:p.zscore_fuse(dense[d][q['query_id']],lex[v][q['query_id']],.6,'branch_min',50) for q in qs}
    rows=[];overall=[];bycat=[];byscope=[];coverage=load('gold_erk_coverage.json');covered={r['query_id'] for r in coverage if r['all_gold_have_erk']};coverage_results=[]
    for name,ranks in methods.items():
        rr=[]
        for q in qs:
            top=[m for m,_ in ranks[q['query_id']][:10]]
            assert len(top)==len(set(top))==10 and set(top)<=set(scope_order[q['scope']])
            row={'method':name,'query_id':q['query_id'],'scope':q['scope'],'category':q['category'],'top10_memory_ids':';'.join(top),**base.score(q['gold'],top)}
            rows.append(row);rr.append(row)
        def agg(sub,extra):return {'method':name,**extra,'n':len(sub),**{m:float(np.mean([r[m] for r in sub])) for m in METRICS}}
        overall.append(agg(rr,{}))
        for cat in ['1','2','3','4']:bycat.append(agg([r for r in rr if r['category']==cat],{'category':cat}))
        for sc in sorted(set(q['scope'] for q in qs)):byscope.append(agg([r for r in rr if r['scope']==sc],{'scope':sc}))
        for label,ss in [('all_gold_with_erk',[r for r in rr if r['query_id'] in covered]),('some_gold_without_erk',[r for r in rr if r['query_id'] not in covered])]:
            if ss:coverage_results.append(agg(ss,{'coverage_group':label}))
    lookup={r['method']:r for r in overall}
    assert abs(lookup['H_FrozenRaw_RawERK']['MRR@10']-.5498223635003742)<1e-12
    csvout('overall.csv',overall);csvout('by_category.csv',bycat);csvout('by_conversation.csv',byscope);csvout('per_query.csv',rows);csvout('by_gold_erk_coverage.csv',coverage_results)
    contrasts=[('H_LocalRaw_RawERK','H_LocalERK_ERK'),('H_LocalRaw_ERK','H_LocalERK_ERK'),('H_LocalRaw_RawERK','H_LocalERK_RawERK'),('H_LocalERK_RawERK','H_LocalERK_ERK'),('H_LocalRaw_RawERK','H_LocalRaw_ERK'),('H_LocalRaw_RawERK','H_LocalRaw_Raw'),('BM25_RawERK','BM25_ERK'),('BM25_RawERK','BM25_Raw'),('BM25_ERK','BM25_Raw'),('Dense_LocalRaw','Dense_LocalERK'),('H_FrozenRaw_RawERK','H_FrozenRaw_ERK'),('H_LocalRaw_RawERK','H_FrozenRaw_RawERK')]
    stats=[];by_method={name:[r for r in rows if r['method']==name] for name in methods}
    for a,b in contrasts:
        log('Bootstrap '+a+' - '+b)
        for metric in ['MRR@10','Recall@10']:
            ar=by_method[a];br=by_method[b];assert [r['query_id'] for r in ar]==[r['query_id'] for r in br]
            diff=np.array([x[metric]-y[metric] for x,y in zip(ar,br)])
            stats.append({'a':a,'b':b,'metric':metric,'n':len(diff),**bootstrap(diff,[q['scope'] for q in qs])})
    save('paired_bootstrap.json',stats)
    contract=load('contract.json');changed=[f for f,h in contract['sources'].items() if sha(Path(f))!=h]
    assert not changed,changed
    save('completion.json',{'status':'complete','methods':len(methods),'queries':len(qs),'paired_comparisons':len(stats),'source_hashes_unchanged':True,'reader_calls':0,'judge_calls':0,'runtime_note':'offline local CPU encoder; primary matched encoder contrasts separate from frozen reference'})
    report(overall,bycat,stats,localcheck,coverage_results)
    log('COMPLETE')
def report(overall,bycat,stats,localcheck,coverage_results):
    lines=['# Raw-preserving necessity：第一阶段完整结果','','## 实验设置','','1,146 held-out valid-gold queries / 8 conversations；canonical gold-v2。所有原数据只读。BM25 与 fusion 使用既有冻结配置：各分支 Top-50、最终 Top-10、α=.6、总体标准差、branch-min 和稳定排序。未重新抽取字段，未调 test 参数。','','Dense_LocalRaw 与 Dense_LocalERK 均使用同一本地 BGE revision、float32 CPU、CLS/L2、1500 字符与 512 token 上限。两者共用冻结 query 向量。FrozenRaw 为原冻结参考，不与 LocalRaw 混称完全相同运行。Raw dense 输入包含 speaker: text；ERK 只含既有 E/R/K 标签和字段。空 ERK 保留 memory ID、dense 为零向量、BM25 遵守原零分稳定排序，无 Raw 回填。','','本轮仅检验检索，不检验 Reader 上下文保真；无 Reader/Judge 调用。ERK 来自既有 session observation 对齐字段，结果不能推广为任意抽取器的充分性。','','## 2×2 matched-local 主对照','','A=H_LocalERK_ERK；B=H_LocalRaw_ERK；C=H_LocalERK_RawERK；D=H_LocalRaw_RawERK。A 全检索表示不附加原文，D 为当前架构的匹配本地编码实例。H_FrozenRaw_RawERK 是原论文数值复现参考。','','## 所有方法整体结果','','| 方法 | n | MRR@10 | Hit@1 | Hit@5 | Hit@10 | Recall@10 | nDCG@10 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in overall:lines.append('| '+r['method']+' | '+str(r['n'])+' | '+' | '.join(f'{r[m]:.6f}' for m in METRICS)+' |')
    lines+=['','## 配对统计','','10,000 次 conversation-cluster bootstrap，seed=20261012，query-weighted micro mean；辅助 query-level seed=20261013。边际 95% percentile CI，未做多重比较调整。仅八个 clusters。每项的前 100 次聚类重采样由独立直接拼接核验。差值为 A−B；所有预定比较如实列出。','','| A − B | 指标 | Δ | Cluster 95% CI | Query 95% CI |','|---|---|---:|---|---|']
    for r in stats:lines.append(f"| {r['a']} − {r['b']} | {r['metric']} | {r['delta']:+.6f} | [{r['cluster_ci95'][0]:+.6f}, {r['cluster_ci95'][1]:+.6f}] | [{r['query_ci95'][0]:+.6f}, {r['query_ci95'][1]:+.6f}] |")
    lines+=['','## 按问题类型完整结果','','类别 1=Multi-Hop，2=Temporal，3=Open-Domain，4=Single-Hop。','','| 方法 | 类别 | n | MRR@10 | Recall@10 | Hit@10 | nDCG@10 |','|---|---|---:|---:|---:|---:|---:|']
    for r in bycat:lines.append(f"| {r['method']} | {r['category']} | {r['n']} | "+' | '.join(f'{r[m]:.6f}' for m in ['MRR@10','Recall@10','Hit@10','nDCG@10'])+' |')
    lines+=['','## 覆盖率与实现诊断','','下列覆盖分组是解释性诊断，完整 1,146 问结果仍是主结果。不能只保留有 ERK 的问题后宣称整体效果。','','| 方法 | Gold ERK 覆盖分组 | n | MRR@10 | Recall@10 |','|---|---|---:|---:|---:|']
    for r in coverage_results:
        if r['method'] in ['H_LocalRaw_RawERK','H_LocalERK_ERK','BM25_ERK','BM25_RawERK']:lines.append(f"| {r['method']} | {r['coverage_group']} | {r['n']} | {r['MRR@10']:.6f} | {r['Recall@10']:.6f} |")
    lines+=['','LocalRaw 与冻结 Raw 的向量/排名一致性（不是逐位相同承诺）：','','```json',json.dumps(localcheck,indent=2),'```','','## 文件索引','','- overall.csv / by_category.csv / by_conversation.csv：完整六指标汇总。','- per_query.csv：全部方法逐题排名与六指标。','- paired_bootstrap.json：全部预定差值与两种 CI。','- representation_stats.json / lexical_diagnostics.json：空文本、长度和零匹配。','- gold_erk_coverage.json / by_gold_erk_coverage.csv：gold 结构覆盖诊断。','- documents.json / memory_inputs.json：实际渲染文本及编码输入。','- lexical_top50.json / frozen_dense_top50.json / local_dense_top50.json：分支分数与候选。','- contract.json / encoder_manifest.json / parity_gate.json / completion.json：设置、版本、哈希与完成检查。','- local_raw_vectors.npy / local_erk_vectors.npy：新编码数组；旧数组保持不动。','']
    (OUT/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','encode','analyze','all']);a=parser.parse_args()
    if a.stage in ['prepare','all']:prepare()
    if a.stage in ['encode','all']:encode()
    if a.stage in ['analyze','all']:analyze()
