"""Remaining stage A matrix; fixtures generated without QA/gold access."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import json,sys,time,re,threading,traceback,dataclasses,hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import run_initial as base
from run_initial import ROOT,OUT,on,state,readcsv,sha
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def log(stage,**x):
    save('stress_status.json',dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**x));print(stage,x,flush=True)
def main():
    assert json.loads((OUT/'initial_complete.json').read_text())['failed']==0
    if (OUT/'stress_observations.jsonl').exists():raise RuntimeError('Existing observations: review before restarting')
    import importlib.util
    spec=importlib.util.spec_from_file_location('a',ROOT/'experiments/retrieval/build_fusion_ablation_aligned_v2.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);p=a.load_p1_module();c=p.load_json(a.P1_CONFIG)
    # Load original corpus before query/gold: fixture decisions use only raw memory fields.
    records=readcsv(ROOT/'data/locomo_memory_records.csv')
    mids=(ROOT/'data/locomo_memory_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines();mv=np.load(ROOT/'data/locomo_memory_bge_large.npy');vectors=dict(zip(mids,mv))
    scopeids=sorted({r['user_id'] for r in records});fixture=[]
    for s in scopeids:
        eligible=[r for r in records if r['user_id']==s and re.search(r'\b\d+\b',r['text'])]
        # At most32 corpus-ordered numeric counterfactuals/scope, independent of queries/results.
        for r in eligible[:32]:
            changed=re.sub(r'\b\d+\b',lambda m:str(int(m.group())+7),r['text'],count=1)
            fixture.append(dict(scope=s,source_id=r['memory_id'],raw=changed,dense_text=r['speaker']+': '+changed,kind='synthetic_first_integer_plus7'))
    save('counterfactual_inputs.json',fixture)
    modelpath=Path('C:/Users/PC/.cache/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09')
    meta=dict(input_sha256=sha(OUT/'counterfactual_inputs.json'),model=str(modelpath),device='cpu',max_seq_length=512,normalize=True)
    if (OUT/'counterfactual_vectors.npy').exists():
        assert json.loads((OUT/'counterfactual_vector_protocol.json').read_text())==meta
        cv=np.load(OUT/'counterfactual_vectors.npy')
    else:
        log('encoding_counterfactuals',n=len(fixture))
        import torch
        torch.set_num_threads(4)
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer(str(modelpath),device='cpu');model.max_seq_length=512
        parts=[]
        for start in range(0,len(fixture),16):
            parts.append(model.encode([r['dense_text'] for r in fixture[start:start+16]],batch_size=8,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False))
            log('encoding_counterfactuals',done=min(start+16,len(fixture)),total=len(fixture))
        cv=np.concatenate(parts);np.save(OUT/'counterfactual_vectors.npy',cv);save('counterfactual_vector_protocol.json',meta)
        del model
    assert len(cv)==len(fixture)
    # Quality labels are accessed only after independent fixture construction.
    gr=[r for r in readcsv(a.GOLD_PATH) if r['split']=='test' and r['gold_status']=='mapped' and r['gold_memory_ids']]
    gold={r['query_id']:r['gold_memory_ids'].split(';') for r in gr}
    qs=[q for q in p.load_queries(ROOT/c['paths']['queries']) if q.query_id in gold];assert len(qs)==1146
    inp=p.load_inputs(c,ROOT,set(gold));docs={m:p.build_document(m,('E','R','K'),inp,c['renderer']) for m in inp.memory_records}
    qids=(ROOT/'data/locomo_qa_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines();qvec=dict(zip(qids,np.load(ROOT/'data/locomo_qa_bge_large.npy')))
    refs={}
    for line in (OUT/'initial_observations.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row['condition']=='S0':refs[row['query_id']]=row
    seeds=(20261014,20261015,20261016);summ=[];threadlocal=threading.local();registry={}
    sp0=on._bm25_search;de0=on._dense_search
    def wrap(fn,kind):
        def f(snapshot,*args,**kw):
            if hasattr(threadlocal,'trace'):threadlocal.trace.append((kind,registry.get(id(snapshot),'UNKNOWN')))
            return fn(snapshot,*args,**kw)
        return f
    on._bm25_search=wrap(sp0,'sparse');on._dense_search=wrap(de0,'dense')
    def load(ix,s,dd,vv,version=1):
        ix.load_scope(s,dd,vv,version);registry[id(ix._sparse[s])]=s;registry[id(ix._dense[s])]=s
    def check(ix,s,q,condition,seed=0,mult=0,order='',roundno=0):
        threadlocal.trace=[];start=time.perf_counter_ns()
        row=dict(condition=condition,seed=seed,multiplier=mult,order=order,round=roundno,scope=s,query_id=q.query_id)
        try:
            r=ix.search(s,q.question,qvec[q.query_id]);ref=refs[q.query_id]['result'];d=dataclasses.asdict(r)
            equal={k:[v[0] for v in d[k]]==[v[0] for v in ref[k]] for k in ('top10','dense_top50','sparse_top50')}
            drift=max([abs(v-dict(ref[k]).get(m,v)) for k in equal for m,v in d[k]] or [0.])
            provenance=threadlocal.trace==[('sparse',s),('dense',s)]
            foreign=sum(m not in docs or m not in inp.conversation_memories[s] for k in ('dense_top50','sparse_top50') for m,v in d[k])
            content=all(ix._documents[s][m]==docs[m] and ix._sparse_versions[(s,m)]==1 and ix._dense_versions[(s,m)]==1 for m,v in r.top10)
            row.update(ok=all(equal.values()) and drift<=1e-7 and provenance and not foreign and content,
                equal=equal,max_score_delta=drift,scope_trace=list(threadlocal.trace),snapshot_scope_ok=provenance,
                foreign_candidates=foreign,candidate_occurrences=len(r.dense_top50)+len(r.sparse_top50),content_version_ok=content,
                metrics=a.score(gold[q.query_id],[m for m,v in r.top10]),result=d)
        except Exception:row.update(ok=False,error=traceback.format_exc())
        row.update(start_ns=start,end_ns=time.perf_counter_ns())
        with (OUT/'stress_observations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        return row
    files=[Path(__file__),Path(on.__file__),OUT/'counterfactual_inputs.json',OUT/'counterfactual_vectors.npy']
    hashes={str(f):sha(f) for f in files}
    save('stress_protocol.json',dict(seeds=seeds,multipliers=[1,5,10],orders=['target_first','external_first'],
        fixtures='S2 alternating exact clones and numeric synthetic variants; S3 original raw vectors with repeated E/K lexical fields',
        counterfactual_pool_per_scope=32,external_scopes=5,concurrent_rounds=10,reads_per_round=5,
        scope='online retrieval only; no authorization, relation backend or result cache claims',hashes=hashes))
    for s in sorted({q.conversation_id for q in qs}):
        ids=inp.conversation_memories[s];dd={m:docs[m] for m in ids};vv={m:vectors[m] for m in ids};targetqs=[q for q in qs if q.conversation_id==s]
        referenceindex=on.ScopedOnlineRetrievalIndex();load(referenceindex,s,dd,vv);expected=state(referenceindex,s)
        fids=[i for i,r in enumerate(fixture) if r['scope']==s]
        for condition in ('S2','S3'):
            for seed in seeds:
                for mult in (1,5,10):
                    rng=np.random.default_rng(seed);ext=[({}, {}) for _ in range(5)]
                    for j in range(len(ids)*mult):
                        source=ids[int(rng.integers(len(ids)))];text=docs[source];vec=vectors[source]
                        if condition=='S2' and j%2 and fids:
                            fi=fids[int(rng.integers(len(fids)))];text=fixture[fi]['raw'];vec=cv[fi]
                        elif condition=='S3':
                            feature=inp.memory_features[source];cue=str(feature.get('entities',''))+' '+str(feature.get('keywords',''))
                            text+='\n'+' '.join([cue]*20)
                        key=f'foreign-{j:06d}';ext[j%5][0][key]=text;ext[j%5][1][key]=vec
                    for order in ('target_first','external_first'):
                        registry.clear();ix=on.ScopedOnlineRetrievalIndex()
                        if order=='target_first':load(ix,s,dd,vv)
                        for j,(ed,ev) in enumerate(ext):load(ix,f'external-{j}',ed,ev,7)
                        if order=='external_first':load(ix,s,dd,vv)
                        failures=0
                        for q in targetqs:failures+=not check(ix,s,q,condition,seed,mult,order)['ok']
                        rec=dict(scope=s,condition=condition,seed=seed,multiplier=mult,order=order,n=len(targetqs),failures=failures,state_unchanged=state(ix,s)==expected)
                        summ.append(rec);save('stress_summary.json',summ);log('stress_cell_complete',**rec)
        # Controlled same-ID external updates: genuine shifted raw/vector pairs, no target writes.
        registry.clear();ix=on.ScopedOnlineRetrievalIndex();load(ix,s,dd,vv)
        shifted=ids[1:]+ids[:1];ed=dict(zip(ids,[dd[m] for m in shifted]));ev=dict(zip(ids,[vv[m] for m in shifted]));other='collision-'+s
        load(ix,other,ed,ev,7)
        for mode in ('interleaved','concurrent'):
            for roundno in range(10):
                mid=ids[roundno];replacement=ids[(roundno+2)%len(ids)];version=8+roundno+(10 if mode=='concurrent' else 0)
                event=threading.Barrier(2)
                def update():
                    if mode=='concurrent':event.wait(timeout=30)
                    start=time.perf_counter_ns();ok1=ix.upsert_sparse(other,mid,dd[replacement],version);ok2=ix.upsert_dense(other,mid,vv[replacement],version)
                    registry[id(ix._sparse[other])]=other;registry[id(ix._dense[other])]=other
                    return dict(start_ns=start,end_ns=time.perf_counter_ns(),accepted=ok1 and ok2,visible=ix.is_visible(other,mid,version)==(True,True))
                def reads():
                    if mode=='concurrent':event.wait(timeout=30)
                    return [check(ix,s,targetqs[(roundno*5+j)%len(targetqs)],'S5-'+mode,roundno=roundno) for j in range(5)]
                # Exercise external query before returning to A.
                ix.search(other,targetqs[0].question,qvec[targetqs[0].query_id])
                if mode=='concurrent':
                    with ThreadPoolExecutor(2) as pool:
                        uf=pool.submit(update);rf=pool.submit(reads);u=uf.result();rr=rf.result()
                else:u=update();rr=reads()
                post=check(ix,s,targetqs[roundno%len(targetqs)],'S5-'+mode+'-post',roundno=roundno)
                rec=dict(scope=s,condition='S5-'+mode,round=roundno,update=u,n=6,failures=sum(not x['ok'] for x in rr+[post]),
                    reads_overlapping_update=sum(x['start_ns']<u['end_ns'] and x['end_ns']>u['start_ns'] for x in rr),state_unchanged=state(ix,s)==expected)
                summ.append(rec);save('stress_summary.json',summ)
        # Fault control: deliberately global mixed-scope index, outside production code.
        broken=on.ScopedOnlineRetrievalIndex();mixed=dict(dd);mixv=dict(vv)
        for m in ids:mixed['000-foreign-'+m]=dd[m];mixv['000-foreign-'+m]=vv[m]
        load(broken,'fault-global',mixed,mixv)
        leaks=0;changed=0
        for q in targetqs:
            r=broken.search('fault-global',q.question,qvec[q.query_id]);leaks+=any(m.startswith('000-foreign-') for m,v in r.top10)
            changed += [m for m,v in r.top10]!=[m for m,v in refs[q.query_id]['result']['top10']]
        summ.append(dict(scope=s,condition='fault_control',n=len(targetqs),leaked_queries=leaks,changed_queries=changed,detected=leaks>0));save('stress_summary.json',summ)
        log('target_scope_complete',scope=s)
    for f,h in hashes.items():assert sha(Path(f))==h
    save('stress_complete.json',dict(cells=len(summ),failed_queries=sum(r.get('failures',0) for r in summ),
        state_failures=sum(not r.get('state_unchanged',True) for r in summ),fault_controls_detected=sum(r.get('detected',False) for r in summ),
        overlapping_reads=sum(r.get('reads_overlapping_update',0) for r in summ),validation_pending=True))
    log('stress_run_complete_validation_pending')
if __name__=='__main__':
    try:main()
    except Exception:log('failed',error=traceback.format_exc());raise
