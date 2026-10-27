"""Stage A initial matrix: S0, S1, S4; actual online search, no backend/API writes."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv,hashlib,traceback,time,dataclasses
from pathlib import Path
import importlib.util
import numpy as np
ROOT=Path('D:/memorytable/cassandra-kg-memory');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cassmem.retrieval import online as on
def save(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def status(stage,**x):
    save('status.json',dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**x));print(stage,x,flush=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def state(index,s):
    sp=index._sparse[s];de=index._dense[s]
    return dict(documents=digest(index._documents[s]),vocabulary=digest({k:int(v) for k,v in sp.vectorizer.vocabulary_.items()}),
        idf=hashlib.sha256(sp.idf.tobytes()).hexdigest(),length=hashlib.sha256(sp.doc_len.tobytes()).hexdigest(),
        avg_dl=sp.avg_dl,dense=hashlib.sha256(de.normalized_embeddings.tobytes()).hexdigest(),
        versions=digest([(m,index._sparse_versions[(s,m)],index._dense_versions[(s,m)]) for m in sp.memory_ids]))
def main():
    if (OUT/'initial_complete.json').exists():raise RuntimeError('Initial run already complete; do not overwrite')
    status('loading_initial_inputs')
    spec=importlib.util.spec_from_file_location('aligned',ROOT/'experiments/retrieval/build_fusion_ablation_aligned_v2.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);p=a.load_p1_module();c=p.load_json(a.P1_CONFIG)
    gr=[r for r in readcsv(a.GOLD_PATH) if r['split']=='test' and r['gold_status']=='mapped' and r['gold_memory_ids']]
    gold={r['query_id']:r['gold_memory_ids'].split(';') for r in gr};assert len(gold)==1146
    queries=[q for q in p.load_queries(ROOT/c['paths']['queries']) if q.query_id in gold]
    inp=p.load_inputs(c,ROOT,set(gold))
    docs={m:p.build_document(m,('E','R','K'),inp,c['renderer']) for m in inp.memory_records}
    mids=(ROOT/'data/locomo_memory_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines()
    qids=(ROOT/'data/locomo_qa_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines()
    mv=np.load(ROOT/'data/locomo_memory_bge_large.npy');qv=np.load(ROOT/'data/locomo_qa_bge_large.npy')
    assert len(set(mids))==len(mv)==len(mids);assert len(set(qids))==len(qv)==len(qids)
    vectors=dict(zip(mids,mv));qvec=dict(zip(qids,qv));assert set(gold)<=set(qvec)
    files=[Path(__file__),Path(on.__file__),a.P1_CONFIG,a.P1_SCRIPT,a.GOLD_PATH]+[ROOT/c['paths'][k] for k in ('memory_records','memory_features','queries','dense_scores')]+[ROOT/'data'/f for f in ('locomo_memory_ids_bge.txt','locomo_qa_ids_bge.txt','locomo_memory_bge_large.npy','locomo_qa_bge_large.npy')]
    hashes={str(f):sha(f) for f in files}
    save('initial_protocol.json',dict(conditions=['S0','S1','S4'],remaining=['S2','S3','S5','fault_control'],n=1146,
        target_scopes=sorted({q.conversation_id for q in queries}),depth=50,alpha=.6,
        source_hashes=hashes,score_tolerance=1e-7,baseline='same online code, not offline .5498',
        collision='same IDs with cyclically shifted genuine documents and their corresponding vectors; version7 vs target1',
        tracing='read-only scorer wrappers verify actual snapshot object identity per requested scope',cache='no result cache present'))
    # Wrappers observe actual sparse/dense snapshots used, not fabricated scope labels.
    original_sparse=on._bm25_search;original_dense=on._dense_search;trace=[];registry={}
    def sparse(snapshot,*args,**kw):
        trace.append(('sparse',registry.get(id(snapshot),'UNKNOWN')));return original_sparse(snapshot,*args,**kw)
    def dense(snapshot,*args,**kw):
        trace.append(('dense',registry.get(id(snapshot),'UNKNOWN')));return original_dense(snapshot,*args,**kw)
    on._bm25_search=sparse;on._dense_search=dense
    rows=[];baseline={};summaries=[]
    def load(ix,s,dd,vv,version=1):
        ix.load_scope(s,dd,vv,version);registry[id(ix._sparse[s])]=s;registry[id(ix._dense[s])]=s
    def evaluate(ix,s,label,order,expected_state=None):
        failures=0;count=0
        for q in queries:
            if q.conversation_id!=s:continue
            count+=1;trace.clear();row=dict(condition=label,order=order,scope=s,query_id=q.query_id)
            try:
                r=ix.search(s,q.question,qvec[q.query_id]);ids=[m for m,_ in r.top10];metrics=a.score(gold[q.query_id],ids)
                if label=='S0':baseline[q.query_id]=r
                ref=baseline[q.query_id]
                seq=lambda v:[m for m,_ in v]
                same=seq(r.top10)==seq(ref.top10)
                branch_same=seq(r.dense_top50)==seq(ref.dense_top50) and seq(r.sparse_top50)==seq(ref.sparse_top50)
                provenance=trace==[('sparse',s),('dense',s)]
                leak=sum(m not in inp.conversation_memories[s] for m,_ in r.dense_top50+r.sparse_top50)
                maxdiff=0.
                for current,old in ((r.top10,ref.top10),(r.dense_top50,ref.dense_top50),(r.sparse_top50,ref.sparse_top50)):
                    oldmap=dict(old)
                    for m,v in current:
                        if m in oldmap:maxdiff=max(maxdiff,abs(v-oldmap[m]))
                content_ok=all(ix._documents[s][m]==docs[m] and ix.is_visible(s,m,1)==(True,True) and
                    ix._sparse_versions[(s,m)]==1 and ix._dense_versions[(s,m)]==1 for m in ids)
                row.update(ok=same and branch_same and provenance and not leak and content_ok and maxdiff<=1e-7,
                    top10_equal=same,branches_equal=branch_same,actual_snapshot_scope_ok=provenance,foreign_candidates=leak,
                    candidate_occurrences=len(r.dense_top50)+len(r.sparse_top50),content_version_ok=content_ok,max_score_delta=maxdiff,
                    trace=list(trace),result=dataclasses.asdict(r),metrics=metrics)
            except Exception:row.update(ok=False,error=traceback.format_exc())
            failures+=not row['ok'];rows.append(row)
            with (OUT/'initial_observations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        st=state(ix,s);unchanged=expected_state is None or st==expected_state
        summaries.append(dict(condition=label,order=order,scope=s,queries=count,failed_queries=failures,state_unchanged=unchanged,state=st))
        save('initial_summary.json',summaries);status('condition_complete',condition=label,scope=s,order=order,queries=count,failures=failures,state_unchanged=unchanged)
        return st
    targets=sorted({q.conversation_id for q in queries})
    for s in targets:
        dd={m:docs[m] for m in inp.conversation_memories[s]};vv={m:vectors[m] for m in dd}
        registry.clear();ix=on.ScopedOnlineRetrievalIndex();load(ix,s,dd,vv);expected=evaluate(ix,s,'S0','target_only')
        for condition in ('S1','S4'):
            for order in ('target_first','external_first'):
                registry.clear();ix=on.ScopedOnlineRetrievalIndex()
                if order=='target_first':load(ix,s,dd,vv)
                if condition=='S1':
                    for other,ids in inp.conversation_memories.items():
                        if other!=s:load(ix,other,{m:docs[m] for m in ids},{m:vectors[m] for m in ids})
                else:
                    ids=list(dd);shift=ids[1:]+ids[:1]
                    load(ix,'collision-'+s,dict(zip(ids,[dd[m] for m in shift])),dict(zip(ids,[vv[m] for m in shift])),7)
                    assert all(ix.is_visible('collision-'+s,m,7)==(True,True) for m in ids)
                if order=='external_first':load(ix,s,dd,vv)
                evaluate(ix,s,condition,order,expected)
    for f,h in hashes.items():assert sha(Path(f))==h
    save('initial_complete.json',dict(queries=1146,observations=len(rows),failed=sum(not r['ok'] for r in rows),
        state_failures=sum(not r['state_unchanged'] for r in summaries),remaining=['S2','S3','S5','fault_control']))
    status('initial_matrix_complete',observations=len(rows),failed=sum(not r['ok'] for r in rows))
if __name__=='__main__':
    try:main()
    except Exception:status('failed',error=traceback.format_exc());raise
