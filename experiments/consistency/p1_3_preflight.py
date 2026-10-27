"""P1-3 measurement preflight. Real online index, no live database writes.

Uses source-bound vectors, forced independent sparse/dense propagation and
positive-control histories. These results are instrumentation validation,
NOT the four-backend 100K experiment.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import csv,json,hashlib,sys,time,threading
from dataclasses import replace
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments/serving_100k'))
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex,_bm25_search,_dense_search,zscore_fuse
from cassmem.representation.graph_event import digest
from run_graph_canonical_gate_v2 import build_graph_records
OUT=ROOT/'results/consistency/p1_3_20261012'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def classify(read,ready_ns,next_update_ns=None):
    """ready is immutable declaration history, not recomputed from this read."""
    if ready_ns is None or read['end_ns']<ready_ns:phase='pre_ready'
    elif read['start_ns']<ready_ns:phase='straddles_ready'
    elif next_update_ns is not None and read['end_ns']>=next_update_ns:phase='overlaps_successor'
    else:phase='post_ready'
    errors=[k for k in ['projection','sparse','dense','relation'] if not read['checks'][k]]
    return {'phase':phase,'mismatched_views':errors,'post_ready_violation':phase=='post_ready' and bool(errors)}

def capture(index,scope,mid,question,qv,target):
    start=time.perf_counter_ns()
    with index._scope_locks[scope]:
        sparse=index._sparse[scope];dense=index._dense[scope]
        sv=index._sparse_versions.get((scope,mid));dv=index._dense_versions.get((scope,mid))
        sn=time.perf_counter_ns()
    # Version evidence is captured with the actual immutable snapshot references.
    si=sparse.memory_ids.index(mid);di=dense.memory_ids.index(mid)
    sr=_bm25_search(sparse,question,50);dr=_dense_search(dense,qv,50)
    got=zscore_fuse(dr,sr,.6,10)
    expected_vector=target['vector']/max(float(np.linalg.norm(target['vector'])),1e-12)
    return {'start_ns':start,'snapshot_ns':sn,'end_ns':time.perf_counter_ns(),'sparse_version':sv,'dense_version':dv,
            'sparse_text_match':sparse.documents[si]==target['rawerk'],
            'dense_vector_match':bool(np.array_equal(dense.normalized_embeddings[di],expected_vector)),
            'top10':[x[0] for x in got],'sparse_top50':[x[0] for x in sr],'dense_top50':[x[0] for x in dr],
            'observed_document_sha256':hashlib.sha256(sparse.documents[si].encode()).hexdigest(),
            'observed_normalized_vector_sha256':hashlib.sha256(dense.normalized_embeddings[di].tobytes()).hexdigest()}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sourcefiles=[ROOT/'src/cassmem/retrieval/online.py',ROOT/'src/cassmem/backend/live_cells_graph.py',ROOT/'src/cassmem/representation/graph_event.py',ROOT/'data/locomo_memory_bge_large.npy',ROOT/'data/frozen_retrieval/p3_memory_features.csv']
    protocol={'run_id':'p1_3_20261012','state':'preflight_only_live_runs_pending_resource_release','primary_cells':['cassandra-base','cassandra-materialized','neo4j-native','neo4j-materialized'],
      'resources':{'cpus':8,'memory_gib':12,'heap_gib':5,'neo4j_page_cache_gib':3,'one_database_at_a_time':True},
      'A':{'corpus':100000,'workers':32,'updates_s':[1,5],'reads_per_update':19,'updates_per_scenario':200,'types_each':50,'types':['insert','raw_revision','structured_revision','relation_replacement_removal'],'repetitions':3},
      'B':{'workers':32,'updates_s':5,'repetitions':3,'same_id_versions':[1,2,3],'schedule':['overlap_predecessor','after_ready'],'superseded':'separately classified, never removed silently'},
      'C':{'workers':32,'updates_s':5,'repetitions':3,'delayed_components':['sparse','dense','relation'],'delay_ms':[100,500],'natural_and_injected_results':'separate'},
      'D':{'workers':32,'updates_s':5,'repetitions':3,'hot_scope_update_fraction':.8,'scope':'fixed from initial corpus before runs'},
      'timing':'perf_counter_ns; t_index is existing check-completion declaration; queue, raw call and independent observations separate; upstream preparation excluded',
      'read_eligibility':'start after t_index, end before successor begins; straddling/overlap reported separately',
      'readiness':'exact target version plus content; stored declaration immutable; independent observer must not redefine readiness to hide failures',
      'relation':'full target relation set plus negative checks for removed associations; unversioned rows use content match, not invented versions',
      'cross_backend':'same scope state/content fingerprint and reference checkpoints, not wall-clock times; other scopes remain active',
      'duration':'interval-bounded observation, end-of-run unresolved events censored; no assumed zero-duration failures',
      'instrumentation':'capture actual snapshots without joint write publication locks; record diagnostic load and paired overhead control',
      'formal_precondition':'resource isolation, live smoke and independent checker validation; original code failures are retained observations',
      'no_reader_judge':True,'original_sources':{str(p.relative_to(ROOT)):sha(p) for p in sourcefiles}}
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))==protocol
    else:save('protocol.json',protocol)
    checks=[]
    for name,start,end,ready,nxt,bad,phase,violation in [
      ('consistent_after_ready',101,110,100,None,[],'post_ready',False),
      ('old_dense_after_ready',101,110,100,None,['dense'],'post_ready',True),
      ('content_wrong_despite_version',101,110,100,None,['sparse'],'post_ready',True),
      ('removed_relation_remains',101,110,100,None,['relation'],'post_ready',True),
      ('pre_ready_mixed',10,20,100,None,['sparse','dense'],'pre_ready',False),
      ('straddling_read',90,110,100,None,['dense'],'straddles_ready',False),
      ('successor_overlap',101,120,100,115,['dense'],'overlaps_successor',False),
      ('never_ready',101,120,None,None,['projection'],'pre_ready',False)]:
        r={'start_ns':start,'end_ns':end,'checks':{k:k not in bad for k in ['projection','sparse','dense','relation']}}
        result=classify(r,ready,nxt);assert result['phase']==phase and result['post_ready_violation']==violation
        checks.append({'control':name,**result,'expected_violation':violation})
    save('checker_positive_controls.json',{'status':'PASS','tests':checks})
    records=build_graph_records();scope=records[0].scope_id
    selected=[r for r in records if r.scope_id==scope][:64];old=selected[0];donor=selected[1]
    ids=(ROOT/'data/locomo_memory_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines()
    vv=np.load(ROOT/'data/locomo_memory_bge_large.npy');vectors=dict(zip(ids,vv))
    # Replace source content with another existing record and its correctly bound vector;
    # this is a preflight revision fixture, not a fabricated benchmark QA example.
    new=replace(donor,scope_id=scope,memory_id=old.memory_id,version=2)
    assert new.raw_text!=old.raw_text
    index=ScopedOnlineRetrievalIndex()
    index.load_scope(scope,{r.memory_id:r.rawerk for r in selected},{r.memory_id:vectors[r.memory_id] for r in selected})
    target={'rawerk':new.rawerk,'vector':vectors[donor.memory_id]};q=new.raw_text;qv=target['vector']
    before=capture(index,scope,old.memory_id,q,qv,target)
    public=index.search(scope,q,qv)
    assert before['top10']==[m for m,_ in public.top10]
    half=threading.Event();resume=threading.Event();updated=threading.Event();errors=[]
    def writer():
        try:
            index.upsert_sparse(scope,old.memory_id,new.rawerk,2);half.set()
            assert resume.wait(5),'preflight reader timed out'
            index.upsert_dense(scope,old.memory_id,target['vector'],2);updated.set()
        except BaseException as e:errors.append(repr(e));half.set();updated.set()
    t=threading.Thread(target=writer);t.start();assert half.wait(5)
    intermediate=[capture(index,scope,old.memory_id,q,qv,target) for _ in range(10)]
    resume.set();assert updated.wait(5);t.join();assert not errors,errors
    ready=time.perf_counter_ns();after=[capture(index,scope,old.memory_id,q,qv,target) for _ in range(10)]
    assert all(r['sparse_version']==2 and r['dense_version']==1 for r in intermediate)
    assert all(r['sparse_version']==r['dense_version']==2 and r['sparse_text_match'] and r['dense_vector_match'] for r in after)
    reference=ScopedOnlineRetrievalIndex()
    reference.load_scope(scope,{r.memory_id:new.rawerk if r.memory_id==old.memory_id else r.rawerk for r in selected},
                         {r.memory_id:target['vector'] if r.memory_id==old.memory_id else vectors[r.memory_id] for r in selected},version=2)
    expected=[m for m,_ in reference.search(scope,q,qv).top10]
    assert all(r['top10']==expected for r in after)
    rejected_sparse=not index.upsert_sparse(scope,old.memory_id,old.rawerk,1)
    rejected_dense=not index.upsert_dense(scope,old.memory_id,vectors[old.memory_id],1)
    assert rejected_sparse and rejected_dense
    index.upsert_sparse(scope,old.memory_id,new.rawerk,3);index.upsert_dense(scope,old.memory_id,target['vector'],3)
    visible=index.is_visible(scope,old.memory_id,2);assert visible==(True,True)
    save('index_history.json',{'fixture':'64 source records, real original online index, deterministic forced interleaving, no database','before':before,'between_sparse_dense':intermediate,'ready_ns':ready,'after':after,'expected_top10':expected})
    summary={'status':'PASS','scope_records':64,'checker_controls':8,'forced_mixed_index_observations':len(intermediate),'post_ready_checked_reads':len(after),'post_ready_index_content_mismatches':0,'reference_top10_matches':len(after),'stale_upserts_rejected':rejected_sparse and rejected_dense,'at_least_visibility_accepts_later_version':all(visible),'source_files_unchanged':all(sha(ROOT/p)==h for p,h in protocol['original_sources'].items()),'database_experiment_started':False,'limits':'controlled index preflight; not real-backend version consistency, not natural failure rates; current representation-cost run untouched'}
    assert summary['source_files_unchanged']
    save('preflight_summary.json',summary)
    save('status.json',{'phase':'local_preflight_complete','live_database_phase':'waiting_for_resource_release_and_live_smoke','database_writes':0,'container_changes':0})
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
