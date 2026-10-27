"""Harness unit tests and independent schema/coverage/barrier validation."""
import sys,json,threading,time,copy,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'D:/memorytable/cassandra-kg-memory/src')
import numpy as np
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex,_build_dense

def audit(cases,events,reads,states):
    errors=[]
    def check(ok,msg):
        if not ok:errors.append(msg)
    expected={c['case_id'] for c in cases}
    check({e.get('case_id') for e in events}==expected,'case coverage')
    allrows=events+reads+states
    keys=[(r.get('case_id'),r.get('seq')) for r in allrows]
    check(len(keys)==len(set(keys)),'unique event_id')
    for c in cases:
        cid=c['case_id'];es=[e for e in events if e.get('case_id')==cid];rs=[r for r in reads if r.get('case_id')==cid]
        combined=sorted([r for r in allrows if r.get('case_id')==cid],key=lambda r:r['seq'])
        check([r['seq'] for r in combined]==list(range(len(combined))),'sequence coverage '+cid)
        check(all(x['ns']<=y['ns'] for x,y in zip(combined,combined[1:])),'monotonic '+cid)
        for v in (2,3):
            check(sum(e['kind']=='submit' and e.get('version')==v for e in es)==1,'offered event '+cid)
            check(sum(e['kind']=='update_end' and e.get('version')==v for e in es)==1,'completion event '+cid)
        check(any(r.get('tag')=='initial' for r in rs) and all(any(r.get('tag')=='final_'+str(i) for r in rs) for i in range(3)),'read windows '+cid)
        check(any(r.get('tag')=='concurrent' for r in rs),'concurrent reader '+cid)
        for r in rs:
            required={'start_ns','end_ns','snapshot_ns','query_end_ns','target_at_start','graph','projection','sparse_document','dense_vector_sha256','sparse_version','dense_version','relation_memberships','ids','top10','sparse_candidates','dense_candidates'}
            check(required<=set(r),'read schema '+cid)
            if required<=set(r):check(r['start_ns']<=r['snapshot_ns']<=r['query_end_ns']<=r['end_ns']<=r['ns'],'read clocks '+cid)
        if c['blocked_stage']:
            pick=lambda kind,version:[e for e in es if e['kind']==kind and e.get('version')==version]
            h=pick('hook_reached',2);rel=pick('hook_released',2);ready=pick('ready',3)
            st=[e for e in pick('stage_start',2) if e['stage']==c['blocked_stage']]
            check(len(h)==len(rel)==len(ready)==len(st)==1 and h[0]['ns']<ready[0]['ns']<rel[0]['ns']<=st[0]['ns'],'barrier order '+cid)
    return errors

def main(out):
    fixtures=json.loads((out/'fixtures.json').read_text(encoding='utf-8'));results={}
    for f in fixtures:
        for v in f['versions']:
            vec=np.asarray(v['vector'],dtype=np.float32);norm=_build_dense({'x':vec}).normalized_embeddings[0]
            assert hashlib.sha256(norm.tobytes()).hexdigest()==v['normalized_vector_sha256']
    results['150_frozen_dense_content_hashes']=True
    idx=ScopedOnlineRetrievalIndex();idx.load_scope('s',{'m':'first apples'},{'m':np.array([1,0],dtype=np.float32)})
    assert idx.upsert_sparse('s','m','third oranges',3) and idx.upsert_dense('s','m',[0,1],3)
    assert not idx.upsert_sparse('s','m','second bananas',2) and not idx.upsert_dense('s','m',[1,1],2)
    assert idx._sparse['s'].documents[0]=='third oranges';results['original_index_rejects_old_version']=True
    hit,release=threading.Event(),threading.Event();trace=[]
    def old():
        trace.append(('hook',time.perf_counter_ns()));hit.set();assert release.wait(2);trace.append(('v2_attempt',time.perf_counter_ns()))
    with ThreadPoolExecutor(1) as pool:
        future=pool.submit(old);assert hit.wait(2);trace.append(('v3_done',time.perf_counter_ns()));release.set();future.result()
    assert [x[0] for x in trace]==['hook','v3_done','v2_attempt'];results['barrier_order_unit']=True
    if (out/'complete.json').exists():
        loadlines=lambda n:[json.loads(s) for s in (out/n).read_text(encoding='utf-8').splitlines()]
        cases=json.loads((out/'schedules.json').read_text(encoding='utf-8'));events=loadlines('timeline.jsonl');reads=loadlines('reads.jsonl');states=loadlines('component_states.jsonl')
        errors=audit(cases,events,reads,states);results['live_schema_coverage_clock_barrier_errors']=errors
        # Mutate only in-memory copies; never touch raw evidence.
        assert audit(cases,events+events[:1],reads,states);results['duplicate_detected']=True
        assert audit(cases,[e for e in events if not(e['kind']=='submit' and e.get('version')==2)],reads,states);results['missing_offered_detected']=True
        rr=copy.deepcopy(reads);rr[0].pop('graph');assert audit(cases,events,rr,states);results['schema_error_detected']=True
        rr=copy.deepcopy(reads);rr[0]['end_ns']=rr[0]['start_ns']-1;assert audit(cases,events,rr,states);results['clock_error_detected']=True
        ee=copy.deepcopy(events);r=next(e for e in ee if e['kind']=='hook_released');r['ns']=0;assert audit(cases,ee,reads,states);results['wrong_order_detected']=True
        assert not errors,errors
    (out/'harness_tests.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results,indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
