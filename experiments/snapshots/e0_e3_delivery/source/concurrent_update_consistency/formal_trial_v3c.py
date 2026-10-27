"""Instrumented consistency workload, not a P2-2 throughput rerun.

Reads use captured immutable scorer snapshots and log target content/version.
No write-lock is added around backend reads. Same memory revised once per phase.
"""
from __future__ import annotations
import argparse, dataclasses, gc, hashlib, json, random, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from runtime import HERE,docker,NEO
from common import np,write_json,write_csv,sha,ROOT
from benchmark import data,make_index,backend
from cassmem.representation.graph_event import digest
from cassmem.retrieval.online import _bm25_search,_dense_search,zscore_fuse,ScopedOnlineRetrievalIndex
from preflight import observation
from neo4j_collected_v3 import Neo4jCollectedGraphCells

def get_backend(cell,keyspace):
    if cell.startswith('cassandra'):
        return backend(keyspace)
    info=json.loads(docker('inspect',NEO).stdout)[0]
    auth=next(e.split('=',1)[1] for e in info['Config']['Env'] if e.startswith('NEO4J_AUTH='))
    user,pw=auth.split('/',1)
    return Neo4jCollectedGraphCells('bolt://127.0.0.1:17687',user,pw)

def score(sparse,dense,question,qvec,candidates):
    allowed=set(candidates)
    sr=[x for x in _bm25_search(sparse,question,max(50,len(sparse.memory_ids))) if x[0] in allowed][:50]
    dr=[x for x in _dense_search(dense,qvec,max(50,len(dense.memory_ids))) if x[0] in allowed][:50]
    return zscore_fuse(dr,sr),sr,dr

def phase_class(start,end,arrival,ready):
    if ready and start>=ready: return 'post_ready'
    if ready and end>=ready: return 'crosses_ready'
    if start<arrival: return 'before_arrival'
    return 'pre_ready'

class Trial:
    def __init__(self,a):
        self.a=a; self.records,self.grouped,self.queries,self.mv,self.qv=data(a.memories)
        self.byid={r.memory_id:r for r in self.records}
        self.b=get_backend(a.cell,a.keyspace)
        self.index=make_index(self.grouped,self.mv,'RawERK')
        self.touched=[]; self.mutex=threading.Lock()

    def events(self,count,phase):
        rng=random.Random(20261013+self.a.rep+(1000 if phase=='measured' else 0))
        eligible=[r for r in self.records if r.edges]
        rng.shuffle(eligible)
        if count>len(eligible): raise ValueError('Insufficient unique revision targets')
        result=[]
        for i,old in enumerate(eligible[:count]):
            others=[r for r in self.grouped[old.scope_id] if r.raw_text!=old.raw_text and r.edges
                    and {e['relation'] for e in old.edges}-{e['relation'] for e in r.edges}]
            if not others: raise ValueError('No genuine revision source for scope')
            source=rng.choice(others)
            mid=old.memory_id if self.a.mode=='revision' else f'consistency_{phase}_{i:06d}::{old.memory_id}'
            target=dataclasses.replace(source,memory_id=mid,version=old.version+1)
            vec=self.mv[source.memory_id.split('::')[-1]]
            assert hashlib.sha256(np.asarray(vec,dtype=np.float32).tobytes()).hexdigest()==target.embedding_sha256
            result.append({'event_id':i,'old':old if self.a.mode=='revision' else None,'target':target,
                           'source':source,'vector':vec,'query':rng.choice(self.queries[old.scope_id.split('::')[-1]]),
                           'arrival_ns':0,'ready_ns':0})
        return result

    def update(self,e,scheduled,submitted):
        t=e['target']; started=time.perf_counter_ns()
        row={'kind':'update','event_id':e['event_id'],'scheduled_ns':scheduled,'submit_ns':submitted,
             'start_ns':started,'status':'error','memory_id':t.memory_id,'scope':t.scope_id}
        with self.mutex: self.touched.append(e)
        try:
            row['raw_call_start_ns']=time.perf_counter_ns(); self.b.commit_raw(self.a.cell,t)
            row['raw_ack_ns']=time.perf_counter_ns(); self.b.write_structured(self.a.cell,t)
            row['structured_end_ns']=time.perf_counter_ns()
            actual=self.b.fetch_graph(self.a.cell,t.scope_id,t.memory_id)
            if actual is None or digest(actual)!=digest(t.graph_projection()): raise AssertionError('canonical mismatch before index update')
            if self.b.candidate_projection(self.a.cell,t.scope_id,t.memory_id)!=t.rawerk: raise AssertionError('projection mismatch')
            self.index.upsert_sparse(t.scope_id,t.memory_id,t.rawerk,t.version)
            row['sparse_end_ns']=time.perf_counter_ns()
            self.index.upsert_dense(t.scope_id,t.memory_id,e['vector'],t.version)
            row['dense_end_ns']=time.perf_counter_ns()
            checked=observation(self.b,self.a.cell,e['old'],t,self.index,e['vector'])
            row['ready_check']=checked
            if not checked['strict_ready']: raise AssertionError('strict target check failed')
            e['ready_ns']=checked['end_ns']; row['ready_ns']=e['ready_ns']; row['status']='ok'
        except Exception as exc: row['error']=repr(exc)
        row['end_ns']=time.perf_counter_ns()
        return row

    def read(self,e,scheduled,submitted,followup=False):
        t=e['target']; old=e['old']; s=t.scope_id; mid=t.memory_id
        row={'kind':'read','event_id':e['event_id'],'scheduled_ns':scheduled,'submit_ns':submitted,
             'start_ns':time.perf_counter_ns(),'status':'error','followup':followup}
        try:
            ids=self.b.ids(self.a.cell,s)
            row['candidate_read_end_ns']=time.perf_counter_ns()
            # Capture version sidecars with the SAME snapshots actually scored.
            with self.index._scope_locks[s]:
                sp,de=self.index._sparse[s],self.index._dense[s]
                sv=self.index._sparse_versions.get((s,mid),-1); dv=self.index._dense_versions.get((s,mid),-1)
                spv={m:self.index._sparse_versions[(s,m)] for m in sp.memory_ids}
                dev={m:self.index._dense_versions[(s,m)] for m in de.memory_ids}
                row['snapshot_ns']=time.perf_counter_ns()
            ranking,sr,dr=score(sp,de,e['query']['question'],self.qv[e['query']['query_id']],ids)
            row['ranking_end_ns']=time.perf_counter_ns()
            doc=sp.documents[sp.memory_ids.index(mid)] if mid in sp.memory_ids else None
            vector=de.normalized_embeddings[de.memory_ids.index(mid)] if mid in de.memory_ids else None
            targetvec=e['vector']/max(float(np.linalg.norm(e['vector'])),1e-12)
            graph=self.b.fetch_graph(self.a.cell,s,mid)
            projection=self.b.candidate_projection(self.a.cell,s,mid)
            oldrels={x['relation'] for x in old.edges} if old else set()
            newrels={x['relation'] for x in t.edges}
            memberships={rel:mid in self.b.ids_by_relation(self.a.cell,s,rel) for rel in sorted(oldrels|newrels)}
            target_rel=all(memberships[r]==(r in newrels) for r in memberships)
            cv=t.version if graph and digest(graph)==digest(t.graph_projection()) else (
                old.version if old and graph and digest(graph)==digest(old.graph_projection()) else -1)
            sparse_ok=sv==t.version and doc==t.rawerk
            dense_ok=dv==t.version and vector is not None and bool(np.allclose(vector,targetvec,rtol=1e-6,atol=1e-7))
            row.update(status='ok',canonical_version=cv,sparse_version=sv,dense_version=dv,
                canonical_ok=cv==t.version,projection_ok=projection==t.rawerk,sparse_ok=sparse_ok,dense_ok=dense_ok,
                relation_ok=target_rel,target_in_candidates=mid in ids,target_in_top10=mid in {m for m,_ in ranking},
                strict_target_ok=bool(cv==t.version and projection==t.rawerk and sparse_ok and dense_ok and target_rel and mid in ids),
                known_mixed_versions=len({v for v in (cv,sv,dv) if v>=0})>1,
                stale_relation=any(memberships[r] for r in oldrels-newrels),
                target_sparse_candidate_version=sv if mid in {m for m,_ in sr} else None,
                target_dense_candidate_version=dv if mid in {m for m,_ in dr} else None,
                top10=[{'id':m,'score':float(v),'sparse_snapshot_version':spv.get(m,-1),'dense_snapshot_version':dev.get(m,-1)} for m,v in ranking],
                observed_graph_hash=digest(graph) if graph else None,expected_graph_hash=digest(t.graph_projection()),
                relation_memberships=memberships)
            if not row['strict_target_ok']:
                row['observed_graph']=graph
                row['observed_projection']=projection
                row['sparse_document']=doc
        except Exception as exc: row['error']=repr(exc)
        row['end_ns']=time.perf_counter_ns()
        return row

    def phase(self,name,duration,rate,out):
        events=self.events(int(round(duration*rate)),name)
        write_json(out/(name+'_trace.json'),[{'event_id':e['event_id'],'old':dataclasses.asdict(e['old']) if e['old'] else None,
             'target':dataclasses.asdict(e['target']),'source_id':e['source'].memory_id,'query_id':e['query']['query_id']} for e in events])
        start=time.perf_counter_ns()+2_000_000_000; end=start+int(duration*1e9); deadline=end+120_000_000_000
        futures=[]; submissions=[]
        with ThreadPoolExecutor(max_workers=32) as pool:
            proc=subprocess.Popen([sys.executable,'-B',str(HERE.parent/'representation_serving_cost/arrival_emitter.py'),
                 '--start-ns',str(start),'--count',str(len(events)*20),'--ops-rate',str(rate*20),
                 '--log',str(out/(name+'_emitter.jsonl'))],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
            try:
                for line in proc.stdout:
                    ordinal,scheduled,emitted=map(int,line.strip().split(','))
                    assert ordinal==len(futures)
                    e=events[ordinal//20]; submitted=time.perf_counter_ns()
                    submissions.append({'ordinal':ordinal,'scheduled_ns':scheduled,'emit_ns':emitted,'submit_ns':submitted})
                    if ordinal%20==0:
                        e['arrival_ns']=submitted
                        futures.append(pool.submit(self.update,e,scheduled,submitted))
                    else: futures.append(pool.submit(self.read,e,scheduled,submitted))
                    submissions[-1]['enqueue_return_ns']=time.perf_counter_ns()
                if proc.wait(timeout=20): raise RuntimeError(proc.stderr.read())
                done,pending=wait(futures,timeout=max(0,(deadline-time.perf_counter_ns())/1e9))
                write_json(out/(name+'_deadline.json'),{'deadline_ns':deadline,'pending':len(pending),'observed_ns':time.perf_counter_ns()})
                rows=[f.result() for f in futures]
                # Explicit post-ready probes are separate from offered business reads.
                follow=[]
                for e in events:
                    if e['ready_ns']:
                        now=time.perf_counter_ns(); follow.append(pool.submit(self.read,e,now,now,True))
                rows.extend(f.result() for f in follow)
            finally:
                if proc.poll() is None: proc.terminate(); proc.wait(timeout=10)
        for r in rows:
            e=events[r['event_id']]
            r['ready_ns']=e['ready_ns']
            r['phase_class']=phase_class(r['start_ns'],r['end_ns'],e['arrival_ns'],e['ready_ns'])
            r['queue_ms']=(r['start_ns']-r['submit_ns'])/1e6
            if r['kind']=='update':
                r['ready_censored']=not e['ready_ns'] or e['ready_ns']>deadline
                r['arrival_to_ready_ms']=(e['ready_ns']-e['arrival_ns'])/1e6 if e['ready_ns'] else None
        with (out/(name+'_observations.jsonl')).open('w',encoding='utf-8') as h:
            for r in rows: h.write(json.dumps(r,ensure_ascii=False)+'\n')
        write_csv(out/(name+'_submissions.csv'),submissions)
        reads=[r for r in rows if r['kind']=='read']; post=[r for r in reads if r['phase_class']=='post_ready' and r['status']=='ok']
        lag=np.percentile([(r['submit_ns']-r['scheduled_ns'])/1e6 for r in submissions],95)
        enqueue_lag=np.percentile([(r['enqueue_return_ns']-r['scheduled_ns'])/1e6 for r in submissions],95)
        summary={'phase':name,'updates':len(events),'business_reads':len(events)*19,
          'actual_operations':len(futures),'observation_reads':len(reads),'post_ready_valid_reads':len(post),
          'post_ready_target_failures':sum(not r['strict_target_ok'] for r in post),
          'post_ready_stale_relation':sum(r['stale_relation'] for r in post),
          'mixed_reads_all_windows':sum(r.get('known_mixed_versions',False) for r in reads),
          'errors':sum(r['status']!='ok' for r in rows),'ready_updates':sum(bool(e['ready_ns']) for e in events),
          'deadline_censored_updates':sum(r.get('ready_censored',False) for r in rows if r['kind']=='update'),
          'submission_lag_p95_ms':float(lag),
          'enqueue_return_lag_p95_ms':float(enqueue_lag),
          'emitter_lag_p95_ms':float(np.percentile([(r['emit_ns']-r['scheduled_ns'])/1e6 for r in submissions],95)),
          'switch_interval_seconds':sys.getswitchinterval(),
          'offered_load_valid':bool(len(futures)==len(events)*20 and lag<=50 and max(r['submit_ns'] for r in submissions)<=end+1_000_000_000),
          'clock':'actual submission -> strict check completion; planned clock recorded separately',
          'followups_not_in_offered_load':sum(r.get('followup',False) for r in reads)}
        # Per-event sampled abnormal intervals; do not claim exact continuous duration.
        intervals=[]
        for e in events:
            er=[r for r in reads if r['event_id']==e['event_id'] and r['status']=='ok']
            bad=[r for r in er if not r['strict_target_ok']]
            last=max((r['end_ns'] for r in bad),default=None)
            recovery=min((r['end_ns'] for r in er if r['strict_target_ok'] and last and r['start_ns']>last),default=None)
            intervals.append({'event_id':e['event_id'],'bad_observations':len(bad),
                'first_bad_start_ns':min((r['start_ns'] for r in bad),default=None),'last_bad_end_ns':last,
                'first_later_good_end_ns':recovery,'unrecovered_in_observation_window':bool(bad and not recovery),
                'max_observation_start_gap_ms':max(((y-x)/1e6 for x,y in zip(sorted(r['start_ns'] for r in er),sorted(r['start_ns'] for r in er)[1:])),default=None)})
        write_json(out/(name+'_intervals.json'),intervals)
        write_json(out/(name+'_summary.json'),summary)
        return events,summary

    def checkpoint(self,events,out):
        touched={e['target'].memory_id:e for e in events}
        grouped={s:{r.memory_id:r for r in rr} for s,rr in self.grouped.items()}
        for mid,e in touched.items(): grouped[e['target'].scope_id][mid]=e['target']
        results=[]
        for scope in sorted(grouped):
            recs=grouped[scope]; idx=ScopedOnlineRetrievalIndex()
            embeds={mid:(touched[mid]['vector'] if mid in touched else self.mv[mid.split('::')[-1]]) for mid in recs}
            idx.load_scope(scope,{mid:r.rawerk for mid,r in recs.items()},embeds)
            q=self.queries[scope.split('::')[-1]][0]
            ids=self.b.ids(self.a.cell,scope)
            expected=idx.search(scope,q['question'],self.qv[q['query_id']],recs)
            actual=self.index.search(scope,q['question'],self.qv[q['query_id']],ids)
            target_checks=[]
            for mid,e in touched.items():
                if e['target'].scope_id==scope:
                    g=self.b.fetch_graph(self.a.cell,scope,mid)
                    target_checks.append({'id':mid,'matches':g is not None and digest(g)==digest(e['target'].graph_projection())})
            results.append({'scope':scope,'query_id':q['query_id'],
                'logical_state_hash':hashlib.sha256(json.dumps([r.memory_projection() for _,r in sorted(recs.items())],sort_keys=True).encode()).hexdigest(),
                'candidate_match':ids==sorted(recs),'top10_match':[x[0] for x in actual.top10]==[x[0] for x in expected.top10],
                'actual_top10':[x[0] for x in actual.top10],'expected_top10':[x[0] for x in expected.top10],
                'target_projection_checks':target_checks})
        write_json(out/'checkpoint.json',results)
        return all(r['candidate_match'] and r['top10_match'] and all(t['matches'] for t in r['target_projection_checks']) for r in results)

    def cleanup(self):
        for e in self.touched:
            if e['old']:
                self.b.commit_raw(self.a.cell,e['old']); self.b.write_structured(self.a.cell,e['old'])
            else: self.b.delete(self.a.cell,e['target'])
        self.touched=[]

def main():
    # Common client runtime policy for every backend; never change arrival timestamps or thresholds.
    sys.setswitchinterval(0.0005)
    p=argparse.ArgumentParser(); p.add_argument('--cell',required=True); p.add_argument('--mode',choices=['insert','revision'],required=True)
    p.add_argument('--rate',type=float,required=True); p.add_argument('--rep',type=int,required=True)
    p.add_argument('--memories',type=int,default=100000);p.add_argument('--keyspace',default='repcost_consistency_formal_v1')
    p.add_argument('--warmup',type=float,default=60);p.add_argument('--seconds',type=float,default=120)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'manifest.json',{'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
       'adapter_sha256':sha(ROOT/'src/cassmem/backend/live_cells_graph.py'),'runner_sha256':sha(Path(__file__)),
       'candidate_wrapper_sha256':sha(HERE/'neo4j_collected_v3.py'),'switch_interval_seconds':sys.getswitchinterval(),
       'instrumentation':'All business reads additionally audit target canonical/projection/relation; immutable snapshots scored and versions logged; not P2-2 throughput comparable'})
    t=Trial(a)
    try:
        assert sum(len(t.b.ids(a.cell,s)) for s in t.grouped)==len(t.records)
        if isinstance(t.b,Neo4jCollectedGraphCells):
            checks=t.b.validate_candidate_equivalence(a.cell,t.grouped)
            write_json(a.output/'candidate_equivalence.json',checks)
            assert all(r['exact_match'] for r in checks)
        if a.warmup:
            _,warm=t.phase('warmup',a.warmup,a.rate,a.output)
            t.cleanup(); del t.index; gc.collect();t.index=make_index(t.grouped,t.mv,'RawERK')
        events,result=t.phase('measured',a.seconds,a.rate,a.output)
        result['checkpoint_pass']=t.checkpoint(events,a.output)
        result['status']='PASS' if result['errors']==0 and result['post_ready_target_failures']==0 and result['checkpoint_pass'] and result['offered_load_valid'] and not result['deadline_censored_updates'] else 'REVIEW_REQUIRED'
        write_json(a.output/'summary.json',result); print(json.dumps(result),flush=True)
    finally:
        try: t.cleanup()
        finally: t.b.close()

if __name__=='__main__': main()
