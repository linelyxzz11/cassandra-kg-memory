"""Isolated diagnostic, not a performance trial. Never changes serving code."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'representation_serving_cost'))
from common import np, write_json, write_csv, build_graph_records, vectors, sha, ROOT
from benchmark import backend
from environment import docker, safe_config, NAME, ORIGINALS
from cassmem.representation.graph_event import digest
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex
import dataclasses
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor


def observation(b, cell, old, target, index, vector):
    start = time.perf_counter_ns()
    scope, mid = target.scope_id, target.memory_id
    graph = b.fetch_graph(cell, scope, mid)
    projection = b.candidate_projection(cell, scope, mid)
    old_rels = {e['relation'] for e in old.edges} if old else set()
    new_rels = {e['relation'] for e in target.edges}
    stale_relations = [rel for rel in sorted(old_rels-new_rels)
                       if mid in b.ids_by_relation(cell, scope, rel)]
    new_relation_ok = all(mid in b.ids_by_relation(cell, scope, rel) for rel in new_rels)
    with index._scope_locks[scope]:
        sparse, dense = index._sparse[scope], index._dense[scope]
        sv = index._sparse_versions.get((scope, mid), -1)
        dv = index._dense_versions.get((scope, mid), -1)
        sparse_ok = mid in sparse.memory_ids and sparse.documents[sparse.memory_ids.index(mid)] == target.rawerk
        expected = vector / max(float(np.linalg.norm(vector)), 1e-12)
        dense_ok = mid in dense.memory_ids and np.allclose(dense.normalized_embeddings[dense.memory_ids.index(mid)], expected, rtol=1e-6, atol=1e-7)
    graph_ok = graph is not None and digest(graph) == digest(target.graph_projection())
    checks = dict(canonical_ok=graph_ok, projection_ok=projection == target.rawerk,
                  sparse_ok=bool(sparse_ok and sv == target.version),
                  dense_ok=bool(dense_ok and dv == target.version),
                  new_relation_ok=new_relation_ok, old_relation_removed=not stale_relations)
    return dict(start_ns=start, end_ns=time.perf_counter_ns(), **checks,
                raw_version=graph['memory']['version'] if graph else -1,
                sparse_version=sv, dense_version=dv, stale_relations=';'.join(stale_relations),
                strict_ready=all(checks.values()),
                legacy_checks_pass=all(v for k,v in checks.items() if k != 'old_relation_removed'),
                canonical_hash=digest(graph) if graph else '', expected_hash=digest(target.graph_projection()))


def index_detector_tests():
    idx = ScopedOnlineRetrievalIndex()
    a, b = np.eye(2, dtype=np.float32)
    idx.load_scope('test', {'m':'old apples'}, {'m':a})
    idx.upsert_sparse('test','m','new bananas',2)
    assert idx._sparse_versions[('test','m')] == 2 and idx._dense_versions[('test','m')] == 1
    idx.upsert_dense('test','m',b,2)
    assert idx.is_visible('test','m',2) == (True,True)
    assert idx._sparse['test'].documents[0] == 'new bananas'
    assert np.allclose(idx._dense['test'].normalized_embeddings[0],b)
    return {'mixed_snapshot_detected':True,'updated_content_verified':True}


def run(out):
    # Actual frozen source records/vectors, changed source content under stable target ID.
    records = build_graph_records()
    mv = vectors('memory')
    first = next(r for r in records if r.edges)
    second = next(r for r in records if r.scope_id == first.scope_id and r.edges
                  and set(e['relation'] for e in first.edges)-set(e['relation'] for e in r.edges)
                  and r.raw_text != first.raw_text)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'detector_tests.json', index_detector_tests())
    state = {'originals':{n:safe_config(n) for n in ORIGINALS},'experiment_before':safe_config(NAME),
             'status':'PREPARING','output':str(out)}
    write_json(out/'environment.json',state)
    b = None
    try:
        assert state['experiment_before']['state'] != 'running', 'Experiment container already in use'
        for n in ORIGINALS:
            if state['originals'][n]['state'] == 'running':
                docker('stop','--time','60',n)
        docker('start',NAME)
        for attempt in range(90):
            p = docker('exec',NAME,'cqlsh','-e','SELECT release_version FROM system.local;',check=False,timeout=30)
            if p.returncode == 0:
                break
            time.sleep(2)
        else:
            raise RuntimeError('Cassandra not ready')
        state['experiment'] = safe_config(NAME)
        assert state['experiment']['resources'] == {'Memory':12*1024**3,'NanoCpus':8*10**9,'MemorySwap':12*1024**3}
        state['nodetool_info'] = docker('exec',NAME,'nodetool','info').stdout
        state['version'] = p.stdout
        keyspace = 'repcost_consistency_' + out.name.replace('-','_')
        state['keyspace'] = keyspace
        b = backend(keyspace)
        all_rows, summaries = [], []
        for cell in b.names:
            for mode in ('insert','revision'):
                for rep in range(3):
                    scope = f'consistency_{mode}_{rep}'
                    mid = 'target'
                    old = dataclasses.replace(first,scope_id=scope,memory_id=mid,version=1)
                    target = dataclasses.replace(second,scope_id=scope,memory_id=mid,version=2)
                    vec = mv[second.memory_id]
                    assert hashlib.sha256(np.asarray(vec,dtype=np.float32).tobytes()).hexdigest() == target.embedding_sha256
                    background = dataclasses.replace(first,scope_id=scope,memory_id='background')
                    b.insert(cell,background)
                    docs = {'background':background.rawerk}
                    embeds = {'background':mv[first.memory_id]}
                    if mode == 'revision':
                        b.insert(cell,old)
                        docs[mid], embeds[mid] = old.rawerk, mv[first.memory_id]
                    idx = ScopedOnlineRetrievalIndex()
                    idx.load_scope(scope, docs, embeds)
                    stop = threading.Event()
                    obs = []
                    def reader():
                        while not stop.is_set():
                            try:
                                obs.append(observation(b,cell,old if mode=='revision' else None,target,idx,vec))
                            except Exception as exc:
                                obs.append({'error':repr(exc),'start_ns':time.perf_counter_ns()})
                            stop.wait(.02)
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(reader)
                        clocks = {'arrival_ns':time.perf_counter_ns()}
                        try:
                            b.commit_raw(cell,target)
                            clocks['raw_ack_ns'] = time.perf_counter_ns()
                            b.write_structured(cell,target)
                            clocks['structured_end_ns'] = time.perf_counter_ns()
                            idx.upsert_sparse(scope,mid,target.rawerk,2)
                            clocks['sparse_end_ns'] = time.perf_counter_ns()
                            idx.upsert_dense(scope,mid,vec,2)
                            clocks['dense_end_ns'] = time.perf_counter_ns()
                            final = observation(b,cell,old if mode=='revision' else None,target,idx,vec)
                            clocks['checks_end_ns'] = final['end_ns']
                            time.sleep(1)
                        finally:
                            stop.set()
                            future.result()
                    post = [r for r in obs if r['start_ns'] >= clocks['checks_end_ns']]
                    summaries.append(dict(cell=cell,mode=mode,rep=rep,**clocks,final=final,
                        observations=len(obs),post_checks_observations=len(post),
                        post_checks_failures=sum(not r.get('strict_ready',False) for r in post)))
                    all_rows.extend(dict(cell=cell,mode=mode,rep=rep,**r) for r in obs)
                    write_json(out/'trials.json',summaries)
                    write_csv(out/'observations.csv',all_rows)
                    print(cell,mode,rep,'strict_ready=',final['strict_ready'],'stale=',final['stale_relations'],flush=True)
        result = {'status':'PASS' if all(r['final']['strict_ready'] for r in summaries) else 'REVISION_PREFLIGHT_FAILED',
                  'trial_count':len(summaries),'failed_trials':sum(not r['final']['strict_ready'] for r in summaries),
                  'scope':'Diagnostic 2-memory scopes, no injected delays; not 100K performance or formal incidence estimates',
                  'source_hashes':{str(p):sha(p) for p in (ROOT/'src/cassmem/backend/live_cells_graph.py',ROOT/'src/cassmem/retrieval/online.py')},
                  'source_ids':[first.memory_id,second.memory_id], 'trials':summaries}
        write_json(out/'summary.json',result)
        print(result['status'],flush=True)
    finally:
        if b:
            b.close()
        if state['experiment_before']['state'] != 'running':
            docker('stop','--time','60',NAME)
        for n in ORIGINALS:
            if state['originals'][n]['state']=='running':
                docker('start',n)
        state['status']='RESTORED_ORIGINALS'
        write_json(out/'environment.json',state)


if __name__=='__main__':
    run(HERE/'results'/('preflight_'+time.strftime('%Y%m%d_%H%M%S')))
