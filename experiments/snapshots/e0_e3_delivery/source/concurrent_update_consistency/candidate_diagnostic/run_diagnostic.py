"""Read-only, same-ID query/driver diagnostic; no serving implementation edits."""
import sys, os, json, time, hashlib, subprocess, threading, argparse, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from formal_trial import data, make_index, score
from common import np, write_json, ROOT, sha
from cassmem.backend.live_cells_graph import Neo4jGraphCells
from cassmem.representation.graph_event import digest
import neo4j

NAME = 'cassmem-consistency-20261013-v2-neo4j'
QUERY = 'MATCH (m:LWV2NativeMemory {scope_id:$s}) RETURN m.memory_id AS id'
COLLECT = 'MATCH (m:LWV2NativeMemory {scope_id:$s}) RETURN collect(m.memory_id) AS ids'
TLS = threading.local()
now = time.perf_counter_ns

def stats(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'mean': float(a.mean()), 'p50': float(np.percentile(a,50)),
            'p95':float(np.percentile(a,95)), 'max':float(a.max())} if len(a) else {}

def ids_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()

def timed_ids(driver, scope, variant):
    TLS.acquire_ns = 0
    TLS.track = True
    row = {'scope':scope, 'variant':variant}
    began = now(); cpu = time.thread_time_ns()
    try:
        with driver.session(database='neo4j') as s:
            created = now()
            result = s.run(QUERY if variant=='stream' else COLLECT, s=scope)
            ran = now()
            records = list(result)
            fetched = now()
            summary = result.consume()
            consumed = now()
        closed = now()
        ids = sorted(r['id'] for r in records) if variant=='stream' else sorted(records[0]['ids'])
        ended = now()
        row.update(session_create_ms=(created-began)/1e6, run_ms=(ran-created)/1e6,
            fetch_ms=(fetched-ran)/1e6, consume_ms=(consumed-fetched)/1e6,
            close_ms=(closed-consumed)/1e6, extract_sort_ms=(ended-closed)/1e6,
            candidate_ms=(ended-began)/1e6, acquire_ms=TLS.acquire_ns/1e6,
            client_thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6,
            server_available_ms=summary.result_available_after,
            server_consumed_ms=summary.result_consumed_after,
            bolt_records=len(records), ids=len(ids), ids_hash=ids_hash(ids))
        return ids, row
    finally:
        TLS.track = False

def main(out):
    out.mkdir(parents=True, exist_ok=False)
    driver=None
    try:
        info=json.loads(subprocess.run(['docker','inspect',NAME],check=True,capture_output=True,text=True).stdout)[0]
        auth=next(x.split('=',1)[1] for x in info['Config']['Env'] if x.startswith('NEO4J_AUTH='))
        user,pw=auth.split('/',1)
        driver=neo4j.GraphDatabase.driver('bolt://127.0.0.1:17687',auth=(user,pw),max_connection_pool_size=128)
        for attempt in range(45):
            try: driver.verify_connectivity(); break
            except Exception:
                if attempt==44:raise
                time.sleep(2)
        original_acquire=driver._pool.acquire
        def acquire(*args,**kwargs):
            began=now()
            try:return original_acquire(*args,**kwargs)
            finally:
                if getattr(TLS,'track',False):TLS.acquire_ns+=now()-began
        driver._pool.acquire=acquire
        # Use original adapter read methods without its schema-creating constructor.
        backend=Neo4jGraphCells.__new__(Neo4jGraphCells); backend.d=driver;backend.database='neo4j'
        records,grouped,queries,mv,qv=data(100000)
        index=make_index(grouped,mv,'RawERK'); byid={r.memory_id:r for r in records}
        qmap={q['query_id']:q for qs in queries.values() for q in qs}
        trace=json.loads((HERE.parent/'results/formal_20261013_v2/neo4j-native_revision_5_r0/measured_trace.json').read_text(encoding='utf-8'))
        tasks=[(byid[e['old']['memory_id']],qmap[e['query_id']]) for e in trace]
        with driver.session(database='neo4j') as s:
            indexes=json.loads(json.dumps([r.data() for r in s.run('SHOW INDEXES')],default=str))
            count=s.run('MATCH (m:LWV2NativeMemory) RETURN count(m) AS n').single()['n']
        assert count==100000,count
        write_json(out/'protocol.json',{'neo4j_python_driver':neo4j.__version__,'python':sys.version,
            'container':NAME,'image':info['Image'],'resources':{k:info['HostConfig'][k] for k in ('Memory','MemorySwap','NanoCpus')},
            'database_memories':count,'scope_count':len(grouped),'queries':{'stream':QUERY,'collect':COLLECT},
            'scope_distribution':'cycle original revision_5_r0 update trace scopes/queries',
            'micro_workers':[1,32],'micro_queries_per_point':192,'micro_reps':3,
            'read_control_workers':32,'read_control_rate':95,'read_control_seconds':8,'read_control_reps':3,
            'read_control_arms':['stream_audit','stream_light','collect_audit','collect_light'],
            'boundary':'read-only diagnostic; no updates, no freshness inference; extra audit disabled only in light arms',
            'timing':'pool acquire is nested in run; server counters overlap client elapsed; fetch combines server streaming/network/decoding; thread CPU is not pure decode',
            'warmup':'171 scopes queried via both forms; each point 32 warmup reads; no cold-cache claim',
            'sources':{str(p):sha(p) for p in (Path(__file__),ROOT/'src/cassmem/backend/live_cells_graph.py',HERE.parent/'formal_trial.py')}})
        write_json(out/'indexes.json',indexes)
        expected={s:sorted(r.memory_id for r in rs) for s,rs in grouped.items()}
        gates=[]
        for scope in sorted(grouped):
            a,ar=timed_ids(driver,scope,'stream'); b,br=timed_ids(driver,scope,'collect')
            gates.append({'scope':scope,'n':len(a),'exact_match':a==b==expected[scope]})
        write_json(out/'candidate_gates.json',gates)
        assert all(r['exact_match'] for r in gates)
        profiles=[]
        for scope in [min(grouped,key=lambda s:len(grouped[s])),max(grouped,key=lambda s:len(grouped[s]))]:
            for variant,query in [('stream',QUERY),('collect',COLLECT)]:
                with driver.session(database='neo4j') as s:
                    r=s.run('PROFILE '+query,s=scope);list(r);summary=r.consume()
                profiles.append({'scope':scope,'size':len(grouped[scope]),'variant':variant,'profile':summary.profile})
        write_json(out/'profiles.json',profiles)
        summaries=[]
        for workers in (1,32):
            for rep in range(3):
                for variant in (('stream','collect') if rep%2==0 else ('collect','stream')):
                    label=f'micro_{workers}_{variant}_r{rep}'
                    write_json(out/'progress.json',{'status':'RUNNING','point':label})
                    def micro(i):
                        scope=tasks[i%len(tasks)][0].scope_id
                        ids,row=timed_ids(driver,scope,variant)
                        row['exact_ids']=ids==expected[scope]
                        return row
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        list(pool.map(micro,range(32)))
                        began=now();observations=list(pool.map(micro,range(192)));elapsed=(now()-began)/1e9
                    with (out/(label+'.jsonl')).open('w',encoding='utf-8') as h:
                        for row in observations:h.write(json.dumps(row)+'\n')
                    summary={'point':label,'seconds':elapsed,'throughput':192/elapsed,'id_failures':sum(not r['exact_ids'] for r in observations)}
                    for k in ('candidate_ms','run_ms','fetch_ms','acquire_ms','client_thread_cpu_ms','server_available_ms','server_consumed_ms','extract_sort_ms'):
                        summary[k]=stats([r[k] for r in observations if r[k] is not None])
                    summaries.append(summary);write_json(out/'micro_summary.json',summaries)
                    print(label,round(elapsed,2),flush=True)
        reference={}
        for target,q in tasks:
            key=(target.scope_id,q['query_id'])
            if key not in reference:reference[key]=[m for m,_ in index.search(target.scope_id,q['question'],qv[q['query_id']],expected[target.scope_id]).top10]
        def read(i,scheduled,emitted,submitted,variant,audit):
            start=now(); target,q=tasks[i%len(tasks)];s=target.scope_id;mid=target.memory_id
            ids,detail=timed_ids(driver,s,variant)
            with index._scope_locks[s]:
                sp,de=index._sparse[s],index._dense[s]
                # Keep original full sidecar snapshot capture in both read arms.
                spv={m:index._sparse_versions[(s,m)] for m in sp.memory_ids}
                dev={m:index._dense_versions[(s,m)] for m in de.memory_ids}
            ranking,_,_=score(sp,de,q['question'],qv[q['query_id']],ids);scored=now()
            ok=None
            if audit:
                graph=backend.fetch_graph('neo4j-native',s,mid)
                projection=backend.candidate_projection('neo4j-native',s,mid)
                rels={e['relation'] for e in target.edges}
                memberships={rel:mid in backend.ids_by_relation('neo4j-native',s,rel) for rel in sorted(rels)}
                doc=sp.documents[sp.memory_ids.index(mid)]
                vector=de.normalized_embeddings[de.memory_ids.index(mid)]
                tv=mv[mid.split('::')[-1]];tv=tv/max(float(np.linalg.norm(tv)),1e-12)
                ok=bool(graph and digest(graph)==digest(target.graph_projection()) and projection==target.rawerk and
                    doc==target.rawerk and np.allclose(vector,tv,rtol=1e-6,atol=1e-7) and all(memberships.values()) and
                    spv[mid]==target.version and dev[mid]==target.version)
            end=now()
            return dict(detail,ordinal=i,scheduled_ns=scheduled,emitted_ns=emitted,submit_ns=submitted,start_ns=start,end_ns=end,
                queue_ms=(start-submitted)/1e6,submission_lag_ms=(submitted-scheduled)/1e6,
                active_ms=(end-start)/1e6,total_ms=(end-submitted)/1e6,audit_ms=(end-scored)/1e6,
                candidate_ok=ids==expected[s],top10_ok=[m for m,_ in ranking]==reference[(s,q['query_id'])],audit_ok=ok)
        arms=[('stream',True),('stream',False),('collect',True),('collect',False)];results=[]
        for rep in range(3):
            order=arms[rep:]+arms[:rep]
            for variant,audit in order:
                label=f'reads_{variant}_{"audit" if audit else "light"}_r{rep}'
                write_json(out/'progress.json',{'status':'RUNNING','point':label})
                with ThreadPoolExecutor(max_workers=32) as pool:
                    warm=[pool.submit(read,i,now(),now(),now(),variant,audit) for i in range(32)]
                    for f in warm:f.result()
                    start=now()+2_000_000_000
                    proc=subprocess.Popen([sys.executable,'-B',str(HERE.parents[1]/'representation_serving_cost/arrival_emitter.py'),
                        '--start-ns',str(start),'--count','760','--ops-rate','95','--log',str(out/(label+'_emitter.jsonl'))],
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
                    pending=[]
                    try:
                        for line in proc.stdout:
                            ordinal,scheduled,emitted=map(int,line.strip().split(','));submitted=now()
                            pending.append(pool.submit(read,ordinal,scheduled,emitted,submitted,variant,audit))
                        if proc.wait(timeout=20):raise RuntimeError(proc.stderr.read())
                        observations=[f.result(timeout=120) for f in pending]
                    finally:
                        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
                with (out/(label+'.jsonl')).open('w',encoding='utf-8') as h:
                    for row in observations:h.write(json.dumps(row)+'\n')
                summary={'point':label,'n':len(observations),'seconds_to_drain':(max(r['end_ns'] for r in observations)-start)/1e9,
                    'candidate_failures':sum(not r['candidate_ok'] for r in observations),
                    'top10_failures':sum(not r['top10_ok'] for r in observations),
                    'audit_failures':sum(r['audit_ok'] is False for r in observations),
                    'queued_at_injection_end':sum(r['submit_ns']<=start+8e9<r['start_ns'] for r in observations)}
                for k in ('candidate_ms','run_ms','fetch_ms','acquire_ms','client_thread_cpu_ms','server_available_ms','server_consumed_ms',
                          'queue_ms','submission_lag_ms','active_ms','total_ms','audit_ms'):
                    summary[k]=stats([r[k] for r in observations if r[k] is not None])
                summary['load_valid']=bool(len(observations)==760 and summary['submission_lag_ms']['p95']<=50 and max(r['submit_ns'] for r in observations)<=start+9e9)
                results.append(summary);write_json(out/'read_summary.json',results)
                print(label,round(summary['seconds_to_drain'],2),'valid',summary['load_valid'],flush=True)
        write_json(out/'progress.json',{'status':'COMPLETE','micro_points':len(summaries),'read_points':len(results)})
    except BaseException as exc:
        write_json(out/'error.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        if driver:driver.close()
        stopped=subprocess.run(['docker','stop','--time','60',NAME],capture_output=True,text=True)
        write_json(out/'restoration.json',{'container':NAME,'initial_state':'stopped','stop_returncode':stopped.returncode})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
