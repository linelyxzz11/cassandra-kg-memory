"""Freeze inputs and schedules before seeing any ordinary-load outcomes."""
import platform
import random
import shutil
import importlib.metadata
import support as S

def schedules(initial,source,queries):
    src_by_scope=S.collections.defaultdict(list)
    qs_by_scope=S.collections.defaultdict(list)
    by_scope=S.collections.defaultdict(list)
    for r in source:src_by_scope[r.scope_id].append(r)
    for q in queries:qs_by_scope[q['conversation_id']].append(q)
    for r in initial:by_scope[r.scope_id].append(r)
    all_traces={}
    for seed in (20261015,20261016,20261017):
        rng=random.Random(seed)
        selected_scopes=rng.sample(sorted(by_scope),16)
        fixed=[rng.choice(by_scope[s]) for s in selected_scopes]
        unique=rng.sample(initial,180)
        order=list(range(16));rng.shuffle(order)
        targets={
            'disjoint':unique,
            'repeated':[fixed[order[(i//4)%16]] for i in range(180)],
            'hotspot':[fixed[0] if i%5!=4 else fixed[1+(i//5)%15] for i in range(180)],
        }
        for pattern,rs in targets.items():
            prng=random.Random(seed+{'disjoint':0,'repeated':100,'hotspot':200}[pattern])
            versions=S.collections.Counter()
            last_source={}
            trace=[]
            for i,r in enumerate(rs):
                s=r.scope_id.split('::')[-1]
                prev=last_source.get(r.memory_id,r.memory_id.split('::')[-1])
                candidates=[x for x in src_by_scope[s] if x.memory_id!=prev and x.raw_text.strip()]
                new=prng.choice(candidates);q=prng.choice(qs_by_scope[s])
                versions[r.memory_id]+=1
                trace.append(dict(update_index=i,phase='warmup' if i<60 else 'measured',scope=r.scope_id,
                                  memory_id=r.memory_id,version=versions[r.memory_id]+1,
                                  source_memory_id=new.memory_id,query_id=q['query_id']))
                last_source[r.memory_id]=new.memory_id
            all_traces[f'{pattern}_{seed}']=trace
    return all_traces

def main():
    if S.RUN.exists():raise RuntimeError('Existing run directory; refuse overwrite')
    env=S.environment_gate()
    source,initial,mv,qv,queries=S.data()
    traces=schedules(initial,source,queries)
    matrix=[]
    for si,seed in enumerate((20261015,20261016,20261017)):
        for pi,pattern in enumerate(('disjoint','repeated','hotspot')):
            for arm in (('uncoordinated','serialized') if (si+pi)%2==0 else ('serialized','uncoordinated')):
                matrix.append(dict(id=f'{len(matrix)+1:02d}_{pattern}_{seed}_{arm}',seed=seed,pattern=pattern,arm=arm,trace=f'{pattern}_{seed}'))
    protocol=dict(
        experiment_id='ordinary-concurrency-20261021-v1',question='Cost and safety of existing single-process per-key update serialization under ordinary concurrent traffic, without injected stalls.',
        container=env,cell=S.CELL,keyspace='oc20261021_v1',memories=100000,scopes=171,
        matrix=dict(points=18,arms=['uncoordinated','serialized'],patterns=['disjoint','repeated','hotspot'],seeds=[20261015,20261016,20261017]),
        traffic=dict(update_rate_per_s=1,read_rate_per_s=19,workers=32,warmup_s=60,injection_s=120,drain_s=120,
                     spacing='Uniform 50 ms operation spacing. One update followed by 19 reads. No injected write delay or barrier.',
                     disjoint='180 distinct keys, sampled uniformly without replacement from the expanded corpus.',
                     repeated='16 fixed keys in 16 sampled scopes; blocks of four consecutive revisions, rotating through a seeded key permutation.',
                     hotspot='80 percent of revisions to one key, remaining 20 percent rotating over the other 15 fixed keys.',
                     query='One frozen same-conversation query per update, repeated for the following 19 reads; not an independent QA benchmark.',
                     revisions='Replace raw/ERK/graph/vector payload with an existing frozen source record from the same conversation; increment target version. Warmup state retained into measurement.'),
        mechanisms=dict(serialized='Unmodified SerializedUpdateService: per-key lock, version admission, exact canonical/ready checks. Single process only.',
                        uncoordinated='E3 raw->structured->canonical check->sparse->dense->ready pipeline; same check routines and existing per-channel version guards; no per-key lock/admission.',
                        scope_locks='Existing copy-on-write per-scope index locks retained in both arms; different memory IDs may still contend on a scope.',
                        treatment='Existing coordination policy as a whole, including admission/rejection behavior; not a pure threading.Lock microbenchmark.'),
        measurement=dict(clock='Same-host perf_counter_ns; planned, emitted, independent receiver enqueue, executor dispatch, worker start retained.',
                         ready='Exact target version check succeeds at that observation. It is not a global publication barrier or a guarantee against future updates.',
                         read_capture='No ordinary-read backend graph/relations audit. Immutable sparse/dense snapshot versions and target text/vector fingerprints logged; both arms identical.',
                         update_capture='Stage hooks and ready outcome retained. No added post-ready read in timed update path.',
                         final_validation='After all requests finish, full per-key raw/graph/projection/relation/sparse/dense candidate check against highest offered version. All ever-used relation keys checked.',
                         overlap='Report same-key service intervals and raw-through-dense mutation intervals separately; queued requests do not establish simultaneous mutations.',
                         supersession='Ready/stale admission/canonical-guard rejection/ready-guard rejection/infrastructure error are distinct; never count a rejected revision as ready.',
                         latency='arrival-to-ready percentiles among confirmed-ready updates plus denominator and deadline completion; queue/wait/write/query/response reported separately.',
                         inference='No conflict under this fixed offered load does not establish race freedom. Read snapshot checks are not atomic backend snapshots.'),
        validity=dict(receive_lag_p95_ms_max=50,last_receive_slack_s=1,all_event_ids=True,source_hashes=True,only_selected_container=True,
                      correctness_failures='Keep as measured outcomes, not an excuse to omit the point. Infrastructure/log corruption stops progression.'),
        boundaries=dict(single_machine=True,precomputed_extraction_embedding=True,api_calls=0,top_branch_depth=50,alpha=.6,top_k=10,
                        no_scope_global_statistics=True,no_os_cache_flush=True,baseline_is_reconciled_v1_each_point=True,
                        evidence='Do not pool timing with E0/E3: this is a new ordinary-load protocol.'),
        smoke=dict(points=6,updates_each=8,duration_s=8,drain_s=2,memories=100000,excluded_from_formal=True),
        failure_policy='Preserve every attempted point; no implicit retry or overwrite. Interrupted point needs an explicit new attempt. Completed point files are immutable.',
        reset_policy='Only new experiment keyspace; restore touched per-memory structures and every known relation key to v1 between points. Never truncate old databases.',
    )
    S.RUN.mkdir()
    S.save(S.RUN/'protocol.json',protocol);S.save(S.RUN/'formal_matrix.json',matrix);S.save(S.RUN/'schedules.json',traces)
    oracle={}
    for key,trace in traces.items():
        records=S.resolve(trace,initial,source,mv)
        oracle[key]={mid:{str(v):S.expected_payload(r,vec) for v,(r,vec) in versions.items()} for (_,mid),versions in records.items()}
    S.save(S.RUN/'oracle.json',oracle)
    S.save(S.RUN/'initial_manifest.json',dict(memories=len(initial),scopes=len({r.scope_id for r in initial}),
            records_sha256=S.hashlib.sha256(''.join(S.digest(r.graph_projection()) for r in initial).encode()).hexdigest()))
    S.save(S.RUN/'environment.json',dict(**env,python=S.sys.version,python_executable=S.sys.executable,platform=platform.platform(),
            packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn','cassandra-driver','psutil')},
            docker_server=S.docker('version','--format','{{json .Server}}')))
    paths=[*S.HERE.glob('*.py'),S.RECEIVER,S.Path('D:/memorytable/representation_serving_cost/arrival_emitter.py'),
           S.Path('D:/memorytable/representation_serving_cost/common.py'),S.Path('D:/memorytable/cassmem_e0_preflight_20261015/instrumentation.py'),
           *[S.ROOT/x for x in ('src/cassmem/serving/serialized_updates.py','src/cassmem/backend/live_cells_graph.py',
            'src/cassmem/retrieval/online.py','src/cassmem/representation/graph_event.py','src/cassmem/serving/environment.py',
            'experiments/serving_100k/run_graph_100k_load_gate_v2.py','experiments/serving_100k/run_graph_canonical_gate_v2.py',
            'data/locomo_memory_records.csv','data/frozen_retrieval/p3_memory_features.csv','data/frozen_retrieval/p3_queries.csv',
            'data/locomo_memory_bge_large.npy','data/locomo_qa_bge_large.npy','data/locomo_memory_ids_bge.txt','data/locomo_qa_ids_bge.txt')],
           *S.RUN.glob('*.json')]
    S.save(S.RUN/'source_manifest.json',{str(p):S.sha(p) for p in paths})
    snapshot=S.RUN/'code_snapshot';snapshot.mkdir()
    for i,p in enumerate(paths):
        if p.suffix=='.py':shutil.copy2(p,snapshot/f'{i:02d}_{p.name}')
    print(S.RUN,flush=True)

if __name__=='__main__':main()
