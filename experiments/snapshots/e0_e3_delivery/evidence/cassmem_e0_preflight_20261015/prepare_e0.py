import sys,json,hashlib,dataclasses,random,collections,platform,subprocess,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=Path('D:/memorytable/cassandra-kg-memory')
sys.path.insert(0,'D:/memorytable/representation_serving_cost')
from common import build_graph_records,vectors,rows,np
from run_graph_100k_load_gate_v2 import expand
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    source=build_graph_records();expanded=expand(source);mv=vectors('memory');qv=vectors('qa');queries=rows(ROOT/'data/frozen_retrieval/p3_queries.csv')
    group=collections.defaultdict(list);qgroup=collections.defaultdict(list)
    for r in source:group[r.scope_id].append(r)
    for q in queries:qgroup[q['conversation_id']].append(q)
    seeds=[20261015,20261016,20261017];traces={}
    for rate in (.25,.5,1.):
        for seed in seeds:
            rng=random.Random(seed);selected=rng.sample(expanded,int(180*rate));events=[]
            for i,old in enumerate(selected):
                base_scope=old.scope_id.split('::')[-1];new=rng.choice([r for r in group[base_scope] if r.raw_text!=old.raw_text]);q=rng.choice(qgroup[base_scope])
                events.append(dict(phase='warmup' if i<int(60*rate) else 'measured',memory_id=old.memory_id,scope=old.scope_id,source_memory_id=new.memory_id,version=2,query_id=q['query_id']))
            traces[f'{rate}_{seed}']=events
    fixtures=json.loads(Path('D:/memorytable/cassmem_e3_preflight_20261015/smoke_f0978e6a51a6/fixtures.json').read_text(encoding='utf-8'))[:10]
    for f in fixtures:
        q=qgroup[f['source_scope']][0];f['query']=dict(query_id=q['query_id'],question=q['question'],vector=qv[q['query_id']].tolist())
    cells=['cassandra-base','cassandra-materialized','neo4j-native','neo4j-materialized']
    p=dict(experiment_id='E0-serialized-update-preflight-20261015',cells=cells,
        formal=dict(memories=100000,rates=[.25,.5,1.],seeds=seeds,repeats=3,points=36,warmup_s=60,injection_s=120,drain_s=120,offered_updates=2520,offered_reads=47880),
        smoke=dict(memories=10,rate=.5,seed=20261015,warmup_s=4,injection_s=12,drain_s=10,updates_per_cell=6,reads_per_cell=114),
        shared_workers=32,reads_per_update=19,update_mode='unique existing-ID revisions to version 2; warmup targets disjoint from measured targets',
        formal_trace='Same frozen trace for all four layouts; reads attached to each update use its frozen query_id, repeated 19 times. Independent-process uniform spacing across update plus 19 reads.',
        initial_state='All records version 1. Fresh/reconciled per-point backend/index. Warmup state retained; warmup targets never reused in measured injection.',
        resources=dict(cpus=8,memory_gib=12,cassandra_heap='6G',neo4j_heap='5G',neo4j_pagecache='3G',single_node=True,engines_run_sequentially=True),
        validity=dict(submission_lag_p95_ms_max=50,last_enqueue_slack_s=1,execution_errors_max=0,require_all_offered=True),
        deadlines='Injection planned start + injection duration + drain duration. No reset at late last enqueue. All offered updates denominator. Late completions separately retained.',
        clocks='Monotonic perf_counter_ns; application arrival=actual enqueue; planned-arrival latency separate. Backend call observed client-side, not DB internal commit.',
        caches='No OS cache flush; verify loaded state then fixed warmup. Warmed service; explicit cold-cache comparison not included.',
        batching='Cassandra one existing logged structured batch per event; Neo4j one existing auto-commit statement per operation. No cross-event transaction batching.',
        connection_policy='Existing Cassandra native v4 driver connections/multiplexing, LOCAL_ONE RF1; Neo4j max pool128. No extra application connection semaphore.',
        unavailable_metrics={'driver_pool_wait':'not separately exposed; included in backend call, null not zero','relation_cleanup_only':'integrated in structured write; report full structured maintenance; isolated cleanup null'},
        observability='Thread-local original scope lock timing; per-ID service hooks; canonical and index/relations ready checks; GC and host/client/container CPU/RAM samples.',
        no_formal_launch=True,fixture_sha=hashlib.sha256(json.dumps(fixtures,sort_keys=True).encode()).hexdigest(),
        code_hashes={str(x):sha(x) for x in HERE.glob('*.py')})
    out=HERE/('e0_'+hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:12]);out.mkdir(exist_ok=False)
    save(out/'protocol.json',p);save(out/'fixtures.json',fixtures);save(out/'formal_update_traces.json',traces)
    save(out/'formal_matrix.json',[dict(layout=cell,rate=rate,seed=seed,trace=f'{rate}_{seed}') for cell in cells for rate in (.25,.5,1.) for seed in seeds])
    paths=[*HERE.glob('*.py'),ROOT/'src/cassmem/serving/serialized_updates.py',ROOT/'src/cassmem/backend/live_cells_graph.py',ROOT/'src/cassmem/retrieval/online.py',ROOT/'src/cassmem/representation/graph_event.py',
       ROOT/'experiments/serving_100k/run_graph_100k_load_gate_v2.py',ROOT/'experiments/serving_100k/run_graph_canonical_gate_v2.py',Path('D:/memorytable/representation_serving_cost/arrival_emitter.py'),
       ROOT/'data/locomo_memory_records.csv',ROOT/'data/frozen_retrieval/p3_memory_features.csv',ROOT/'data/frozen_retrieval/p3_queries.csv',ROOT/'data/locomo_memory_bge_large.npy',ROOT/'data/locomo_qa_bge_large.npy',ROOT/'data/locomo_memory_ids_bge.txt',ROOT/'data/locomo_qa_ids_bge.txt',
       out/'protocol.json',out/'fixtures.json',out/'formal_update_traces.json',out/'formal_matrix.json']
    save(out/'source_manifest.json',{str(x):sha(x) for x in paths});(out/'code_snapshot').mkdir()
    for x in HERE.glob('*.py'):shutil.copy2(x,out/'code_snapshot'/x.name)
    import psutil
    save(out/'environment.json',dict(python=sys.version,os=platform.platform(),cpu=platform.processor(),logical_cpus=psutil.cpu_count(),ram=psutil.virtual_memory().total,
       pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),git_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
       docker_version=subprocess.check_output(['docker','version','--format','{{json .Server}}'],text=True)))
    print(out)
if __name__=='__main__':main()
