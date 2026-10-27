"""Sequential, resource-matched campaign with original-container restoration."""
import argparse,json,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from runtime import HERE,capture,switch,restore
from common import write_json,sha,ROOT
from formal_trial_v3 import data,get_backend

CELLS=['cassandra-base','cassandra-materialized','neo4j-native']

def load(cell,keyspace,n,out):
    records,grouped,_,_,_=data(n);b=get_backend(cell,keyspace)
    try:
        counts={s:len(b.ids(cell,s)) for s in grouped}
        if any(counts.values()):
            raise RuntimeError('Refusing to load a nonempty campaign scope')
        began=time.perf_counter()
        def scope_load(recs):
            for r in recs: b.insert(cell,r)
            return len(recs)
        if cell.startswith('neo4j'):
            # Different scopes per worker avoid shared-entity transaction conflicts.
            total=0
            with ThreadPoolExecutor(max_workers=8) as pool:
                for count in pool.map(scope_load,grouped.values()):
                    total+=count;print(f'LOAD {cell} {total}/{len(records)}',flush=True)
        else:
            with ThreadPoolExecutor(max_workers=16) as pool:
                for offset in range(0,len(records),256):
                    list(pool.map(lambda r:b.insert(cell,r),records[offset:offset+256]))
                    if offset%5120==0: print(f'LOAD {cell} {offset}/{len(records)}',flush=True)
        actual=sum(len(b.ids(cell,s)) for s in grouped)
        assert actual==len(records)
        write_json(out/f'load_{cell}.json',{'records':actual,'seconds':time.perf_counter()-began})
    finally:b.close()

def pair_check(out):
    pairs=[]
    for mode in ('insert','revision'):
        for rate in (1,5):
            for rep in range(3):
                found={cell:out/f'{cell}_{mode}_{rate}_r{rep}'/'checkpoint.json' for cell in CELLS}
                if not all(p.exists() for p in found.values()):continue
                rows={cell:json.loads(p.read_text(encoding='utf-8')) for cell,p in found.items()}
                maps={cell:{r['scope']:r for r in rr} for cell,rr in rows.items()}
                for scope in maps[CELLS[0]]:
                    rs=[maps[c][scope] for c in CELLS]
                    same=len({r['logical_state_hash'] for r in rs})==1
                    pairs.append({'mode':mode,'rate':rate,'rep':rep,'scope':scope,'same_logical_state':same,
                      'ordered_top10_match':same and all(r['actual_top10']==rs[0]['actual_top10'] for r in rs),
                      'all_candidate_oracle_checks':all(r['candidate_match'] for r in rs)})
    write_json(out/'cross_backend_checkpoints.json',pairs)

def run(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    state=capture();write_json(out/'environment.json',state)
    keyspace='repcost_consistency_'+out.name.replace('-','_')
    sources={str(p):sha(p) for p in (ROOT/'src/cassmem/backend/live_cells_graph.py',HERE/'formal_trial_v3.py',HERE/'run_matrix_v3.py',HERE/'neo4j_collected_v3.py',HERE/'runtime.py')}
    write_json(out/'protocol.json',{'cells':CELLS,'mode':'smoke' if a.smoke else 'formal','memories':300 if a.smoke else 100000,
        'warmup_seconds':0 if a.smoke else 60,'injection_seconds':3 if a.smoke else 120,'drain_seconds':120,
        'rates':[1] if a.smoke else [1,5],'repetitions':1 if a.smoke else 3,'read_update_ratio':'19:1','workers':32,
        'client_switch_interval_seconds':0.0001,'neo4j_candidate_return':'collect IDs; exact scope parity validated per point',
        'cost_boundary':'prebuilt extraction/vectors; includes diagnostic target read checks; not a throughput comparison',
        'load_validity':'all requests; submission lag p95 <=50ms; final submission within planned end+1s',
        'sources':sources,'keyspace':keyspace})
    progress={'status':'STARTING','completed':[],'planned':6 if a.smoke else 36}
    write_json(out/'progress.json',progress)
    try:
        current=None
        for cell in CELLS:
            engine='cassandra' if cell.startswith('cassandra') else 'neo4j'
            if engine!=current:
                progress.update(status='SWITCHING',cell=cell);write_json(out/'progress.json',progress)
                state[engine]=switch(engine);write_json(out/'environment.json',state);current=engine
            progress.update(status='LOADING',cell=cell);write_json(out/'progress.json',progress)
            load(cell,keyspace,300 if a.smoke else 100000,out)
            for rep in range(1 if a.smoke else 3):
                # Rotate workload/rate order across repetitions; all receive same seeded trace.
                points=[(mode,rate) for mode in ('insert','revision') for rate in ([1] if a.smoke else [1,5])]
                points=points[rep:]+points[:rep]
                for mode,rate in points:
                    for path,h in sources.items():
                        if sha(Path(path))!=h:raise RuntimeError('Source changed during campaign: '+path)
                    name=f'{cell}_{mode}_{rate}_r{rep}'
                    progress.update(status='RUNNING',point=name);write_json(out/'progress.json',progress)
                    command=[sys.executable,'-B',str(HERE/'formal_trial_v3.py'),'--cell',cell,'--mode',mode,'--rate',str(rate),
                        '--rep',str(rep),'--memories',str(300 if a.smoke else 100000),'--keyspace',keyspace,
                        '--warmup',str(0 if a.smoke else 60),'--seconds',str(3 if a.smoke else 120),'--output',str(out/name)]
                    print('RUN',name,flush=True)
                    with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                        p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                    if p.returncode:raise RuntimeError(f'{name} failed; inspect its log')
                    result=json.loads((out/name/'summary.json').read_text(encoding='utf-8'))
                    progress['completed'].append({'point':name,'status':result['status'],'errors':result['errors'],
                        'post_ready_target_failures':result['post_ready_target_failures'],'load_valid':result['offered_load_valid'],
                        'enqueue_return_lag_p95_ms':result['enqueue_return_lag_p95_ms']})
                    write_json(out/'progress.json',progress)
                    print('DONE',name,result['status'],flush=True)
                    if a.smoke and (result['errors'] or result['post_ready_target_failures'] or not result['checkpoint_pass']):
                        raise RuntimeError('Smoke semantic gate failed')
        pair_check(out)
        progress.update(status='COMPLETE');write_json(out/'progress.json',progress)
    except BaseException as exc:
        progress.update(status='FAILED',error=repr(exc));write_json(out/'progress.json',progress);raise
    finally:
        restore(state);write_json(out/'environment.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
