"""Real Neo4j before/after semantic preflight, private container only."""
import time,dataclasses,importlib.util
from runtime import HERE,capture,switch,restore,neo_backend
from common import write_json,build_graph_records,vectors,sha,ROOT
from preflight import observation
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex

def run(out):
    out.mkdir(parents=True,exist_ok=False)
    state=capture(); write_json(out/'environment.json',state)
    try:
        state['experiment']=switch('neo4j'); write_json(out/'environment.json',state)
        b=neo_backend()
        state['version']=[dict(r) for r in b.run('CALL dbms.components() YIELD name,versions RETURN name,versions')]
        state['memory_settings']=[dict(r) for r in b.run("SHOW SETTINGS YIELD name,value WHERE name IN ['server.memory.heap.initial_size','server.memory.heap.max_size','server.memory.pagecache.size'] RETURN name,value")]
        rows=build_graph_records(); mv=vectors('memory')
        first=next(r for r in rows if r.edges)
        second=next(r for r in rows if r.scope_id==first.scope_id and r.edges and {e['relation'] for e in first.edges}-{e['relation'] for e in r.edges} and r.raw_text!=first.raw_text)
        spec=importlib.util.spec_from_file_location('baseline_neo',HERE/'baseline_source/live_cells_graph_before_neo4j_cleanup.py')
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        baseline=mod.Neo4jGraphCells.__new__(mod.Neo4jGraphCells); baseline.d=b.d; baseline.database=b.database
        results=[]
        try:
            for label,adapter in [('baseline',baseline),('fixed',b)]:
                for cell in b.names:
                    for mode in ('insert','revision'):
                        for rep in range(3):
                            scope=f'{out.name}_{label}_{cell}_{mode}_{rep}'
                            old=dataclasses.replace(first,scope_id=scope,memory_id='target',version=1)
                            target=dataclasses.replace(second,scope_id=scope,memory_id='target',version=2)
                            bg=dataclasses.replace(first,scope_id=scope,memory_id='background')
                            adapter.insert(cell,bg)
                            docs={'background':bg.rawerk}; embeds={'background':mv[first.memory_id]}
                            if mode=='revision':
                                adapter.insert(cell,old); docs['target']=old.rawerk; embeds['target']=mv[first.memory_id]
                            idx=ScopedOnlineRetrievalIndex(); idx.load_scope(scope,docs,embeds)
                            adapter.commit_raw(cell,target); adapter.write_structured(cell,target)
                            idx.upsert_sparse(scope,'target',target.rawerk,2); idx.upsert_dense(scope,'target',mv[second.memory_id],2)
                            obs=observation(adapter,cell,old if mode=='revision' else None,target,idx,mv[second.memory_id])
                            results.append({'implementation':label,'cell':cell,'mode':mode,'rep':rep,**obs,
                                            'actual_graph':adapter.fetch_graph(cell,scope,'target'),'expected_graph':target.graph_projection()})
                            print(label,cell,mode,rep,obs['strict_ready'],flush=True)
            # Empty structure target regression, preserving raw/vector consistency.
            for cell in b.names:
                scope=out.name+'_empty_'+cell
                old=dataclasses.replace(first,scope_id=scope,memory_id='target',version=1)
                target=dataclasses.replace(old,version=2,entities='',relations='',keywords='',triples='')
                b.insert(cell,old); idx=ScopedOnlineRetrievalIndex(); idx.load_scope(scope,{'target':old.rawerk},{'target':mv[first.memory_id]})
                b.commit_raw(cell,target); b.write_structured(cell,target)
                idx.upsert_sparse(scope,'target',target.rawerk,2); idx.upsert_dense(scope,'target',mv[first.memory_id],2)
                obs=observation(b,cell,old,target,idx,mv[first.memory_id])
                results.append({'implementation':'fixed','cell':cell,'mode':'empty','rep':0,**obs})
        finally: b.close()
        fixed=[r for r in results if r['implementation']=='fixed']
        summary={'status':'PASS' if all(r['strict_ready'] for r in fixed) else 'FAIL','trials':results,
                 'fixed_passed':sum(r['strict_ready'] for r in fixed),'fixed_total':len(fixed),
                 'adapter_sha256':sha(ROOT/'src/cassmem/backend/live_cells_graph.py')}
        write_json(out/'summary.json',summary)
        assert summary['status']=='PASS'
    finally:
        restore(state); write_json(out/'environment.json',state)

if __name__=='__main__': run(HERE/'results'/('neo_preflight_'+time.strftime('%Y%m%d_%H%M%S')))
