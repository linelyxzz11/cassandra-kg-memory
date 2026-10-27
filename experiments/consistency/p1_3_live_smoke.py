"""Original-adapter revision smoke, restricted to new P1-3 containers.

No resets/deletes, no schema or serving fixes. Every unexpected stale relation
is retained as an observation. This is a gate before the100K timed campaign.
"""
import argparse,os,json,subprocess,time,hashlib,sys
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments/serving_100k'))
from cassmem.backend.live_cells_graph import CassandraGraphCells,Neo4jGraphCells
from run_graph_canonical_gate_v2 import build_graph_records
OUT=ROOT/'results/consistency/p1_3_20261012'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--container',required=True)
    ap.add_argument('--cell',required=True,choices=['cassandra-base','cassandra-materialized','neo4j-native','neo4j-materialized'])
    a=ap.parse_args()
    assert a.container.startswith('cassmem-p1-3-20261012-'),'refuse non-P1-3 container'
    info=json.loads(subprocess.check_output(['docker','inspect',a.container],text=True))[0]
    assert info['State']['Running']
    engine='cassandra' if a.cell.startswith('cassandra') else 'neo4j'
    assert a.container.endswith(engine)
    private_port='9042/tcp' if engine=='cassandra' else '7687/tcp'
    mappings=info['NetworkSettings']['Ports'][private_port];assert len(mappings)>=1
    port=int(mappings[0]['HostPort'])
    suffix=str(time.time_ns());scope='p1_3_smoke_'+suffix;mid='target_'+suffix
    source=build_graph_records()[0]
    old=replace(source,scope_id=scope,memory_id=mid,entities='Alice, Paris, London',relations='lives_in, visited',keywords='travel',triples='(Alice, lives_in, Paris); (Alice, visited, London)',version=1)
    new=replace(old,entities='Alice, Rome',relations='lives_in',keywords='relocation',triples='(Alice, lives_in, Rome)',version=2)
    removed=replace(new,entities='Alice',relations='',keywords='profile',triples='',version=3)
    if engine=='cassandra':
        os.environ['CASSANDRA_PORT']=str(port)
        backend=CassandraGraphCells('127.0.0.1',keyspace='p1_3_smoke_20261012')
    else:
        auth=next((x.split('=',1)[1] for x in info['Config']['Env'] if x.startswith('NEO4J_AUTH=')),None)
        assert auth and '/' in auth,'private container auth unavailable'
        user,password=auth.split('/',1)
        backend=Neo4jGraphCells(f'bolt://127.0.0.1:{port}',user,password)
    observations=[]
    try:
        backend.insert(a.cell,old)
        for target in [old,new,removed]:
            if target.version>1:
                backend.commit_raw(a.cell,target);backend.write_structured(a.cell,target)
            graph=backend.fetch_graph(a.cell,scope,mid)
            projection=backend.candidate_projection(a.cell,scope,mid)
            actual={r:mid in backend.ids_by_relation(a.cell,scope,r) for r in ['lives_in','visited']}
            expected={r:any(e['relation']==r for e in target.edges) for r in actual}
            observations.append({'version':target.version,'actual_graph':graph,'expected_graph':target.graph_projection(),
              'canonical_exact':graph==target.graph_projection(),'rawerk_exact':projection==target.rawerk,
              'relation_membership':actual,'expected_membership':expected,
              'old_relation_remaining':[r for r in actual if actual[r] and not expected[r]],
              'new_relation_missing':[r for r in actual if expected[r] and not actual[r]]})
    finally:backend.close()
    folder=OUT/'live_smoke';folder.mkdir(parents=True,exist_ok=True)
    result={'phase':'original_adapter_smoke','cell':a.cell,'container':a.container,'isolated_scope':scope,
       'source_sha256':hashlib.sha256((ROOT/'src/cassmem/backend/live_cells_graph.py').read_bytes()).hexdigest(),
       'observations':observations,'correctness_failures':sum(bool(not r['canonical_exact'] or r['old_relation_remaining'] or r['new_relation_missing']) for r in observations),
       'scope_preserved_for_inspection':True,'production_fixes':False}
    (folder/f'{a.cell}_{suffix}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'cell':a.cell,'versions_checked':len(observations),'canonical_matches':sum(r['canonical_exact'] for r in observations),'relation_memberships':[{k:r[k] for k in ['version','old_relation_remaining','new_relation_missing']} for r in observations]},indent=2))
if __name__=='__main__':main()
