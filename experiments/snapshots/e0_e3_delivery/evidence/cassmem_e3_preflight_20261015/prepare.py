"""Freeze fixtures without queries, gold, API calls or production modifications."""
import os,sys,json,hashlib,random,dataclasses,subprocess,platform,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('D:/memorytable/cassandra-kg-memory')
sys.path.insert(0,'D:/memorytable/representation_serving_cost')
from common import build_graph_records,vectors,np,sha,write_json

def main():
    records=build_graph_records(); mv=vectors('memory'); rng=random.Random(20261015)
    eligible=[r for r in records if r.edges]; rng.shuffle(eligible); fixtures=[]
    for old in eligible:
        rel=lambda r:{e['relation'] for e in r.edges}
        choices=[r for r in records if r.scope_id==old.scope_id and r.raw_text!=old.raw_text and rel(old)-rel(r) and rel(r)-rel(old)]
        rng.shuffle(choices)
        pair=next(((x,y) for x in choices for y in choices if x.raw_text!=y.raw_text and rel(x)-rel(y) and rel(y)-rel(x)),None)
        if pair is None:continue
        versions=[]
        for v,source in enumerate((old,*pair),1):
            r=dataclasses.replace(source,memory_id=old.memory_id,version=v)
            vec=np.asarray(mv[source.memory_id],dtype=np.float32)
            matrix=vec[None,:];norm=(matrix/np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-12))[0]
            assert hashlib.sha256(vec.tobytes()).hexdigest()==r.embedding_sha256
            versions.append(dict(record=dataclasses.asdict(r),graph=r.graph_projection(),rawerk=r.rawerk,
                vector=vec.tolist(),normalized_vector_sha256=hashlib.sha256(norm.tobytes()).hexdigest(),source_memory_id=source.memory_id,
                required_relations=sorted(rel(r))))
        fixtures.append(dict(fixture_id=len(fixtures),memory_id=old.memory_id,source_scope=old.scope_id,versions=versions))
        if len(fixtures)==50:break
    assert len(fixtures)==50 and len({f['memory_id'] for f in fixtures})==50
    protocol=dict(experiment='E3-original-dynamic-v2-preflight',seed=20261015,fixture_count=50,smoke_fixture_ids=list(range(5)),
        cells=['cassandra-base','cassandra-materialized','neo4j-native'],
        scenarios={'control':None,'v2_sparse_delayed':'sparse','v2_dense_delayed':'dense',
          'v2_relation_cleanup_delayed':'structured','v2_all_maintenance_finishes_after_v3':'structured'},
        alias_boundary='Structured cleanup and publication are one existing API; relation and all-maintenance scenarios share the before-structured barrier. Original sequential pipeline and canonical check are preserved, including aborts.',
        fixture_semantics='Real frozen memory contents reassigned as deterministic revision payloads; not claims of natural temporal revisions. No QA/gold selection.',
        schedule='Submit v2, wait at selected hook; submit and complete v3 and strict readiness probe; release v2; drain and final probes. Control completes v2 before v3.',
        reader_interval_ms=20,barrier_timeout_s=60,worker_timeout_s=120,post_final_probes=3,
        resources={'container_memory':'3g','cpus':2,'heap':'1G','neo4j_pagecache':'512M'},
        metric_boundary='Correctness smoke only; sampled cross-component observations are not atomic snapshots. Query latency and extra audit latency reported separately, not serving capacity.',
        semantics='Expected target is latest submitted at read start. Final-ready means v3 strict check passed, even if delayed old work remains. Unknown component versions remain unknown.',
        baseline='Existing backend + ScopedOnlineRetrievalIndex unmodified; stale sparse/dense versions already rejected; no new fencing.',
        fixture_digest=hashlib.sha256(json.dumps(fixtures,sort_keys=True).encode()).hexdigest())
    protocol['code_hashes']={p.name:sha(p) for p in HERE.glob('*.py')}
    ph=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest(); out=HERE/('smoke_'+ph[:12])
    out.mkdir(exist_ok=False);write_json(out/'protocol.json',protocol);write_json(out/'fixtures.json',fixtures)
    (out/'code_snapshot').mkdir()
    for script in HERE.glob('*.py'):shutil.copy2(script,out/'code_snapshot'/script.name)
    write_json(out/'schedules.json',[dict(case_id=f'{cell}__{scenario}__{i}',cell=cell,scenario=scenario,fixture_id=i,blocked_stage=stage)
        for cell in protocol['cells'] for scenario,stage in protocol['scenarios'].items() for i in protocol['smoke_fixture_ids']])
    paths=[ROOT/'src/cassmem/backend/live_cells_graph.py',ROOT/'src/cassmem/retrieval/online.py',ROOT/'src/cassmem/representation/graph_event.py',
       ROOT/'experiments/serving_100k/run_graph_canonical_gate_v2.py',Path('D:/memorytable/concurrent_update_consistency/formal_trial.py'),Path('D:/memorytable/concurrent_update_consistency/preflight.py'),
       ROOT/'data/locomo_memory_records.csv',ROOT/'data/frozen_retrieval/p3_memory_features.csv',ROOT/'data/locomo_memory_ids_bge.txt',ROOT/'data/locomo_memory_bge_large.npy',
       *sorted(HERE.glob('*.py')),out/'protocol.json',out/'fixtures.json',out/'schedules.json']
    write_json(out/'source_manifest.json',{str(p):sha(p) for p in paths})
    import psutil
    env=dict(python=sys.version,executable=sys.executable,os=platform.platform(),cpu=platform.processor(),logical_cpus=os.cpu_count(),ram=psutil.virtual_memory().total,
        git_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True),
        pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),
        docker_info=json.loads(subprocess.check_output(['docker','info','--format','{{json .}}'],text=True)))
    # Docker info contains no container Env or credentials. Keep only operational fields.
    env['docker_info']={k:env['docker_info'].get(k) for k in ('ServerVersion','OSType','OperatingSystem','NCPU','MemTotal','Driver')}
    write_json(out/'environment.json',env);print(out)
if __name__=='__main__':main()
