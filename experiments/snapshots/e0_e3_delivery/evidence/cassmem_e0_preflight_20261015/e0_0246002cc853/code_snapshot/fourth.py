import sys,json,hashlib,shutil,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'D:/memorytable/cassmem_e3_serialized_20261015')
from live import SerializedCase
import importlib.util
spec=importlib.util.spec_from_file_location('e0_environment',HERE/'environment.py');env=importlib.util.module_from_spec(spec);spec.loader.exec_module(env)
BASE=Path('D:/memorytable/cassmem_e3_serialized_20261015/smoke_95adb3ae31e5')
def main():
    p=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'));p['cells']=['neo4j-materialized'];p['supplement']='fourth-layout-only, resource budget 8CPU/12GiB'
    p['resources']=dict(cpus=8,memory_gib=12,heap='5G',pagecache='3G')
    p['entry_hashes']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (Path(__file__),HERE/'environment.py')}
    out=HERE/('fourth_'+hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:12]);out.mkdir(exist_ok=False)
    env.save(out/'protocol.json',p);shutil.copy2(BASE/'fixtures.json',out/'fixtures.json')
    cases=[dict(case_id=f'neo4j-materialized__{sc}__{i}',cell='neo4j-materialized',scenario=sc,fixture_id=i,blocked_stage=stage) for sc,stage in p['scenarios'].items() for i in range(5)]
    env.save(out/'schedules.json',cases)
    paths=[Path('D:/memorytable/cassandra-kg-memory/src/cassmem/serving/serialized_updates.py'),Path('D:/memorytable/cassandra-kg-memory/src/cassmem/backend/live_cells_graph.py'),Path('D:/memorytable/cassandra-kg-memory/src/cassmem/retrieval/online.py'),Path('D:/memorytable/cassmem_e3_serialized_20261015/live.py'),Path('D:/memorytable/cassmem_e3_preflight_20261015/runner.py'),out/'fixtures.json',out/'protocol.json',out/'schedules.json']
    env.save(out/'source_manifest.json',{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths})
    state=env.suspend_old(out);b=None
    try:
        b,name=env.start('neo4j',out);fixtures=json.loads((out/'fixtures.json').read_text(encoding='utf-8'))
        for c in cases:SerializedCase(b,c,fixtures[c['fixture_id']],out,p).run();print('complete',c['case_id'],flush=True)
        env.save(out/'complete.json',dict(cases=45,validation_pending=True))
    except Exception:env.save(out/'failure.json',dict(error=traceback.format_exc()));raise
    finally:
        if b:env.stop(b,name)
        env.restore(state,out)
    print(out,flush=True)
if __name__=='__main__':main()
