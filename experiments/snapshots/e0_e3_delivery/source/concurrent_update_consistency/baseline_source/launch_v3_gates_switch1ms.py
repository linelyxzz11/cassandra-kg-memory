"""Gate v3 on all three100K layouts; never relax validity/ready cutoffs."""
import os,sys,json,argparse,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
os.environ['CASSMEM_CONSISTENCY_NEO_CONTAINER']='cassmem-consistency-20261013-v3gate-neo4j'
os.environ['CASSMEM_CONSISTENCY_NEO_VOLUME']='cassmem_consistency_20261013_v3gate_neo4j'
from run_matrix_v3 import load,CELLS
from runtime import capture,switch,restore
from common import write_json,sha,ROOT

def main():
    out=HERE/'results/gates_20261013_v3';out.mkdir(parents=True,exist_ok=False)
    active=subprocess.run(['docker','ps','--format','{{.Names}}'],check=True,capture_output=True,text=True).stdout.strip()
    if active:raise RuntimeError('Refusing to interrupt running containers: '+active)
    state=capture();write_json(out/'environment.json',state)
    progress={'status':'STARTING','completed':[],'planned':6}
    keyspace='repcost_consistency_gates_20261013_v3'
    write_json(out/'protocol.json',{'memories':100000,'cells':CELLS,'rate':5,'modes':['revision','insert'],
        'warmup':10,'seconds':30,'drain_seconds':120,'workers':32,'read_update_ratio':'19:1',
        'required':'all requests, original submit p95<=50ms/end+1s, enqueue-return p95<=50ms, original strict-ready checks and checkpoint pass, no deadline censoring',
        'source_hashes':{str(p):sha(p) for p in (HERE/'formal_trial_v3.py',HERE/'neo4j_collected_v3.py',Path(__file__),ROOT/'src/cassmem/backend/live_cells_graph.py')}})
    try:
        current=None
        for cell in CELLS:
            engine='cassandra' if cell.startswith('cassandra') else 'neo4j'
            if engine!=current:
                progress.update(status='SWITCHING',cell=cell);write_json(out/'progress.json',progress)
                state[engine]=switch(engine);current=engine
            progress.update(status='LOADING',cell=cell);write_json(out/'progress.json',progress)
            load(cell,keyspace,100000,out)
            for mode in ('revision','insert'):
                name=f'{cell}_{mode}_5_r0';progress.update(status='RUNNING',point=name);write_json(out/'progress.json',progress)
                cmd=[sys.executable,'-B','-u',str(HERE/'formal_trial_v3.py'),'--cell',cell,'--mode',mode,'--rate','5','--rep','0',
                     '--memories','100000','--keyspace',keyspace,'--warmup','10','--seconds','30','--output',str(out/name)]
                print('RUN',name,flush=True)
                with (out/(name+'.log')).open('w',encoding='utf-8') as h:
                    p=subprocess.run(cmd,stdout=h,stderr=subprocess.STDOUT)
                if p.returncode:raise RuntimeError('Trial failed: '+name)
                result=json.loads((out/name/'summary.json').read_text(encoding='utf-8'))
                ok=result['status']=='PASS' and result['enqueue_return_lag_p95_ms']<=50
                progress['completed'].append({'point':name,'gate_pass':ok,'summary':result})
                write_json(out/'progress.json',progress);print('DONE',name,'gate',ok,'lag',result['enqueue_return_lag_p95_ms'],flush=True)
                if not ok:
                    progress.update(status='GATE_FAILED');write_json(out/'progress.json',progress);return
        progress.update(status='GATES_PASSED');write_json(out/'progress.json',progress)
    except BaseException as exc:
        progress.update(status='ERROR',error=repr(exc));write_json(out/'progress.json',progress);raise
    finally:
        restore(state);write_json(out/'environment.json',state)

if __name__=='__main__':main()
