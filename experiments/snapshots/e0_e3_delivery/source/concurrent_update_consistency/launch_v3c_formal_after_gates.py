"""Launch one fresh formal matrix only after six strict pilot gates pass."""
import os,sys,json,time,argparse,subprocess
from pathlib import Path
import psutil

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'representation_serving_cost'))
os.environ['CASSMEM_CONSISTENCY_NEO_CONTAINER']='cassmem-consistency-20261013-v3c-neo4j'
os.environ['CASSMEM_CONSISTENCY_NEO_VOLUME']='cassmem_consistency_20261013_v3c_neo4j'
from common import write_json,sha

def main(pid):
    gate=HERE/'results/gates_20261013_v3c'
    status=HERE/'results/formal_20261013_v3c_launch_status.json'
    if status.exists():raise RuntimeError('Preserving prior launch status')
    write_json(status,{'status':'WAITING_FOR_GATES','gate_pid':pid})
    try:
        process=psutil.Process(pid)
        process.wait()
    except psutil.NoSuchProcess:pass
    progress=json.loads((gate/'progress.json').read_text(encoding='utf-8'))
    if progress['status']!='GATES_PASSED' or len(progress['completed'])!=6 or not all(r['gate_pass'] for r in progress['completed']):
        write_json(status,{'status':'NOT_STARTED_GATE_FAILED','gate_status':progress['status']});return
    protocol=json.loads((gate/'protocol.json').read_text(encoding='utf-8'))
    for filename,expected in protocol['source_hashes'].items():
        if sha(Path(filename))!=expected:raise RuntimeError('Source drift after gates: '+filename)
    active=subprocess.run(['docker','ps','--format','{{.Names}}'],check=True,capture_output=True,text=True).stdout.strip()
    if active:raise RuntimeError('Unexpected active containers after gates: '+active)
    others=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        if proc.pid==os.getpid() or not proc.info['name'].lower().startswith('python'):continue
        cmd=' '.join(proc.info['cmdline'] or []).lower()
        if 'memorytable' in cmd:others.append(proc.pid)
    if others:raise RuntimeError('Other workspace Python jobs active: '+str(others))
    from run_matrix_v3c import run
    write_json(status,{'status':'FORMAL_STARTED','gates':str(gate),'planned':36})
    run(argparse.Namespace(output=HERE/'results/formal_20261013_v3c',smoke=False))
    write_json(status,{'status':'FORMAL_COMPLETE','gates':str(gate)})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gate-pid',type=int,required=True)
    args=parser.parse_args()
    try:main(args.gate_pid)
    except BaseException as exc:
        write_json(HERE/'results/formal_20261013_v3c_launch_error.json',{'error':repr(exc)});raise
