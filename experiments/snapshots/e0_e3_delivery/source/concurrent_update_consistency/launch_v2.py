"""Fresh v2 campaign; preserve all v1 evidence and database namespaces."""
import argparse
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ['CASSMEM_CONSISTENCY_NEO_CONTAINER'] = 'cassmem-consistency-20261013-v2-neo4j'
os.environ['CASSMEM_CONSISTENCY_NEO_VOLUME'] = 'cassmem_consistency_20261013_v2_neo4j'

from run_matrix import run
from common import sha, write_json, ROOT

if __name__ == '__main__':
    out = HERE / 'results/formal_20261013_v2'
    if out.exists():
        raise RuntimeError('Preserving existing v2 output; no automatic overwrite/resume')
    running = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'],
                             check=True, capture_output=True, text=True).stdout.splitlines()
    if any(n != 'cassmem-repcost-20261012-cassandra' for n in running):
        raise RuntimeError('Unexpected running container; resource ownership must be checked')
    image = subprocess.run(['docker','inspect','cassmem-consistency-20261013-neo4j',
                            '--format','{{.Image}}'],check=True,capture_output=True,text=True).stdout.strip()
    # Require the locally available Neo4j tag to resolve to the prior experiment image.
    tagged = subprocess.run(['docker','image','inspect','neo4j:5.26','--format','{{.Id}}'],
                             check=True,capture_output=True,text=True).stdout.strip()
    if image != tagged:
        raise RuntimeError('Neo4j image drift; inspect before starting')
    paths = [HERE / n for n in ('runtime.py','formal_trial.py','run_matrix.py','preflight.py','launch_v2.py')]
    paths += [ROOT/'src/cassmem/retrieval/online.py', ROOT/'src/cassmem/backend/live_cells_graph.py']
    write_json(HERE/'results/formal_20261013_v2_launch.json', {
        'initial_running_containers':running,
        'neo4j_image':image,
        'sources':{str(p):sha(p) for p in paths},
        'tests':'8 existing cleanup/scoring/boundary tests passed; Cassandra 5.0.8 live query passed',
        'prior_results':'formal_20261013_v1 preserved; process absent; incomplete point not reused',
        'coverage':'FORMAL_PROTOCOL.md 36-point matrix; older B/C/D extensions not included'
    })
    run(argparse.Namespace(output=out, smoke=False))
