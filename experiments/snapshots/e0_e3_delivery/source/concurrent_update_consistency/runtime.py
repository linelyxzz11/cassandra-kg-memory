"""Isolated experiment lifecycle; preserve all previous volumes and results."""
import sys, json, time, secrets, os
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'representation_serving_cost'))
from common import write_json
from environment import docker,safe_config,NAME as CASS,ORIGINALS
NEO=os.environ.get('CASSMEM_CONSISTENCY_NEO_CONTAINER','cassmem-consistency-20261013-neo4j')
VOLUME=os.environ.get('CASSMEM_CONSISTENCY_NEO_VOLUME','cassmem_consistency_20261013_neo4j')
assert NEO.startswith('cassmem-consistency-') and VOLUME.startswith('cassmem_consistency_')

def neo_backend():
    from cassmem.backend.live_cells_graph import Neo4jGraphCells
    info=json.loads(docker('inspect',NEO).stdout)[0]
    auth=next(e.split('=',1)[1] for e in info['Config']['Env'] if e.startswith('NEO4J_AUTH='))
    user,pw=auth.split('/',1)
    return Neo4jGraphCells('bolt://127.0.0.1:17687',user,pw)

def capture():
    return {'originals':{n:safe_config(n) for n in ORIGINALS},'status':'PREPARING'}

def switch(engine):
    for name in (*ORIGINALS,CASS,NEO):
        p=docker('inspect',name,check=False)
        if p.returncode==0 and json.loads(p.stdout)[0]['State']['Running']:
            docker('stop','--time','60',name)
    name=CASS if engine=='cassandra' else NEO
    if engine=='neo4j' and docker('inspect',NEO,check=False).returncode:
        if docker('volume','inspect',VOLUME,check=False).returncode==0:
            raise RuntimeError('Unexpected existing Neo4j volume')
        docker('volume','create',VOLUME)
        docker('run','-d','--name',NEO,'--cpus','8','--memory','12g','--memory-swap','12g',
               '-p','127.0.0.1:17687:7687','-v',f'{VOLUME}:/data',
               '-e','NEO4J_AUTH=neo4j/'+secrets.token_hex(20),
               '-e','NEO4J_server_memory_heap_initial__size=5G',
               '-e','NEO4J_server_memory_heap_max__size=5G',
               '-e','NEO4J_server_memory_pagecache_size=3G','neo4j:5.26')
    else:
        docker('start',name)
    for _ in range(90):
        try:
            if engine=='cassandra':
                p=docker('exec',name,'cqlsh','-e','SELECT release_version FROM system.local;',check=False,timeout=20)
                if p.returncode: raise RuntimeError('not ready')
            else:
                b=neo_backend(); b.close()
            break
        except Exception:
            time.sleep(2)
    else: raise RuntimeError('Database startup timed out')
    info=safe_config(name)
    assert info['resources']=={'Memory':12*1024**3,'NanoCpus':8*10**9,'MemorySwap':12*1024**3}
    return info

def restore(state):
    for name in (CASS,NEO):
        p=docker('inspect',name,check=False)
        if p.returncode==0 and json.loads(p.stdout)[0]['State']['Running']:
            docker('stop','--time','60',name)
    for name,before in state['originals'].items():
        if before['state']=='running': docker('start',name)
    state['status']='RESTORED_ORIGINALS'
