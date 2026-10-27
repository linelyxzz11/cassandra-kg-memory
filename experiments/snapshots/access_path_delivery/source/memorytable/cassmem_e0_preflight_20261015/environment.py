import json,subprocess,secrets,time,os
from pathlib import Path
import sys
sys.path.insert(0,'D:/memorytable/cassandra-kg-memory/src')
from cassmem.backend.live_cells_graph import CassandraGraphCells,Neo4jGraphCells
OLD='cassmem-consistency-scale-20261013-100000-neo4j'
def docker(*a):return subprocess.check_output(['docker',*a],text=True,stderr=subprocess.PIPE).strip()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def suspend_old(out):
    running=docker('ps','--format','{{.Names}}').splitlines()
    if set(running)-{OLD}:raise RuntimeError('Other containers running; resource ownership must be reviewed')
    state=dict(original_running=running,stopped=False)
    save(out/'original_state.json',state)
    if OLD in running:
        info=json.loads(docker('inspect',OLD))[0];env=dict(v.split('=',1) for v in info['Config']['Env'] if '=' in v)
        from neo4j import GraphDatabase
        with GraphDatabase.driver('bolt://127.0.0.1:17687',auth=('neo4j',env['NEO4J_AUTH'].split('/',1)[1])) as d:
            with d.session() as s:
                active=list(s.run('SHOW TRANSACTIONS YIELD currentQuery RETURN currentQuery'))
                if any(r['currentQuery'] and not r['currentQuery'].startswith('SHOW TRANSACTIONS') for r in active):raise RuntimeError('Old database has active transactions')
        docker('stop','-t','60',OLD);state['stopped']=True;save(out/'original_state.json',state)
    return state
def restore(state,out):
    if state['stopped']:docker('start',OLD)
    save(out/'restored.json',dict(originals=state['original_running'],running=docker('ps','--format','{{.Names}}').splitlines()))
def start(engine,out):
    name='cassmem-e0-preflight-'+time.strftime('%Y%m%d%H%M%S')+'-'+engine
    port=49042 if engine=='cassandra' else 47687;password=secrets.token_urlsafe(24)
    image='2c6576f9b8e0' if engine=='cassandra' else '0b5d3ab6ec1b'
    args=['run','-d','--name',name,'--cpus','8','--memory','12g','--memory-swap','12g','-p',f'127.0.0.1:{port}:'+('9042' if engine=='cassandra' else '7687'),'-v',name+'-data:'+('/var/lib/cassandra' if engine=='cassandra' else '/data')]
    args+=['-e','MAX_HEAP_SIZE=6G','-e','HEAP_NEWSIZE=1G'] if engine=='cassandra' else ['-e','NEO4J_AUTH=neo4j/'+password,'-e','NEO4J_server_memory_heap_initial__size=5G','-e','NEO4J_server_memory_heap_max__size=5G','-e','NEO4J_server_memory_pagecache_size=3G']
    docker(*args,image);save(out/(engine+'_container.json'),dict(name=name,port=port,image=json.loads(docker('inspect',name))[0]['Image']))
    b=None
    try:
        for attempt in range(120):
            try:
                if engine=='cassandra':
                    docker('exec',name,'cqlsh','-e','SELECT release_version FROM system.local;');os.environ['CASSANDRA_PORT']=str(port)
                    b=CassandraGraphCells(keyspace='e0_preflight_'+time.strftime('%Y%m%d%H%M%S'))
                    from cassandra import ConsistencyLevel
                    b.s.default_timeout=30;b.s.default_consistency_level=ConsistencyLevel.LOCAL_ONE
                    for p in b.p.values():p.consistency_level=ConsistencyLevel.LOCAL_ONE
                else:
                    from neo4j import GraphDatabase
                    with GraphDatabase.driver(f'bolt://127.0.0.1:{port}',auth=('neo4j',password),connection_timeout=3) as d:d.verify_connectivity()
                    b=Neo4jGraphCells(f'bolt://127.0.0.1:{port}','neo4j',password)
                break
            except Exception:
                if attempt==119:raise
                time.sleep(2)
        info=json.loads(docker('inspect',name))[0]
        meta=dict(name=name,image=info['Image'],memory=info['HostConfig']['Memory'],nano_cpus=info['HostConfig']['NanoCpus'],memory_swap=info['HostConfig']['MemorySwap'])
        assert meta['memory']==12*1024**3 and meta['nano_cpus']==8*10**9
        if engine=='cassandra':
            meta.update(version=docker('exec',name,'cqlsh','-e','SELECT release_version FROM system.local;'),
                config=docker('exec',name,'sh','-c',"grep -E '^(commitlog_sync|commitlog_sync_period_in_ms|commitlog_sync_group_window_in_ms|disk_access_mode|concurrent_reads|concurrent_writes):' /etc/cassandra/cassandra.yaml"),
                consistency='LOCAL_ONE',replication_factor=1,driver_protocol=4,request_timeout_s=30,core_connections_local=b.cluster.get_core_connections_per_host(0))
        else:
            meta.update(version=[dict(r) for r in b.run('CALL dbms.components() YIELD name,versions RETURN name,versions')],
                indexes=[dict(r) for r in b.run('SHOW INDEXES YIELD name,type,state,labelsOrTypes,properties RETURN name,type,state,labelsOrTypes,properties')],
                pool_max=128,transactions='one auto-commit statement per backend operation',durability='server transaction commit acknowledgment; no cross-index atomic publication')
        save(out/(engine+'_environment.json'),meta);return b,name
    except Exception:
        if b:b.close()
        docker('stop','-t','30',name);raise
def stop(b,name):
    try:b.close()
    finally:docker('stop','-t','30',name)
