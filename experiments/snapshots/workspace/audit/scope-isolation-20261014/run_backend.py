"""Stage B: isolated live-backend scope noninterference. Never resets existing data."""
import os,sys,json,time,subprocess,traceback,hashlib,secrets,threading
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('D:/memorytable/cassandra-kg-memory');OUT=Path(__file__).resolve().parent/os.environ.get('SCOPE_B_OUTPUT','backend')
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
from cassmem.backend.live_cells_graph import CassandraGraphCells,Neo4jGraphCells
from cassmem.representation.graph_event import GraphRecord
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def log(stage,**kw):
    save('status.json',dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw));print(stage,kw,flush=True)
def docker(*args):return subprocess.check_output(['docker',*args],text=True,stderr=subprocess.STDOUT).strip()
def record(scope,i,external=False,version=1):
    place='Rome' if external else 'Paris';rel='visited' if external else 'lives_in'
    raw=f'Alice {rel} {place}. Record {i}, version {version}.'
    return GraphRecord(scope,f'm{i:03d}',version,raw,f'Alice, {place}',rel,'city',f'(Alice, {rel}, {place})',hashlib.sha256(raw.encode()).hexdigest())
def main():
    from cassandra.cluster import Cluster
    from neo4j import GraphDatabase
    if (OUT/'protocol.json').exists():raise RuntimeError('Existing stage B evidence: review before restart')
    images={'cassandra':'2c6576f9b8e0','neo4j':'0b5d3ab6ec1b'}
    source=ROOT/'src/cassmem/backend/live_cells_graph.py';source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    runid=time.strftime('%Y%m%d%H%M%S');prefix='cassmem-scope-b-'+runid
    protocol=dict(runid=runid,containers=prefix,images=images,source_hash=source_hash,records_per_scope=24,
        orders=['A_first','B_first'],cells=['cassandra-base','cassandra-materialized','neo4j-native'],
        resources=dict(memory='3g',cpus=2,cassandra_heap='1G',neo4j_heap='1G',neo4j_pagecache='512M'),
        test='correctness only, not performance; same IDs/entities, shared and disjoint relation labels; external updates only',
        data='synthetic fixtures; no LoCoMo quality claims',scope_isolation='fresh containers+volumes and per-cell/order scopes',
        update_rounds=10,reads_per_concurrent_round=3,original_containers_untouched=True)
    save('protocol.json',protocol);observations=[];events=[]
    def check(backend,cell,A,phase,roundno=-1):
        t=time.perf_counter_ns();s=A[0].scope_id;expected_ids=sorted(r.memory_id for r in A)
        row=dict(cell=cell,scope=s,phase=phase,round=roundno,start_ns=t)
        try:
            actual_ids=backend.ids(cell,s);graphs=[];projections=[]
            for r in A:
                g=backend.fetch_graph(cell,s,r.memory_id);projection=backend.candidate_projection(cell,s,r.memory_id)
                graphs.append(dict(memory_id=r.memory_id,equal=g==r.graph_projection(),actual=g))
                projections.append(dict(memory_id=r.memory_id,equal=projection==r.rawerk,actual=projection))
            relations={rel:backend.ids_by_relation(cell,s,rel) for rel in ('lives_in','visited','works_in')}
            expected={rel:sorted(r.memory_id for r in A if any(e['relation']==rel for e in r.edges)) for rel in relations}
            row.update(ok=actual_ids==expected_ids and all(g['equal'] for g in graphs) and all(p['equal'] for p in projections) and relations==expected,
                actual_ids=actual_ids,graphs=graphs,projections=projections,relations=relations,expected_relations=expected)
        except Exception:row.update(ok=False,error=traceback.format_exc())
        row['end_ns']=time.perf_counter_ns();observations.append(row)
        with (OUT/'observations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        return row
    for engine in ('cassandra','neo4j'):
        reuse=os.environ.get('SCOPE_B_REUSE_CASSANDRA') if engine=='cassandra' else None
        name=reuse or prefix+'-'+engine;port=29042 if engine=='cassandra' else 27687;password=secrets.token_urlsafe(24)
        volume=name+'-data'
        args=['run','-d','--name',name,'--memory','3g','--cpus','2','-p',f'127.0.0.1:{port}:'+('9042' if engine=='cassandra' else '7687'),'-v',volume+(':/var/lib/cassandra' if engine=='cassandra' else ':/data')]
        args+=['-e','MAX_HEAP_SIZE=1G','-e','HEAP_NEWSIZE=256M'] if engine=='cassandra' else ['-e','NEO4J_AUTH=neo4j/'+password,'-e','NEO4J_server_memory_heap_initial__size=1G','-e','NEO4J_server_memory_heap_max__size=1G','-e','NEO4J_server_memory_pagecache_size=512M']
        if reuse:
            assert reuse=='cassmem-scope-b-20261014161714-cassandra'
            info=json.loads(docker('inspect',reuse))[0]
            assert info['State']['Running'] and info['HostConfig']['Memory']==3*1024**3
            assert info['NetworkSettings']['Ports']['9042/tcp'][0]['HostPort']=='29042'
        else:
            docker('volume','create',volume);docker(*args,images[engine])
        log('waiting_database',engine=engine,container=name)
        backend=None;last_error=None
        try:
            for attempt in range(60):
                try:
                    if engine=='cassandra':
                        os.environ['CASSANDRA_PORT']=str(port);backend=CassandraGraphCells(keyspace='scope_b_'+runid)
                    else:backend=Neo4jGraphCells(f'bolt://127.0.0.1:{port}','neo4j',password)
                    break
                except Exception as e:
                    last_error=type(e).__name__;time.sleep(3)
            if backend is None:raise RuntimeError('Database readiness failed: '+str(last_error))
            info=json.loads(docker('inspect',name))[0]
            save(engine+'_environment.json',dict(name=name,image=info['Image'],memory=info['HostConfig']['Memory'],nano_cpus=info['HostConfig']['NanoCpus'],mounts=info['Mounts'],ports=info['NetworkSettings']['Ports']))
            cells=['cassandra-base','cassandra-materialized'] if engine=='cassandra' else ['neo4j-native']
            for cell in cells:
                for order in ('A_first','B_first'):
                    s=runid+'_'+cell+'_'+order;A=[record(s+'_A',i) for i in range(24)];B=[record(s+'_B',i,True,7) for i in range(24)]
                    # Half share relation label as well as entity ID, but use different destination.
                    B=[replace(r,relations='lives_in',triples='(Alice, lives_in, Rome)') if i%2==0 else r for i,r in enumerate(B)]
                    for group in ([A,B] if order=='A_first' else [B,A]):
                        for r in group:backend.insert(cell,r)
                        if group is A:check(backend,cell,A,'after_A_insert')
                    check(backend,cell,A,'after_both_insert')
                    # Query B explicitly; then read A again through the same backend instance.
                    for r in B:backend.fetch_graph(cell,r.scope_id,r.memory_id)
                    check(backend,cell,A,'after_B_queries')
                    for mode in ('interleaved','concurrent'):
                        for roundno in range(10):
                            i=roundno%len(B);old=B[i];new=replace(old,version=old.version+1,raw_text=f'Alice works in London, update {mode} {roundno}.',entities='Alice, London',relations='works_in',triples='(Alice, works_in, London)',keywords='employment')
                            new=replace(new,embedding_sha256=hashlib.sha256(new.raw_text.encode()).hexdigest());barrier=threading.Barrier(2)
                            def writer():
                                if mode=='concurrent':barrier.wait(timeout=30)
                                start=time.perf_counter_ns();backend.commit_raw(cell,new);backend.write_structured(cell,new)
                                end=time.perf_counter_ns();actual=backend.fetch_graph(cell,new.scope_id,new.memory_id)
                                return dict(start_ns=start,end_ns=end,external_graph_equal=actual==new.graph_projection(),actual=actual)
                            def reader():
                                if mode=='concurrent':barrier.wait(timeout=30)
                                return [check(backend,cell,A,mode,roundno) for _ in range(3)]
                            if mode=='concurrent':
                                with ThreadPoolExecutor(2) as pool:
                                    wf=pool.submit(writer);rf=pool.submit(reader);w=wf.result();rs=rf.result()
                            else:w=writer();rs=reader()
                            B[i]=new;check(backend,cell,A,mode+'_post',roundno)
                            events.append(dict(cell=cell,order=order,mode=mode,round=roundno,writer=w,overlap_reads=sum(r['start_ns']<w['end_ns'] and r['end_ns']>w['start_ns'] for r in rs)))
                            save('update_events.json',events)
                        log('mode_complete',cell=cell,order=order,mode=mode,observations=len(observations),failed=sum(not r['ok'] for r in observations))
            backend.close();backend=None
        finally:
            if backend is not None:backend.close()
            # Stop only the container created by this run; retain its data volume and metadata.
            docker('stop','-t','30',name)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    save('complete.json',dict(observations=len(observations),failed_scope_checks=sum(not r['ok'] for r in observations),
        update_events=len(events),external_update_mismatches=sum(not e['writer']['external_graph_equal'] for e in events),
        overlapping_reads=sum(e['overlap_reads'] for e in events),validation_pending=True))
    log('stage_B_execution_complete_validation_pending')
if __name__=='__main__':
    try:main()
    except Exception:log('failed',error=traceback.format_exc());raise
