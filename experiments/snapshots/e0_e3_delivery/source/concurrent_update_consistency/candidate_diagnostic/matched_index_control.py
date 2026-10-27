"""Scope-index-hinted control; separate process and artifacts from original suite."""
import json, subprocess, time, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import run_diagnostic as diag

HERE=Path(__file__).resolve().parent
OUT=HERE/'matched_index_v2'

def main():
    OUT.mkdir(exist_ok=False)
    driver=None
    try:
        info=json.loads(subprocess.run(['docker','inspect',diag.NAME],check=True,capture_output=True,text=True).stdout)[0]
        auth=next(x.split('=',1)[1] for x in info['Config']['Env'] if x.startswith('NEO4J_AUTH='))
        user,pw=auth.split('/',1)
        driver=diag.neo4j.GraphDatabase.driver('bolt://127.0.0.1:17687',auth=(user,pw),max_connection_pool_size=128)
        for attempt in range(45):
            try:driver.verify_connectivity();break
            except Exception:
                if attempt==44:raise
                time.sleep(2)
        original=driver._pool.acquire
        def acquire(*args,**kwargs):
            start=diag.now()
            try:return original(*args,**kwargs)
            finally:
                if getattr(diag.TLS,'track',False):diag.TLS.acquire_ns+=diag.now()-start
        driver._pool.acquire=acquire
        diag.QUERY='MATCH (m:LWV2NativeMemory) USING INDEX m:LWV2NativeMemory(scope_id) WHERE m.scope_id=$s RETURN m.memory_id AS id'
        diag.COLLECT='MATCH (m:LWV2NativeMemory) USING INDEX m:LWV2NativeMemory(scope_id) WHERE m.scope_id=$s RETURN collect(m.memory_id) AS ids'
        trace=json.loads((HERE.parent/'results/formal_20261013_v2/neo4j-native_revision_5_r0/measured_trace.json').read_text(encoding='utf-8'))
        scopes=[e['old']['scope_id'] for e in trace]
        profiles=[];expected={}
        for scope in dict.fromkeys(scopes):
            a,_=diag.timed_ids(driver,scope,'stream');b,_=diag.timed_ids(driver,scope,'collect');assert a==b
            expected[scope]=a
        for variant,q in [('stream',diag.QUERY),('collect',diag.COLLECT)]:
            with driver.session(database='neo4j') as s:
                r=s.run('PROFILE '+q,s=scopes[0]);list(r);p=r.consume().profile
            profiles.append({'variant':variant,'query':q,'profile':p})
        diag.write_json(OUT/'profiles.json',profiles)
        diag.write_json(OUT/'protocol.json',{'workers':32,'n':192,'reps':3,'same_scope_index_hint':True,
            'queries':{'stream':diag.QUERY,'collect':diag.COLLECT},'warmup_per_point':32,'source_sha256':diag.sha(Path(__file__))})
        summaries=[]
        for rep in range(3):
            for variant in (('stream','collect') if rep%2==0 else ('collect','stream')):
                label=f'{variant}_r{rep}'
                def worker(i):
                    ids,row=diag.timed_ids(driver,scopes[i%len(scopes)],variant)
                    row['exact_ids']=ids==expected[scopes[i%len(scopes)]]
                    return row
                with ThreadPoolExecutor(max_workers=32) as pool:
                    list(pool.map(worker,range(32)));start=diag.now();rows=list(pool.map(worker,range(192)));elapsed=(diag.now()-start)/1e9
                with (OUT/(label+'.jsonl')).open('w') as h:
                    for row in rows:h.write(json.dumps(row)+'\n')
                summary={'point':label,'elapsed':elapsed,'id_failures':sum(not r['exact_ids'] for r in rows)}
                for k in ('candidate_ms','run_ms','fetch_ms','acquire_ms','client_thread_cpu_ms','server_available_ms','server_consumed_ms'):
                    summary[k]=diag.stats([r[k] for r in rows if r[k] is not None])
                summaries.append(summary);diag.write_json(OUT/'summary.json',summaries);print(label,elapsed,flush=True)
        diag.write_json(OUT/'status.json',{'status':'COMPLETE'})
    finally:
        if driver:driver.close()
        p=subprocess.run(['docker','stop','--time','60',diag.NAME],capture_output=True,text=True)
        diag.write_json(OUT/'restoration.json',{'returncode':p.returncode})

if __name__=='__main__':main()
