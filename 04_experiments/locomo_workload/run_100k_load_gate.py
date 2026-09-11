"""Materialize the deterministic 100K trace into all four live cells."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from live_cells import CassandraCells,Neo4jCells,Record,digest_projection
from run_canonical_live_gate import ROOT,build,env

OUT=ROOT/'05_reports'/'locomo_workload_100k'
TARGET=100_000

def file_sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def expand(source):
    out=[]; n=len(source)
    for ordinal in range(TARGET):
        replica,index=divmod(ordinal,n); r=source[index]; prefix=f"lr{replica:06d}::"
        out.append(Record(prefix+r.scope_id,prefix+r.memory_id,r.version,r.raw_text,r.entities,r.relations,r.keywords,r.triples,r.embedding_sha256))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reset',action='store_true');ap.add_argument('--cassandra-workers',type=int,default=32);ap.add_argument('--neo4j-workers',type=int,default=8);a=ap.parse_args();env()
    manifest=json.loads((ROOT/'05_reports/locomo_workload_manifests/trace_manifest_100000.json').read_text(encoding='utf-8'))
    source_paths={'memory_csv':ROOT/'01_data/locomo_memory_records.csv','qa_csv':ROOT/'01_data/locomo_qa_records.csv','feature_csv':ROOT/'02_artifacts/p3_memory_features.csv','memory_ids':ROOT/'01_data/locomo_memory_ids_bge.txt','qa_ids':ROOT/'01_data/locomo_qa_ids_bge.txt'}
    bad_hash={k:{'expected':manifest['source_sha256'][k],'actual':file_sha256(p)} for k,p in source_paths.items() if file_sha256(p)!=manifest['source_sha256'][k]}
    if bad_hash: raise RuntimeError(f'Source hash gate failed: {bad_hash}')
    records=expand(build()); scopes=sorted({r.scope_id for r in records}); pairs={(r.scope_id,r.memory_id) for r in records}; ids={r.memory_id for r in records}
    preload={'protocol_id':'locomo-shaped-100k-v1','manifest_sha256':manifest['manifest_sha256'],'target_memories':len(records),'namespace_count':len(scopes),'unique_memory_ids':len(ids),'unique_scope_memory_pairs':len(pairs),'complete_replica_count':17,'partial_replica_memory_count':6,'partial_namespace_count':1,'source_hash_gate':'PASS'}
    if not(len(records)==len(ids)==len(pairs)==100000 and len(scopes)==171):raise RuntimeError(f'Expansion cardinality gate failed: {preload}')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'preload_gate.json').write_text(json.dumps(preload,indent=2)+'\n',encoding='utf-8')
    cass=CassandraCells(os.getenv('CASSANDRA_HOST','127.0.0.1'));neo=Neo4jCells(os.getenv('NEO4J_URI','bolt://localhost:7687'),os.getenv('NEO4J_USER','neo4j'),os.getenv('NEO4J_PASSWORD'),os.getenv('NEO4J_DATABASE','neo4j'))
    stage=[]; total_start=time.time()
    try:
        if a.reset:
            t=time.time();cass.reset();neo.reset();stage.append({'stage':'reset','seconds':time.time()-t})
        for cell in cass.names:
            t=time.time()
            with ThreadPoolExecutor(max_workers=a.cassandra_workers) as ex:
                for i,_ in enumerate(ex.map(lambda r:cass.insert(cell,r),records),1):
                    if i%10000==0:print(f'{cell} loaded {i}/{TARGET}',flush=True)
            stage.append({'stage':f'load:{cell}','seconds':time.time()-t})
        for cell in neo.names:
            t=time.time();neo.insert_many(cell,records,batch=500,workers=a.neo4j_workers);stage.append({'stage':f'load:{cell}','seconds':time.time()-t});print(f'{cell} loaded {TARGET}/{TARGET}',flush=True)
        counts={};namespace_counts={}
        for cell in cass.names+neo.names:
            adapter=cass if cell.startswith('cassandra') else neo
            counts[cell]=sum(len(adapter.ids(cell,s)) for s in scopes);namespace_counts[cell]=sum(bool(adapter.ids(cell,s)) for s in scopes)
        # Deterministic 1,000-record cross-cell digest gate; full counts remain exact.
        sample=[records[(i*100003)%TARGET] for i in range(1000)];diff=[]
        for r in sample:
            expected=digest_projection(r.projection())
            for cell in cass.names+neo.names:
                adapter=cass if cell.startswith('cassandra') else neo;p=adapter.fetch(cell,r.scope_id,r.memory_id);observed=digest_projection(p) if p else ''
                if observed!=expected:diff.append({'memory_id':r.memory_id,'cell':cell,'expected':expected,'observed':observed})
        status='PASS' if all(v==TARGET for v in counts.values()) and all(v==171 for v in namespace_counts.values()) and not diff else 'FAIL'
        summary={'status':status,'target_memories':TARGET,'expected_namespaces':171,'counts':counts,'namespace_counts':namespace_counts,'digest_sample_memories':len(sample),'digest_comparisons':len(sample)*4,'digest_mismatches':len(diff),'stage_times':stage,'total_seconds':time.time()-total_start}
        (OUT/'load_gate_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        with (OUT/'load_gate_digest_diffs.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=['memory_id','cell','expected','observed']);w.writeheader();w.writerows(diff)
        print(json.dumps(summary,indent=2));
        if status!='PASS':raise SystemExit(2)
    finally:cass.close();neo.close()
if __name__=='__main__':main()
