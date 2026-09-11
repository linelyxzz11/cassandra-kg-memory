"""Execute one frozen 100K 95:5 workload cell/concurrency/repetition."""
from __future__ import annotations
import argparse,csv,json,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from live_cells import CassandraCells,Neo4jCells,Record,digest_projection
from online_retrieval import ScopedOnlineRetrievalIndex
from run_canonical_live_gate import ROOT,build,env
from run_100k_load_gate import expand
OUT=ROOT/'05_reports'/'locomo_workload_100k'/'runs'
def load_jsonl(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x]
def percentile(x,p):return float(np.percentile(np.asarray(x,dtype=float),p)) if x else 0.0
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cell',required=True,choices=['cassandra-base','cassandra-materialized','neo4j-native','neo4j-materialized']);ap.add_argument('--concurrency',required=True,type=int,choices=[1,8,16,32,64]);ap.add_argument('--rep',required=True,type=int,choices=[0,1,2]);a=ap.parse_args();env()
 source=build();records=expand(source);byid={r.memory_id:r for r in records};byscope={}
 for r in records:byscope.setdefault(r.scope_id,[]).append(r)
 mids=(ROOT/'01_data/locomo_memory_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines();marr=np.load(ROOT/'01_data/locomo_memory_bge_large.npy',mmap_mode='r');mv={m:marr[i] for i,m in enumerate(mids)}
 qids=(ROOT/'01_data/locomo_qa_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines();qarr=np.load(ROOT/'01_data/locomo_qa_bge_large.npy',mmap_mode='r');qv={q:qarr[i] for i,q in enumerate(qids)}
 trace_dir=ROOT/'05_reports/locomo_workload_100k/traces';warm=load_jsonl(trace_dir/f'rep{a.rep}_warmup.jsonl');measured=load_jsonl(trace_dir/f'rep{a.rep}_measured.jsonl');targets={x['target_memory_id'] for x in warm+measured if x['op_type']=='update'}
 cass=CassandraCells(os.getenv('CASSANDRA_HOST','127.0.0.1')) if a.cell.startswith('cassandra') else None;neo=Neo4jCells(os.getenv('NEO4J_URI','bolt://localhost:7687'),os.getenv('NEO4J_USER','neo4j'),os.getenv('NEO4J_PASSWORD'),os.getenv('NEO4J_DATABASE','neo4j')) if a.cell.startswith('neo4j') else None;backend=cass or neo
 idx=ScopedOnlineRetrievalIndex();print('building_index',flush=True)
 for scope,rr in byscope.items():
  if scope.startswith('lr000017::'):continue
  base=[r for r in rr if r.memory_id not in targets];idx.load_scope(scope,{r.memory_id:r.rawerk for r in base},{r.memory_id:mv[r.memory_id.split('::',1)[1]] for r in base})
 for t in targets:backend.delete(a.cell,byid[t])
 def execute(op,phase):
  t0=time.perf_counter_ns();scope=op['scope_id'];backend0=time.perf_counter_ns()
  if op['op_type']=='read':
   ids=backend.ids(a.cell,scope);backend1=time.perf_counter_ns();res=idx.search(scope,op['question'],qv[op['source_qa_id']],ids);end=time.perf_counter_ns();return {'op_id':op['op_id'],'phase':phase,'op_type':'read','status':'ok','latency_ms':(end-t0)/1e6,'backend_ms':(backend1-backend0)/1e6,'index_fusion_ms':(end-backend1)/1e6,'fresh_hit_at_10':'','projection_ok':'','candidate_count':len(ids)}
  r=byid[op['target_memory_id']];backend.insert(a.cell,r);commit=time.perf_counter_ns();p=backend.fetch(a.cell,r.scope_id,r.memory_id);visible=time.perf_counter_ns();idx.upsert_sparse(scope,r.memory_id,r.rawerk,2);sparse=time.perf_counter_ns();idx.upsert_dense(scope,r.memory_id,mv[op['source_memory_id']],2);dense=time.perf_counter_ns();ids=backend.ids(a.cell,scope);bf=time.perf_counter_ns();res=idx.search(scope,op['question'],qv[op['source_qa_id']],ids);end=time.perf_counter_ns();top={x for x,_ in res.top10};return {'op_id':op['op_id'],'phase':phase,'op_type':'update','status':'ok','latency_ms':(end-t0)/1e6,'backend_ms':(commit-t0)/1e6,'index_fusion_ms':(end-visible)/1e6,'fresh_hit_at_10':int(r.memory_id in top),'projection_ok':int(bool(p) and digest_projection(p)==digest_projection(r.projection())),'candidate_count':len(ids)}
 def run_phase(ops,name):
  start=time.perf_counter()
  with ThreadPoolExecutor(max_workers=a.concurrency) as ex:result=list(ex.map(lambda op:execute(op,name),ops))
  return result,time.perf_counter()-start
 try:
  _,warm_s=run_phase(warm,'warmup');rows,measured_s=run_phase(measured,'measured');OUT.mkdir(parents=True,exist_ok=True);stem=f'{a.cell}_c{a.concurrency}_r{a.rep}'
  with (OUT/f'{stem}_events.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  lat=[x['latency_ms'] for x in rows];reads=[x for x in rows if x['op_type']=='read'];updates=[x for x in rows if x['op_type']=='update'];summary={'status':'PASS' if len(rows)==5000 and all(x['status']=='ok' for x in rows) and all(x['projection_ok']==1 for x in updates) else 'FAIL','cell':a.cell,'concurrency':a.concurrency,'repetition':a.rep,'warmup_operations':500,'measured_operations':len(rows),'reads':len(reads),'updates':len(updates),'warmup_seconds':warm_s,'measured_seconds':measured_s,'throughput_ops_s':len(rows)/measured_s,'latency_p50_ms':percentile(lat,50),'latency_p95_ms':percentile(lat,95),'latency_p99_ms':percentile(lat,99),'fresh_hit_at_10':sum(x['fresh_hit_at_10'] for x in updates)/len(updates)};(OUT/f'{stem}_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2))
 finally:backend.close()
if __name__=='__main__':main()
