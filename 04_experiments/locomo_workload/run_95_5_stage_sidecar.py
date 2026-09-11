"""Replay a 95:5 point and retain detailed timings for update events only."""
from __future__ import annotations
import argparse,csv,json,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from live_cells import CassandraCells,Neo4jCells,Record,digest_projection
from online_retrieval import ScopedOnlineRetrievalIndex
from run_canonical_live_gate import ROOT,build,env
from run_100k_load_gate import expand
OUT=ROOT/'05_reports'/'locomo_workload_100k'/'stage_sidecar'
def load_jsonl(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cell',required=True,choices=['cassandra-base','cassandra-materialized','neo4j-native','neo4j-materialized']);ap.add_argument('--concurrency',required=True,type=int,choices=[1,8,16,32,64]);ap.add_argument('--rep',required=True,type=int,choices=[0,1,2]);a=ap.parse_args();env()
 records=expand(build());byid={r.memory_id:r for r in records};byscope={}
 for r in records:byscope.setdefault(r.scope_id,[]).append(r)
 mids=(ROOT/'01_data/locomo_memory_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines();ma=np.load(ROOT/'01_data/locomo_memory_bge_large.npy',mmap_mode='r');mv={m:ma[i] for i,m in enumerate(mids)}
 qids=(ROOT/'01_data/locomo_qa_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines();qa=np.load(ROOT/'01_data/locomo_qa_bge_large.npy',mmap_mode='r');qv={q:qa[i] for i,q in enumerate(qids)}
 td=ROOT/'05_reports/locomo_workload_100k/traces';warm=load_jsonl(td/f'rep{a.rep}_warmup.jsonl');measured=load_jsonl(td/f'rep{a.rep}_measured.jsonl');targets={x['target_memory_id'] for x in warm+measured if x['op_type']=='update'}
 backend=CassandraCells(os.getenv('CASSANDRA_HOST','127.0.0.1')) if a.cell.startswith('cassandra') else Neo4jCells(os.getenv('NEO4J_URI'),os.getenv('NEO4J_USER'),os.getenv('NEO4J_PASSWORD'),os.getenv('NEO4J_DATABASE','neo4j'))
 idx=ScopedOnlineRetrievalIndex();print('building_index',flush=True)
 for scope,rr in byscope.items():
  if scope.startswith('lr000017::'):continue
  base=[r for r in rr if r.memory_id not in targets];idx.load_scope(scope,{r.memory_id:r.rawerk for r in base},{r.memory_id:mv[r.memory_id.split('::',1)[1]] for r in base})
 for t in targets:backend.delete(a.cell,byid[t])
 def op(x,keep):
  scope=x['scope_id']
  if x['op_type']=='read':
   ids=backend.ids(a.cell,scope);idx.search(scope,x['question'],qv[x['source_qa_id']],ids);return None
  r=byid[x['target_memory_id']];t0=time.perf_counter_ns();backend.insert(a.cell,r);tc=time.perf_counter_ns();p=backend.fetch(a.cell,scope,r.memory_id);tr=time.perf_counter_ns();idx.upsert_sparse(scope,r.memory_id,r.rawerk,2);ts=time.perf_counter_ns();idx.upsert_dense(scope,r.memory_id,mv[x['source_memory_id']],2);tdense=time.perf_counter_ns();ids=backend.ids(a.cell,scope);tf=time.perf_counter_ns();result,timing=idx.search_with_timings(scope,x['question'],qv[x['source_qa_id']],ids);tt=time.perf_counter_ns();top={m for m,_ in result.top10}
  if not keep:return None
  return {'op_id':x['op_id'],'cell':a.cell,'concurrency':a.concurrency,'repetition':a.rep,'qa_id':x['qa_id'],'scope_id':scope,'target_memory_id':r.memory_id,'commit_ms':(tc-t0)/1e6,'rawerk_visible_ms':(tr-t0)/1e6,'sparse_index_visible_ms':(ts-t0)/1e6,'dense_index_visible_ms':(tdense-t0)/1e6,'candidate_fetch_ms':(tf-tdense)/1e6,**timing,'time_to_top10_ms':(tt-t0)/1e6 if r.memory_id in top else '','fresh_hit_at_10':int(r.memory_id in top),'projection_ok':int(bool(p) and digest_projection(p)==digest_projection(r.projection())),'candidate_count':len(ids)}
 def phase(xs,keep):
  with ThreadPoolExecutor(max_workers=a.concurrency) as ex:return [r for r in ex.map(lambda z:op(z,keep),xs) if r]
 try:
  phase(warm,False);rows=phase(measured,True);OUT.mkdir(parents=True,exist_ok=True);stem=f'{a.cell}_c{a.concurrency}_r{a.rep}'
  with (OUT/f'{stem}_updates.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  status='PASS' if len(rows)==250 and all(r['projection_ok']==1 for r in rows) else 'FAIL';summary={'status':status,'cell':a.cell,'concurrency':a.concurrency,'repetition':a.rep,'update_events':len(rows),'fresh_hit_at_10':sum(r['fresh_hit_at_10'] for r in rows)/len(rows)};(OUT/f'{stem}_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary))
  if status!='PASS':raise SystemExit(2)
 finally:backend.close()
if __name__=='__main__':main()
