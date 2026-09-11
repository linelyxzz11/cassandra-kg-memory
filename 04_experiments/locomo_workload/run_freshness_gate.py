"""Run all 1,533 unambiguous Cat1-4 freshness events on four live cells."""
from __future__ import annotations
import csv,json,os,time,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from live_cells import CassandraCells,Neo4jCells,Record,digest_projection
from online_retrieval import ScopedOnlineRetrievalIndex,render_rawerk
from run_canonical_live_gate import ROOT,build,env,rows
OUT=ROOT/'05_reports'/'locomo_workload_canonical_gate'
def pct(xs,p):return float(np.percentile(np.asarray(xs,dtype=float),p)) if xs else 0.0
def main():
 env(); records=build(); byid={r.memory_id:r for r in records}; byscope={}
 for r in records:byscope.setdefault(r.scope_id,[]).append(r)
 eligible=rows(ROOT/'05_reports/locomo_workload_profile/freshness_eligible_cat1_4.csv')
 mids=(ROOT/'01_data/locomo_memory_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines(); mvec=np.load(ROOT/'01_data/locomo_memory_bge_large.npy',mmap_mode='r'); mv={m:mvec[i] for i,m in enumerate(mids)}
 qids=(ROOT/'01_data/locomo_qa_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines(); qvec=np.load(ROOT/'01_data/locomo_qa_bge_large.npy',mmap_mode='r'); qv={q:qvec[i] for i,q in enumerate(qids)}
 cass=CassandraCells(os.getenv('CASSANDRA_HOST','127.0.0.1'));neo=Neo4jCells(os.getenv('NEO4J_URI','bolt://localhost:7687'),os.getenv('NEO4J_USER','neo4j'),os.getenv('NEO4J_PASSWORD'),os.getenv('NEO4J_DATABASE','neo4j'))
 out=[]; cells=cass.names+neo.names
 try:
  for qi,q in enumerate(eligible):
   gold=json.loads(q['gold_memory_ids'])[0]; target=byid[gold]; scope=target.scope_id
   base=[r for r in byscope[scope] if r.memory_id!=gold]; docs={r.memory_id:r.rawerk for r in base}; embs={r.memory_id:mv[r.memory_id] for r in base}
   def run_cell(cell):
    a=cass if cell.startswith('cassandra') else neo; a.delete(cell,target); idx=ScopedOnlineRetrievalIndex();idx.load_scope(scope,docs,embs)
    t0=time.perf_counter_ns();a.insert(cell,target);tc=time.perf_counter_ns();p=a.fetch(cell,scope,gold);tr=time.perf_counter_ns()
    projection_ok=bool(p) and digest_projection(p)==digest_projection(target.projection())
    idx.upsert_sparse(scope,gold,target.rawerk,2);ts=time.perf_counter_ns();idx.upsert_dense(scope,gold,mv[gold],2);td=time.perf_counter_ns()
    candidate_ids=a.ids(cell,scope);tf0=time.perf_counter_ns();res=idx.search(scope,q['question'],qv[q['qa_id']],candidate_ids);tt=time.perf_counter_ns();top=[x for x,_ in res.top10];hit=gold in top
    return {'qa_id':q['qa_id'],'category':q['category'],'cell':cell,'scope_id':scope,'gold_memory_id':gold,'projection_ok':int(projection_ok),'candidate_count':len(candidate_ids),'candidate_parity':int(candidate_ids==sorted(r.memory_id for r in byscope[scope])),'commit_ms':(tc-t0)/1e6,'rawerk_ms':(tr-t0)/1e6,'sparse_index_ms':(ts-t0)/1e6,'dense_index_ms':(td-t0)/1e6,'backend_fetch_ms':(tf0-td)/1e6,'fusion_ms':(tt-tf0)/1e6,'time_to_top10_ms':(tt-t0)/1e6 if hit else '','fresh_hit_at_10':int(hit),'gold_rank':top.index(gold)+1 if hit else '','top10_ids':json.dumps(top)}
   with ThreadPoolExecutor(max_workers=4) as ex: out.extend(ex.map(run_cell,cells))
   if (qi+1)%100==0:print(f'completed_questions={qi+1}/{len(eligible)}',flush=True)
  OUT.mkdir(parents=True,exist_ok=True); fields=list(out[0]);
  with (OUT/'freshness_events.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
  summary={}
  for cell in cells:
   rr=[x for x in out if x['cell']==cell];lat=[float(x['time_to_top10_ms']) for x in rr if x['time_to_top10_ms']!='']
   summary[cell]={'events':len(rr),'projection_pass':sum(x['projection_ok'] for x in rr),'candidate_parity_pass':sum(x['candidate_parity'] for x in rr),'fresh_hit_at_10':sum(x['fresh_hit_at_10'] for x in rr)/len(rr),'time_to_top10_hit_p50_ms':pct(lat,50),'p95_ms':pct(lat,95),'p99_ms':pct(lat,99)}
  status='PASS' if all(v['events']==1533 and v['projection_pass']==1533 and v['candidate_parity_pass']==1533 for v in summary.values()) else 'FAIL'
  payload={'status':status,'question_count':len(eligible),'event_count':len(out),'summary':summary};(OUT/'freshness_gate_summary.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8');print(json.dumps(payload,indent=2))
 finally:cass.close();neo.close()
if __name__=='__main__':main()
