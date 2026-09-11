"""Load 5,882 canonical memories into four live cells and verify digests."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from live_cells import CassandraCells,Neo4jCells,Record,digest_projection

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'05_reports'/'locomo_workload_canonical_gate'
def env():
 p=ROOT/'.env'
 if p.exists():
  for line in p.read_text(encoding='utf-8-sig').splitlines():
   if line.strip() and not line.lstrip().startswith('#') and '=' in line:
    k,v=line.split('=',1);os.environ.setdefault(k.strip(),v.strip().strip('"').strip("'"))
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def build():
 mem=rows(ROOT/'01_data/locomo_memory_records.csv'); feat={x['memory_id']:x for x in rows(ROOT/'02_artifacts/p3_memory_features.csv')}
 ids=(ROOT/'01_data/locomo_memory_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines(); vec=np.load(ROOT/'01_data/locomo_memory_bge_large.npy',mmap_mode='r')
 vh={mid:hashlib.sha256(np.asarray(vec[i],dtype=np.float32).tobytes()).hexdigest() for i,mid in enumerate(ids)}
 return [Record(x['sample_id'],x['memory_id'],1,x['text'],feat[x['memory_id']]['entities'],feat[x['memory_id']]['relations'],feat[x['memory_id']]['keywords'],feat[x['memory_id']]['triples'],vh[x['memory_id']]) for x in mem]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reset',action='store_true');a=ap.parse_args();env(); records=build(); expected={r.memory_id:digest_projection(r.projection()) for r in records}
 cass=CassandraCells(os.getenv('CASSANDRA_HOST','127.0.0.1')); neo=Neo4jCells(os.getenv('NEO4J_URI','bolt://localhost:7687'),os.getenv('NEO4J_USER','neo4j'),os.getenv('NEO4J_PASSWORD'),os.getenv('NEO4J_DATABASE','neo4j'))
 OUT.mkdir(parents=True,exist_ok=True); start=time.time()
 try:
  if a.reset:cass.reset();neo.reset()
  for cell in cass.names:
   with ThreadPoolExecutor(max_workers=24) as ex:list(ex.map(lambda r:cass.insert(cell,r),records))
  for cell in neo.names:neo.insert_many(cell,records)
  counts={}
  scopes=sorted({r.scope_id for r in records})
  for cell in cass.names+neo.names:
   adapter=cass if cell.startswith('cassandra') else neo
   counts[cell]=sum(len(adapter.ids(cell,s)) for s in scopes)
  diffs=[]
  for i,r in enumerate(records):
   for cell in cass.names+neo.names:
    adapter=cass if cell.startswith('cassandra') else neo; p=adapter.fetch(cell,r.scope_id,r.memory_id); observed=digest_projection(p) if p else ''
    if observed!=expected[r.memory_id]:diffs.append({'memory_id':r.memory_id,'cell':cell,'expected':expected[r.memory_id],'observed':observed})
  summary={'status':'PASS' if all(v==5882 for v in counts.values()) and not diffs else 'FAIL','expected_count':5882,'counts':counts,'projection_comparisons':5882*4,'mismatch_count':len(diffs),'elapsed_s':time.time()-start}
  (OUT/'canonical_gate_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
  with (OUT/'canonical_projection_diffs.csv').open('w',encoding='utf-8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['memory_id','cell','expected','observed']);w.writeheader();w.writerows(diffs)
  print(json.dumps(summary,indent=2))
  if summary['status']!='PASS':raise SystemExit(2)
 finally:cass.close();neo.close()
if __name__=='__main__':main()
