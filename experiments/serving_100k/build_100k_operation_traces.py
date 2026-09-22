"""Freeze deterministic 95:5 operation traces for the 100K main matrix."""
from __future__ import annotations
import csv,hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results'/'serving_100k'/'locomo_workload_100k'/'traces';SEED=20260803
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build(rep):
 qa=read(ROOT/'data/locomo_qa_records.csv'); eligible=read(ROOT/'results/serving_100k/locomo_workload_profile/freshness_eligible_cat1_4.csv')
 reads=[];updates=[]
 for r in range(17):
  pre=f'lr{r:06d}::'
  for q in qa:reads.append({'op_type':'read','qa_id':pre+q['qa_id'],'source_qa_id':q['qa_id'],'scope_id':pre+q['sample_id'],'question':q['question'],'category':q['category']})
  for q in eligible:
   gold=json.loads(q['gold_memory_ids'])[0];updates.append({'op_type':'update','qa_id':pre+q['qa_id'],'source_qa_id':q['qa_id'],'scope_id':pre+q['sample_id'],'question':q['question'],'category':q['category'],'target_memory_id':pre+gold,'source_memory_id':gold})
 # One logical arrival per target per run.
 unique={}
 for x in updates:unique.setdefault(x['target_memory_id'],x)
 updates=list(unique.values());rng=random.Random(SEED+rep);rng.shuffle(updates)
 def phase(name,nread,nupdate,offset):
  chosen_updates=updates[offset:offset+nupdate];ops=[dict(rng.choice(reads)) for _ in range(nread)]+[dict(x) for x in chosen_updates];rng.shuffle(ops)
  for i,x in enumerate(ops):x['op_id']=f'r{rep}_{name}_{i:05d}';x['phase']=name;x['ordinal']=i
  return ops
 warm=phase('warmup',475,25,0);measured=phase('measured',4750,250,25)
 return warm,measured
def main():
 OUT.mkdir(parents=True,exist_ok=True); manifest={'protocol_id':'locomo-100k-95r5u-v1','seed':SEED,'repetitions':[]}
 for rep in range(3):
  warm,measured=build(rep)
  paths=[]
  for name,data in [('warmup',warm),('measured',measured)]:
   p=OUT/f'rep{rep}_{name}.jsonl';p.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in data),encoding='utf-8');paths.append({'phase':name,'path':str(p.relative_to(ROOT)).replace('\\','/'),'rows':len(data),'sha256':digest(p),'reads':sum(x['op_type']=='read' for x in data),'updates':sum(x['op_type']=='update' for x in data),'unique_update_targets':len({x.get('target_memory_id') for x in data if x['op_type']=='update'})})
  manifest['repetitions'].append({'repetition':rep,'files':paths})
 (OUT/'trace_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
