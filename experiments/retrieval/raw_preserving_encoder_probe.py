"""Offline encoder provenance probe; no network or benchmark tuning."""
import csv, json, os, time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/retrieval/raw_preserving_stage1_20261012'
MODEL=Path('C:/Users/PC/.cache/huggingface/hub/models--BAAI--bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09')
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModel.from_pretrained(MODEL,local_files_only=True).eval()
    report={'model_path':str(MODEL),'torch':torch.__version__,'device':'cpu','threads':8,'checks':[]}
    for kind,source,idcol,arrfile,idsfile in [('memory','locomo_memory_records.csv','memory_id','locomo_memory_bge_large.npy','locomo_memory_ids_bge.txt'),('query','locomo_qa_records.csv','qa_id','locomo_qa_bge_large.npy','locomo_qa_ids_bge.txt')]:
        data=rows(ROOT/'data'/source); lookup={r[idcol]:r for r in data}
        ids=(ROOT/'data'/idsfile).read_text(encoding='utf-8-sig').splitlines()
        arr=np.load(ROOT/'data'/arrfile)
        indices=np.linspace(0,len(ids)-1,12,dtype=int)
        texts=[((lookup[ids[i]]['speaker']+': '+lookup[ids[i]]['text']) if kind=='memory' else lookup[ids[i]]['question'])[:1500] for i in indices]
        start=time.time()
        with torch.inference_mode():
            batch=tok(texts,padding=True,truncation=True,max_length=512,return_tensors='pt')
            hidden=model(**batch).last_hidden_state
            mask=batch['attention_mask'].unsqueeze(-1)
            pools={'cls':hidden[:,0],'mean':(hidden*mask).sum(1)/mask.sum(1)}
            target=arr[indices];target=target/np.linalg.norm(target,axis=1,keepdims=True)
            for name,vec in pools.items():
                v=torch.nn.functional.normalize(vec,p=2,dim=1).numpy()
                cos=(v*target).sum(1)
                report['checks'].append({'kind':kind,'pooling':name,'n':len(indices),'cos_min':float(cos.min()),'cos_mean':float(cos.mean()),'max_abs_error':float(np.abs(v-target).max()),'ids':[ids[i] for i in indices]})
        report[kind+'_probe_seconds']=time.time()-start
    (OUT/'encoder_probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
