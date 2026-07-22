#!/usr/bin/env python3
import argparse, hashlib, json, os, random, time
from pathlib import Path
import pandas as pd, requests

def h(s): return hashlib.sha256(str(s).encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args(); cp=Path(a.config).resolve(); cfg=json.loads(cp.read_text(encoding='utf-8')); base=cp.parent
    def p(v):
        x=Path(v); return x if x.is_absolute() else (base/x).resolve()
    df=pd.read_csv(p(cfg['input'])); c=cfg['columns']; api=cfg['api']; url=(api.get('base_url') or os.environ[api.get('base_url_env','LLM_BASE_URL')]).rstrip('/')+api.get('endpoint','/chat/completions'); key=api.get('api_key') or os.environ[api.get('api_key_env','LLM_API_KEY')]; model=api.get('model') or os.environ[api.get('model_env','LLM_MODEL')]
    tmpl=cfg['prompt_template']; ph=h(tmpl); cachep=p(cfg.get('cache','summary_cache.jsonl')); cache={}
    if cachep.exists():
        for line in cachep.read_text(encoding='utf-8').splitlines():
            if line.strip(): x=json.loads(line); cache[x['memory_id']]=x
    s=requests.Session(); done=[]
    for i,row in enumerate(df.to_dict('records'),1):
        mid=str(row[c['memory_id']])
        if mid in cache: done.append(cache[mid]); continue
        prompt=tmpl.format(speaker=str(row.get(c.get('speaker'),'')),timestamp=str(row.get(c.get('timestamp'),'')),text=str(row[c['raw_text']]))
        body={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':int(api.get('max_tokens',80))}
        for attempt in range(int(api.get('max_retries',6))+1):
            try:
                r=s.post(url,headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},json=body,timeout=float(api.get('timeout_seconds',120))); r.raise_for_status(); summary=r.json()['choices'][0]['message']['content'].strip(); x={'memory_id':mid,'summary':summary,'model':model,'prompt_sha256':ph,'source_sha256':h(row[c['raw_text']])}; cachep.parent.mkdir(parents=True,exist_ok=True)
                with cachep.open('a',encoding='utf-8') as f: f.write(json.dumps(x,ensure_ascii=False)+'\n')
                cache[mid]=x; done.append(x); break
            except Exception:
                if attempt>=int(api.get('max_retries',6)): raise
                time.sleep(min(60,2**attempt+random.random()))
        if i%100==0: print(f'{i}/{len(df)}')
    op=p(cfg['output']); op.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(done).to_csv(op,index=False); print(f'Wrote {len(done)} summaries: {op}')
if __name__=='__main__': main()
