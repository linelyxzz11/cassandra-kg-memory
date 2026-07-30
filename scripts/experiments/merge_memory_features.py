#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

def load(p):
    if p.suffix.lower()=='.csv': return pd.read_csv(p)
    if p.suffix.lower() in {'.jsonl','.ndjson'}: return pd.read_json(p,lines=True)
    if p.suffix.lower()=='.parquet': return pd.read_parquet(p)
    raise ValueError(p)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args(); cp=Path(a.config).resolve(); cfg=json.loads(cp.read_text(encoding='utf-8')); base=cp.parent
    def p(v):
        x=Path(v); return x if x.is_absolute() else (base/x).resolve()
    r=cfg['raw']; out=load(p(r['path'])).rename(columns={r['memory_id']:'memory_id',r['conversation_id']:'conversation_id',r['raw_text']:'raw_text'})[['memory_id','conversation_id','raw_text']]
    out.memory_id=out.memory_id.astype(str)
    if out.memory_id.duplicated().any(): raise ValueError('duplicate raw memory_id')
    for section,cols in [('summary',['summary']),('erk',['entities','relations','keywords']),('triples',['triples'])]:
        sc=cfg.get(section)
        if not sc:
            for c in cols: out[c]=''
            continue
        df=load(p(sc['path'])); ren={sc['memory_id']:'memory_id'}; ren.update({sc[c]:c for c in cols}); df=df.rename(columns=ren)[['memory_id']+cols]; df.memory_id=df.memory_id.astype(str)
        if df.memory_id.duplicated().any(): raise ValueError(f'duplicate {section} memory_id')
        out=out.merge(df,on='memory_id',how='left',validate='one_to_one')
    for c in ['summary','entities','relations','keywords','triples']:
        if c not in out: out[c]=''
        out[c]=out[c].fillna('')
    op=p(cfg['output']); op.parent.mkdir(parents=True,exist_ok=True); out.to_csv(op,index=False); print(f'Wrote {len(out)} rows: {op}')
if __name__=='__main__': main()
