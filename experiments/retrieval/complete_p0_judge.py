"""Resumable, explicitly authorized P0 Judge completion via user-selected relay.

Only pending exact prompts are submitted. Secrets are read from environment or
the repository .env and never printed or saved. Historical caches are read-only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/p0_audit_20261012'
OUT=BASE/'judge_completion_v2_uiuihao'
ENDPOINT='https://api.uiuihao.com/v1/chat/completions'
MODEL='gpt-4o-2024-08-06'
HALT=threading.Event()

def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def rows(path):
    if not path.exists():return []
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
def save(name,data):
    # Same-directory atomic checkpoint; only this script's generated files.
    temp=OUT/(name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(temp,OUT/name)
def append(name,row):
    with (OUT/name).open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def key():
    value=os.environ.get('OPENAI_API_KEY','').strip()
    if not value:
        p=ROOT/'.env'
        if p.exists():
            for line in p.read_text(encoding='utf-8-sig').splitlines():
                if '=' not in line or line.lstrip().startswith('#'):continue
                name,value0=line.split('=',1)
                if name.strip()=='OPENAI_API_KEY':value=value0.strip().strip('"\'');break
    if not value or any(s in value.lower() for s in ['your_api','your-api','replace_me','placeholder']):
        raise RuntimeError('No configured usable OPENAI_API_KEY; value never logged.')
    return value

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        raise RuntimeError('HTTP redirect refused to protect Authorization header')

def call(item,secret):
    h,prompt=item
    body=json.dumps(dict(model=MODEL,messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=80,response_format={'type':'json_object'}),ensure_ascii=False).encode()
    errors=[]
    for attempt in range(1,4):
        if HALT.is_set():raise RuntimeError('cancelled_after_fatal_error')
        start=time.perf_counter()
        try:
            req=urllib.request.Request(ENDPOINT,data=body,headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'},method='POST')
            opener=urllib.request.build_opener(NoRedirect())
            with opener.open(req,timeout=60) as response:
                result=json.loads(response.read().decode('utf-8'))
            if result.get('model')!=MODEL:
                HALT.set();raise RuntimeError('returned_model_mismatch')
            raw=str(result['choices'][0]['message']['content'] or '').strip()
            parsed=json.loads(raw)
            label=parsed.get('label')
            if label not in {'CORRECT','WRONG'}:raise ValueError('invalid_label')
            return dict(judge_key=h,label=label,raw_response=raw,model_returned=result['model'],
                        usage=result.get('usage') or {},attempts=attempt,latency_seconds=round(time.perf_counter()-start,4),
                        endpoint=ENDPOINT,judged_at_utc=utc(),transient_errors=errors)
        except urllib.error.HTTPError as exc:
            # Never log server response bodies: they may contain reflected secrets.
            code=exc.code;errors.append('HTTP_'+str(code))
            if code in {400,401,402,403,404}:
                HALT.set();raise RuntimeError('HTTP_'+str(code)) from None
        except Exception as exc:
            name=str(exc) if str(exc) in {'returned_model_mismatch','invalid_label'} else type(exc).__name__
            errors.append(name)
            if HALT.is_set():raise RuntimeError(name) from None
        if attempt<3:time.sleep(2**attempt)
    raise RuntimeError('request_failed:'+','.join(errors))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers<1 or args.workers>4:raise ValueError('workers must be 1..4')
    OUT.mkdir(parents=True,exist_ok=True)
    prompts=read(BASE/'judge_completion_pending_prompts.json')
    mapping=read(BASE/'judge_completion_row_map.json')
    preflight=read(BASE/'judge_completion_preflight.json')
    original=ROOT.parent/'tmp/hipporag2-reader-full5-eval/judge_unique.jsonl'
    inputs={str(p):sha(p) for p in [BASE/'judge_completion_pending_prompts.json',BASE/'judge_completion_row_map.json',original]}
    contract=dict(endpoint=ENDPOINT,model=MODEL,temperature=0,max_tokens=80,response_format={'type':'json_object'},
                  prompt_template_sha256=preflight['prompt_template_sha256'],inputs_sha256=inputs)
    if (OUT/'contract.json').exists():
        if read(OUT/'contract.json')!=contract:raise RuntimeError('resume_contract_changed')
    else:save('contract.json',contract)
    cache=rows(OUT/'judgments_new.jsonl')
    if len({r['judge_key'] for r in cache})!=len(cache):raise RuntimeError('duplicate_saved_judgment')
    for r in cache:
        if r['judge_key'] not in prompts or r['model_returned']!=MODEL or r['label'] not in {'CORRECT','WRONG'}:raise RuntimeError('invalid_saved_judgment')
    completed={r['judge_key'] for r in cache}
    pending=[(h,p) for h,p in sorted(prompts.items()) if h not in completed]
    for h,p in pending:
        if hashlib.sha256(p.encode()).hexdigest()!=h:raise RuntimeError('pending_prompt_hash_mismatch')
    pending=pending[:args.limit] if args.limit else pending
    secret=key() if pending else ''
    started=utc();count=0;failed=[];last=time.monotonic()
    def checkpoint(status):
        current=rows(OUT/'judgments_new.jsonl')
        usage={k:sum(int(r.get('usage',{}).get(k,0) or 0) for r in current) for k in ['prompt_tokens','completion_tokens','total_tokens']}
        data=dict(status=status,started_at_utc=started,updated_at_utc=utc(),required_new_unique=len(prompts),
                  completed_new_unique=len(current),remaining=len(prompts)-len(current),workers=args.workers,
                  successful_this_invocation=count,failed_this_invocation=len(failed),usage_successful_requests=usage,
                  original_files_unchanged=all(sha(Path(p))==h for p,h in inputs.items()))
        save('status.json',data)
        print(json.dumps(data),flush=True)
    checkpoint('running')
    iterator=iter(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active={}
        for _ in range(args.workers):
            item=next(iterator,None)
            if item:active[pool.submit(call,item,secret)]=item[0]
        while active:
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for fut in done:
                h=active.pop(fut)
                try:
                    row=fut.result();append('judgments_new.jsonl',row);count+=1
                except Exception as exc:
                    error=dict(judge_key=h,error=str(exc),at_utc=utc());failed.append(error);append('errors.jsonl',error)
                if not HALT.is_set():
                    item=next(iterator,None)
                    if item:active[pool.submit(call,item,secret)]=item[0]
            if time.monotonic()-last>=30:
                checkpoint('running' if not HALT.is_set() else 'halting');last=time.monotonic()
    total=len(rows(OUT/'judgments_new.jsonl'))
    checkpoint('complete' if total==len(prompts) else 'stopped_on_error' if failed else 'partial_limit')
    if failed:raise SystemExit(2)

if __name__=='__main__':
    try:main()
    except RuntimeError as exc:
        print('STOP: '+str(exc),flush=True);raise SystemExit(2)
