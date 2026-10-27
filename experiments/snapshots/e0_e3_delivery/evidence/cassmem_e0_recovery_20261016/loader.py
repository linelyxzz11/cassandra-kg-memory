"""Pre-measurement only: one loader per scope, bounded idempotent retries."""
import collections,json,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
def load_scope_safe(backend,cell,records,log_path,workers=12,sleep=time.sleep):
    groups=collections.defaultdict(list)
    for r in records:groups[r.scope_id].append(r)
    lock=threading.Lock();retries=[];success=0
    def load_group(rs):
        nonlocal success
        for r in rs:
            for attempt in range(1,6):
                try:
                    backend.insert(cell,r)
                    with lock:success+=1
                    break
                except Exception as exc:
                    code=getattr(exc,'code',None)
                    entry=dict(scope=r.scope_id,memory_id=r.memory_id,attempt=attempt,code=code,type=type(exc).__name__,retry=code=='Neo.TransientError.Transaction.Outdated' and attempt<5)
                    with lock:
                        retries.append(entry)
                        with Path(log_path).open('a',encoding='utf-8') as h:h.write(json.dumps(entry)+'\n')
                    if not entry['retry']:raise
                    sleep(.1*2**(attempt-1))
    with ThreadPoolExecutor(workers) as pool:list(pool.map(load_group,groups.values()))
    assert success==len(records)
    return dict(success=success,retries=len(retries),scopes=len(groups))
