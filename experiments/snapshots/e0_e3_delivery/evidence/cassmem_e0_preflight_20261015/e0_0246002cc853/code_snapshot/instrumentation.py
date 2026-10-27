"""Diagnostic scope-lock timings, no changes to index locking semantics."""
import threading,time
LOCAL=threading.local()
class TimedLock:
    def __init__(self,lock,scope):self.lock,self.scope=lock,scope;self.local=threading.local()
    def __enter__(self):
        start=time.perf_counter_ns();self.lock.acquire();acquired=time.perf_counter_ns()
        if not hasattr(self.local,'stack'):self.local.stack=[]
        self.local.stack.append((start,acquired));return self
    def __exit__(self,*args):
        start,acquired=self.local.stack.pop();self.lock.release();end=time.perf_counter_ns()
        row=getattr(LOCAL,'row',None)
        if row is not None:row.setdefault('scope_locks',[]).append(dict(scope=self.scope,wait_start_ns=start,acquired_ns=acquired,released_ns=end))

def classify(row,deadline):
    if row.get('worker_start_ns') is None:return 'queued_not_started'
    if row.get('response_finish_ns') is None:return 'started_unfinished'
    if row['response_finish_ns']>deadline:return 'completed_after_deadline'
    return 'completed_before_deadline'
