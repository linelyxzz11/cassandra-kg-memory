import unittest,tempfile,threading,time
from pathlib import Path
from types import SimpleNamespace
from loader import load_scope_safe
class Outdated(Exception):code='Neo.TransientError.Transaction.Outdated'
class Fake:
    def __init__(self,permanent=False):self.lock=threading.Lock();self.active=set();self.done=[];self.attempts={};self.max_parallel=0;self.permanent=permanent
    def insert(self,cell,r):
        with self.lock:
            if r.scope_id in self.active:raise AssertionError('same scope concurrency')
            self.active.add(r.scope_id);self.max_parallel=max(self.max_parallel,len(self.active));self.attempts[r.memory_id]=self.attempts.get(r.memory_id,0)+1
        try:
            time.sleep(.003)
            if self.permanent or self.attempts[r.memory_id]==1:raise Outdated('injected transient')
            self.done.append(r.memory_id)
        finally:
            with self.lock:self.active.remove(r.scope_id)
class Tests(unittest.TestCase):
    def test_scope_exclusion_and_retry(self):
        rs=[SimpleNamespace(scope_id=str(s),memory_id=f'{s}-{i}') for s in range(4) for i in range(5)]
        b=Fake()
        with tempfile.TemporaryDirectory() as d:r=load_scope_safe(b,'test',rs,Path(d)/'retry.jsonl',sleep=lambda _:None)
        self.assertEqual(r['success'],20);self.assertEqual(r['retries'],20);self.assertEqual(len(set(b.done)),20);self.assertGreater(b.max_parallel,1)
    def test_retry_cap(self):
        b=Fake(True)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Outdated):load_scope_safe(b,'test',[SimpleNamespace(scope_id='s',memory_id='m')],Path(d)/'retry.jsonl',sleep=lambda _:None)
        self.assertEqual(b.attempts['m'],5)
    def test_non_transient(self):
        class B:
            def insert(self,*a):raise ValueError('not retriable')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):load_scope_safe(B(),'test',[SimpleNamespace(scope_id='s',memory_id='m')],Path(d)/'retry.jsonl',sleep=lambda _:None)
if __name__=='__main__':unittest.main()
