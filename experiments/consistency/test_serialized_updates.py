import sys,unittest,hashlib,threading,dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
from cassmem.serving.serialized_updates import SerializedUpdateService,VersionConflict
from cassmem.representation.graph_event import GraphRecord
from cassmem.retrieval.online import ScopedOnlineRetrievalIndex

class Backend:
    def __init__(self):self.records={};self.calls=[]
    def commit_raw(self,cell,r):self.calls.append(('raw',r.version));self.records[(r.scope_id,r.memory_id)]=r
    def write_structured(self,cell,r):self.calls.append(('structured',r.version));self.records[(r.scope_id,r.memory_id)]=r
    def fetch_graph(self,cell,s,m):return self.records[(s,m)].graph_projection()
    def candidate_projection(self,cell,s,m):return self.records[(s,m)].rawerk
    def ids(self,cell,s):return [m for sc,m in self.records if sc==s]
    def ids_by_relation(self,cell,s,rel):return [m for (sc,m),r in self.records.items() if sc==s and any(e['relation']==rel for e in r.edges)]
def record(v,scope='s',mid='m'):
    vec=np.array([v,4-v],dtype=np.float32);r=GraphRecord(scope,mid,v,f'Alice visits city{v}',f'Alice, city{v}',f'visit{v}','city',f'(Alice, visit{v}, city{v})',hashlib.sha256(vec.tobytes()).hexdigest())
    return r,vec

class Tests(unittest.TestCase):
    def setUp(self):
        self.b=Backend();self.i=ScopedOnlineRetrievalIndex();self.s=SerializedUpdateService(self.b,'test',self.i);self.seed('s','m')
    def seed(self,scope,mid):
        r,v=record(1,scope,mid);self.b.records[(scope,mid)]=r
        if scope in self.i._documents:
            docs={**self.i._documents[scope],mid:r.rawerk};emb={**self.i._embeddings[scope],mid:v}
        else:docs={mid:r.rawerk};emb={mid:v}
        self.i.load_scope(scope,docs,emb);self.s.adopt(r,v)
    def test_reverse_duplicate_conflict(self):
        self.assertEqual(self.s.apply(*record(3))['status'],'applied');n=len(self.b.calls)
        self.assertEqual(self.s.apply(*record(2))['status'],'stale')
        self.assertEqual(self.s.apply(*record(3))['status'],'duplicate')
        r,v=record(3)
        with self.assertRaises(VersionConflict):self.s.apply(dataclasses.replace(r,raw_text='conflicting'),v)
        self.assertEqual(len(self.b.calls),n)
    def test_every_stage_failure_retries(self):
        for stage in ('raw','structured','canonical_check','sparse','dense','ready'):
            with self.subTest(stage=stage):
                self.setUp()
                def hook(event,current,ns):
                    if event=='after' and current==stage:raise RuntimeError('injected')
                with self.assertRaises(RuntimeError):self.s.apply(*record(3),hook=hook)
                n=len(self.b.calls);self.assertEqual(self.s.apply(*record(2))['status'],'stale');self.assertEqual(len(self.b.calls),n)
                self.assertEqual(self.s.apply(*record(3))['status'],'applied');self.assertTrue(self.s.verify(*record(3)))
    def test_same_id_waits_other_keys_progress(self):
        self.seed('s','other');self.seed('other_scope','m');hit=threading.Event();release=threading.Event();waiting=threading.Event();entered=threading.Event()
        def h2(event,stage,ns):
            if event=='before' and stage=='structured':hit.set();self.assertTrue(release.wait(5))
        def h3(event,stage,ns):
            if event=='waiting':waiting.set()
            if event=='before':entered.set()
        with ThreadPoolExecutor(4) as p:
            f2=p.submit(self.s.apply,*record(2),hook=h2)
            try:
                self.assertTrue(hit.wait(5));f3=p.submit(self.s.apply,*record(3),hook=h3);self.assertTrue(waiting.wait(5));self.assertFalse(entered.is_set())
                self.assertEqual(p.submit(self.s.apply,*record(2,mid='other')).result(5)['status'],'applied')
                self.assertEqual(p.submit(self.s.apply,*record(2,scope='other_scope')).result(5)['status'],'applied')
            finally:release.set()
            f2.result(5);f3.result(5)
        self.assertTrue(self.s.verify(*record(3)))
    def test_unadopted_fails_closed(self):
        with self.assertRaises(RuntimeError):self.s.apply(*record(2,mid='missing'))
        self.assertEqual(self.b.calls,[])
    def test_bad_vector_no_write(self):
        r,v=record(2)
        with self.assertRaises(ValueError):self.s.apply(r,v+1)
        self.assertEqual(self.b.calls,[])
    def test_bootstrap_checks_content(self):
        with self.assertRaises(ValueError):SerializedUpdateService(self.b,'test',self.i).adopt(*record(3))
    def test_adopt_cannot_reset_watermark(self):
        self.s.apply(*record(3))
        with self.assertRaises(RuntimeError):self.s.adopt(*record(1))
    def test_wrong_dimension_no_write(self):
        r,_=record(2);v=np.ones(3,dtype=np.float32);r=dataclasses.replace(r,embedding_sha256=hashlib.sha256(v.tobytes()).hexdigest())
        with self.assertRaises(ValueError):self.s.apply(r,v)
        self.assertEqual(self.b.calls,[])

if __name__=='__main__':unittest.main(verbosity=2)
