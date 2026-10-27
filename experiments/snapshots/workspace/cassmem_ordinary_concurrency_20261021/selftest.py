"""Positive/negative validation gates, excluded from measured workloads."""
import copy
import io
import threading
import time
import unittest
import support as S
import prepare
import validate as V

class FakeBackend:
    def __init__(self,r):self.record=r;self.calls=[]
    def fetch_graph(self,c,s,m):return self.record.graph_projection()
    def candidate_projection(self,c,s,m):return self.record.rawerk
    def ids(self,c,s):return [self.record.memory_id]
    def ids_by_relation(self,c,s,rel):return [self.record.memory_id] if rel in {e['relation'] for e in self.record.edges} else []
    def commit_raw(self,c,r):self.calls.append(('raw',r.version));self.record=r
    def write_structured(self,c,r):self.calls.append(('structured',r.version));self.record=r

def fixture(v):
    vec=S.np.array([v,7-v],dtype=S.np.float32)
    r=S.GraphRecord('scope','memory',v,f'Alice visits city {v}','Alice',f'visit{v}','city',f'(Alice, visit{v}, city{v})',S.hash_array(vec))
    return r,vec

class Tests(unittest.TestCase):
    def test_dense_hash_matches_row_normalization(self):
        rng=S.np.random.default_rng(19);a=rng.normal(size=(53,1024)).astype(S.np.float32)
        normalized=a/S.np.maximum(S.np.linalg.norm(a,axis=1,keepdims=True),1e-12)
        for i in range(len(a)):self.assertEqual(S.normalized_hash(a[i]),S.hash_array(normalized[i]))

    def test_both_pipelines_preserve_target_checks(self):
        for arm in ('uncoordinated','serialized'):
            with self.subTest(arm=arm):
                r,v=fixture(1);b=FakeBackend(r);index=S.ScopedOnlineRetrievalIndex();index.load_scope(r.scope_id,{r.memory_id:r.rawerk},{r.memory_id:v})
                svc=S.SerializedUpdateService(b,S.CELL,index) if arm=='serialized' else S.UncoordinatedService(b,index)
                svc.adopt(r,v);hooks=[]
                new,nv=fixture(2);self.assertEqual(svc.apply(new,nv,hook=lambda *x:hooks.append(x))['status'],'applied')
                self.assertEqual([x[1] for x in hooks if x[0]=='before'],['raw','structured','canonical_check','sparse','dense','ready'])
                obs=S.capture(b,index,('scope','memory'),{'visit1','visit2'})
                wanted=S.expected_payload(new,nv)
                self.assertTrue(all(V.checkpoint_matches(obs,wanted).values()))
                bad=copy.deepcopy(obs);bad['dense_version']=1
                self.assertFalse(all(V.checkpoint_matches(bad,wanted).values()))
                bad=copy.deepcopy(obs);bad['relation_memberships']['visit1']=True
                self.assertFalse(all(V.checkpoint_matches(bad,wanted).values()))
                bad=copy.deepcopy(obs);bad['dense_vector_sha256']='bad'
                self.assertFalse(all(V.checkpoint_matches(bad,wanted).values()))
                if arm=='serialized':
                    self.assertEqual(svc.apply(*fixture(1))['status'],'stale')
                    self.assertEqual(svc.apply(*fixture(2))['status'],'duplicate')

    def test_uncoordinated_retains_canonical_guard(self):
        r,v=fixture(1);b=FakeBackend(r);i=S.ScopedOnlineRetrievalIndex();i.load_scope('scope',{'memory':r.rawerk},{'memory':v})
        svc=S.UncoordinatedService(b,i);svc.adopt(r,v)
        def sabotage(event,stage,ns):
            if event=='after' and stage=='structured':b.record=fixture(3)[0]
        with self.assertRaises(AssertionError):svc.apply(*fixture(2),hook=sabotage)
        self.assertEqual(i._sparse_versions[('scope','memory')],1)

    def test_frozen_schedule_properties(self):
        traces=S.load(S.RUN/'schedules.json')
        self.assertEqual(len(traces),9)
        for name,trace in traces.items():
            self.assertEqual(len(trace),180);versions={}
            for t in trace:
                self.assertEqual(t['version'],versions.get(t['memory_id'],1)+1);versions[t['memory_id']]=t['version']
            if name.startswith('disjoint'):self.assertEqual(len(versions),180)
            else:self.assertEqual(len(versions),16)
            if name.startswith('hotspot'):
                self.assertEqual(max(S.collections.Counter(t['memory_id'] for t in trace).values()),144)
        matrix=S.load(S.RUN/'formal_matrix.json');self.assertEqual(len(matrix),18)
        for a,b in zip(matrix[::2],matrix[1::2]):
            self.assertEqual(a['trace'],b['trace']);self.assertNotEqual(a['arm'],b['arm'])

    def test_overlap_distinguishes_keys_and_mutations(self):
        rows=[dict(scope='s',memory_id='a',update_index=0,service_start_ns=0,service_finish_ns=10,raw_start_ns=0,dense_done_ns=5),
              dict(scope='s',memory_id='a',update_index=1,service_start_ns=2,service_finish_ns=20,raw_start_ns=10,dense_done_ns=15),
              dict(scope='s',memory_id='b',update_index=2,service_start_ns=2,service_finish_ns=8,raw_start_ns=3,dense_done_ns=6)]
        self.assertEqual(V.overlap(rows,'service_start_ns','service_finish_ns')['updates_in_same_key_overlap'],2)
        self.assertEqual(V.overlap(rows,'raw_start_ns','dense_done_ns')['updates_in_same_key_overlap'],0)

def ingress(name,count,rate,pause):
    dest=S.RUN/'ingress_tests'/name;dest.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter_ns()+1_000_000_000;deadline=start+int((count/rate+10)*1e9)
    p=S.subprocess.Popen([S.sys.executable,'-B',str(S.RECEIVER),'--start-ns',str(start),'--count',str(count),'--ops-rate',str(rate),'--log',str(dest/'emitter.jsonl')],stdout=S.subprocess.PIPE,stderr=S.subprocess.PIPE,text=True)
    rows=[]
    for line in p.stdout:
        i,planned,emitted,enqueued=map(int,line.strip().split(','));dispatch=time.perf_counter_ns();finished=time.perf_counter_ns()
        rows.append(dict(event_id=i,planned_arrival_ns=planned,emit_ns=emitted,actual_enqueue_ns=enqueued,execution_dispatch_ns=dispatch,
                         worker_start_ns=dispatch,response_finish_ns=finished,absolute_deadline_ns=deadline,completed_by_deadline=finished<=deadline))
        if i==0 and pause:time.sleep(pause)
    if p.wait(timeout=30):raise RuntimeError(p.stderr.read())
    p.stdout.close();p.stderr.close()
    rec=V.lines(dest/'receiver.jsonl');em=V.lines(dest/'emitter.jsonl');ctx=dict(operations=count,absolute_deadline_ns=deadline)
    assert not V.clocks(ctx,rows,rows,rec,em)
    lag=V.pct([(r['actual_enqueue_ns']-r['planned_arrival_ns'])/1e6 for r in rows]);assert lag<=50,lag
    max_dispatch=max((r['execution_dispatch_ns']-r['actual_enqueue_ns'])/1e6 for r in rows)
    if pause:assert max_dispatch>pause*1000-200
    bad=copy.deepcopy(rows);bad[0]['actual_enqueue_ns']+=1;assert V.clocks(ctx,bad,rows,rec,em)
    assert V.clocks(ctx,rows,rows,rec[:-1],em)
    bad=copy.deepcopy(rows);bad[0]['worker_start_ns']=bad[0]['actual_enqueue_ns']-1;assert V.clocks(ctx,bad,rows,rec,em)
    bad=copy.deepcopy(rows);bad[0]['completed_by_deadline']=not bad[0]['completed_by_deadline'];assert V.clocks(ctx,bad,rows,rec,em)
    S.save(dest/'execution.json',rows)
    return dict(name=name,count=count,rate=rate,pause_s=pause,receive_lag_p95_ms=lag,max_dispatch_lag_ms=max_dispatch,negative_controls=4)

def main():
    S.verify_manifest()
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    (S.RUN/'unit_tests.log').write_text(stream.getvalue(),encoding='utf-8');print(stream.getvalue(),flush=True)
    S.save(S.RUN/'unit_tests.json',dict(passed=result.wasSuccessful(),**{'pass':result.wasSuccessful()},tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)))
    if not result.wasSuccessful():raise RuntimeError('Unit gate failed')
    tests=[ingress('normal',60,20,0),ingress('stalled_consumer',100,20,3),ingress('backpressure',1000,500,3)]
    S.save(S.RUN/'ingress_tests.json',dict(**{'pass':True},tests=tests,negative_controls=12))
    S.verify_manifest();print(S.json.dumps(tests),flush=True)

if __name__=='__main__':main()
