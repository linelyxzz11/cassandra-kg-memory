import unittest
from collections import namedtuple
def Row(**kw):return namedtuple('Row',kw.keys())(**kw)
import campaign as C
from cassmem.backend.live_cells_graph import CassandraGraphCells

class Rows:
    def __init__(self,rs):self.rs=rs
    def __iter__(self):return iter(self.rs)
    def one(self):return self.rs[0] if self.rs else None

class FakeSession:
    def __init__(self,r):self.r=r;self.batches=[]
    def execute(self,q,*args,**kw):
        if hasattr(q,'_statements_and_parameters'):
            self.batches.append(q);return Rows([])
        if 'SELECT entity_id' in q:return Rows([Row(entity_id=e) for e in self.r.mentions])
        if 'SELECT ordinal' in q:return Rows([Row(**e) for e in self.r.edges])
        if 'SELECT rawerk' in q:return Rows([Row(rawerk=self.r.rawerk)])
        if 'SELECT raw_text' in q:return Rows([Row(raw_text=self.r.raw_text)])
        if 'SELECT entities' in q:return Rows([Row(entities=self.r.entities,relations=self.r.relations,keywords=self.r.keywords)])
        if 'SELECT memory_id' in q:
            return Rows([Row(memory_id=self.r.memory_id,relation=e['relation']) for e in self.r.edges])
        return Rows([])

class Tests(unittest.TestCase):
    def setUp(self):
        self.r=C.GraphRecord('scope','mid',1,'original','A, B, C','knows, visits','key',
                             '(A, knows, B); (A, visits, C)','vector-hash')
    def backend(self):
        b=object.__new__(CassandraGraphCells);b.s=FakeSession(self.r)
        # Real BatchStatement with unprepared INSERT fixtures, matching each arity.
        sizes={'bf':6,'be':2,'bmention':3,'bedgem':6,'bedges':6,'bedgescope':6,
               'mm':10,'me':2,'mmention':3,'medgem':6,'mrel':6}
        b.p={k:'INSERT INTO test VALUES ('+','.join(['%s']*n)+')' for k,n in sizes.items()}
        return b
    def test_fixture_control(self):
        self.assertTrue(C.eligible(self.r))
        for kind in C.KINDS:
            nr=C.mutate(self.r,kind)
            self.assertEqual(nr.raw_text,self.r.raw_text)
            self.assertEqual(nr.embedding_sha256,self.r.embedding_sha256)
            self.assertEqual(nr.graph_entities,self.r.graph_entities)
            self.assertEqual(nr.version,2)
    def test_batch_statements(self):
        for cell in C.CELLS:
            for kind in C.KINDS:
                b=self.backend();nr=C.mutate(self.r,kind)
                _,event=C.timed(b,lambda:b.write_structured(cell,nr))
                expected=C.expected_batch(self.r,nr,cell)
                calls=event['probe']['calls']
                self.assertEqual(len(calls),3)
                self.assertEqual(calls[-1]['batch_statements'],expected['total'])
                self.assertEqual(calls[-1]['delete_statements'],expected['deletes'])
    def test_projection_calls(self):
        for cell,n in zip(C.CELLS,[2,1]):
            b=self.backend();value,event=C.timed(b,lambda:b.candidate_projection(cell,'scope','mid'))
            self.assertEqual(value,self.r.rawerk);self.assertEqual(event['probe']['cql_calls'],n)
    def test_session_restored_on_failure(self):
        b=self.backend();original=b.s
        with self.assertRaises(ZeroDivisionError):C.timed(b,lambda:1/0)
        self.assertIs(b.s,original)
    def test_negative_checks(self):
        for a,z in [([],['a']),(['a','b'],['a']),(['b','a'],['a','b']),('old','new')]:
            with self.assertRaises(AssertionError):C.require_equal(a,z,'bad')

if __name__=='__main__':unittest.main(verbosity=2)
