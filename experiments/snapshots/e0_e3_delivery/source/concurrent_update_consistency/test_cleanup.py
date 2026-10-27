"""Regression tests for obsolete Cassandra keys; no live database required."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'representation_serving_cost'))
import common
from cassmem.backend.live_cells_graph import CassandraGraphCells
from cassmem.representation.graph_event import GraphRecord


class Batch:
    def __init__(self): self.calls=[]
    def add(self, query, params): self.calls.append((query,params))


class CleanupTests(unittest.TestCase):
    def clean(self, cell, triples='(Alice,newrel,Carol)', entities='Alice,Carol'):
        adapter=CassandraGraphCells.__new__(CassandraGraphCells)
        def execute(query, params):
            if 'SELECT entity_id' in query:
                return [SimpleNamespace(entity_id=x) for x in ('Alice','Bob')]
            return [SimpleNamespace(ordinal=0,src='Alice',relation='oldrel',dst='Bob'),
                    SimpleNamespace(ordinal=1,src='Bob',relation='extra',dst='Alice')]
        adapter.s=SimpleNamespace(execute=execute)
        batch=Batch()
        adapter._remove_obsolete_structure(batch,cell,GraphRecord('s','m',2,'text',entities,'','',triples,'hash'))
        return batch.calls

    def test_base_reused_ordinal_not_deleted(self):
        calls=self.clean('cassandra-base')
        self.assertFalse(any(('edges_by_memory' in q or 'edges_by_scope' in q) and p[-1]==0 for q,p in calls))
        self.assertTrue(any('edges_by_src' in q and 'oldrel' in p for q,p in calls))

    def test_removed_ordinal_deleted(self):
        for cell in ('cassandra-base','cassandra-materialized'):
            self.assertTrue(any('edges_by_memory' in q and p[-1]==1 for q,p in self.clean(cell)))

    def test_materialized_old_relation_deleted(self):
        self.assertTrue(any('by_scope_relation' in q and 'oldrel' in p for q,p in self.clean('cassandra-materialized')))

    def test_shared_entities_preserved(self):
        calls=self.clean('cassandra-base')
        self.assertFalse(any('entities_by_scope' in q for q,p in calls))
        self.assertTrue(any('mentions' in q and p[-1]=='Bob' for q,p in calls))
        self.assertFalse(any('mentions' in q and p[-1]=='Alice' for q,p in calls))

    def test_empty_target_removes_all_edges(self):
        for cell in ('cassandra-base','cassandra-materialized'):
            calls=self.clean(cell,'','')
            self.assertEqual(sum('edges_by_memory' in q for q,p in calls),2)
            self.assertEqual(sum('mentions_by_memory' in q for q,p in calls),2)

    def test_unchanged_edge_not_deleted(self):
        for cell in ('cassandra-base','cassandra-materialized'):
            calls=self.clean(cell,'(Alice,oldrel,Bob);(Bob,extra,Alice)','Alice,Bob')
            self.assertEqual(calls,[])


if __name__=='__main__': unittest.main()
