import csv
import tempfile
import unittest
from pathlib import Path

from c0bench.canonical import canonicalize_csv, load_canonical_edges
from c0bench.models import Edge, QuerySpec
from c0bench.semantics import ReferenceGraph


class C0SemanticsTests(unittest.TestCase):
    def setUp(self):
        self.edges = [
            Edge("g", "a", "likes", "c", "s2"),
            Edge("g", "a", "likes", "b", "s3"),
            Edge("g", "a", "likes", "b", "s1"),
            Edge("g", "b", "suitable_for", "d", "s1"),
            Edge("g", "c", "suitable_for", "d", "s1"),
            Edge("g", "d", "related_to", "a", "s1"),
        ]
        self.graph = ReferenceGraph(self.edges)

    def test_portable_sort_and_fanout(self):
        query = QuerySpec("q1", "g", "a", ("likes",), 1, 1)
        # same relation/destination => source s1 breaks the tie deterministically
        expected = ((self.edges[2].logical_id,),)
        self.assertEqual(self.graph.execute(query).normalized_paths(), expected)

    def test_multi_hop_and_cycle_filter(self):
        query = QuerySpec("q2", "g", "a", ("likes", "suitable_for"), 2, 5)
        paths = self.graph.execute(query).normalized_paths()
        self.assertEqual(len(paths), 3)
        query_cycle = QuerySpec("q3", "g", "a", ("likes", "suitable_for", "related_to"), 3, 5)
        self.assertEqual(self.graph.execute(query_cycle).normalized_paths(), tuple())

    def test_canonicalize_dedupes_exact_logical_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.csv"; out = Path(directory) / "canonical.csv"
            raw.write_text("graph_id,src_id,relation,dst_id,source\ng,a,likes,b,s\ng,a,likes,b,s\n", encoding="utf-8")
            edges, duplicate_count = canonicalize_csv(raw, out)
            self.assertEqual((len(edges), duplicate_count), (1, 1))
            self.assertEqual(len(load_canonical_edges(out)), 1)


if __name__ == "__main__":
    unittest.main()

class Neo4jQueryGenerationTests(unittest.TestCase):
    def test_cypher_uses_portable_source_tiebreaker_and_no_version_column(self):
        from c0bench.executors.neo4j import Neo4jExecutor
        executor = object.__new__(Neo4jExecutor)
        executor.config = {
            "node_label": "KGNode", "relationship_type": "KG_EDGE",
            "node_graph_property": "graph_id", "node_id_property": "node_id",
            "rel_graph_property": "graph_id", "rel_relation_property": "relation",
            "rel_source_property": "source",
        }
        query = QuerySpec("q", "g", "a", ("likes", "suitable_for"), 2, 20)
        cypher, params = executor._cypher(query)
        self.assertIn("coalesce(r0.`source`, '') ASC", cypher)
        self.assertIn("LIMIT $fanout", cypher)
        self.assertNotIn("visible_from_version", cypher)
        self.assertEqual(params["rel_0"], "likes")
