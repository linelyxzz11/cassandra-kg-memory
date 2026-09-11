import os
import tempfile
import unittest
from pathlib import Path

from p5_1_protocol import LogicalEvent, bm25_rank, logical_event_digest, materialize_event
from run_p5_1_v3 import aggregate_configurations, load_local_env


class ProtocolTests(unittest.TestCase):
    def test_unique_probe_token_puts_target_first(self):
        documents = [
            ("m1", "E: alex R: visited K: paris"),
            ("m2", "E: alex R: visited K: london P: p5v3_probe_42"),
            ("m3", "E: alex R: visited K: tokyo"),
        ]
        ranked = bm25_rank("p5v3_probe_42 alex visited", documents, top_k=3)
        self.assertEqual(ranked[0][0], "m2")

    def test_tie_break_is_memory_id_ascending(self):
        ranked = bm25_rank("absent", [("b", "x"), ("a", "x")], top_k=2)
        self.assertEqual([item[0] for item in ranked], ["a", "b"])

    def test_materialized_event_has_collision_free_edge_and_probe(self):
        source = {
            "update_id": "evt_1",
            "memory_id": "mem_1",
            "version": 1,
            "raw_text": "raw",
            "raw_erk_text": "raw erk",
            "src_id": "alice",
            "dst_id": "bob",
            "relation": "knows",
        }
        event = materialize_event(source, "scope", "formal", 7)
        self.assertIn("formal_00000007", event.edge_id)
        self.assertIn(event.probe_token, event.raw_erk_text)
        self.assertEqual(event.run_scope, "scope")

    def test_digest_changes_with_logical_event(self):
        base = LogicalEvent("u", "s", "m", 1, "r", "e", "a", "b", "rel", "edge", "probe")
        changed = LogicalEvent("u", "s", "m", 2, "r", "e", "a", "b", "rel", "edge", "probe")
        self.assertNotEqual(logical_event_digest([base]), logical_event_digest([changed]))

    def test_aggregate_counts_and_percentiles(self):
        rows = [
            {"backend": "cassandra", "concurrency": 2, "status": "ok", "update_to_topk_ms": 10.0, "update_to_commit_ms": 5.0, "candidate_count": 11, "target_rank": 1},
            {"backend": "cassandra", "concurrency": 2, "status": "ok", "update_to_topk_ms": 20.0, "update_to_commit_ms": 8.0, "candidate_count": 12, "target_rank": 1},
            {"backend": "cassandra", "concurrency": 2, "status": "timeout", "update_to_topk_ms": 99.0, "update_to_commit_ms": 9.0, "candidate_count": 12, "target_rank": -1},
        ]
        summary = aggregate_configurations(rows)[0]
        self.assertEqual(summary["formal_events"], 3)
        self.assertEqual(summary["successful_events"], 2)
        self.assertEqual(summary["timeouts"], 1)
        self.assertEqual(summary["p50_update_to_topk_ms"], 10.0)

    def test_local_env_does_not_override_process_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("P5V3_TEST_SECRET=file-value\nP5V3_NEW_VALUE=new-value\n", encoding="utf-8")
            os.environ["P5V3_TEST_SECRET"] = "process-value"
            os.environ.pop("P5V3_NEW_VALUE", None)
            load_local_env(path)
            self.assertEqual(os.environ["P5V3_TEST_SECRET"], "process-value")
            self.assertEqual(os.environ["P5V3_NEW_VALUE"], "new-value")
            os.environ.pop("P5V3_TEST_SECRET", None)
            os.environ.pop("P5V3_NEW_VALUE", None)


if __name__ == "__main__":
    unittest.main()
