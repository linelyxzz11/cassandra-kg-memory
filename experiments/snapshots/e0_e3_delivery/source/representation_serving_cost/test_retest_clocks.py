import unittest
from freshness_retest import clocks


class ClockTests(unittest.TestCase):
    def row(self):
        return {"arrival_ns":0,"submit_ns":100_000_000,"worker_start_ns":150_000_000,
                "raw_ack_ns":200_000_000,"ready_checked_ns":600_000_000,"ready":True,"op_type":"update"}

    def test_separate_boundaries(self):
        r=clocks(self.row(),1_000_000_000)
        self.assertEqual(r["queue_wait_ms"],50)
        self.assertEqual(r["planned_to_worker_ms"],150)
        self.assertEqual(r["arrival_to_retrievable_ms"],500)
        self.assertEqual(r["planned_arrival_to_retrievable_ms"],600)
        self.assertTrue(r["ready_within_500ms"])
        self.assertFalse(r["planned_ready_within_500ms"])

    def test_late_completion(self):
        self.assertTrue(clocks(self.row(),599_999_999)["ready_censored_at_deadline"])
        self.assertFalse(clocks(self.row(),600_000_000)["ready_censored_at_deadline"])

    def test_failed_check_is_censored(self):
        r=self.row(); r["ready"]=False
        self.assertTrue(clocks(r,1_000_000_000)["ready_censored_at_deadline"])


if __name__=="__main__":
    unittest.main()
