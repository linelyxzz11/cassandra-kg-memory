import unittest,threading
from instrumentation import TimedLock,LOCAL,classify
class Tests(unittest.TestCase):
    def test_deadline_states(self):
        self.assertEqual(classify({},10),'queued_not_started')
        self.assertEqual(classify({'worker_start_ns':1},10),'started_unfinished')
        self.assertEqual(classify({'worker_start_ns':1,'response_finish_ns':10},10),'completed_before_deadline')
        self.assertEqual(classify({'worker_start_ns':1,'response_finish_ns':11},10),'completed_after_deadline')
    def test_recursive_lock_clock(self):
        LOCAL.row={};lock=TimedLock(threading.RLock(),'s')
        with lock:
            with lock:pass
        self.assertEqual(len(LOCAL.row['scope_locks']),2)
        for r in LOCAL.row['scope_locks']:self.assertLessEqual(r['wait_start_ns'],r['acquired_ns']);self.assertLessEqual(r['acquired_ns'],r['released_ns'])
        LOCAL.row=None
if __name__=='__main__':unittest.main(verbosity=2)
