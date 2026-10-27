import unittest
from formal_trial import phase_class,score,np,ScopedOnlineRetrievalIndex

class FormalTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(phase_class(20,30,0,10),'post_ready')
        self.assertEqual(phase_class(5,15,0,10),'crosses_ready')
        self.assertEqual(phase_class(1,5,0,10),'pre_ready')
        self.assertEqual(phase_class(1,5,2,0),'before_arrival')
    def test_scoring_identical(self):
        idx=ScopedOnlineRetrievalIndex();v=np.eye(3,dtype=np.float32)
        idx.load_scope('s',{'a':'apples banana','b':'orange citrus','c':'apple tree'},{m:v[i] for i,m in enumerate('abc')})
        expected=idx.search('s','apple citrus',v[0],['b','c'])
        actual,sr,dr=score(idx._sparse['s'],idx._dense['s'],'apple citrus',v[0],['b','c'])
        self.assertEqual(tuple(actual),expected.top10)
        self.assertEqual(tuple(sr),expected.sparse_top50)
        self.assertEqual(tuple(dr),expected.dense_top50)

if __name__=='__main__':unittest.main()
