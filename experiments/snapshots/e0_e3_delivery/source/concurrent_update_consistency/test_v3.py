import unittest,ast
from pathlib import Path
import formal_trial_v3 as v3
from neo4j_collected_v3 import Neo4jCollectedGraphCells

class V3Tests(unittest.TestCase):
    def test_sorted_ids_empty_and_nonempty(self):
        b=Neo4jCollectedGraphCells.__new__(Neo4jCollectedGraphCells)
        b.run=lambda *a,**kw:[{'ids':['z','a','m']}]
        self.assertEqual(b.ids('neo4j-native','s'),['a','m','z'])
        b.run=lambda *a,**kw:[{'ids':[]}]
        self.assertEqual(b.ids('neo4j-native','missing'),[])

    def test_ready_read_event_and_checkpoint_logic_unchanged(self):
        def methods(name):
            tree=ast.parse((Path(__file__).parent/name).read_text(encoding='utf-8'))
            return {n.name:ast.dump(n) for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Trial'
                    for n in c.body if isinstance(n,ast.FunctionDef)}
        old,new=methods('formal_trial.py'),methods('formal_trial_v3.py')
        for name in ('update','read','events','checkpoint','cleanup'):
            self.assertEqual(old[name],new[name],name)

    def test_scoring(self):
        idx=v3.ScopedOnlineRetrievalIndex();vectors=v3.np.eye(3,dtype=v3.np.float32)
        idx.load_scope('s',dict(a='apple tree',b='apple pear',c='orange'),{m:vectors[i] for i,m in enumerate('abc')})
        expected=idx.search('s','apple',vectors[0],['b','a'])
        result,_,_=v3.score(idx._sparse['s'],idx._dense['s'],'apple',vectors[0],['a','b'])
        self.assertEqual(tuple(result),expected.top10)

if __name__=='__main__':unittest.main()
