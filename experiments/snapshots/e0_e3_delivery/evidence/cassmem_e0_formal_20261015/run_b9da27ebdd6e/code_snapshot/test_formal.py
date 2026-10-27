import tempfile,unittest,copy
from pathlib import Path
import formal as f
from validator import audit_phase,lines,load,content

class Backend:
    def __init__(self,rs):self.rs={(r.scope_id,r.memory_id):r for r in rs}
    def fetch_graph(self,c,s,m):return self.rs[(s,m)].graph_projection()
    def candidate_projection(self,c,s,m):return self.rs[(s,m)].rawerk
    def ids(self,c,s):return sorted(m for ss,m in self.rs if ss==s)
    def ids_by_relation(self,c,s,rel):return sorted(m for (ss,m),r in self.rs.items() if ss==s and rel in {x['relation'] for x in r.edges})
    def commit_raw(self,c,r):self.rs[(r.scope_id,r.memory_id)]=r
    def write_structured(self,c,r):self.rs[(r.scope_id,r.memory_id)]=r

class Tests(unittest.TestCase):
    def test_multiscope_phase_and_negative_validator(self):
        source=f.build_graph_records();mv=f.vectors('memory');qv=f.vectors('qa');qs=f.rows(f.ROOT/'data/frozen_retrieval/p3_queries.csv')
        scopes=sorted({r.scope_id for r in source})[:2]
        initial=[r for s in scopes for r in [x for x in source if x.scope_id==s][:10]]
        fs=[]
        for s in scopes:
            old=next(r for r in initial if r.scope_id==s);new=f.dataclasses.replace(next(r for r in source if r.scope_id==s and r.memory_id!=old.memory_id),memory_id=old.memory_id,version=2)
            src=next(r for r in source if r.scope_id==s and r.memory_id!=old.memory_id)
            q=next(q for q in qs if q['conversation_id']==s)
            fs.append(dict(memory_id=old.memory_id,scope=s,versions=[f.version(old,mv[old.memory_id]),f.version(new,mv[src.memory_id])],query=dict(query_id=q['query_id'],question=q['question'],vector=qv[q['query_id']].tolist())))
        with tempfile.TemporaryDirectory(prefix='e0-formal-tests-') as temp:
            out=Path(temp);trial=f.FormalTrial(Backend(initial),'test',initial,mv,fs,out,{'active':{'rate':1,'drain_s':1}},'test-multiscope')
            trial.phase('measured',2,0);d=out/'measured';ctx=load(d/'context.json');rs=lines(d/'events.jsonl')+lines(d/'reads.jsonl');subs=lines(d/'submissions.jsonl')
            a=audit_phase(ctx,rs,subs,fs);self.assertEqual(a['errors'],[]);self.assertEqual(a['execution_errors'],0);self.assertTrue(a['all_updates_eventually_ready'])
            self.assertEqual({r['query_scope'] for r in rs},set(scopes))
            self.assertTrue(audit_phase(ctx,rs[:-1],subs,fs)['errors'])
            bad=copy.deepcopy(rs);bad[0]['raw_ack_ns']=bad[0]['raw_call_start_ns']-1;self.assertTrue(audit_phase(ctx,bad,subs,fs)['errors'])
            bad=copy.deepcopy(rs);bad[0]['audit']['projection']='WRONG';self.assertTrue(audit_phase(ctx,bad,subs,fs)['errors'])
            late=copy.deepcopy(subs)
            for r in late:r['actual_enqueue_ns']+=100000000
            self.assertFalse(audit_phase(ctx,rs,late,fs)['offered_load_valid'])
            bad=copy.deepcopy(rs);bad[0]['response_finish_ns']=ctx['absolute_deadline_ns']+1
            self.assertFalse(audit_phase(ctx,bad,subs,fs)['completed_before_absolute_drain'])
            bad=copy.deepcopy(rs);bad[0]['post_update_top10']=[];bad[0]['post_update_target_present']=False
            self.assertEqual(audit_phase(ctx,bad,subs,fs)['errors'],[])
            self.assertTrue(content(trial.capture(0),fs[0]))
if __name__=='__main__':unittest.main()
