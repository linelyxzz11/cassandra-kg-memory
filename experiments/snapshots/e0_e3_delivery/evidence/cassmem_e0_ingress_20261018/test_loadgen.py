import sys,time,subprocess,json,copy,ast
from pathlib import Path
import launch as L
ROOT=L.HERE/'loadgen_test'
def trial(name,count,rate,pause):
    d=ROOT/name;d.mkdir(parents=True,exist_ok=False);start=time.perf_counter_ns()+1_000_000_000
    p=subprocess.Popen([sys.executable,'-B',L.RECEIVER,'--start-ns',str(start),'--count',str(count),'--ops-rate',str(rate),'--log',str(d/'emitter.jsonl')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    rows=[]
    for line in p.stdout:
        i,scheduled,emitted,enqueued=map(int,line.strip().split(','));dispatch=time.perf_counter_ns()
        rows.append(dict(event_id=i,planned_arrival_ns=scheduled,emit_ns=emitted,actual_enqueue_ns=enqueued,execution_dispatch_ns=dispatch,worker_start_ns=dispatch,response_finish_ns=time.perf_counter_ns()))
        if i==0 and pause:time.sleep(pause)
    assert p.wait(timeout=10)==0,p.stderr.read();p.stdout.close();p.stderr.close()
    recv=L.V.lines(d/'receiver.jsonl');emit=L.V.lines(d/'emitter.jsonl');ctx={'operations':count}
    assert not L.ingress_check(ctx,rows,rows,recv,emit)
    p95=float(L.F.np.percentile([(x['actual_enqueue_ns']-x['planned_arrival_ns'])/1e6 for x in rows],95))
    lag=max((x['execution_dispatch_ns']-x['actual_enqueue_ns'])/1e6 for x in rows)
    assert p95<=50,p95
    if pause:assert lag>pause*1000-200
    # Negative controls: evidence mutation, missing receiver event, impossible dispatch.
    bad=copy.deepcopy(rows);bad[0]['actual_enqueue_ns']+=1
    assert L.ingress_check(ctx,bad,rows,recv,emit)
    assert L.ingress_check(ctx,rows,rows,recv[:-1],emit)
    bad=copy.deepcopy(rows);bad[0]['execution_dispatch_ns']=bad[0]['actual_enqueue_ns']-1
    assert L.ingress_check(ctx,bad,bad,recv,emit)
    L.F.save(d/'execution_observations.json',rows)
    return dict(name=name,count=count,ops_rate=rate,consumer_pause_s=pause,receive_lag_p95_ms=p95,dispatch_lag_max_ms=lag,negative_tests=3)

def main():
    ROOT.mkdir(exist_ok=False)
    old=ast.parse(L.A.phase_source());new=ast.parse(L.phase_source())
    ex=lambda tree:ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='execute'))
    assert ex(old)==ex(new),'execution/readiness/audit changed'
    results=[trial('normal',60,20,0),trial('stalled_consumer',100,20,3),trial('pipe_backpressure',1000,500,3)]
    L.F.save(ROOT/'result.json',dict(**{'pass':True},tests=results,execution_ast_identical=True,total_events=1160,clock='same-host Windows perf_counter_ns, ordering verified across emitter/receiver/executor',negative_tests=9))
    print(json.dumps(results),flush=True)
if __name__=='__main__':main()
