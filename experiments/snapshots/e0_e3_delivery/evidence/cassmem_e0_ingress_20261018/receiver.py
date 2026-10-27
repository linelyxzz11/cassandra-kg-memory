"""Small independent ingress process; no numpy, database, or audit objects."""
import sys,time,json,threading,queue,subprocess,argparse,traceback
from pathlib import Path
EMITTER='D:/memorytable/representation_serving_cost/arrival_emitter.py'
def main():
    ap=argparse.ArgumentParser()
    for n,t in [('start-ns',int),('count',int),('ops-rate',float),('log',str)]:ap.add_argument('--'+n,type=t,required=True)
    a=ap.parse_args();out=Path(a.log).parent;q=queue.Queue();records=[];sent=[];errors=[]
    def send():
        try:
            while True:
                x=q.get()
                if x is None:break
                t=time.perf_counter_ns()
                sys.stdout.write(f"{x['event_id']},{x['planned_arrival_ns']},{x['emit_ns']},{x['actual_enqueue_ns']}\n");sys.stdout.flush()
                sent.append(dict(event_id=x['event_id'],send_start_ns=t,send_end_ns=time.perf_counter_ns()))
        except BaseException:errors.append(traceback.format_exc())
    th=threading.Thread(target=send);th.start()
    proc=subprocess.Popen([sys.executable,'-B',EMITTER,'--start-ns',str(a.start_ns),'--count',str(a.count),'--ops-rate',str(a.ops_rate),'--log',a.log],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        for line in proc.stdout:
            i,planned,emitted=map(int,line.strip().split(','));assert i==len(records)
            # Both enqueue and forwarding use the queue mutex, so stamp precedes visibility.
            with q.not_full:
                received=time.perf_counter_ns()
                x=dict(event_id=i,planned_arrival_ns=planned,emit_ns=emitted,actual_enqueue_ns=received)
                q._put(x);q.unfinished_tasks+=1;q.not_empty.notify()
            records.append(x)
        if proc.wait(timeout=15):raise RuntimeError(proc.stderr.read())
        assert len(records)==a.count
    finally:
        q.put(None);th.join(timeout=30)
        for name,rs in [('receiver.jsonl',records),('receiver_transport.jsonl',sent)]:
            (out/name).write_text(''.join(json.dumps(x)+'\n' for x in rs),encoding='utf-8')
        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
        proc.stdout.close();proc.stderr.close()
    assert not th.is_alive() and not errors and len(sent)==a.count,errors
if __name__=='__main__':main()
