"""Version-2 freshness clocks and independent offered-arrival process.

Old benchmark and results remain untouched. A fixed absolute deadline is
evaluated separately from eventual correctness and offered-load fidelity.
"""
from __future__ import annotations
import argparse
import gc
import json
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from common import HERE, ARMS, np, source_manifest, write_csv, write_json
from benchmark import Trial, make_index, summarize
import psutil


def clocks(row, deadline):
    row["planned_to_worker_ms"] = (row["worker_start_ns"]-row["arrival_ns"])/1e6
    row["queue_wait_ms"] = (row["worker_start_ns"]-row["submit_ns"])/1e6
    if row["ready"]:
        row["planned_arrival_to_retrievable_ms"] = (row["ready_checked_ns"]-row["arrival_ns"])/1e6
        row["submitted_to_retrievable_ms"] = (row["ready_checked_ns"]-row["submit_ns"])/1e6
        # Main arrival boundary is actual application submission, aligned to section 3.3.
        row["arrival_to_retrievable_ms"] = row["submitted_to_retrievable_ms"]
        row["arrival_to_raw_ack_ms"] = (row["raw_ack_ns"]-row["submit_ns"])/1e6
        row["ready_within_500ms"] = row["submitted_to_retrievable_ms"] <= 500
        row["planned_ready_within_500ms"] = row["planned_arrival_to_retrievable_ms"] <= 500
    row["ready_censored_at_deadline"] = bool(row["op_type"] == "update" and
        (not row["ready"] or row["ready_checked_ns"] > deadline))
    return row


class RetestTrial(Trial):
    def phase(self, phase, duration, rate=0):
        if not rate:
            return super().phase(phase,duration,rate)
        rng = random.Random(20261012+self.args.rep+10000)
        count = int(round(duration*rate))*20
        # All payload construction is outside the measured interval.
        operations = [self.operation(i,phase,rng) for i in range(count)]
        self.peak_pending = self.peak_queued = self.total_updates = 0
        start = time.perf_counter_ns()+1_000_000_000
        deadline = start+int((duration+self.args.drain_seconds)*1e9)
        futures = []
        with ThreadPoolExecutor(max_workers=self.args.concurrency) as pool:
            emitter = subprocess.Popen([sys.executable,"-B",str(HERE/"arrival_emitter.py"),
                 "--start-ns",str(start),"--count",str(count),"--ops-rate",str(rate*20),
                 "--log",str(self.args.output/"emitter_events.jsonl")],stdout=subprocess.PIPE,
                 stderr=subprocess.PIPE,text=True,bufsize=1)
            try:
                for line in emitter.stdout:
                    ordinal,scheduled,emitted = map(int,line.strip().split(","))
                    assert ordinal == len(futures), "Arrival loss/reorder"
                    op = self.register(operations[ordinal],scheduled)
                    op["emit_ns"] = emitted
                    futures.append(pool.submit(self.execute,op))
                code = emitter.wait(timeout=15)
                if code:
                    raise RuntimeError(emitter.stderr.read())
                assert len(futures) == count
                # Absolute deadline, never reset at the last delayed submission.
                done,pending = wait(futures,timeout=max(0,(deadline-time.perf_counter_ns())/1e9))
                write_json(self.args.output/"deadline_observation.json",{
                    "scheduled_start_ns":start,"absolute_deadline_ns":deadline,
                    "observed_ns":time.perf_counter_ns(),"operations_pending_at_observation":len(pending)})
                output = [f.result() for f in futures]  # Preserve late completions explicitly.
            finally:
                if emitter.poll() is None:
                    emitter.terminate()
                    emitter.wait(timeout=15)
        for row in output:
            clocks(row,deadline)
            op = operations[row["ordinal"]]
            row["emit_ns"] = op["emit_ns"]
            row["emitter_lag_ms"] = (op["emit_ns"]-row["arrival_ns"])/1e6
            row["emitter_to_submit_ms"] = (row["submit_ns"]-op["emit_ns"])/1e6
        return output, {"wall_seconds_including_drain":(time.perf_counter_ns()-start)/1e9,
            "offered_window_seconds":duration,"peak_pending_updates":self.peak_pending,
            "peak_queued_updates":self.peak_queued,"planned_operations":count,
            "drain_deadline_exceeded":any(r["ready_censored_at_deadline"] for r in output),
            "scheduled_start_ns":start,"absolute_deadline_ns":deadline}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--keyspace",default="repcost_formal_v1")
    p.add_argument("--memories",type=int,default=100000)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--rate",type=float,required=True)
    p.add_argument("--rep",type=int,required=True)
    p.add_argument("--warmup-seconds",type=float,default=60)
    p.add_argument("--seconds",type=float,default=120)
    p.add_argument("--drain-seconds",type=float,default=120)
    a=p.parse_args()
    a.cell="cassandra-materialized"
    a.concurrency=32
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/"manifest.json",{"sources":source_manifest(),"clock_boundary":"application submission", "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        "load_fidelity_rule":"all submissions present; submission-lag p95 <= 50 ms; last submission <= injection-end + 1 s; fixed before trials"})
    t=RetestTrial(a)
    samples=[]
    stop=threading.Event()
    def sample():
        process=psutil.Process()
        while not stop.wait(1):
            samples.append({"unix":time.time(),"rss":process.memory_info().rss,"host_cpu_percent":psutil.cpu_percent(),"host_available_ram":psutil.virtual_memory().available})
    monitor=threading.Thread(target=sample,daemon=True)
    monitor.start()
    try:
        assert sum(len(t.b.ids(a.cell,s)) for s in t.grouped)==a.memories
        warm,_=t.phase("warmup",a.warmup_seconds)
        assert all(r["status"]=="ok" for r in warm)
        write_csv(a.output/"warmup_events.csv",warm)
        t.cleanup()
        del t.index
        gc.collect()
        t.index=make_index(t.grouped,t.mv,a.arm)
        events,context=t.phase("measured",a.seconds,a.rate)
        write_csv(a.output/"events.csv",events)
        result=summarize(events,context)
        lag=float(np.percentile([r["submission_lag_ms"] for r in events],95))
        last=max(r["submit_ns"] for r in events)
        load_ok=(len(events)==context["planned_operations"] and lag<=50 and
                 last<=context["scheduled_start_ns"]+int((a.seconds+1)*1e9))
        correctness=result["errors"]==0 and result["ready_updates"]==result["updates"]
        result.update(arm=a.arm,rep=a.rep,rate=a.rate,submission_lag_p95_ms=lag,
            execution_correct=correctness,offered_load_valid=load_ok,
            deadline_met=result["ready_censored_at_deadline"]==0,
            planned_ready_within_500ms_count=sum(r.get("planned_ready_within_500ms",False) for r in events),
            status="PASS" if correctness and load_ok and result["ready_censored_at_deadline"]==0 else "REVIEW_REQUIRED")
        write_json(a.output/"summary.json",result)
        print(json.dumps(result),flush=True)
        if not correctness:
            raise RuntimeError("Execution correctness failed")
    finally:
        stop.set(); monitor.join(timeout=2)
        write_csv(a.output/"host_client_samples.csv",samples)
        try:
            t.cleanup()
        finally:
            t.b.close()


if __name__=="__main__":
    main()
