"""18-point freshness retest; restores precisely the initial container states."""
import json
import subprocess
import sys
import threading
import time
from common import HERE,ARMS,write_json,write_csv,source_manifest
from environment import NAME,ORIGINALS,docker,inspect,safe_config

OUT=HERE/"results/freshness_minimal_v2"


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/"logs").mkdir()
    originals={n:inspect(n)["State"]["Running"] for n in ORIGINALS}
    progress={"status":"PREPARING","completed":0,"total":18,"original_running":originals,
              "started_unix":time.time(),"sources":source_manifest()}
    write_json(OUT/"progress.json",progress)
    stop=threading.Event()
    monitor=None
    def sample():
        with (OUT/"container_samples.jsonl").open("w",encoding="utf-8") as h:
            while not stop.is_set():
                try:
                    item={"unix":time.time(),"stats":docker("stats","--no-stream","--format","{{json .}}",NAME,timeout=20).stdout.strip()}
                except Exception as e:
                    item={"unix":time.time(),"error":str(e)}
                h.write(json.dumps(item)+"\n"); h.flush()
                stop.wait(10)
    def child(arguments,label):
        log=OUT/"logs"/(label+".log")
        with log.open("w",encoding="utf-8") as h:
            proc=subprocess.Popen([sys.executable,"-B",str(HERE/"freshness_retest.py"),*arguments],stdout=h,stderr=subprocess.STDOUT)
            write_json(OUT/"current_process.json",{"pid":proc.pid,"log":str(log),"arguments":arguments})
            try:
                code=proc.wait()
            except BaseException:
                proc.terminate(); proc.wait(timeout=90)
                raise
        if code:
            raise RuntimeError(f"Process failed: {log}")
    summaries=[]
    try:
        for n,on in originals.items():
            if on: docker("stop","--time","60",n)
        docker("start",NAME)
        for _ in range(120):
            if docker("exec",NAME,"cqlsh","-e","SELECT release_version FROM system.local;",check=False,timeout=20).returncode==0:
                break
            time.sleep(2)
        else:
            raise RuntimeError("Database unavailable")
        cfg=safe_config(NAME)
        assert cfg["resources"]=={"Memory":12*1024**3,"NanoCpus":8*10**9,"MemorySwap":12*1024**3}
        write_json(OUT/"environment.json",{"container":cfg,"nodetool_info":docker("exec",NAME,"nodetool","info").stdout})
        monitor=threading.Thread(target=sample,daemon=True); monitor.start()
        progress.update(stage="short_preflight",status="RUNNING")
        write_json(OUT/"progress.json",progress)
        child(["--output",str(OUT/"preflight"),"--keyspace","repcost_smoke_v1","--memories","400",
               "--arm","RawERK","--rate","10","--rep","0","--warmup-seconds","2","--seconds","3"],"preflight")
        preflight=json.loads((OUT/"preflight/summary.json").read_text(encoding="utf-8"))
        assert preflight["execution_correct"] and preflight["offered_load_valid"], "Preflight validation failed"
        points=[{"rep":rep,"rate":rate,"arm":arm} for rep in range(3) for rate in (5,10) for arm in ARMS[rep:]+ARMS[:rep]]
        write_json(OUT/"matrix.json",points)
        for i,point in enumerate(points):
            label=f"{i:02d}_{point['arm']}_u{point['rate']}_r{point['rep']}"
            progress.update(stage=label); write_json(OUT/"progress.json",progress)
            telemetry={"before":{cmd:docker("exec",NAME,"nodetool",cmd).stdout for cmd in ("gcstats","compactionstats")}}
            child(["--output",str(OUT/"trials"/label),"--arm",point["arm"],"--rate",str(point["rate"]),"--rep",str(point["rep"])],label)
            telemetry["after"]={cmd:docker("exec",NAME,"nodetool",cmd).stdout for cmd in ("gcstats","compactionstats")}
            write_json(OUT/"trials"/label/"database_telemetry.json",telemetry)
            result=json.loads((OUT/"trials"/label/"summary.json").read_text(encoding="utf-8"))
            summaries.append({"trial":label,**result})
            write_csv(OUT/"all_trials.csv",summaries)
            progress.update(completed=i+1,review_required=sum(s["status"]!="PASS" for s in summaries))
            write_json(OUT/"progress.json",progress)
            print(f"{i+1}/18 {label}: {result['status']}",flush=True)
        progress.update(status="COMPLETED",stage="complete",finished_unix=time.time())
    except BaseException as e:
        progress.update(status="FAILED",error=f"{type(e).__name__}: {e}")
        raise
    finally:
        stop.set()
        if monitor: monitor.join(timeout=25)
        write_json(OUT/"progress.json",progress)
        docker("stop","--time","60",NAME)
        for n,on in originals.items():
            if on: docker("start",n)
        progress["original_containers_restored"]=True
        write_json(OUT/"progress.json",progress)


if __name__=="__main__":
    main()
