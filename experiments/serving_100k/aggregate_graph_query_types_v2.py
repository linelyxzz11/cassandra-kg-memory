"""Aggregate the graph-aware v2 query-type matrix."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics

from cassmem.serving.environment import ROOT

BASE=ROOT / "results" / "serving_100k" / "locomo_workload_graph_v2_100k"/"query_types_v2"; RUNS=BASE/"runs"
CELLS=("cassandra-base","cassandra-materialized","neo4j-native","neo4j-materialized"); REPS=(0,1,2)
TYPES=("scope_fetch","relation_filter","candidate_projection","top10_retrieval")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows=[]; missing=[]
    for cell in CELLS:
        for rep in REPS:
            path=RUNS/f"{cell}_c32_r{rep}_summary.json"
            if not path.exists(): missing.append(str(path)); continue
            payload=json.loads(path.read_text(encoding="utf-8"))
            if payload["status"]!="PASS": raise SystemExit(f"Failed run: {path}")
            for row in payload["results"]: row=dict(row);row["source_sha256"]=sha(path);rows.append(row)
    if missing: raise SystemExit("Missing formal summaries:\n"+"\n".join(missing))
    result=[]
    for cell in CELLS:
        for query_type in TYPES:
            group=[row for row in rows if row["cell"]==cell and row["query_type"]==query_type]
            out={"cell":cell,"query_type":query_type,"concurrency":32,"repetitions":3,"operations_per_rep":5000}
            for metric in ("qps","p50_ms","p95_ms","p99_ms","error_rate"):
                values=[float(row[metric]) for row in group];out[f"{metric}_median"]=statistics.median(values);out[f"{metric}_min"]=min(values);out[f"{metric}_max"]=max(values)
            out["status"]="PASS";result.append(out)
    BASE.mkdir(parents=True,exist_ok=True)
    for name,data in (("query_type_runs.csv",rows),("query_type_summary.csv",result)):
        with (BASE/name).open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    manifest={"status":"PASS","protocol_id":"graph-aware-query-types-v2","cells":list(CELLS),"query_types":list(TYPES),"concurrency":32,"repetitions":3,"operations_per_type_per_rep":5000,"aggregation":"median of three run-level statistics; min/max retained","outputs":{name:sha(BASE/name) for name in ("query_type_runs.csv","query_type_summary.csv")}}
    (BASE/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8");print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
