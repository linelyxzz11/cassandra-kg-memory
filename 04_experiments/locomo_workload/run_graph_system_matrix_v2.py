"""Resumable sequential launcher for the two graph-aware system matrices."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
CELLS=("cassandra-base","cassandra-materialized","neo4j-native","neo4j-materialized")


def passed(path, expected_operations=None):
    if not path.exists(): return False
    try: payload=json.loads(path.read_text(encoding="utf-8"))
    except Exception: return False
    if payload.get("status")!="PASS": return False
    return expected_operations is None or payload.get("measured_operations")==expected_operations


def run(command, label):
    print(f"START {label}",flush=True); subprocess.run(command,check=True,cwd=HERE); print(f"DONE {label}",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--experiment",choices=["main","queries","all"],default="all");args=parser.parse_args()
    if args.experiment in ("main","all"):
        out=ROOT/"05_reports"/"locomo_workload_graph_v2_100k"/"main_95_5_v2"/"runs"
        for cell in CELLS:
            for concurrency in (1,8,16,32,64):
                for rep in (0,1,2):
                    summary=out/f"{cell}_c{concurrency}_r{rep}_summary.json";label=f"main {cell} c={concurrency} rep={rep}"
                    if passed(summary,5000): print(f"SKIP {label}",flush=True);continue
                    run([sys.executable,str(HERE/"run_graph_100k_workload_v2.py"),"--cell",cell,"--concurrency",str(concurrency),"--rep",str(rep)],label)
        run([sys.executable,str(HERE/"aggregate_graph_100k_workload_v2.py")],"aggregate main")
    if args.experiment in ("queries","all"):
        out=ROOT/"05_reports"/"locomo_workload_graph_v2_100k"/"query_types_v2"/"runs"
        for cell in CELLS:
            for rep in (0,1,2):
                summary=out/f"{cell}_c32_r{rep}_summary.json";label=f"queries {cell} c=32 rep={rep}"
                if passed(summary): print(f"SKIP {label}",flush=True);continue
                run([sys.executable,str(HERE/"run_graph_query_types_v2.py"),"--cell",cell,"--rep",str(rep),"--concurrency","32"],label)
        run([sys.executable,str(HERE/"aggregate_graph_query_types_v2.py")],"aggregate queries")


if __name__=="__main__":main()
