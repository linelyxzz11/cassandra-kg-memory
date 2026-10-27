"""Standard-library-only, separate-process arrival scheduler."""
import argparse
import json
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-ns", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--ops-rate", type=float, required=True)
    p.add_argument("--log", required=True)
    a = p.parse_args()
    records = []
    for ordinal in range(a.count):
        scheduled = a.start_ns + int(ordinal / a.ops_rate * 1e9)
        delay = (scheduled-time.perf_counter_ns())/1e9
        if delay > 0:
            time.sleep(delay)
        emitted = time.perf_counter_ns()
        sys.stdout.write(f"{ordinal},{scheduled},{emitted}\n")
        sys.stdout.flush()
        records.append({"ordinal":ordinal,"scheduled_ns":scheduled,"emit_ns":emitted,"send_end_ns":time.perf_counter_ns()})
    with open(a.log,"w",encoding="utf-8") as h:
        for r in records:
            h.write(json.dumps(r)+"\n")


if __name__ == "__main__":
    main()
