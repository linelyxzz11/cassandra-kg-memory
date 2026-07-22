from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from .executors.base import BackendExecutor
from .models import QuerySpec


def run_trace(executor: BackendExecutor, manifest: Iterable[QuerySpec], out_path: str | Path) -> dict:
    """Sequential trace runner for C0 smoke testing. C1's concurrent driver reuses this manifest."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = []
    reads = writes = errors = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for event in manifest:
            started = time.perf_counter_ns()
            record = {"query_id": event.query_id, "op_type": event.op_type, "system": executor.system_name}
            try:
                if event.op_type == "read":
                    result = executor.execute(event)
                    reads += 1
                    record.update({"raw_edges_read": result.raw_edges_read, "cache_hits": result.cache_hits, "cache_misses": result.cache_misses, "result_path_count": len(result.normalized_paths())})
                elif event.op_type == "write":
                    executor.apply_write(event)
                    writes += 1
                else:
                    raise ValueError(f"Unsupported op_type={event.op_type}")
            except Exception as exc:  # noqa: BLE001
                errors += 1
                record["error"] = repr(exc)
            record["latency_ms"] = (time.perf_counter_ns() - started) / 1_000_000
            latencies.append(record["latency_ms"])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    latencies.sort()
    def pct(p: float) -> float | None:
        return None if not latencies else latencies[min(len(latencies) - 1, int((len(latencies) - 1) * p))]
    return {"system": executor.system_name, "events": len(latencies), "reads": reads, "writes": writes, "errors": errors,
            "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None, "p50_latency_ms": pct(.50), "p95_latency_ms": pct(.95), "p99_latency_ms": pct(.99)}
