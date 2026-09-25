"""Isolated, resource-matched rerun of the graph-aware serving experiments.

This launcher never removes a container or volume. It stops the two original
containers temporarily, uses separately named Docker volumes and non-default
ports, and restarts the originals in a finally block. Run only after Docker
Desktop's data disk has been moved to a filesystem with adequate free space.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = ROOT / ".venv-onnx" / "Scripts" / "python.exe"
OLD_CONTAINERS = ("cassandra", "neo4j-kg")
IMAGE_IDS = {
    "cassandra": "sha256:2c6576f9b8e0271d3d30a4142e3a79cd69d60fa1e5ecf32dc641a352d372e9ff",
    "neo4j": "sha256:0b5d3ab6ec1b866890dbfb53bf4fe1cf039f9e03c96165599a403005b7e7bcc3",
}
CELLS = {
    "cassandra": ("cassandra-base", "cassandra-materialized"),
    "neo4j": ("neo4j-native", "neo4j-materialized"),
}
MEMORY = "12g"
CPUS = "8"


def container_name(backend: str, run_id: str) -> str:
    return f"cassmem-{run_id.replace('_', '-')}-{backend}"


def volume_name(backend: str, run_id: str) -> str:
    return f"cassmem_{run_id}_{backend}_data"


def docker(*args: str, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(("docker", *args), text=True, capture_output=True, check=check, env=env)


def present(name: str) -> bool:
    return docker("inspect", name, check=False).returncode == 0


def running(name: str) -> bool:
    if not present(name):
        return False
    return docker("inspect", "--format", "{{.State.Running}}", name).stdout.strip() == "true"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_python(script: Path, arguments: list[str], *, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {script.name} {' '.join(arguments)}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run((str(PYTHON), str(script), *arguments), cwd=ROOT, env=env,
                                stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"{script.name} failed ({result.returncode}); inspect {log}")


def load_local_password() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("NEO4J_PASSWORD="):
            return line.partition("=")[2].strip().strip("\"'")
    raise RuntimeError("NEO4J_PASSWORD is absent from ignored .env")


def run_environment(backend: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(HERE), env.get("PYTHONPATH", "")))
    env["CASSANDRA_HOST"] = "127.0.0.1"
    env["CASSANDRA_PORT"] = "19042"
    env["NEO4J_URI"] = "bolt://127.0.0.1:17687"
    env["NEO4J_USER"] = "neo4j"
    if backend == "neo4j":
        env["NEO4J_PASSWORD"] = load_local_password()
    return env


def wait_ready(backend: str, env: dict[str, str], name: str, timeout_s: int = 300) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if backend == "cassandra":
            probe = docker("exec", name, "cqlsh", "-e", "DESCRIBE KEYSPACES", check=False)
            if probe.returncode == 0:
                return
        else:
            code = "from neo4j import GraphDatabase; import os; d=GraphDatabase.driver(os.environ['NEO4J_URI'],auth=(os.environ['NEO4J_USER'],os.environ['NEO4J_PASSWORD'])); d.verify_connectivity(); d.close()"
            probe = subprocess.run((str(PYTHON), "-c", code), env=env, capture_output=True, text=True)
            if probe.returncode == 0:
                return
        time.sleep(5)
    raise RuntimeError(f"{backend} did not become ready in {timeout_s}s")


def start_new_container(backend: str, *, phase: str, run_id: str) -> str:
    name = container_name(backend, run_id)
    volume = volume_name(backend, run_id)
    if phase == "preflight":
        if present(name) or docker("volume", "inspect", volume, check=False).returncode == 0:
            raise RuntimeError(f"Preflight refuses to reuse {name} or {volume}; choose a fresh protocol ID")
        docker("volume", "create", volume)
        args = ["run", "-d", "--name", name, "--cpus", CPUS, "--memory", MEMORY,
                "--memory-swap", MEMORY, "-v", f"{volume}:{'/var/lib/cassandra' if backend == 'cassandra' else '/data'}"]
        if backend == "cassandra":
            args += ["-p", "19042:9042", "-e", "MAX_HEAP_SIZE=5G", IMAGE_IDS[backend]]
            docker(*args)
        else:
            password = load_local_password()
            neo_env = os.environ.copy()
            neo_env["NEO4J_AUTH"] = f"neo4j/{password}"
            args += ["-p", "17687:7687", "-e", "NEO4J_AUTH",
                     "-e", "NEO4J_server_memory_heap_initial__size=5G",
                     "-e", "NEO4J_server_memory_heap_max__size=5G",
                     "-e", "NEO4J_server_memory_pagecache_size=3G", IMAGE_IDS[backend]]
            docker(*args, env=neo_env)
            neo_env["NEO4J_AUTH"] = ""
    else:
        if not present(name):
            raise RuntimeError(f"Missing preflight container {name}")
        if not running(name):
            docker("start", name)
    return name


def safe_inspect(name: str) -> dict:
    item = json.loads(docker("inspect", name).stdout)[0]
    host = item["HostConfig"]
    return {
        "name": item["Name"], "created": item["Created"], "image_id": item["Image"],
        "cpu_quota": host["CpuQuota"], "nano_cpus": host["NanoCpus"],
        "memory_bytes": host["Memory"], "memory_swap_bytes": host["MemorySwap"],
        "mounts": [{"type": m["Type"], "name": m.get("Name"), "destination": m["Destination"]} for m in item["Mounts"]],
        "ports": item["NetworkSettings"]["Ports"],
        "env_names": sorted(value.partition("=")[0] for value in item["Config"].get("Env", [])),
    }


def cgroup_snapshot(name: str) -> dict[str, str]:
    paths = ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/memory.events",
             "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.peak",
             "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/io.stat")
    snapshot = {}
    for path in paths:
        result = docker("exec", name, "cat", path, check=False)
        snapshot[path.rsplit("/", 1)[-1]] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    return snapshot


def verify_memory_config(name: str, backend: str) -> dict:
    if backend == "cassandra":
        info = docker("exec", name, "nodetool", "info").stdout
        match = re.search(r"Heap Memory \(MB\)\s*:\s*[\d.]+\s*/\s*([\d.]+)", info)
        if not match or not 5000 <= float(match.group(1)) <= 5300:
            raise RuntimeError("Cassandra did not start with the expected 5 GiB maximum heap")
        return {"observed_heap_max_mb": float(match.group(1)), "nodetool_info": info}
    conf = docker("exec", name, "sh", "-c", "grep '^server.memory.' /var/lib/neo4j/conf/neo4j.conf").stdout
    expected = ("server.memory.heap.initial_size=5G", "server.memory.heap.max_size=5G", "server.memory.pagecache.size=3G")
    if not all(line in conf for line in expected):
        raise RuntimeError("Neo4j configuration did not report the expected 5G/3G memory settings")
    return {"neo4j_memory_settings": conf}


class StatsCapture:
    def __init__(self, name: str, path: Path):
        self.name, self.path = name, path
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.samples = 0
        self.errors: list[str] = []

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        def copy():
            with self.path.open("w", encoding="utf-8") as handle:
                while not self.stop.is_set():
                    result = docker("stats", "--no-stream", "--format", "{{json .}}", self.name, check=False)
                    try:
                        if result.returncode or not result.stdout.strip():
                            raise RuntimeError(result.stderr.strip() or "empty docker stats result")
                        item = json.loads(result.stdout)
                        item["sampled_at_utc"] = datetime.now(timezone.utc).isoformat()
                        handle.write(json.dumps(item) + "\n")
                        handle.flush()
                        self.samples += 1
                    except (json.JSONDecodeError, RuntimeError) as error:
                        self.errors.append(str(error))
                    self.stop.wait(2)

        self.thread = threading.Thread(target=copy, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=10)
        if self.samples == 0:
            raise RuntimeError(f"No Docker resource samples captured for {self.name}: {self.errors[:3]}")


def trace_manifest() -> dict:
    path = ROOT / "results" / "serving_100k" / "locomo_workload_100k" / "traces" / "trace_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for rep in data["repetitions"]:
        for item in rep["files"]:
            if sha(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"Trace hash mismatch: {item['path']}")
    return data


def verified_storage_location() -> dict:
    settings_path = Path(os.environ["APPDATA"]) / "Docker" / "settings-store.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    distro_dir = Path(settings["CustomWslDistroDir"])
    disk = distro_dir / "disk" / "docker_data.vhdx"
    if distro_dir.drive.upper() != "D:" or not disk.is_file():
        raise RuntimeError("Docker Desktop data disk is not verified on D")
    return {"wsl_distro_dir": str(distro_dir), "vhd_path": str(disk)}


def run_backend(backend: str, phase: str, base: Path, run_id: str) -> None:
    env = run_environment(backend)
    name = start_new_container(backend, phase=phase, run_id=run_id)
    wait_ready(backend, env, name)
    backend_dir = base / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    write_json(backend_dir / f"{phase}_container.json", safe_inspect(name))
    write_json(backend_dir / f"{phase}_memory_config.json", verify_memory_config(name, backend))
    write_json(backend_dir / f"{phase}_cgroup_before.json", cgroup_snapshot(name))
    if phase == "preflight":
        load_args = ["--backend", backend, "--reset", "--output-dir", str(backend_dir / "load")]
        if backend == "neo4j":
            # Dataset preparation is outside the timed workload. Serial writes avoid
            # conflicting MERGEs of shared graph entities during initial loading.
            load_args += ["--neo4j-workers", "1"]
        run_python(HERE / "run_graph_100k_load_gate_v2.py",
                   load_args,
                   env=env, log=backend_dir / "load.log")
        with StatsCapture(name, backend_dir / "preflight_stats.jsonl"):
            for cell in CELLS[backend]:
                run_python(HERE / "run_graph_100k_workload_v2.py",
                           ["--cell", cell, "--concurrency", "1", "--rep", "0", "--warmup-limit", "20",
                            "--measured-limit", "100", "--output", str(backend_dir / "preflight_main")],
                           env=env, log=backend_dir / f"preflight_{cell}_main.log")
                run_python(ROOT / "experiments" / "freshness" / "run_experiment11_freshness_v2.py",
                           ["--cell", cell, "--smoke", "--output-dir", str(backend_dir / "preflight_freshness")],
                           env=env, log=backend_dir / f"preflight_{cell}_freshness.log")
    else:
        if not (backend_dir / "preflight_pass.json").exists():
            raise RuntimeError(f"Missing successful preflight for {backend}")
        with StatsCapture(name, backend_dir / "formal_stats.jsonl"):
            for cell in CELLS[backend]:
                for concurrency in (1, 8, 16, 32, 64):
                    for rep in (0, 1, 2):
                        stem = f"{cell}_c{concurrency}_r{rep}"
                        summary = base / "main" / "runs" / f"{stem}_summary.json"
                        if summary.exists() and json.loads(summary.read_text(encoding="utf-8")).get("status") == "PASS":
                            continue
                        run_python(HERE / "run_graph_100k_workload_v2.py",
                                   ["--cell", cell, "--concurrency", str(concurrency), "--rep", str(rep),
                                    "--output", str(base / "main" / "runs")],
                                   env=env, log=backend_dir / "logs" / f"{stem}.log")
                freshness_manifest = base / "freshness" / "runs" / f"{cell}_c32_manifest.json"
                if not (freshness_manifest.exists() and json.loads(freshness_manifest.read_text(encoding="utf-8")).get("status") == "PASS"):
                    run_python(ROOT / "experiments" / "freshness" / "run_experiment11_freshness_v2.py",
                               ["--cell", cell, "--output-dir", str(base / "freshness" / "runs")],
                               env=env, log=backend_dir / "logs" / f"{cell}_freshness.log")
    write_json(backend_dir / f"{phase}_cgroup_after.json", cgroup_snapshot(name))
    state = json.loads(docker("inspect", name).stdout)[0]["State"]
    if state.get("OOMKilled"):
        raise RuntimeError(f"{backend} container was OOM-killed during {phase}")
    write_json(backend_dir / f"{phase}_pass.json", {"status": "PASS", "backend": backend})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "formal"), required=True)
    parser.add_argument("--backend", choices=("cassandra", "neo4j"), required=True)
    parser.add_argument("--run-id", default="resmatch_20260924_v3r2")
    parser.add_argument("--storage-migrated", action="store_true", help="Confirm Docker data disk is no longer on the nearly full C drive")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", args.run_id):
        raise SystemExit("run-id must be a simple lowercase identifier")
    if not args.storage_migrated:
        raise SystemExit("Refusing to create benchmark volumes until Docker Desktop data disk is moved to D")
    if not PYTHON.exists():
        raise SystemExit(f"Required environment not found: {PYTHON}")
    storage = verified_storage_location()
    traces = trace_manifest()
    base = ROOT / "results" / "serving_100k" / "resource_matched_v3" / args.run_id
    base.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol_id": args.run_id, "container_cpus": CPUS, "container_memory": MEMORY,
        "swap_disabled": True, "one_database_at_a_time": True,
        "neo4j_heap": "5G", "neo4j_page_cache": "3G", "cassandra_heap": "5G",
        "image_ids": IMAGE_IDS, "trace_manifest": traces,
        "storage": storage,
        "trace_manifest_sha256": sha(ROOT / "results" / "serving_100k" / "locomo_workload_100k" / "traces" / "trace_manifest.json"),
        "launcher_sha256": sha(Path(__file__)),
        "adapter_sha256": sha(ROOT / "src" / "cassmem" / "backend" / "live_cells_graph.py"),
        "loader_sha256": sha(HERE / "run_graph_100k_load_gate_v2.py"),
        "main_workload_sha256": sha(HERE / "run_graph_100k_workload_v2.py"),
        "freshness_workload_sha256": sha(ROOT / "experiments" / "freshness" / "run_experiment11_freshness_v2.py"),
        "platform": platform.platform(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "latency_boundaries": {
            "main_latency_ms": "worker start to completed operation; excludes executor admission wait",
            "freshness_queue_wait_ms": "submit to worker start",
            "freshness_raw_write_call_ms": "client call begin to acknowledged return; includes driver, network, server",
            "freshness_t_commit_ms": "submit to raw write acknowledged; queue-inclusive legacy field",
        },
    }
    protocol_path = base / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        for key in ("container_cpus", "container_memory", "swap_disabled", "image_ids", "storage",
                    "trace_manifest_sha256", "launcher_sha256", "adapter_sha256", "loader_sha256",
                    "main_workload_sha256", "freshness_workload_sha256"):
            if old.get(key) != protocol.get(key):
                raise RuntimeError(f"Frozen protocol changed: {key}; choose a new run-id")
    else:
        write_json(protocol_path, protocol)
    previous = {name: running(name) for name in OLD_CONTAINERS}
    try:
        for name, was_running in previous.items():
            if was_running:
                docker("stop", "--time", "120", name)
        for other in CELLS:
            other_name = container_name(other, args.run_id)
            if other != args.backend and running(other_name):
                docker("stop", "--time", "120", other_name)
        run_backend(args.backend, args.phase, base, args.run_id)
        if args.phase == "formal" and all((base / backend / "formal_pass.json").exists() for backend in CELLS):
            run_python(HERE / "aggregate_graph_100k_workload_v2.py", ["--base", str(base / "main")],
                       env=run_environment(args.backend), log=base / "main_aggregate.log")
    finally:
        new_name = container_name(args.backend, args.run_id)
        if running(new_name):
            docker("stop", "--time", "120", new_name)
        for name, was_running in previous.items():
            if was_running and not running(name):
                docker("start", name)


if __name__ == "__main__":
    main()
