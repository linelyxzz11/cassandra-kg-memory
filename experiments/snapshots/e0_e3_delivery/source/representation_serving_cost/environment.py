"""Manage only the explicitly authorized experiment container and originals."""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from common import HERE, write_json

NAME = "cassmem-repcost-20261012-cassandra"
VOLUME = "cassmem_repcost_20261012_data"
IMAGE = "sha256:2c6576f9b8e0271d3d30a4142e3a79cd69d60fa1e5ecf32dc641a352d372e9ff"
STATE = HERE / "results" / "environment_20261012.json"
ORIGINALS = ("cassandra", "neo4j-kg")


def docker(*args, check=True, timeout=180):
    p = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f"docker {args[0]} failed: {p.stderr}")
    return p


def inspect(name):
    return json.loads(docker("inspect", name).stdout)[0]


def safe_config(name):
    info = inspect(name)
    return {"name": name, "image": info["Image"], "state": info["State"]["Status"],
            "resources": {k: info["HostConfig"][k] for k in ("Memory", "NanoCpus", "MemorySwap")},
            "mounts": [{k: m.get(k) for k in ("Type", "Name", "Destination")} for m in info["Mounts"]],
            "ports": info["NetworkSettings"]["Ports"]}


def start():
    if STATE.exists() or docker("inspect", NAME, check=False).returncode == 0:
        raise RuntimeError("Environment already exists; inspect saved state instead of overwriting it")
    if docker("volume", "inspect", VOLUME, check=False).returncode == 0:
        raise RuntimeError("Refusing to reuse an existing volume")
    state = {"originals": {n: safe_config(n) for n in ORIGINALS}, "experiment": NAME,
             "authorization": "User confirmed temporary switch on 2026-10-12", "status": "PREPARING"}
    write_json(STATE, state)
    try:
        for name in ORIGINALS:
            if state["originals"][name]["state"] == "running":
                docker("stop", "--time", "60", name)
        docker("volume", "create", VOLUME)
        docker("run", "-d", "--name", NAME, "--cpus", "8", "--memory", "12g", "--memory-swap", "12g",
               "-p", "19042:9042", "-v", f"{VOLUME}:/var/lib/cassandra", "-e", "MAX_HEAP_SIZE=5G", IMAGE)
        for attempt in range(90):
            result = docker("exec", NAME, "cqlsh", "-e", "SELECT release_version FROM system.local;", check=False, timeout=30)
            if result.returncode == 0:
                break
            time.sleep(2)
        else:
            raise RuntimeError("Cassandra readiness timed out")
        config = safe_config(NAME)
        assert config["resources"] == {"Memory": 12 * 1024**3, "NanoCpus": 8 * 10**9, "MemorySwap": 12 * 1024**3}
        heap = docker("exec", NAME, "nodetool", "info").stdout
        import re
        match = re.search(r"Heap Memory \(MB\)\s*:\s*[\d.]+\s*/\s*([\d.]+)", heap)
        assert match and 5000 <= float(match.group(1)) <= 5300, heap
        state.update(status="READY", container=config, version=result.stdout, nodetool_info=heap,
                     cassandra_yaml=docker("exec", NAME, "cat", "/etc/cassandra/cassandra.yaml").stdout)
        write_json(STATE, state)
        print("READY: isolated Cassandra, 8 CPUs / 12 GiB / 5 GiB heap", flush=True)
    except BaseException:
        restore()
        raise


def restore():
    if not STATE.exists():
        return
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if docker("inspect", NAME, check=False).returncode == 0:
        docker("stop", "--time", "60", NAME)
    for name, before in state["originals"].items():
        if before["state"] == "running":
            docker("start", name)
    state["status"] = "RESTORED_ORIGINALS"
    write_json(STATE, state)
    print("Original containers restored; all volumes retained", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "restore"))
    args = parser.parse_args()
    start() if args.action == "start" else restore()
