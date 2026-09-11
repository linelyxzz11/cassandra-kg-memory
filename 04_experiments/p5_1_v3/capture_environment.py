"""Capture a post-run, credential-redacted backend/runtime snapshot."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_p5_1_v3 import ROOT, load_local_env


def main() -> int:
    output = Path(sys.argv[1])
    load_local_env(ROOT / ".env")
    from cassandra.cluster import Cluster
    from neo4j import GraphDatabase

    cluster = Cluster([os.environ.get("CASSANDRA_HOST", "127.0.0.1")], protocol_version=4)
    session = cluster.connect()
    cass = session.execute(
        "SELECT release_version, cluster_name, data_center, rack FROM system.local"
    ).one()
    session.shutdown()
    cluster.shutdown()

    password = os.environ.get("NEO4J_PASSWORD", "")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required")
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), password),
    )
    with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as neo_session:
        neo = neo_session.run(
            "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition"
        ).single(strict=True)
    driver.close()

    snapshot = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_timing": "immediately after formal run; post-run snapshot",
        "credentials_included": False,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": sys.version,
        },
        "drivers": {
            "cassandra-driver": importlib.metadata.version("cassandra-driver"),
            "neo4j": importlib.metadata.version("neo4j"),
        },
        "cassandra": {
            "release_version": cass.release_version,
            "cluster_name": cass.cluster_name,
            "data_center": cass.data_center,
            "rack": cass.rack,
        },
        "neo4j": {
            "name": neo["name"],
            "versions": list(neo["versions"]),
            "edition": neo["edition"],
        },
    }
    output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
