from __future__ import annotations

import argparse
import json
from pathlib import Path

from .canonical import canonicalize_csv, load_canonical_edges, read_manifest
from .executors.cassandra import CassandraExecutor
from .executors.neo4j import Neo4jExecutor
from .manifest import generate_manifest_from_plan
from .runner import run_trace
from .semantics import ReferenceGraph
from .validation import validate_static_reads


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_executors(config: dict, names: list[str]):
    output = {}
    for name in names:
        if name == "cassandra_naive":
            output[name] = CassandraExecutor(config["cassandra"], "naive")
        elif name == "cassandra_opt":
            output[name] = CassandraExecutor(config["cassandra"], "opt")
        elif name == "neo4j":
            output[name] = Neo4jExecutor(config["neo4j"])
        else:
            raise ValueError(f"Unknown system: {name}")
    return output


def profile(config: dict) -> dict:
    cass = config["cassandra"]
    index = bool(cass.get("relation_index", {}).get("enabled", False))
    return {
        "cassandra_host": f"{cass.get('hosts', ['127.0.0.1'])[0]}:{cass.get('port', 9042)}",
        "keyspace": cass["keyspace"],
        "auth": "enabled" if cass.get("username") else "none",
        "tables": cass["tables"],
        "profile": "4-table (relation index enabled)" if index else "3-table compatible (relation index disabled)",
        "c0_identity": "logical SHA-1(graph_id, src_id, relation, dst_id, source)",
        "requires_visible_from_version": False,
        "timeuuid_migration_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="C0 schema-compatible Cassandra/Neo4j harness")
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("canonicalize")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p = subs.add_parser("generate-manifest")
    p.add_argument("--plan", required=True); p.add_argument("--output", required=True); p.add_argument("--count", required=True, type=int); p.add_argument("--seed", required=True, type=int)
    p = subs.add_parser("profile")
    p.add_argument("--config", required=True)
    p = subs.add_parser("validate")
    p.add_argument("--config", required=True); p.add_argument("--canonical", required=True); p.add_argument("--manifest", required=True); p.add_argument("--systems", nargs="+", required=True, choices=["cassandra_naive", "cassandra_opt", "neo4j"]); p.add_argument("--report-dir", required=True)
    p = subs.add_parser("run")
    p.add_argument("--config", required=True); p.add_argument("--manifest", required=True); p.add_argument("--system", required=True, choices=["cassandra_naive", "cassandra_opt", "neo4j"]); p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "canonicalize":
        edges, duplicates = canonicalize_csv(args.input, args.output)
        print(json.dumps({"canonical_edges": len(edges), "duplicates_collapsed": duplicates, "output": args.output}, ensure_ascii=False)); return
    if args.command == "generate-manifest":
        records = generate_manifest_from_plan(args.plan, args.output, args.count, args.seed)
        print(json.dumps({"events": len(records), "output": args.output}, ensure_ascii=False)); return
    config = load_config(args.config)
    if args.command == "profile":
        print(json.dumps(profile(config), ensure_ascii=False, indent=2)); return
    manifest = read_manifest(args.manifest, config.get("graph_id"), config.get("semantic", {}).get("cycle_policy", "path"))
    if args.command == "validate":
        reference = ReferenceGraph(load_canonical_edges(args.canonical))
        executors = build_executors(config, args.systems)
        try:
            summary = validate_static_reads(reference, executors, manifest, args.report_dir)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            if not summary["all_pass"]: raise SystemExit(2)
        finally:
            for executor in executors.values(): executor.close()
        return
    if args.command == "run":
        executor = build_executors(config, [args.system])[args.system]
        try:
            summary = run_trace(executor, manifest, args.out)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            if summary["errors"]: raise SystemExit(2)
        finally:
            executor.close()

if __name__ == "__main__":
    main()
