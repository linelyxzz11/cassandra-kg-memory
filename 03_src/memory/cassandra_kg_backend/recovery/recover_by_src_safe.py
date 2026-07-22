import argparse
import csv
import hashlib
import json
import os
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent_with_args


KEYSPACE = "ai_memory"
TABLE = "kg_edges_by_src"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def logical_key(row):
    return (
        row["graph_id"],
        row["src_id"],
        row["relation"],
        row["dst_id"],
        row["source"],
    )


def norm_row(raw, graph_id):
    lower = {k.strip().lower(): v for k, v in raw.items()}

    def pick(*names, default=None):
        for n in names:
            if n in lower and lower[n] not in (None, ""):
                return str(lower[n])
        return default

    src_id = pick("src_id", "src", "source_id")
    dst_id = pick("dst_id", "dst", "target_id")
    relation = pick("relation", "rel", "predicate")
    source = pick("source", default="synthetic")
    src_type = pick("src_type", default="ENTITY")
    dst_type = pick("dst_type", default="ENTITY")

    conf_raw = pick("confidence", default="1.0")
    try:
        confidence = float(conf_raw)
    except Exception:
        confidence = 1.0

    if not src_id or not dst_id or not relation:
        raise ValueError(f"CSV row missing src_id/dst_id/relation-like columns: {raw}")

    return {
        "graph_id": graph_id,
        "src_id": src_id,
        "relation": relation,
        "dst_id": dst_id,
        "source": source,
        "src_type": src_type,
        "dst_type": dst_type,
        "confidence": confidence,
    }


def load_csv_edges(csv_path: Path, graph_id: str):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        print(f"[CSV] columns={reader.fieldnames}")
        for raw in reader:
            rows.append(norm_row(raw, graph_id))

    keys = [logical_key(r) for r in rows]
    dup_count = len(keys) - len(set(keys))
    print(f"[CSV] raw_rows={len(rows)} distinct_logical={len(set(keys))} duplicate_logical={dup_count}")

    if dup_count != 0:
        c = Counter(keys)
        examples = [k for k, v in c.items() if v > 1][:20]
        raise RuntimeError(f"CSV itself has duplicate logical edges. examples={examples}")

    return rows


def connect(host="127.0.0.1", port=9042):
    cluster = Cluster([host], port=port)
    session = cluster.connect(KEYSPACE)
    return cluster, session


def print_schema(cluster, out_dir: Path):
    ks = cluster.metadata.keyspaces[KEYSPACE]
    table = ks.tables[TABLE]
    schema = table.as_cql_query()
    print("\n[SCHEMA]")
    print(schema)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "by_src_schema.txt").write_text(schema, encoding="utf-8")

    pk = [c.name for c in table.partition_key]
    ck = [c.name for c in table.clustering_key]
    print(f"[PK] partition_key={pk} clustering_key={ck}")
    return pk, ck


def make_manifest(csv_path: Path, graph_id: str, manifest_path: Path, regen=False):
    meta_path = manifest_path.with_suffix(".meta.json")
    source_sha = sha256_file(csv_path)

    if manifest_path.exists() and meta_path.exists() and not regen:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source_sha256") == source_sha and meta.get("graph_id") == graph_id:
            print(f"[MANIFEST] reuse existing {manifest_path}")
            return

    print(f"[MANIFEST] generating {manifest_path}")
    rows = load_csv_edges(csv_path, graph_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "graph_id", "src_id", "relation", "dst_id", "source",
            "src_type", "dst_type", "confidence", "edge_id", "created_at",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        edge_ids = set()
        for r in rows:
            # uuid.uuid1() is valid for Cassandra timeuuid.
            eid = str(uuid.uuid1())
            if eid in edge_ids:
                raise RuntimeError("uuid.uuid1 collision, extremely unexpected")
            edge_ids.add(eid)
            out = dict(r)
            out["edge_id"] = eid
            out["created_at"] = created_at.isoformat()
            writer.writerow(out)

    meta = {
        "graph_id": graph_id,
        "source_csv": str(csv_path),
        "source_sha256": source_sha,
        "manifest_rows": len(rows),
        "logical_distinct": len({logical_key(r) for r in rows}),
        "edge_id_distinct": len(edge_ids),
        "created_at": created_at.isoformat(),
        "generated_at": now_iso(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[MANIFEST] rows={meta['manifest_rows']} edge_id_distinct={meta['edge_id_distinct']}")


def load_manifest(manifest_path: Path):
    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "graph_id": r["graph_id"],
                "src_id": r["src_id"],
                "relation": r["relation"],
                "dst_id": r["dst_id"],
                "source": r["source"],
                "src_type": r.get("src_type") or "ENTITY",
                "dst_type": r.get("dst_type") or "ENTITY",
                "confidence": float(r.get("confidence") or 1.0),
                "edge_id": uuid.UUID(r["edge_id"]),
                "created_at": datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")),
            })

    keys = [logical_key(r) for r in rows]
    eids = [r["edge_id"] for r in rows]
    print(f"[MANIFEST] loaded rows={len(rows)} logical_distinct={len(set(keys))} edge_id_distinct={len(set(eids))}")

    if len(rows) != len(set(keys)):
        raise RuntimeError("Manifest has duplicate logical keys")
    if len(rows) != len(set(eids)):
        raise RuntimeError("Manifest has duplicate edge_id")

    return rows


def truncate_table(session):
    print(f"[TRUNCATE] {KEYSPACE}.{TABLE}")
    session.execute(f"TRUNCATE {KEYSPACE}.{TABLE}")
    time.sleep(2.0)
    print("[TRUNCATE] done")


def insert_rows(session, rows, concurrency=64, max_retries=3):
    cql = f"""
    INSERT INTO {TABLE}
    (graph_id, src_id, relation, dst_id, edge_id,
     src_type, dst_type, confidence, source, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    print("[INSERT CQL]")
    print(cql)

    stmt = session.prepare(cql)

    def params_from_row(r):
        return (
            r["graph_id"], r["src_id"], r["relation"], r["dst_id"], r["edge_id"],
            r["src_type"], r["dst_type"], r["confidence"], r["source"], r["created_at"],
        )

    pending = list(rows)
    total_ok = 0
    attempts_log = []
    start = time.time()

    for attempt in range(1, max_retries + 2):
        if not pending:
            break

        print(f"[IMPORT] attempt={attempt} pending={len(pending)} concurrency={concurrency}")
        params = [params_from_row(r) for r in pending]

        results = execute_concurrent_with_args(
            session,
            stmt,
            params,
            concurrency=concurrency,
            raise_on_first_error=False,
        )

        failed = []
        for r, (success, result_or_exc) in zip(pending, results):
            key_str = "|".join(logical_key(r))
            key_hash = hashlib.sha1(key_str.encode("utf-8")).hexdigest()
            attempts_log.append({
                "logical_key_hash": key_hash,
                "edge_id": str(r["edge_id"]),
                "attempt": attempt,
                "success": bool(success),
                "error": None if success else repr(result_or_exc),
            })
            if success:
                total_ok += 1
            else:
                failed.append(r)

        print(f"[IMPORT] attempt={attempt} ok_this_round={len(pending)-len(failed)} failed={len(failed)}")
        pending = failed

    elapsed = time.time() - start
    print(f"[IMPORT] done ok={total_ok} still_failed={len(pending)} elapsed_s={elapsed:.1f} rows_per_s={len(rows)/max(elapsed,0.001):.1f}")

    if pending:
        raise RuntimeError(f"Import failed after retries: {len(pending)} rows still failed")

    return {
        "attempted_success_count_including_retries": total_ok,
        "input_rows": len(rows),
        "elapsed_s": elapsed,
        "rows_per_s": len(rows) / max(elapsed, 0.001),
        "attempts_log": attempts_log,
    }


def validate_graph(session, manifest_rows, out_dir: Path, label: str, dump_dups=True):
    graph_id = manifest_rows[0]["graph_id"]
    expected = {logical_key(r) for r in manifest_rows}
    src_ids = sorted({r["src_id"] for r in manifest_rows})

    select_stmt = session.prepare(
        f"""
        SELECT src_id, relation, dst_id, source, edge_id, created_at
        FROM {TABLE}
        WHERE graph_id=? AND src_id=?
        """
    )

    raw_rows = []
    for i, src in enumerate(src_ids, 1):
        rs = session.execute(select_stmt, (graph_id, src))
        for row in rs:
            raw_rows.append({
                "graph_id": graph_id,
                "src_id": row.src_id,
                "relation": row.relation,
                "dst_id": row.dst_id,
                "source": row.source,
                "edge_id": str(row.edge_id),
                "created_at": str(row.created_at),
            })
        if i % 1000 == 0:
            print(f"[VALIDATE] scanned src_partitions={i}/{len(src_ids)} raw_so_far={len(raw_rows)}")

    by_key = defaultdict(list)
    for r in raw_rows:
        k = (r["graph_id"], r["src_id"], r["relation"], r["dst_id"], r["source"])
        by_key[k].append(r)

    actual = set(by_key.keys())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    dup_groups = {k: v for k, v in by_key.items() if len(v) > 1}

    summary = {
        "label": label,
        "graph_id": graph_id,
        "expected_logical_edges": len(expected),
        "by_src_raw_rows": len(raw_rows),
        "by_src_distinct_logical_edges": len(actual),
        "duplicate_logical_edges": len(dup_groups),
        "duplicate_physical_rows_extra": sum(len(v) - 1 for v in dup_groups.values()),
        "missing_logical_edges": len(missing),
        "extra_logical_edges": len(extra),
        "validated_at": now_iso(),
    }

    print("[VALIDATE SUMMARY]")
    print(json.dumps(summary, indent=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if dump_dups:
        dup_path = out_dir / f"{label}_duplicate_groups.jsonl"
        with dup_path.open("w", encoding="utf-8") as f:
            for idx, (k, rows) in enumerate(dup_groups.items()):
                if idx >= 1000:
                    break
                obj = {
                    "logical_key": {
                        "graph_id": k[0], "src_id": k[1], "relation": k[2], "dst_id": k[3], "source": k[4],
                    },
                    "count": len(rows),
                    "edge_ids": [r["edge_id"] for r in rows],
                    "created_at": [r["created_at"] for r in rows],
                }
                f.write(json.dumps(obj) + "\n")

    return summary


def run_probe(session, out_dir: Path):
    graph_id = f"restore_probe_{int(time.time())}"
    rows = []
    created_at = datetime.now(timezone.utc)
    for i in range(1000):
        rows.append({
            "graph_id": graph_id,
            "src_id": f"probe_src_{i:06d}",
            "relation": "probe_rel",
            "dst_id": f"probe_dst_{i:06d}",
            "source": "restore_probe",
            "src_type": "ENTITY",
            "dst_type": "ENTITY",
            "confidence": 1.0,
            "edge_id": uuid.uuid1(),
            "created_at": created_at,
        })

    print(f"[PROBE] graph_id={graph_id} rows=1000")
    import_result = insert_rows(session, rows, concurrency=1, max_retries=0)
    summary = validate_graph(session, rows, out_dir, "probe_1000", dump_dups=True)
    summary["import"] = {k: v for k, v in import_result.items() if k != "attempts_log"}
    (out_dir / "probe_1000_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if summary["duplicate_logical_edges"] != 0:
        raise RuntimeError("Probe produced duplicates. Inspect schema/Cassandra state.")
    print("[PROBE] PASS: Cassandra does not duplicate a clean 1000-row serial insert.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9042)
    parser.add_argument("--out-dir", default=r"D:\memorytable\cassandra-kg-memory\reports\data_recovery")
    parser.add_argument("--probe", action="store_true", help="Run isolated 1000-edge probe, no truncate.")
    parser.add_argument("--truncate", action="store_true", help="TRUNCATE kg_edges_by_src before restore.")
    parser.add_argument("--csv", help="Source CSV path.")
    parser.add_argument("--graph-id", help="Graph id to import.")
    parser.add_argument("--label", default="restore")
    parser.add_argument("--manifest", help="Manifest path. Default: out-dir/<label>_manifest.csv")
    parser.add_argument("--regen-manifest", action="store_true")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster, session = connect(args.host, args.port)
    try:
        print_schema(cluster, out_dir)

        if args.probe:
            run_probe(session, out_dir)
            return

        if not args.csv or not args.graph_id:
            print("Need --csv and --graph-id unless --probe is used.")
            sys.exit(2)

        csv_path = Path(args.csv)
        manifest_path = Path(args.manifest) if args.manifest else out_dir / f"{args.label}_manifest.csv"

        if args.truncate:
            truncate_table(session)

        make_manifest(csv_path, args.graph_id, manifest_path, regen=args.regen_manifest)
        rows = load_manifest(manifest_path)

        import_result = insert_rows(session, rows, concurrency=args.concurrency, max_retries=args.max_retries)

        attempts_path = out_dir / f"{args.label}_import_attempts.jsonl"
        with attempts_path.open("w", encoding="utf-8") as f:
            for obj in import_result["attempts_log"]:
                f.write(json.dumps(obj) + "\n")

        summary = validate_graph(session, rows, out_dir, args.label, dump_dups=True)
        summary["source_csv"] = str(csv_path)
        summary["manifest"] = str(manifest_path)
        summary["import_elapsed_s"] = round(import_result["elapsed_s"], 3)
        summary["import_rows_per_s"] = round(import_result["rows_per_s"], 1)
        summary["concurrency"] = args.concurrency
        summary["max_retries"] = args.max_retries

        (out_dir / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if (
            summary["by_src_raw_rows"] == summary["expected_logical_edges"]
            and summary["by_src_distinct_logical_edges"] == summary["expected_logical_edges"]
            and summary["duplicate_logical_edges"] == 0
            and summary["missing_logical_edges"] == 0
            and summary["extra_logical_edges"] == 0
        ):
            print("[RESTORE] PHYSICAL CLEAN PASS")
        else:
            print("[RESTORE] PHYSICAL CLEAN FAIL")
            print(f"See duplicate file: {out_dir / (args.label + '_duplicate_groups.jsonl')}")
            sys.exit(1)

    finally:
        cluster.shutdown()


if __name__ == "__main__":
    main()