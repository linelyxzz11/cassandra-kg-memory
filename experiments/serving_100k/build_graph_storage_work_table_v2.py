"""Create the publication-facing storage/work table from the frozen 100K gate."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "serving_100k" / "locomo_workload_graph_v2_100k" / "graph_100k_load_gate_summary.json"
OUT = ROOT / "results" / "serving_100k" / "locomo_workload_graph_v2_100k" / "storage_work_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("target_memories") != 100000:
        raise ValueError("100K graph-aware load gate is not publication-ready")
    c = payload["cassandra_table_counts"]
    n = payload["neo4j_graph_counts"]
    mutations = payload["attempted_logical_mutations"]
    load_seconds = {item["stage"].split(":", 1)[1]: item["seconds"] for item in payload["stage_times"] if item["stage"].startswith("load:")}
    rows = [
        {
            "cell": "Cassandra-base",
            "memory_rows_or_nodes": c["g_base_memory_by_scope"],
            "feature_rows_or_nodes": c["g_base_features_by_memory"],
            "entity_dictionary_records": c["g_base_entities_by_scope"],
            "mention_records": c["g_base_mentions_by_memory"],
            "semantic_edges": c["g_base_edges_by_memory"],
            "physical_edge_records": c["g_base_edges_by_memory"] + c["g_base_edges_by_src"] + c["g_base_edges_by_scope"],
            "materialized_candidate_records": 0,
            "actual_storage_records": payload["actual_storage_records"]["cassandra-base"],
            "logical_mutations_mean": mutations["cassandra-base"]["mean"],
            "logical_mutations_p95": mutations["cassandra-base"]["p95"],
            "load_seconds": load_seconds["cassandra-base"],
        },
        {
            "cell": "Cassandra-materialized",
            "memory_rows_or_nodes": c["g_mat_memory_by_scope"],
            "feature_rows_or_nodes": 0,
            "entity_dictionary_records": c["g_mat_entities_by_scope"],
            "mention_records": c["g_mat_mentions_by_memory"],
            "semantic_edges": c["g_mat_edges_by_memory"],
            "physical_edge_records": c["g_mat_edges_by_memory"],
            "materialized_candidate_records": c["g_mat_by_scope_relation"],
            "actual_storage_records": payload["actual_storage_records"]["cassandra-materialized"],
            "logical_mutations_mean": mutations["cassandra-materialized"]["mean"],
            "logical_mutations_p95": mutations["cassandra-materialized"]["p95"],
            "load_seconds": load_seconds["cassandra-materialized"],
        },
        {
            "cell": "Neo4j-native",
            "memory_rows_or_nodes": n["LWV2NativeMemory"],
            "feature_rows_or_nodes": n["LWV2NativeFeature"],
            "entity_dictionary_records": n["LWV2NativeEntity"],
            "mention_records": n["LWV2_NATIVE_MENTIONS"],
            "semantic_edges": n["LWV2_NATIVE_REL"],
            "physical_edge_records": n["LWV2_NATIVE_REL"],
            "materialized_candidate_records": 0,
            "actual_storage_records": payload["actual_storage_records"]["neo4j-native"],
            "logical_mutations_mean": mutations["neo4j-native"]["mean"],
            "logical_mutations_p95": mutations["neo4j-native"]["p95"],
            "load_seconds": load_seconds["neo4j-native"],
        },
        {
            "cell": "Neo4j-materialized",
            "memory_rows_or_nodes": n["LWV2MatMemory"],
            "feature_rows_or_nodes": 0,
            "entity_dictionary_records": n["LWV2MatEntity"],
            "mention_records": n["LWV2_MAT_MENTIONS"],
            "semantic_edges": n["LWV2_MAT_REL"],
            "physical_edge_records": n["LWV2_MAT_REL"],
            "materialized_candidate_records": n["LWV2MatCandidate"],
            "actual_storage_records": payload["actual_storage_records"]["neo4j-materialized"],
            "logical_mutations_mean": mutations["neo4j-materialized"]["mean"],
            "logical_mutations_p95": mutations["neo4j-materialized"]["p95"],
            "load_seconds": load_seconds["neo4j-materialized"],
        },
    ]
    for row in rows:
        if row["memory_rows_or_nodes"] != 100000 or row["semantic_edges"] != payload["edges"]:
            raise ValueError(f"semantic work mismatch for {row['cell']}")
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "table9a_storage_work.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Table 9A: Actual storage work at 100K",
        "",
        "All four cells store the same 100,000 logical memories, 76,475 entity dictionary records, 129,156 mentions, and 42,944 semantic edges. Physical records differ because the native and query-materialized schemas encode that semantic work differently.",
        "",
        "| Cell | Memory | Feature | Entity | Mentions | Semantic edges | Physical edge rows/rels | Materialized candidates | Total records | Writes/event mean | Writes/event p95 | Load s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['cell']} | {row['memory_rows_or_nodes']} | {row['feature_rows_or_nodes']} | {row['entity_dictionary_records']} | "
            f"{row['mention_records']} | {row['semantic_edges']} | {row['physical_edge_records']} | {row['materialized_candidate_records']} | "
            f"{row['actual_storage_records']} | {row['logical_mutations_mean']:.3f} | {row['logical_mutations_p95']:.1f} | {row['load_seconds']:.2f} |"
        )
    report += [
        "",
        "Interpretation: `actual_storage_records` counts rows/nodes/relationships, not bytes on disk. Cross-engine byte footprint is not reported because Cassandra SSTable and Neo4j store files include different compaction, transaction-log, and allocation overheads; presenting raw directory size as a fair schema comparison would be misleading.",
        "",
    ]
    report_path = OUT / "TABLE9A_STORAGE_WORK.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    manifest_path = OUT / "manifest.json"
    manifest = {
        "status": "PASS",
        "protocol_id": "graph-aware-storage-work-v2-100k",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": sha256(SOURCE)},
        "semantic_parity": {"memories": 100000, "entities": 76475, "mentions": 129156, "edges": 42944},
        "outputs": [
            {"path": str(csv_path.relative_to(ROOT)), "sha256": sha256(csv_path)},
            {"path": str(report_path.relative_to(ROOT)), "sha256": sha256(report_path)},
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(rows), "output": str(OUT)}))


if __name__ == "__main__":
    main()
