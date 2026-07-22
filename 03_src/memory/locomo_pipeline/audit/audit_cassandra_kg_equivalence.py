import argparse
import csv
from pathlib import Path
from cassandra.cluster import Cluster

BASE = Path("D:/memorytable/cassandra-kg-memory")

EDGES_CSV = str(BASE / "results/locomo_kg_edges_spacy.csv")
MEMORY_CSV = str(BASE / "results/locomo_memory_records.csv")
OUT_AUDIT = str(BASE / "results/audit/cassandra_online_vs_offline_kg_signal_audit.csv")


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_offline_kg_set():
    edges = load_csv(EDGES_CSV)
    ev_ids = set()
    for r in edges:
        ev = r["evidence"].strip()
        if ev:
            ev_ids.add(ev)

    memories = load_csv(MEMORY_CSV)
    kg_set = set()
    for r in memories:
        mid = r["memory_id"].strip()
        for ev in ev_ids:
            if mid.endswith("_" + ev) or mid.endswith(ev):
                kg_set.add(mid)
                break

    kg_per_ev = {}
    for r in edges:
        ev = r["evidence"].strip()
        if ev not in kg_per_ev:
            kg_per_ev[ev] = r

    return kg_set, ev_ids, kg_per_ev


def query_cassandra_kg_set(host, port, keyspace):
    cluster = Cluster([host], port=port)
    session = cluster.connect(keyspace)

    query = "SELECT source FROM kg_edges_by_src"
    rows = session.execute(query)

    ev_ids = set()
    source_records = []
    for row in rows:
        src = row.source
        source_records.append(src)
        if "|" in src:
            _, ev = src.rsplit("|", 1)
            ev = ev.strip()
            if ev:
                ev_ids.add(ev)

    cluster.shutdown()

    memories = load_csv(MEMORY_CSV)
    kg_set = set()
    for r in memories:
        mid = r["memory_id"].strip()
        for ev in ev_ids:
            if mid.endswith("_" + ev) or mid.endswith(ev):
                kg_set.add(mid)
                break

    return kg_set, ev_ids, source_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9042)
    parser.add_argument("--keyspace", default="ai_memory")
    args = parser.parse_args()

    print("=" * 70)
    print("Offline-vs-Cassandra KG Signal Equivalence Audit")
    print("=" * 70)

    print("\n[1] Building OFFLINE KG set from CSV...")
    kg_off, ev_off, _ = build_offline_kg_set()
    print(f"    Unique evidence IDs (CSV): {len(ev_off)}")
    print(f"    KG memory set (CSV):       {len(kg_off)}")

    print(f"\n[2] Querying ONLINE KG set from Cassandra ({args.host}:{args.port}/{args.keyspace})...")
    kg_on, ev_on, src_records = query_cassandra_kg_set(args.host, args.port, args.keyspace)
    print(f"    Total source rows (C*):    {len(src_records)}")
    print(f"    Unique evidence IDs (C*):  {len(ev_on)}")
    print(f"    KG memory set (C*):        {len(kg_on)}")

    print(f"\n[3] Comparison...")
    same_set = (kg_off == kg_on)
    only_off = kg_off - kg_on
    only_on = kg_on - kg_off
    intersection = kg_off & kg_on

    same_set_rate = int(same_set)
    overlap_pct = 100 * len(intersection) / max(len(kg_off), 1)

    print(f"    Sets identical:            {same_set}")
    print(f"    Intersection:              {len(intersection)}")
    print(f"    Only in offline (CSV):     {len(only_off)}")
    print(f"    Only in online (C*):       {len(only_on)}")
    print(f"    Overlap rate:              {overlap_pct:.2f}%")

    results = [{
        "metric": "same_kg_set",
        "value": same_set_rate,
        "details": f"Offline={len(kg_off)}, Online={len(kg_on)}, Intersection={len(intersection)}, OnlyOff={len(only_off)}, OnlyOn={len(only_on)}",
    }, {
        "metric": "overlap_pct",
        "value": f"{overlap_pct:.4f}",
        "details": f"{len(intersection)}/{max(len(kg_off),1)}",
    }, {
        "metric": "offline_ev_count",
        "value": len(ev_off),
        "details": "Unique evidence IDs from CSV",
    }, {
        "metric": "online_ev_count",
        "value": len(ev_on),
        "details": f"Unique evidence IDs from Cassandra ({len(src_records)} source rows)",
    }, {
        "metric": "offline_kg_memory_count",
        "value": len(kg_off),
        "details": "Memory IDs with KG edges (CSV mapping)",
    }, {
        "metric": "online_kg_memory_count",
        "value": len(kg_on),
        "details": "Memory IDs with KG edges (Cassandra mapping)",
    }, {
        "metric": "conclusion",
        "value": "EQUIVALENT" if same_set else "MISMATCH",
        "details": "KG signal equivalence confirmed" if same_set else f"Discrepancy: only_off={len(only_off)}, only_on={len(only_on)}",
    }]

    fieldnames = ["metric", "value", "details"]
    write_csv(OUT_AUDIT, results, fieldnames)

    print(f"\n[4] Audit saved: {OUT_AUDIT}")

    if len(only_off) > 0 or len(only_on) > 0:
        detail_rows = []
        for mid in sorted(only_off):
            detail_rows.append({"side": "offline_only", "memory_id": mid})
        for mid in sorted(only_on):
            detail_rows.append({"side": "online_only", "memory_id": mid})
        detail_path = str(BASE / "results/audit/cassandra_online_vs_offline_kg_signal_discrepancies.csv")
        write_csv(detail_path, detail_rows, ["side", "memory_id"])
        print(f"    Discrepancy details: {detail_path}")

    print("\n" + "=" * 70)
    if same_set:
        print("CONCLUSION: Cassandra online KG signal == offline CSV KG signal.")
        print("Dense-bge+KG boost can be served identically through Cassandra.")
    else:
        print(f"WARNING: Mismatch detected ({len(only_off)} offline-only, {len(only_on)} online-only).")
        print("Investigate discrepancy before claiming equivalence.")
    print("=" * 70)


if __name__ == "__main__":
    main()
