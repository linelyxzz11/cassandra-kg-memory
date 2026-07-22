"""
C0-A minimal correctness gate:
Cassandra naive traversal (workers=1) vs frontier-parallel traversal.

Only uses:
    ai_memory.kg_edges_by_src

Does NOT use:
    cache
    relation index
    Neo4j
    LoCoMo import
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from cassandra.cluster import Cluster


SELECT_ONE_HOP = """
SELECT graph_id, src_id, relation, dst_id, source
FROM kg_edges_by_src
WHERE graph_id = ? AND src_id = ?
"""


@dataclass(frozen=True, order=True)
class Edge:
    graph_id: str
    src_id: str
    relation: str
    dst_id: str
    source: str

    def logical_id(self) -> tuple[str, str, str, str, str]:
        return (
            self.graph_id,
            self.src_id,
            self.relation,
            self.dst_id,
            self.source,
        )


@dataclass(frozen=True)
class PathState:
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C0-A: compare Cassandra naive vs frontier-parallel traversal."
    )

    parser.add_argument("--graph-id", required=True)
    parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated seed nodes, e.g. user_000001,user_000002",
    )
    parser.add_argument("--hop", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--fanout", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)

    parser.add_argument(
        "--relation-path",
        default="",
        help=(
            "Optional comma-separated relation list. "
            "Must contain exactly one relation per hop. "
            "Example: likes,suitable_for"
        ),
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9042)
    parser.add_argument("--keyspace", default="ai_memory")
    parser.add_argument("--out", required=True)

    return parser.parse_args()


def parse_relation_path(raw: str, hop: int) -> tuple[str, ...] | None:
    relations = tuple(item.strip() for item in raw.split(",") if item.strip())

    if not relations:
        return None

    if len(relations) != hop:
        raise ValueError(
            f"--relation-path has {len(relations)} relations, "
            f"but --hop is {hop}. "
            "Provide one relation per hop, or omit --relation-path."
        )

    return relations


class CassandraOneHopReader:
    def __init__(self, host: str, port: int, keyspace: str) -> None:
        self.cluster = Cluster([host], port=port)
        self.session = self.cluster.connect(keyspace)
        self.statement = self.session.prepare(SELECT_ONE_HOP)

    def fetch(self, graph_id: str, src_id: str) -> list[Edge]:
        rows = self.session.execute(self.statement, (graph_id, src_id))

        edges = [
            Edge(
                graph_id=str(row.graph_id),
                src_id=str(row.src_id),
                relation=str(row.relation),
                dst_id=str(row.dst_id),
                source="" if row.source is None else str(row.source),
            )
            for row in rows
        ]

        # Fixed deterministic ordering.
        return sorted(
            edges,
            key=lambda edge: (
                edge.relation,
                edge.dst_id,
                edge.source,
            ),
        )

    def close(self) -> None:
        self.cluster.shutdown()


def traverse(
    reader: CassandraOneHopReader,
    graph_id: str,
    seed_id: str,
    hop: int,
    fanout: int,
    workers: int,
    relation_path: tuple[str, ...] | None,
) -> tuple[set[tuple[tuple[str, str, str, str, str], ...]], dict]:
    """
    Same traversal semantics for naive and parallel modes.

    Only difference:
        workers=1  -> serial one-hop fetches
        workers>1  -> frontier-level parallel one-hop fetches
    """

    frontier = [
        PathState(
            nodes=(seed_id,),
            edges=(),
        )
    ]

    raw_edges_read = 0
    one_hop_queries = 0

    for depth in range(hop):
        if not frontier:
            break

        # Deduplicate source nodes for physical Cassandra reads.
        frontier_sources = sorted({state.nodes[-1] for state in frontier})
        fetched_by_source: dict[str, list[Edge]] = {}

        # Naive / serial mode.
        if workers == 1:
            for src_id in frontier_sources:
                rows = reader.fetch(graph_id, src_id)
                fetched_by_source[src_id] = rows
                raw_edges_read += len(rows)
                one_hop_queries += 1

        # Frontier parallel mode.
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_source = {
                    executor.submit(reader.fetch, graph_id, src_id): src_id
                    for src_id in frontier_sources
                }

                for future in as_completed(future_to_source):
                    src_id = future_to_source[future]
                    rows = future.result()

                    fetched_by_source[src_id] = rows
                    raw_edges_read += len(rows)
                    one_hop_queries += 1

        required_relation = None
        if relation_path is not None:
            required_relation = relation_path[depth]

        next_frontier: list[PathState] = []

        for path_state in frontier:
            current_node = path_state.nodes[-1]
            candidates = fetched_by_source[current_node]

            # Relation filter first.
            if required_relation is not None:
                candidates = [
                    edge
                    for edge in candidates
                    if edge.relation == required_relation
                ]

            # Path-cycle prevention, then deterministic fanout.
            selected_edges: list[Edge] = []

            for edge in candidates:
                if edge.dst_id in path_state.nodes:
                    continue

                selected_edges.append(edge)

                if len(selected_edges) >= fanout:
                    break

            for edge in selected_edges:
                next_frontier.append(
                    PathState(
                        nodes=path_state.nodes + (edge.dst_id,),
                        edges=path_state.edges + (edge,),
                    )
                )

        frontier = next_frontier

    final_paths = {
        tuple(edge.logical_id() for edge in state.edges)
        for state in frontier
    }

    stats = {
        "raw_edges_read": raw_edges_read,
        "one_hop_queries": one_hop_queries,
        "result_paths": len(final_paths),
    }

    return final_paths, stats


def latency_summary(values: list[float]) -> dict:
    if not values:
        return {
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }

    ordered = sorted(values)

    def percentile(p: float) -> float:
        index = int((len(ordered) - 1) * p)
        return ordered[index]

    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }


def main() -> int:
    args = parse_args()

    if args.fanout < 1:
        raise ValueError("--fanout must be at least 1")

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    seeds = [
        item.strip()
        for item in args.seeds.split(",")
        if item.strip()
    ]

    if not seeds:
        raise ValueError("--seeds must contain at least one seed node")

    relation_path = parse_relation_path(args.relation_path, args.hop)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    naive_latencies = []
    parallel_latencies = []
    mismatches = []

    reader = CassandraOneHopReader(
        host=args.host,
        port=args.port,
        keyspace=args.keyspace,
    )

    try:
        for index, seed_id in enumerate(seeds, start=1):
            # Naive: serial.
            start = time.perf_counter_ns()

            naive_paths, naive_stats = traverse(
                reader=reader,
                graph_id=args.graph_id,
                seed_id=seed_id,
                hop=args.hop,
                fanout=args.fanout,
                workers=1,
                relation_path=relation_path,
            )

            naive_ms = (time.perf_counter_ns() - start) / 1_000_000
            naive_latencies.append(naive_ms)

            # Parallel: same semantics, only concurrent one-hop fetches.
            start = time.perf_counter_ns()

            parallel_paths, parallel_stats = traverse(
                reader=reader,
                graph_id=args.graph_id,
                seed_id=seed_id,
                hop=args.hop,
                fanout=args.fanout,
                workers=args.workers,
                relation_path=relation_path,
            )

            parallel_ms = (time.perf_counter_ns() - start) / 1_000_000
            parallel_latencies.append(parallel_ms)

            matched = naive_paths == parallel_paths

            if not matched:
                mismatches.append(
                    {
                        "query_id": f"c0-a-{index:04d}",
                        "graph_id": args.graph_id,
                        "seed_id": seed_id,
                        "hop": args.hop,
                        "fanout": args.fanout,
                        "relation_path": (
                            list(relation_path)
                            if relation_path is not None
                            else None
                        ),
                        "missing_in_parallel": [
                            [list(edge) for edge in path]
                            for path in sorted(naive_paths - parallel_paths)
                        ],
                        "unexpected_in_parallel": [
                            [list(edge) for edge in path]
                            for path in sorted(parallel_paths - naive_paths)
                        ],
                        "naive_stats": naive_stats,
                        "parallel_stats": parallel_stats,
                    }
                )

            print(
                f"[{index}/{len(seeds)}] "
                f"seed={seed_id} | "
                f"paths={len(naive_paths)} | "
                f"naive={naive_ms:.3f} ms | "
                f"parallel={parallel_ms:.3f} ms | "
                f"{'MATCH' if matched else 'MISMATCH'}"
            )

    finally:
        reader.close()

    mismatch_file = out_dir / "mismatches.jsonl"

    with mismatch_file.open("w", encoding="utf-8") as file:
        for mismatch in mismatches:
            file.write(json.dumps(mismatch, ensure_ascii=False) + "\n")

    summary = {
        "experiment": "C0-A Cassandra naive vs frontier-parallel semantic gate",
        "graph_id": args.graph_id,
        "seeds": seeds,
        "hop": args.hop,
        "fanout": args.fanout,
        "relation_path": (
            list(relation_path)
            if relation_path is not None
            else None
        ),
        "naive_workers": 1,
        "parallel_workers": args.workers,
        "cache": "disabled",
        "relation_index": "disabled",
        "checked_queries": len(seeds),
        "disagreements": len(mismatches),
        "all_pass": len(mismatches) == 0,
        "naive_latency": latency_summary(naive_latencies),
        "parallel_latency": latency_summary(parallel_latencies),
        "mismatch_file": str(mismatch_file),
    }

    summary_file = out_dir / "summary.json"

    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== C0-A SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())