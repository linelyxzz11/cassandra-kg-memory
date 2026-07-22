from __future__ import annotations

import json
import random
from pathlib import Path

from .canonical import write_manifest
from .models import QuerySpec


def generate_manifest_from_plan(plan_path: str | Path, output_path: str | Path, count: int, random_seed: int) -> list[QuerySpec]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    graph_id = str(plan["graph_id"])
    templates = plan.get("queries", [])
    if not templates:
        raise ValueError("query plan has no queries")
    weights = [float(item.get("weight", 1.0)) for item in templates]
    if any(weight <= 0 for weight in weights):
        raise ValueError("all query template weights must be > 0")
    rng = random.Random(random_seed)
    records = []
    for index in range(1, count + 1):
        item = rng.choices(templates, weights=weights, k=1)[0]
        records.append(QuerySpec.from_mapping({
            "query_id": f"read-{index:08d}", "graph_id": graph_id, "seed_id": item["seed_id"],
            "relation_path": item["relation_path"], "hop": item["hop"], "fanout": item["fanout"],
            "op_type": "read", "random_seed": random_seed, "cycle_policy": item.get("cycle_policy", "path"),
        }, graph_id))
    write_manifest(output_path, records)
    return records
