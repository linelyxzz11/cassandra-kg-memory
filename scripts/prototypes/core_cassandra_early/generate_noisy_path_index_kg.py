import argparse
import csv
import random
from pathlib import Path


NOISE_RELATIONS = [
    "mentions",
    "studies",
    "uses",
    "has_feature",
    "supports",
    "requires",
    "located_in",
    "prevents",
]


def make_edge(src_id, src_type, relation, dst_id, dst_type, source):
    return {
        "src_id": src_id,
        "src_type": src_type,
        "relation": relation,
        "dst_id": dst_id,
        "dst_type": dst_type,
        "confidence": round(random.uniform(0.55, 0.99), 2),
        "source": source,
    }


def add_noise_edges(edges, src_id, src_type, prefix, noise_per_node):
    for i in range(1, noise_per_node + 1):
        relation = NOISE_RELATIONS[(i - 1) % len(NOISE_RELATIONS)]
        dst_id = f"{prefix}_noise_{i:05d}"

        edges.append(
            make_edge(
                src_id=src_id,
                src_type=src_type,
                relation=relation,
                dst_id=dst_id,
                dst_type="noise",
                source="noisy_path_index_noise_edge",
            )
        )


def generate(branching, start_noise, noise_per_node):
    edges = []
    start = "user_path_mix"

    # Level 1: start node has target likes edges plus many irrelevant edges.
    for i in range(1, branching + 1):
        pref = f"preference_path_{i:05d}"
        need = f"need_path_{i:05d}"
        state = f"state_path_{i:05d}"
        strategy = f"strategy_path_{i:05d}"

        edges.append(
            make_edge(start, "user", "likes", pref, "preference", "path_target_edge")
        )

        edges.append(
            make_edge(pref, "preference", "suitable_for", need, "need", "path_target_edge")
        )

        edges.append(
            make_edge(need, "need", "related_to", state, "state", "path_target_edge")
        )

        edges.append(
            make_edge(state, "state", "suggests", strategy, "strategy", "path_target_edge")
        )

        # Add irrelevant relation edges to each intermediate path node.
        add_noise_edges(
            edges=edges,
            src_id=pref,
            src_type="preference",
            prefix=f"pref_{i:05d}",
            noise_per_node=noise_per_node,
        )

        add_noise_edges(
            edges=edges,
            src_id=need,
            src_type="need",
            prefix=f"need_{i:05d}",
            noise_per_node=noise_per_node,
        )

        add_noise_edges(
            edges=edges,
            src_id=state,
            src_type="state",
            prefix=f"state_{i:05d}",
            noise_per_node=noise_per_node,
        )

    add_noise_edges(
        edges=edges,
        src_id=start,
        src_type="user",
        prefix="start",
        noise_per_node=start_noise,
    )

    random.shuffle(edges)

    return edges


def write_csv(edges, output):
    output_path = Path(output)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "src_id",
            "src_type",
            "relation",
            "dst_id",
            "dst_type",
            "confidence",
            "source",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a noisy multi-hop path graph for relation-aware index benchmarks."
    )

    parser.add_argument("--branching", type=int, default=20)
    parser.add_argument("--start-noise", type=int, default=500)
    parser.add_argument("--noise-per-node", type=int, default=200)
    parser.add_argument("--output", default="edges_noisy_path_index.csv")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    edges = generate(
        branching=args.branching,
        start_noise=args.start_noise,
        noise_per_node=args.noise_per_node,
    )

    write_csv(edges, args.output)

    print("Noisy path index KG generated.")
    print("------------------------------")
    print(f"Branching       : {args.branching}")
    print(f"Start noise     : {args.start_noise}")
    print(f"Noise per node  : {args.noise_per_node}")
    print(f"Total edges     : {len(edges)}")
    print(f"Output file     : {args.output}")


if __name__ == "__main__":
    main()