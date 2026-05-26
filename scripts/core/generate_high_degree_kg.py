import argparse
import csv
import random
from pathlib import Path


def make_node_id(node_type, index):
    return f"{node_type}_{index:06d}"


def random_confidence():
    return round(random.uniform(0.55, 0.99), 2)


def add_edge(edges, seen, src_id, src_type, relation, dst_id, dst_type, source):
    key = (src_id, relation, dst_id)

    if key in seen:
        return False

    seen.add(key)

    edges.append({
        "src_id": src_id,
        "src_type": src_type,
        "relation": relation,
        "dst_id": dst_id,
        "dst_type": dst_type,
        "confidence": random_confidence(),
        "source": source,
    })

    return True


def generate_high_degree_kg(high_degree, tail_depth, noise_edges, seed):
    random.seed(seed)

    edges = []
    seen = set()
    root_node = "user_big"

    for index in range(1, high_degree + 1):
        entity = make_node_id("entity", index)
        feature = make_node_id("feature", index)
        need = make_node_id("need", index)
        state = make_node_id("state", index)
        strategy = make_node_id("strategy", index)

        chain = [
            (root_node, "user", "mentions", entity, "entity", "high_degree_root_edge"),
            (entity, "entity", "has_feature", feature, "feature", "high_degree_tail_edge"),
            (feature, "feature", "supports", need, "need", "high_degree_tail_edge"),
            (need, "need", "related_to", state, "state", "high_degree_tail_edge"),
            (state, "state", "suggests", strategy, "strategy", "high_degree_tail_edge"),
        ]

        for item in chain[:tail_depth]:
            src_id, src_type, relation, dst_id, dst_type, source = item

            add_edge(
                edges=edges,
                seen=seen,
                src_id=src_id,
                src_type=src_type,
                relation=relation,
                dst_id=dst_id,
                dst_type=dst_type,
                source=source,
            )

    for _ in range(noise_edges):
        src_id = make_node_id("entity", random.randint(1, high_degree))
        dst_id = make_node_id("entity", random.randint(1, high_degree))

        if src_id == dst_id:
            continue

        add_edge(
            edges=edges,
            seen=seen,
            src_id=src_id,
            src_type="entity",
            relation="related_to",
            dst_id=dst_id,
            dst_type="entity",
            source="noise_entity_related_to",
        )

    random.shuffle(edges)

    return edges


def write_csv(edges, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "src_id",
        "src_type",
        "relation",
        "dst_id",
        "dst_type",
        "confidence",
        "source",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)


def main():
    parser = argparse.ArgumentParser(
        description="Generate high-degree node KG data."
    )

    parser.add_argument(
        "--high-degree",
        type=int,
        default=5000,
        help="Number of outgoing edges from user_big."
    )

    parser.add_argument(
        "--tail-depth",
        type=int,
        default=4,
        help="Chain depth including user_big to entity."
    )

    parser.add_argument(
        "--noise-edges",
        type=int,
        default=1000,
        help="Number of random noise edges."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="edges_high_degree_21k.csv",
        help="Output CSV file."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()

    if not 1 <= args.tail_depth <= 5:
        raise ValueError("--tail-depth must be between 1 and 5.")

    edges = generate_high_degree_kg(
        high_degree=args.high_degree,
        tail_depth=args.tail_depth,
        noise_edges=args.noise_edges,
        seed=args.seed,
    )

    write_csv(edges, args.output)

    print("High-degree KG generated successfully.")
    print("High-degree node: user_big")
    print(f"user_big outgoing mentions edges: {args.high_degree}")
    print(f"Tail depth: {args.tail_depth}")
    print(f"Noise edges target: {args.noise_edges}")
    print(f"Total edges generated: {len(edges)}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
