import argparse
import csv
import random
from pathlib import Path


RELATIONS = [
    "likes",
    "studies",
    "mentions",
    "uses",
    "related_to",
    "has_feature",
    "supports",
    "requires",
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


def generate_edges(src_id, target_relation, target_count, noise_count):
    edges = []

    # Target relation edges are the ones the index query should retrieve.
    for i in range(1, target_count + 1):
        dst_id = f"{target_relation}_target_{i:06d}"
        edges.append(
            make_edge(
                src_id=src_id,
                src_type="user",
                relation=target_relation,
                dst_id=dst_id,
                dst_type="entity",
                source="multirelation_target_edge",
            )
        )

    other_relations = [
        relation for relation in RELATIONS
        if relation != target_relation
    ]

    # Noise relation edges simulate irrelevant outgoing edges under the same src_id.
    for i in range(1, noise_count + 1):
        relation = other_relations[(i - 1) % len(other_relations)]
        dst_id = f"{relation}_noise_{i:06d}"

        edges.append(
            make_edge(
                src_id=src_id,
                src_type="user",
                relation=relation,
                dst_id=dst_id,
                dst_type="entity",
                source="multirelation_noise_edge",
            )
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
        description="Generate a multi-relation high-degree node for relation index benchmarks."
    )

    parser.add_argument("--src", default="user_mix")
    parser.add_argument("--target-relation", default="likes")
    parser.add_argument("--target-count", type=int, default=1000)
    parser.add_argument("--noise-count", type=int, default=9000)
    parser.add_argument("--output", default="edges_multirelation_index_10k.csv")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    edges = generate_edges(
        src_id=args.src,
        target_relation=args.target_relation,
        target_count=args.target_count,
        noise_count=args.noise_count,
    )

    write_csv(edges, args.output)

    print("Multi-relation index KG generated.")
    print("----------------------------------")
    print(f"Source node      : {args.src}")
    print(f"Target relation  : {args.target_relation}")
    print(f"Target edges     : {args.target_count}")
    print(f"Noise edges      : {args.noise_count}")
    print(f"Total edges      : {len(edges)}")
    print(f"Output file      : {args.output}")


if __name__ == "__main__":
    main()