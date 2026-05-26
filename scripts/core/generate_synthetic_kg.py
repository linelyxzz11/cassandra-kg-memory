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


def add_preference_chain(edges, seen, user_id, chain_id):
    preference = make_node_id("preference", chain_id)
    need = make_node_id("need", chain_id)
    state = make_node_id("state", chain_id)
    strategy = make_node_id("strategy", chain_id)
    action = make_node_id("action", chain_id)

    chain = [
        (user_id, "user", "likes", preference, "preference"),
        (preference, "preference", "suitable_for", need, "need"),
        (need, "need", "related_to", state, "state"),
        (state, "state", "suggests", strategy, "strategy"),
        (strategy, "strategy", "leads_to", action, "action"),
    ]

    for src_id, src_type, relation, dst_id, dst_type in chain:
        add_edge(
            edges=edges,
            seen=seen,
            src_id=src_id,
            src_type=src_type,
            relation=relation,
            dst_id=dst_id,
            dst_type=dst_type,
            source="structured_preference_chain",
        )


def add_entity_feature_chain(edges, seen, chain_id):
    entity = make_node_id("entity", chain_id)
    feature = make_node_id("feature", chain_id)
    need = make_node_id("need", chain_id + 100000)
    state = make_node_id("state", chain_id + 100000)
    strategy = make_node_id("strategy", chain_id + 100000)
    action = make_node_id("action", chain_id + 100000)

    chain = [
        (entity, "entity", "has_feature", feature, "feature"),
        (feature, "feature", "supports", need, "need"),
        (need, "need", "related_to", state, "state"),
        (state, "state", "suggests", strategy, "strategy"),
        (strategy, "strategy", "leads_to", action, "action"),
    ]

    for src_id, src_type, relation, dst_id, dst_type in chain:
        add_edge(
            edges=edges,
            seen=seen,
            src_id=src_id,
            src_type=src_type,
            relation=relation,
            dst_id=dst_id,
            dst_type=dst_type,
            source="structured_entity_feature_chain",
        )


def add_study_chain(edges, seen, user_id, chain_id):
    topic = make_node_id("topic", chain_id)
    technology = make_node_id("technology", chain_id)
    task = make_node_id("task", chain_id)
    design = make_node_id("design", chain_id)
    mechanism = make_node_id("mechanism", chain_id)

    chain = [
        (user_id, "user", "studies", topic, "topic"),
        (topic, "topic", "includes", technology, "technology"),
        (technology, "technology", "used_for", task, "task"),
        (task, "task", "requires", design, "design"),
        (design, "design", "uses", mechanism, "mechanism"),
    ]

    for src_id, src_type, relation, dst_id, dst_type in chain:
        add_edge(
            edges=edges,
            seen=seen,
            src_id=src_id,
            src_type=src_type,
            relation=relation,
            dst_id=dst_id,
            dst_type=dst_type,
            source="structured_study_chain",
        )


def add_task_design_chain(edges, seen, chain_id):
    task = make_node_id("task", chain_id + 100000)
    design = make_node_id("design", chain_id + 100000)
    mechanism = make_node_id("mechanism", chain_id + 100000)
    problem = make_node_id("problem", chain_id)
    solution = make_node_id("solution", chain_id)
    outcome = make_node_id("outcome", chain_id)

    chain = [
        (task, "task", "requires", design, "design"),
        (design, "design", "uses", mechanism, "mechanism"),
        (mechanism, "mechanism", "prevents", problem, "problem"),
        (problem, "problem", "solved_by", solution, "solution"),
        (solution, "solution", "improves", outcome, "outcome"),
    ]

    for src_id, src_type, relation, dst_id, dst_type in chain:
        add_edge(
            edges=edges,
            seen=seen,
            src_id=src_id,
            src_type=src_type,
            relation=relation,
            dst_id=dst_id,
            dst_type=dst_type,
            source="structured_task_design_chain",
        )


NOISE_RULES = [
    ("user", "mentions", "entity"),
    ("user", "likes", "preference"),
    ("user", "studies", "topic"),
    ("entity", "has_feature", "feature"),
    ("entity", "located_in", "location"),
    ("entity", "related_to", "entity"),
    ("feature", "supports", "need"),
    ("preference", "suitable_for", "need"),
    ("need", "related_to", "state"),
    ("state", "suggests", "strategy"),
    ("strategy", "leads_to", "action"),
    ("topic", "related_to", "topic"),
    ("topic", "includes", "technology"),
    ("technology", "used_for", "task"),
    ("task", "requires", "design"),
    ("design", "uses", "mechanism"),
    ("mechanism", "prevents", "problem"),
]


def add_noise_edge(edges, seen, user_ids, node_range):
    src_type, relation, dst_type = random.choice(NOISE_RULES)

    if src_type == "user":
        src_id = random.choice(user_ids)
    else:
        src_id = make_node_id(src_type, random.randint(1, node_range))

    dst_id = make_node_id(dst_type, random.randint(1, node_range))

    add_edge(
        edges=edges,
        seen=seen,
        src_id=src_id,
        src_type=src_type,
        relation=relation,
        dst_id=dst_id,
        dst_type=dst_type,
        source="noise_random_triple",
    )


def generate_synthetic_kg(
    total_edges,
    user_count,
    anchor_count,
    chains_per_anchor,
    noise_ratio,
    seed,
):
    random.seed(seed)

    edges = []
    seen = set()

    user_ids = [
        make_node_id("user", index)
        for index in range(1, user_count + 1)
    ]

    anchor_users = user_ids[:anchor_count]
    target_structured_edges = int(total_edges * (1 - noise_ratio))

    chain_id = 1

    for user_id in anchor_users:
        for _ in range(chains_per_anchor):
            add_preference_chain(edges, seen, user_id, chain_id)
            chain_id += 1

            add_study_chain(edges, seen, user_id, chain_id)
            chain_id += 1

    while len(edges) < target_structured_edges:
        template_type = chain_id % 4

        if template_type == 0:
            user_id = random.choice(user_ids)
            add_preference_chain(edges, seen, user_id, chain_id)
        elif template_type == 1:
            add_entity_feature_chain(edges, seen, chain_id)
        elif template_type == 2:
            user_id = random.choice(user_ids)
            add_study_chain(edges, seen, user_id, chain_id)
        else:
            add_task_design_chain(edges, seen, chain_id)

        chain_id += 1

    node_range = max(100, total_edges // 5)

    while len(edges) < total_edges:
        add_noise_edge(edges, seen, user_ids, node_range)

    random.shuffle(edges)

    return edges, anchor_users, target_structured_edges, total_edges - target_structured_edges


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
        description="Generate structured synthetic KG edges."
    )

    parser.add_argument(
        "--edges",
        type=int,
        default=1000,
        help="Total number of edges to generate."
    )

    parser.add_argument(
        "--users",
        type=int,
        default=100,
        help="Total number of user nodes."
    )

    parser.add_argument(
        "--anchors",
        type=int,
        default=5,
        help="Number of fixed benchmark start users."
    )

    parser.add_argument(
        "--chains-per-anchor",
        type=int,
        default=5,
        help="Number of preference and study chain pairs per anchor user."
    )

    parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.30,
        help="Noise edge ratio."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="edges_structured_1k.csv",
        help="Output CSV file."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()

    if args.anchors > args.users:
        raise ValueError("--anchors cannot be greater than --users.")

    if not 0 <= args.noise_ratio < 1:
        raise ValueError("--noise-ratio must be in [0, 1).")

    edges, anchor_users, structured_count, noise_count = generate_synthetic_kg(
        total_edges=args.edges,
        user_count=args.users,
        anchor_count=args.anchors,
        chains_per_anchor=args.chains_per_anchor,
        noise_ratio=args.noise_ratio,
        seed=args.seed,
    )

    write_csv(edges, args.output)

    print("Synthetic structured KG generated successfully.")
    print(f"Total edges: {len(edges)}")
    print(f"Structured edges target: {structured_count}")
    print(f"Noise edges target: {noise_count}")
    print(f"Users: {args.users}")
    print(f"Anchor users: {', '.join(anchor_users)}")
    print(f"Chains per anchor: {args.chains_per_anchor}")
    print(f"Noise ratio: {args.noise_ratio}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()