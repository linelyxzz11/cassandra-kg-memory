import argparse
import json
from collections import Counter
from pathlib import Path


def load_json(file_path):
    with Path(file_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def as_list(data):
    # LoCoMo may be released either as a list or as a dict containing a list.
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "samples", "conversations"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Cannot find a top-level sample list in the JSON file.")


def short_text(value, max_len=120):
    text = str(value).replace("\n", " ").strip()

    if len(text) > max_len:
        return text[:max_len] + "..."

    return text


def inspect_top_level(data, samples):
    print("LoCoMo Dataset Inspection")
    print("-------------------------")

    print(f"Top-level type: {type(data).__name__}")

    if isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())}")

    print(f"Number of samples: {len(samples)}")
    print()


def inspect_sample_keys(samples):
    print("Sample Key Overview")
    print("-------------------")

    key_counter = Counter()

    for sample in samples:
        if isinstance(sample, dict):
            key_counter.update(sample.keys())

    for key, count in key_counter.most_common():
        print(f"{key:<24} {count}")

    print()


def find_conversation(sample):
    for key in ["conversation", "conversations", "dialogue", "dialogues", "messages"]:
        value = sample.get(key)

        if isinstance(value, list):
            return key, value

    return None, []


def flatten_conversation_items(conversation):
    # Handles both direct turn lists and session-level nested structures.
    flat_items = []

    for item in conversation:
        if isinstance(item, dict):
            if "dialogue" in item and isinstance(item["dialogue"], list):
                for turn in item["dialogue"]:
                    if isinstance(turn, dict):
                        flat_items.append(turn)
            elif "turns" in item and isinstance(item["turns"], list):
                for turn in item["turns"]:
                    if isinstance(turn, dict):
                        flat_items.append(turn)
            else:
                flat_items.append(item)

    return flat_items


def inspect_conversation(samples, max_samples):
    print("Conversation Structure")
    print("----------------------")

    total_turns = 0
    conversation_key_counter = Counter()
    turn_key_counter = Counter()

    for sample in samples[:max_samples]:
        if not isinstance(sample, dict):
            continue

        conv_key, conversation = find_conversation(sample)

        if conv_key is None:
            continue

        conversation_key_counter[conv_key] += 1
        flat_turns = flatten_conversation_items(conversation)
        total_turns += len(flat_turns)

        for turn in flat_turns:
            turn_key_counter.update(turn.keys())

    print(f"Samples inspected: {min(max_samples, len(samples))}")
    print(f"Detected conversation keys: {dict(conversation_key_counter)}")
    print(f"Total flattened turns in inspected samples: {total_turns}")
    print()

    print("Common turn keys:")
    for key, count in turn_key_counter.most_common(20):
        print(f"{key:<24} {count}")

    print()


def find_qa(sample):
    for key in ["qa", "qas", "questions", "question_answering"]:
        value = sample.get(key)

        if isinstance(value, list):
            return key, value

    return None, []


def inspect_qa(samples, max_examples):
    print("QA Structure")
    print("------------")

    total_qa = 0
    qa_key_counter = Counter()
    category_counter = Counter()
    evidence_count = 0
    examples = []

    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue

        qa_key, qa_items = find_qa(sample)

        if qa_key is None:
            continue

        for qa_index, qa in enumerate(qa_items):
            if not isinstance(qa, dict):
                continue

            total_qa += 1
            qa_key_counter.update(qa.keys())

            category = (
                qa.get("category")
                or qa.get("type")
                or qa.get("question_type")
                or "UNKNOWN"
            )
            category_counter[category] += 1

            evidence = qa.get("evidence") or qa.get("evidences") or qa.get("gold_evidence")

            if evidence:
                evidence_count += 1

            if len(examples) < max_examples:
                examples.append({
                    "sample_index": sample_index,
                    "qa_index": qa_index,
                    "category": category,
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                    "evidence": evidence,
                })

    print(f"Total QA items: {total_qa}")
    print(f"QA items with evidence field: {evidence_count}")
    print()

    print("Common QA keys:")
    for key, count in qa_key_counter.most_common(20):
        print(f"{key:<24} {count}")

    print()

    print("QA category distribution:")
    for category, count in category_counter.most_common():
        print(f"{category:<32} {count}")

    print()

    print("QA examples:")
    for item in examples:
        print("-" * 80)
        print(f"sample_index: {item['sample_index']}")
        print(f"qa_index    : {item['qa_index']}")
        print(f"category    : {item['category']}")
        print(f"question    : {short_text(item['question'])}")
        print(f"answer      : {short_text(item['answer'])}")
        print(f"evidence    : {short_text(item['evidence'])}")

    print()


def inspect_optional_fields(samples):
    print("Optional Memory Fields")
    print("----------------------")

    fields = [
        "observation",
        "observations",
        "session_summary",
        "event_summary",
        "persona",
    ]

    for field in fields:
        count = sum(
            1
            for sample in samples
            if isinstance(sample, dict) and field in sample
        )
        print(f"{field:<24} {count}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect LoCoMo JSON structure before building adapters."
    )

    parser.add_argument(
        "--file",
        default="external_data/locomo10.json",
        help="Path to LoCoMo JSON file.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Number of samples used for detailed structure inspection.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Number of QA examples to print.",
    )

    args = parser.parse_args()

    data = load_json(args.file)
    samples = as_list(data)

    inspect_top_level(data, samples)
    inspect_sample_keys(samples)
    inspect_conversation(samples, args.max_samples)
    inspect_qa(samples, args.max_examples)
    inspect_optional_fields(samples)


if __name__ == "__main__":
    main()