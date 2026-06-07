#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Print LoCoMo QA examples by numeric category.

Usage:
  python scripts/memory/inspect_locomo_categories.py --file external_data/locomo10.json --max-per-category 3
"""
import argparse
import json
from collections import defaultdict, Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--max-per-category", type=int, default=3)
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    buckets = defaultdict(list)
    counts = Counter()
    for sample in data:
        sample_id = sample.get("sample_id")
        for q in sample.get("qa", []):
            cat = q.get("category")
            counts[cat] += 1
            if len(buckets[cat]) < args.max_per_category:
                buckets[cat].append((sample_id, q))

    print("Category distribution:")
    for cat, n in sorted(counts.items()):
        print(f"  category {cat}: {n}")
    print()

    for cat in sorted(buckets):
        print("=" * 80)
        print(f"Category {cat}")
        for sample_id, q in buckets[cat]:
            print(f"- sample={sample_id} evidence={q.get('evidence')}")
            print(f"  Q: {q.get('question')}")
            if "answer" in q:
                print(f"  A: {q.get('answer')}")
            if "adversarial_answer" in q:
                print(f"  adversarial_answer: {q.get('adversarial_answer')}")
            print()

if __name__ == "__main__":
    main()
