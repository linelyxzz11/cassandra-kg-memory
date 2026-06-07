#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inspect official LoCoMo JSON structure.

Usage:
  python scripts/memory/inspect_locomo_conversation.py --file external_data/locomo10.json --sample-index 0
"""
import argparse
import json
import re
from collections import Counter

def brief(value, max_len=240):
    text = repr(value)
    return text if len(text) <= max_len else text[:max_len] + "..."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-qa", type=int, default=5)
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample = data[args.sample_index]
    conv = sample["conversation"]

    print(f"Top-level type: {type(data).__name__}")
    print(f"Number of samples: {len(data)}")
    print(f"Sample index: {args.sample_index}")
    print(f"Sample id: {sample.get('sample_id')}")
    print(f"Sample keys: {list(sample.keys())}")
    print()

    print("Conversation keys overview:")
    for k, v in conv.items():
        if re.match(r"session_\d+$", k):
            print(f"  {k}: {type(v).__name__}, len={len(v) if isinstance(v, list) else 'NA'}")
        elif re.match(r"session_\d+_date_time$", k):
            print(f"  {k}: {v}")
        elif k in ("speaker_a", "speaker_b"):
            print(f"  {k}: {v}")
    print()

    print("First sessions and turns:")
    shown_sessions = 0
    for k, v in conv.items():
        m = re.match(r"session_(\d+)$", k)
        if not (m and isinstance(v, list)):
            continue
        session_id = int(m.group(1))
        dt = conv.get(f"session_{session_id}_date_time", "")
        print(f"[session_{session_id}] date_time={dt}, turns={len(v)}")
        for turn in v[:args.max_turns]:
            print(f"  {turn.get('dia_id')} | {turn.get('speaker')}: {turn.get('text')}")
        shown_sessions += 1
        if shown_sessions >= 2:
            break
    print()

    print("Observation overview:")
    obs = sample.get("observation", {})
    for k, v in list(obs.items())[:3]:
        print(f"  {k}:")
        if isinstance(v, dict):
            for speaker, items in v.items():
                print(f"    {speaker}: {len(items)} observations")
                for item in items[:2]:
                    print(f"      {brief(item)}")
    print()

    print("Session summary overview:")
    ss = sample.get("session_summary", {})
    if isinstance(ss, dict):
        for k, v in list(ss.items())[:3]:
            print(f"  {k}: {brief(v)}")
    else:
        print(brief(ss))
    print()

    print("Event summary overview:")
    es = sample.get("event_summary", {})
    for k, v in list(es.items())[:3]:
        print(f"  {k}: {brief(v)}")
    print()

    print("QA overview:")
    qa = sample.get("qa", [])
    print(f"  QA count: {len(qa)}")
    print(f"  Category distribution: {dict(Counter(q.get('category') for q in qa))}")
    for q in qa[:args.max_qa]:
        print(f"  category={q.get('category')} evidence={q.get('evidence')}")
        print(f"    Q: {q.get('question')}")
        print(f"    A: {q.get('answer', q.get('adversarial_answer'))}")

if __name__ == "__main__":
    main()
