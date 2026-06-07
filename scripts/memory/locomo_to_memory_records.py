#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert official LoCoMo JSON into intermediate memory CSV files.

Outputs:
  locomo_memory_records.csv
  locomo_qa_records.csv
  locomo_evidence_map.csv

Usage:
  python scripts/memory/locomo_to_memory_records.py \
    --file external_data/locomo10.json \
    --out-dir results/locomo
"""
import argparse
import json
import re
from pathlib import Path
import pandas as pd

def extract_evidence_ids(evidence):
    """Extract evidence IDs robustly from strings like D1:3, D8:6; D9:17, D30:05, D:11:26."""
    if not evidence:
        return []
    text = " ".join(str(x) for x in evidence)
    raw = re.findall(r"D\s*:?\s*\d+\s*:\s*\d+", text)
    ids = []
    for item in raw:
        nums = re.findall(r"\d+", item)
        if len(nums) >= 2:
            ids.append(f"D{int(nums[0])}:{int(nums[1])}")
    return ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--out-dir", default="results/locomo")
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    memory_rows = []
    turn_index = {}

    for sample_index, sample in enumerate(data):
        sample_id = sample.get("sample_id", f"sample_{sample_index}")
        conv = sample["conversation"]

        for key, turns in conv.items():
            m = re.match(r"session_(\d+)$", key)
            if not (m and isinstance(turns, list)):
                continue

            session_id = int(m.group(1))
            session_dt = conv.get(f"session_{session_id}_date_time", "")

            for turn_index_in_session, turn in enumerate(turns):
                dia_id = turn.get("dia_id", "")
                memory_id = f"{sample_id}_{dia_id.replace(':', '_')}"
                turn_index[(sample_id, dia_id)] = memory_id

                memory_rows.append({
                    "memory_id": memory_id,
                    "user_id": sample_id,
                    "sample_id": sample_id,
                    "session_id": session_id,
                    "dia_id": dia_id,
                    "turn_index": turn_index_in_session,
                    "speaker": turn.get("speaker", ""),
                    "text": turn.get("text", ""),
                    "timestamp": session_dt,
                    "source": "conversation",
                    "has_image": bool(turn.get("img_url") or turn.get("blip_caption")),
                    "blip_caption": turn.get("blip_caption", ""),
                    "query": turn.get("query", ""),
                })

    qa_rows = []
    evidence_rows = []

    for sample_index, sample in enumerate(data):
        sample_id = sample.get("sample_id", f"sample_{sample_index}")
        for qa_index, qa in enumerate(sample.get("qa", [])):
            qa_id = f"{sample_id}_qa_{qa_index:04d}"
            evidence_ids = extract_evidence_ids(qa.get("evidence", []))

            qa_rows.append({
                "qa_id": qa_id,
                "user_id": sample_id,
                "sample_id": sample_id,
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "adversarial_answer": qa.get("adversarial_answer", ""),
                "category": qa.get("category", ""),
                "evidence_raw": json.dumps(qa.get("evidence", []), ensure_ascii=False),
                "evidence_ids": json.dumps(evidence_ids, ensure_ascii=False),
            })

            for evidence_id in evidence_ids:
                memory_id = turn_index.get((sample_id, evidence_id), "")
                evidence_rows.append({
                    "qa_id": qa_id,
                    "sample_id": sample_id,
                    "evidence_id": evidence_id,
                    "memory_id": memory_id,
                    "matched": bool(memory_id),
                })

    memory_df = pd.DataFrame(memory_rows)
    qa_df = pd.DataFrame(qa_rows)
    evidence_df = pd.DataFrame(evidence_rows)

    memory_path = out_dir / "locomo_memory_records.csv"
    qa_path = out_dir / "locomo_qa_records.csv"
    evidence_path = out_dir / "locomo_evidence_map.csv"

    memory_df.to_csv(memory_path, index=False, encoding="utf-8")
    qa_df.to_csv(qa_path, index=False, encoding="utf-8")
    evidence_df.to_csv(evidence_path, index=False, encoding="utf-8")

    print("LoCoMo conversion completed.")
    print(f"Samples: {len(data)}")
    print(f"Memory records: {len(memory_df)} -> {memory_path}")
    print(f"QA records: {len(qa_df)} -> {qa_path}")
    print(f"Evidence mappings: {len(evidence_df)} -> {evidence_path}")
    print(f"Matched evidence: {int(evidence_df['matched'].sum()) if not evidence_df.empty else 0}")
    print(f"Unmatched evidence: {int((~evidence_df['matched']).sum()) if not evidence_df.empty else 0}")

if __name__ == "__main__":
    main()
