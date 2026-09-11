#!/usr/bin/env python3
"""Launch the two resumable corrected Reader settings without persisting secrets."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "03_src/evaluation/run_locomo_prompt_protocol_gpt4o.py"
QUESTIONS = ROOT / "01_data/locomo_qa_records.csv"
MEMORIES = ROOT / "01_data/locomo_memory_records.csv"
OUTPUT_ROOT = ROOT / "05_reports/locomo_gpt4o_prompt_protocol_corrected_densekg_v2"


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY not set")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    processes = []
    for setting_dir, setting, log_stem in [
        ("setting_a_unified", "a_unified", "setting_a"),
        ("setting_b_category", "b_category", "setting_b"),
    ]:
        output_dir = OUTPUT_ROOT / setting_dir / "Dense+GlobalKG"
        command = [
            sys.executable,
            str(RUNNER),
            "--questions", str(QUESTIONS),
            "--memories", str(MEMORIES),
            "--method", "Dense+GlobalKG",
            "--output-dir", str(output_dir),
            "--setting", setting,
            "--workers", "6",
            "--sleep-min", "0.05",
            "--sleep-max", "0.15",
        ]
        stdout_handle = (OUTPUT_ROOT / f"{log_stem}_run.log").open("a", encoding="utf-8")
        stderr_handle = (OUTPUT_ROOT / f"{log_stem}_error.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )
        processes.append({"setting": setting, "pid": process.pid})
        stdout_handle.close()
        stderr_handle.close()
    launch = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "processes": processes,
        "note": "API key inherited from transient parent environment; not persisted",
    }
    (OUTPUT_ROOT / "launch_manifest.json").write_text(
        json.dumps(launch, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(launch, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
