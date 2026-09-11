"""Run a resumable Mem0/HingeMem-style binary LLM judge over all local reader outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREDICTION_ROOT = ROOT / "05_reports" / "locomo_gpt4o_prompt_protocol"
DEFAULT_OUTPUT = ROOT / "05_reports" / "llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected"
DEFAULT_BASE_URL = "https://api.uiuihao.com/v1"
DEFAULT_MODEL = "gpt-4o-2024-08-06"

CATEGORY_NAMES = {
    "1": "Multi-Hop",
    "2": "Temporal",
    "3": "Open-Domain",
    "4": "Single-Hop",
    "5": "Adversarial",
}

ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.

The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace

The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script. Just return the label CORRECT or WRONG in a json format with the key as "label".
""".strip()

_write_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(value: object) -> str:
    return str(value or "").strip()


def judgment_key(question: str, gold_answer: str, generated_answer: str) -> str:
    payload = json.dumps(
        [normalize(question), normalize(gold_answer), normalize(generated_answer)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_swap(query_id: str) -> bool:
    """Match the frozen Cat5 option order used by the reader-generation script."""
    value = int(hashlib.shake_128(query_id.encode()).hexdigest(4), 16)
    return value % 2 == 1


def resolve_judge_fields(row: dict) -> tuple[str, str, str]:
    question = normalize(row["question"])
    gold_answer = normalize(row["gold_answer"])
    generated_answer = normalize(row["prediction"])
    if str(row["category"]) != "5":
        return question, gold_answer, generated_answer

    # LoCoMo Cat5 stores a plausible adversarial/distractor answer in the
    # answer field; the semantically correct answer is that it was not stated.
    judge_gold = "Not mentioned in the conversation"
    if generated_answer.lower() in {"(a)", "a", "(b)", "b"}:
        swapped = deterministic_swap(row["query_id"])
        option_a = gold_answer if swapped else judge_gold
        option_b = judge_gold if swapped else gold_answer
        judge_generated = option_a if "a" in generated_answer.lower() else option_b
    else:
        judge_generated = generated_answer
    return question, judge_gold, judge_generated


def load_inputs(prediction_root: Path = PREDICTION_ROOT) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    unique: dict[str, dict] = {}
    for setting_dir in ["setting_a_unified", "setting_b_category"]:
        for path in sorted((prediction_root / setting_dir).glob("*/reader_predictions.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    judge_question, judge_gold, judge_generated = resolve_judge_fields(row)
                    key = judgment_key(judge_question, judge_gold, judge_generated)
                    row["judge_key"] = key
                    row["judge_gold_answer"] = judge_gold
                    row["judge_generated_answer"] = judge_generated
                    row["source_file"] = str(path.relative_to(ROOT)).replace("\\", "/")
                    rows.append(row)
                    unique.setdefault(
                        key,
                        {
                            "judge_key": key,
                            "question": judge_question,
                            "gold_answer": judge_gold,
                            "generated_answer": judge_generated,
                        },
                    )
    return rows, unique


def load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if item.get("label") in {"CORRECT", "WRONG"}:
                    cache[item["judge_key"]] = item
    return cache


def extract_label(content: str) -> str:
    try:
        obj = json.loads(content)
        value = str(obj.get("label", "")).strip().upper()
        if value in {"CORRECT", "WRONG"}:
            return value
    except json.JSONDecodeError:
        pass
    labels = re.findall(r"\b(CORRECT|WRONG)\b", content.upper())
    if labels and len(set(labels)) == 1:
        return labels[0]
    raise ValueError(f"Unparseable judge response: {content[:500]!r}")


def post_judgment(
    item: dict,
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    max_retries: int,
) -> dict:
    prompt = ACCURACY_PROMPT.format(
        question=item["question"],
        gold_answer=item["gold_answer"],
        generated_answer=item["generated_answer"],
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 80,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    last_error = ""
    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            content = response_body["choices"][0]["message"]["content"]
            label = extract_label(content)
            usage = response_body.get("usage") or {}
            return {
                **item,
                "label": label,
                "score": 1 if label == "CORRECT" else 0,
                "raw_response": content,
                "model_returned": response_body.get("model"),
                "usage": usage,
                "latency_seconds": round(time.perf_counter() - started, 4),
                "attempts": attempt,
                "judged_at_utc": utc_now(),
            }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:1000]
                except Exception:
                    detail = ""
                last_error = f"HTTP {exc.code}: {detail}"
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
            else:
                last_error = f"{type(exc).__name__}: {exc}"
                retryable = True
            if attempt >= max_retries or not retryable:
                break
            time.sleep(min(60.0, (2 ** (attempt - 1)) + random.random()))
    raise RuntimeError(f"Judge failed after {max_retries} attempts: {last_error}")


def write_protocol(output_dir: Path, args: argparse.Namespace, total: int, unique: int) -> None:
    protocol = {
        "name": "Mem0/HingeMem-compatible binary LLM-as-a-Judge",
        "judge_model_requested": args.model,
        "base_url": args.base_url,
        "temperature": 0,
        "max_tokens": 80,
        "response_format": {"type": "json_object"},
        "labels": ["CORRECT", "WRONG"],
        "score": "mean(CORRECT); reported as percentage",
        "categories": CATEGORY_NAMES,
        "cat5_policy": "included; gold is 'Not mentioned in the conversation'; Setting B option labels are deterministically resolved to option text before judging",
        "overall_policy": "all 1,986 LoCoMo questions, including Cat5",
        "prompt": ACCURACY_PROMPT,
        "prompt_sha256": hashlib.sha256(ACCURACY_PROMPT.encode("utf-8")).hexdigest(),
        "prediction_rows": total,
        "unique_judgments": unique,
        "deduplication": "exact normalized (question, gold_answer, generated_answer) SHA-256 cache",
        "reference": {
            "HingeMem": "High-mem.pdf, Sec. 4.1.2: same prompt template as Mem0",
            "Mem0": "mem0ai/mem0 commit 2b58775, evaluation/metrics/llm_judge.py",
        },
        "secret_policy": "API key read only from OPENAI_API_KEY and never written",
        "created_at_utc": utc_now(),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def aggregate(rows: list[dict], cache: dict[str, dict], output_dir: Path) -> None:
    scored_path = output_dir / "scored_predictions.jsonl"
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    with scored_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            judged = cache[row["judge_key"]]
            score = int(judged["score"])
            category = str(row["category"])
            method = row["method"]
            setting = row["setting"]
            grouped[(method, setting, category)].append(score)
            grouped[(method, setting, "Overall")].append(score)
            out = {
                "query_id": row["query_id"],
                "category": category,
                "category_name": CATEGORY_NAMES[category],
                "method": method,
                "setting": setting,
                "setting_label": row.get("setting_label"),
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "prediction": row["prediction"],
                "judge_gold_answer": row["judge_gold_answer"],
                "judge_generated_answer": row["judge_generated_answer"],
                "judge_key": row["judge_key"],
                "judge_label": judged["label"],
                "judge_score": score,
                "source_file": row["source_file"],
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")

    category_order = ["4", "1", "2", "3", "5", "Overall"]
    summary_rows: list[dict] = []
    preferred_methods = ["BM25", "Dense-bge", "RRF_compact", "Dense+GlobalKG", "ZScore-Raw", "ZScore-RawERK"]
    observed_methods = {str(row["method"]) for row in rows}
    methods = [method for method in preferred_methods if method in observed_methods]
    methods.extend(sorted(observed_methods - set(methods)))
    settings = [
        setting
        for setting in ["a_unified", "b_category"]
        if any(str(row["setting"]) == setting for row in rows)
    ]
    for method in methods:
        for setting in settings:
            for category in category_order:
                values = grouped[(method, setting, category)]
                if not values:
                    continue
                summary_rows.append(
                    {
                        "method": method,
                        "setting": setting,
                        "setting_label": "Cat.✗" if setting == "a_unified" else "Cat.✓",
                        "category": category,
                        "category_name": "Overall" if category == "Overall" else CATEGORY_NAMES[category],
                        "n": len(values),
                        "correct": sum(values),
                        "j_score": sum(values) / len(values),
                        "j_percent": 100 * sum(values) / len(values),
                    }
                )
    fields = ["method", "setting", "setting_label", "category", "category_name", "n", "correct", "j_score", "j_percent"]
    with (output_dir / "j_scores_by_method_setting_category.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    wide_fields = ["method", "setting", "setting_label", "single_hop_j", "multi_hop_j", "temporal_j", "open_domain_j", "adversarial_j", "overall_j"]
    category_field = {
        "4": "single_hop_j", "1": "multi_hop_j", "2": "temporal_j",
        "3": "open_domain_j", "5": "adversarial_j", "Overall": "overall_j",
    }
    wide_rows = []
    for method in methods:
        for setting in settings:
            row = {"method": method, "setting": setting, "setting_label": "Cat.✗" if setting == "a_unified" else "Cat.✓"}
            for category in category_order:
                values = grouped[(method, setting, category)]
                row[category_field[category]] = 100 * sum(values) / len(values) if values else ""
            wide_rows.append(row)
    with (output_dir / "j_scores_reader_main_wide.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=wide_fields)
        writer.writeheader()
        writer.writerows(wide_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prediction-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Only judge this many pending unique items; 0 means all")
    parser.add_argument("--seed-cache", type=Path, help="Reuse compatible judgments from another JSONL cache")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.prediction_root = args.prediction_root.resolve()
    rows, unique = load_inputs(args.prediction_root)
    cache_path = args.output_dir / "judgments_unique.jsonl"
    cache = load_cache(cache_path)
    seeded: list[dict] = []
    if args.seed_cache:
        for key, item in load_cache(args.seed_cache.resolve()).items():
            if key in unique and key not in cache:
                cache[key] = item
                seeded.append(item)
        if seeded:
            with cache_path.open("a", encoding="utf-8") as handle:
                for item in seeded:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_protocol(args.output_dir, args, len(rows), len(unique))
    pending = [item for key, item in unique.items() if key not in cache]
    if args.limit > 0:
        pending = pending[: args.limit]
    print(json.dumps({"rows": len(rows), "unique": len(unique), "cached": len(cache), "pending_this_run": len(pending)}, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    failures: list[dict] = []
    with cache_path.open("a", encoding="utf-8", buffering=1) as cache_handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    post_judgment,
                    item,
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                ): item
                for item in pending
            }
            done = 0
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    cache[item["judge_key"]] = result
                    with _write_lock:
                        cache_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                except Exception as exc:
                    failures.append({"judge_key": item["judge_key"], "error": str(exc)})
                done += 1
                if done % 50 == 0 or done == len(pending):
                    elapsed = time.perf_counter() - started
                    rate = done / elapsed if elapsed else 0
                    print(json.dumps({"completed": done, "submitted": len(pending), "cached_total": len(cache), "failures": len(failures), "rate_per_sec": round(rate, 3)}, ensure_ascii=False), flush=True)

    if failures:
        (args.output_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} judgments failed; rerun to resume")
    if args.limit > 0:
        print("LIMITED RUN COMPLETE", flush=True)
        return
    missing = set(unique) - set(cache)
    if missing:
        raise RuntimeError(f"Cache incomplete: {len(missing)} unique judgments missing")

    aggregate(rows, cache, args.output_dir)
    usage = Counter()
    model_returned = Counter()
    attempts = Counter()
    for key in unique:
        item = cache[key]
        for key, value in (item.get("usage") or {}).items():
            if isinstance(value, int):
                usage[key] += value
        model_returned[str(item.get("model_returned"))] += 1
        attempts[int(item.get("attempts", 1))] += 1
    manifest = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "prediction_rows": len(rows),
        "unique_judgments": len(unique),
        "cache_reuse_rows": len(rows) - len(unique),
        "compatible_seeded_judgments": len(seeded),
        "judge_model_requested": args.model,
        "models_returned": dict(model_returned),
        "usage": dict(usage),
        "attempt_histogram": dict(attempts),
        "outputs": [
            "protocol.json", "judgments_unique.jsonl", "scored_predictions.jsonl",
            "j_scores_by_method_setting_category.csv", "j_scores_reader_main_wide.csv",
        ],
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
