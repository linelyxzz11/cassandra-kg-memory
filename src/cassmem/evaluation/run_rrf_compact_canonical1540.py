#!/usr/bin/env python3
"""
Run the RRF_compact Reader on LoCoMo canonical Cat1-4 (n=1540).

RRF_compact definition:
    Dense(raw) + BM25(compact) + weighted RRF

Frozen configuration:
    alpha = 0.6
    RRF k = 10
    Reader top_k = 10
    model = deepseek-v4-flash
    temperature = 0

This script:
1. Loads an existing canonical RRF_compact ranking, OR rebuilds it from
   canonical Dense(raw) and BM25(compact) rankings.
2. Sends the top-10 raw memory texts to the frozen Reader prompt.
3. Saves per-query predictions.
4. Supports resuming without repeating completed API calls.

It does not calculate rF1. After prediction generation, run
locomo_official_eval_v1.py offline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI


METHOD = "RRF_compact"
EXPECTED_N = 1540

DEFAULT_ALPHA = 0.6
DEFAULT_RRF_K = 10
DEFAULT_READER_TOP_K = 10

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


# ============================================================
# Generic loading
# ============================================================

def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}"
                )

            rows.append(row)

    return rows


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return load_csv(path)

    if suffix in {".jsonl", ".ndjson"}:
        return load_jsonl(path)

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))

        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")

        return [row for row in data if isinstance(row, dict)]

    raise ValueError(f"Unsupported file type: {path}")


def first_value(
    row: dict[str, Any],
    field_names: list[str],
    default: str = "",
) -> str:
    for field in field_names:
        if field not in row:
            continue

        value = row[field]

        if value is None:
            continue

        text = str(value).strip()

        if text:
            return text

    return default


def split_memory_ids(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    text = str(value).strip()

    if not text:
        return []

    return [
        item.strip()
        for item in re.split(r"[;,|]", text)
        if item.strip()
    ]


# ============================================================
# Questions
# ============================================================

def load_questions(path: Path) -> dict[str, dict[str, str]]:
    questions: dict[str, dict[str, str]] = {}

    for row in load_records(path):
        query_id = first_value(
            row,
            ["query_id", "qa_id", "question_id", "id"],
        )

        category = first_value(
            row,
            ["category", "cat", "question_type"],
        )

        if not query_id or category not in {"1", "2", "3", "4"}:
            continue

        questions[query_id] = {
            "query_id": query_id,
            "category": category,
            "question": first_value(row, ["question", "query"]),
            "gold_answer": first_value(
                row,
                ["gold_answer", "answer"],
            ),
        }

    if len(questions) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N} canonical Cat1-4 questions, "
            f"but loaded {len(questions)} from {path}"
        )

    return questions


# ============================================================
# Memory loading and ID resolution
# ============================================================

def normalize_session_id(value: Any) -> str:
    text = str(value or "").strip()

    if not text:
        return ""

    if text.startswith("session_"):
        return text

    match = re.search(r"\d+", text)

    if match:
        return f"session_{int(match.group(0))}"

    return text


def memory_aliases(row: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()

    memory_id = first_value(row, ["memory_id", "id"])
    sample_id = first_value(row, ["sample_id"])
    session_id = normalize_session_id(
        first_value(row, ["session_id"])
    )
    dia_id = first_value(row, ["dia_id", "evidence_id"])

    if memory_id:
        aliases.add(memory_id)
        aliases.add(memory_id.replace(":", "_"))

    if sample_id and dia_id:
        dia_underscore = dia_id.replace(":", "_")

        aliases.add(f"{sample_id}_{dia_id}")
        aliases.add(f"{sample_id}_{dia_underscore}")

        if session_id:
            aliases.add(
                f"{sample_id}_{session_id}_{dia_id}"
            )
            aliases.add(
                f"{sample_id}_{session_id}_{dia_underscore}"
            )

    return aliases


def load_memories(
    memory_path: Path,
    evidence_map_path: Path | None,
) -> tuple[dict[str, str], dict[str, str]]:
    memory_text: dict[str, str] = {}
    alias_to_memory_id: dict[str, str] = {}

    for row in load_records(memory_path):
        memory_id = first_value(row, ["memory_id", "id"])
        text = first_value(
            row,
            ["text", "memory_text", "content"],
        )

        if not memory_id:
            continue

        memory_text[memory_id] = text

        for alias in memory_aliases(row):
            alias_to_memory_id[alias] = memory_id

    if evidence_map_path and evidence_map_path.exists():
        for row in load_records(evidence_map_path):
            memory_id = first_value(row, ["memory_id"])
            evidence_id = first_value(row, ["evidence_id"])
            sample_id = first_value(row, ["sample_id"])

            if memory_id not in memory_text:
                continue

            if evidence_id:
                alias_to_memory_id[evidence_id] = memory_id
                alias_to_memory_id[
                    evidence_id.replace(":", "_")
                ] = memory_id

                if sample_id:
                    alias_to_memory_id[
                        f"{sample_id}_{evidence_id}"
                    ] = memory_id

                    alias_to_memory_id[
                        f"{sample_id}_{evidence_id.replace(':', '_')}"
                    ] = memory_id

    if not memory_text:
        raise RuntimeError(
            f"No memory records loaded from {memory_path}"
        )

    return memory_text, alias_to_memory_id


def resolve_memory_id(
    raw_memory_id: str,
    memory_text: dict[str, str],
    alias_to_memory_id: dict[str, str],
) -> str:
    raw_memory_id = str(raw_memory_id or "").strip()

    if not raw_memory_id:
        return ""

    if raw_memory_id in memory_text:
        return raw_memory_id

    if raw_memory_id in alias_to_memory_id:
        return alias_to_memory_id[raw_memory_id]

    underscore_id = raw_memory_id.replace(":", "_")

    if underscore_id in memory_text:
        return underscore_id

    if underscore_id in alias_to_memory_id:
        return alias_to_memory_id[underscore_id]

    # Example:
    # sample_session_1_D1:3 -> sample_D1_3
    match = re.match(
        r"^(.*)_session_(\d+)_(D\d+):(\d+)$",
        raw_memory_id,
    )

    if match:
        candidates = [
            f"{match.group(1)}_{match.group(3)}_{match.group(4)}",
            f"{match.group(1)}_{match.group(3)}:{match.group(4)}",
        ]

        for candidate in candidates:
            if candidate in memory_text:
                return candidate

            if candidate in alias_to_memory_id:
                return alias_to_memory_id[candidate]

    return ""


# ============================================================
# Ranking loading and WRRF construction
# ============================================================

def load_long_ranking(
    path: Path,
) -> dict[str, dict[str, int]]:
    """
    Required fields:
        query_id / qa_id
        memory_id
        rank
    """
    rankings: dict[str, dict[str, int]] = defaultdict(dict)

    for row in load_csv(path):
        query_id = first_value(
            row,
            ["query_id", "qa_id", "question_id"],
        )

        memory_id = first_value(row, ["memory_id"])
        rank_text = first_value(row, ["rank"])

        if not query_id or not memory_id or not rank_text:
            continue

        rank = int(float(rank_text))

        current_rank = rankings[query_id].get(memory_id)

        if current_rank is None or rank < current_rank:
            rankings[query_id][memory_id] = rank

    return dict(rankings)


def load_final_rrf_ranking(
    path: Path,
    top_k: int,
) -> dict[str, list[str]]:
    """
    Supports:

    Long form:
        query_id,memory_id,rank,score

    Wide form:
        query_id,retrieved_memory_ids
        query_id,top10_memory_ids
        query_id,ranked_memory_ids
    """
    rows = load_records(path)

    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    columns = set(rows[0].keys())

    if {"memory_id", "rank"} <= columns:
        long_ranking = load_long_ranking(path)
        output: dict[str, list[str]] = {}

        for query_id, memory_rank_map in long_ranking.items():
            ranked = sorted(
                memory_rank_map.items(),
                key=lambda item: (item[1], item[0]),
            )

            output[query_id] = [
                memory_id
                for memory_id, _ in ranked[:top_k]
            ]

        return output

    candidate_fields = [
        "retrieved_memory_ids",
        "top10_memory_ids",
        "ranked_memory_ids",
        "memory_ids",
    ]

    ranking_field = next(
        (
            field
            for field in candidate_fields
            if field in columns
        ),
        None,
    )

    if ranking_field is None:
        raise RuntimeError(
            f"No ranking field found in {path}. "
            f"Columns: {sorted(columns)}"
        )

    output = {}

    for row in rows:
        query_id = first_value(
            row,
            ["query_id", "qa_id", "question_id"],
        )

        if not query_id:
            continue

        output[query_id] = split_memory_ids(
            row.get(ranking_field)
        )[:top_k]

    return output


def build_weighted_rrf(
    dense_ranking_path: Path,
    bm25_ranking_path: Path,
    alpha: float,
    rrf_k: int,
    top_k: int,
) -> tuple[
    dict[str, list[str]],
    dict[str, list[tuple[str, float]]],
]:
    dense = load_long_ranking(dense_ranking_path)
    bm25 = load_long_ranking(bm25_ranking_path)

    query_ids = sorted(set(dense) | set(bm25))

    final_rankings: dict[str, list[str]] = {}
    final_scores: dict[str, list[tuple[str, float]]] = {}

    for query_id in query_ids:
        dense_ranks = dense.get(query_id, {})
        bm25_ranks = bm25.get(query_id, {})

        candidate_ids = set(dense_ranks) | set(bm25_ranks)
        scored_candidates: list[tuple[str, float, int]] = []

        for memory_id in candidate_ids:
            score = 0.0
            best_rank = 10**9

            if memory_id in dense_ranks:
                dense_rank = dense_ranks[memory_id]

                score += alpha / (rrf_k + dense_rank)
                best_rank = min(best_rank, dense_rank)

            if memory_id in bm25_ranks:
                bm25_rank = bm25_ranks[memory_id]

                score += (
                    (1.0 - alpha)
                    / (rrf_k + bm25_rank)
                )

                best_rank = min(best_rank, bm25_rank)

            scored_candidates.append(
                (memory_id, score, best_rank)
            )

        scored_candidates.sort(
            key=lambda item: (
                -item[1],
                item[2],
                item[0],
            )
        )

        top_candidates = scored_candidates[:top_k]

        final_rankings[query_id] = [
            memory_id
            for memory_id, _, _ in top_candidates
        ]

        final_scores[query_id] = [
            (memory_id, score)
            for memory_id, score, _ in top_candidates
        ]

    return final_rankings, final_scores


def write_fused_ranking(
    path: Path,
    rankings: dict[str, list[str]],
    scores: dict[str, list[tuple[str, float]]] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query_id",
                "memory_id",
                "rank",
                "score",
            ],
        )

        writer.writeheader()

        for query_id in sorted(rankings):
            score_map = dict(scores.get(query_id, [])) if scores else {}

            for rank, memory_id in enumerate(
                rankings[query_id],
                start=1,
            ):
                writer.writerow(
                    {
                        "query_id": query_id,
                        "memory_id": memory_id,
                        "rank": rank,
                        "score": score_map.get(
                            memory_id,
                            "",
                        ),
                    }
                )


# ============================================================
# Frozen Reader prompt
# ============================================================

def build_prompt(
    question: str,
    evidence_items: list[str],
) -> str:
    evidence_lines = []

    for index, text in enumerate(evidence_items, start=1):
        evidence_lines.append(f"[{index}] {text}")

    return "\n".join(
        [
            "Answer the question using only the evidence below.",
            (
                "If the evidence does not contain the answer, "
                "respond exactly with 'Cannot answer'."
            ),
            "Return only the shortest answer. Do not explain.",
            "",
            "Evidence:",
            "\n\n".join(evidence_lines),
            "",
            f"Question: {question}",
            "Answer:",
        ]
    )


# ============================================================
# API and output
# ============================================================

def call_deepseek(
    client: OpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
) -> str:
    retry_waits = [1, 2, 4, 8, 16, 30, 45, 60]
    last_error: Exception | None = None

    for attempt, wait_seconds in enumerate(
        retry_waits,
        start=1,
    ):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                max_tokens=max_tokens,
                timeout=60,
            )

            answer = (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

            if not answer:
                raise RuntimeError("DeepSeek returned an empty answer")

            return answer

        except Exception as exc:
            last_error = exc

            print(
                f"API failure {attempt}/{len(retry_waits)}: "
                f"{exc}"
            )

            if attempt < len(retry_waits):
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"DeepSeek failed after all retries: {last_error}"
    )


def load_completed_predictions(
    path: Path,
) -> set[str]:
    if not path.exists():
        return set()

    completed: set[str] = set()

    for row in load_jsonl(path):
        query_id = first_value(
            row,
            ["query_id", "qa_id"],
        )

        prediction = first_value(
            row,
            [
                "prediction",
                "predicted_answer",
                "response",
                "output",
            ],
        )

        if query_id and prediction:
            completed.add(query_id)

    return completed


def append_prediction(
    path: Path,
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--memories",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--evidence-map",
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--ranking-output",
        required=True,
        type=Path,
    )

    # Option A: existing canonical RRF_compact ranking.
    parser.add_argument(
        "--rrf-ranking",
        type=Path,
    )

    # Option B: rebuild canonical RRF_compact.
    parser.add_argument(
        "--dense-ranking",
        type=Path,
    )

    parser.add_argument(
        "--bm25-compact-ranking",
        type=Path,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
    )

    parser.add_argument(
        "--reader-top-k",
        type=int,
        default=DEFAULT_READER_TOP_K,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max-new-calls",
        type=int,
        default=0,
        help="0 means all missing queries",
    )

    parser.add_argument(
        "--sleep-min",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--sleep-max",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    questions = load_questions(args.questions)

    memory_text, alias_to_memory_id = load_memories(
        args.memories,
        args.evidence_map,
    )

    # --------------------------------------------------------
    # Load or reconstruct canonical RRF_compact ranking
    # --------------------------------------------------------

    if args.rrf_ranking:
        rankings = load_final_rrf_ranking(
            args.rrf_ranking,
            args.reader_top_k,
        )

        ranking_scores = None

    else:
        if (
            args.dense_ranking is None
            or args.bm25_compact_ranking is None
        ):
            raise RuntimeError(
                "Provide either --rrf-ranking, or both "
                "--dense-ranking and --bm25-compact-ranking"
            )

        rankings, ranking_scores = build_weighted_rrf(
            dense_ranking_path=args.dense_ranking,
            bm25_ranking_path=args.bm25_compact_ranking,
            alpha=args.alpha,
            rrf_k=args.rrf_k,
            top_k=args.reader_top_k,
        )

    canonical_query_ids = sorted(questions)

    missing_rankings = [
        query_id
        for query_id in canonical_query_ids
        if query_id not in rankings
    ]

    if missing_rankings:
        raise RuntimeError(
            f"RRF_compact ranking is missing "
            f"{len(missing_rankings)} canonical queries. "
            f"Examples: {missing_rankings[:10]}"
        )

    rankings = {
        query_id: rankings[query_id][
            :args.reader_top_k
        ]
        for query_id in canonical_query_ids
    }

    write_fused_ranking(
        args.ranking_output,
        rankings,
        ranking_scores,
    )

    # --------------------------------------------------------
    # Resolve memory IDs before spending API calls
    # --------------------------------------------------------

    resolved_rankings: dict[str, list[str]] = {}

    for query_id in canonical_query_ids:
        resolved_ids: list[str] = []

        for raw_memory_id in rankings[query_id]:
            resolved_memory_id = resolve_memory_id(
                raw_memory_id,
                memory_text,
                alias_to_memory_id,
            )

            if not resolved_memory_id:
                raise RuntimeError(
                    f"Cannot resolve memory ID "
                    f"{raw_memory_id!r} for query {query_id}"
                )

            resolved_ids.append(resolved_memory_id)

        if len(resolved_ids) != args.reader_top_k:
            raise RuntimeError(
                f"{query_id} has {len(resolved_ids)} resolved "
                f"memories instead of {args.reader_top_k}"
            )

        resolved_rankings[query_id] = resolved_ids

    # --------------------------------------------------------
    # Resume completed predictions
    # --------------------------------------------------------

    completed_query_ids = load_completed_predictions(
        args.output
    )

    pending_query_ids = [
        query_id
        for query_id in canonical_query_ids
        if query_id not in completed_query_ids
    ]

    print(f"Canonical questions: {len(canonical_query_ids)}")
    print(f"Already completed: {len(completed_query_ids)}")
    print(f"Pending API calls: {len(pending_query_ids)}")
    print(f"Prediction output: {args.output}")
    print(f"Ranking output: {args.ranking_output}")

    if not pending_query_ids:
        print("All 1540 RRF_compact predictions already exist.")
        return

    api_key = os.environ.get(
        "DEEPSEEK_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY environment variable is not set"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
    )

    new_calls = 0

    for query_id in pending_query_ids:
        if (
            args.max_new_calls > 0
            and new_calls >= args.max_new_calls
        ):
            print(
                f"Stopped after {new_calls} new API calls "
                f"because --max-new-calls was reached."
            )
            break

        question_row = questions[query_id]
        resolved_ids = resolved_rankings[query_id]

        evidence_items = [
            memory_text[memory_id]
            for memory_id in resolved_ids
        ]

        prompt = build_prompt(
            question_row["question"],
            evidence_items,
        )

        predicted_answer = call_deepseek(
            client=client,
            prompt=prompt,
            model=args.model,
            max_tokens=args.max_tokens,
        )

        output_row = {
            "query_id": query_id,
            "qa_id": query_id,
            "category": question_row["category"],
            "method": METHOD,
            "question": question_row["question"],
            "gold_answer": question_row["gold_answer"],
            "prediction": predicted_answer,
            "predicted_answer": predicted_answer,
            "top10_memory_ids": ";".join(
                rankings[query_id]
            ),
            "resolved_memory_ids": ";".join(
                resolved_ids
            ),
            "model": args.model,
            "temperature": 0,
            "top_k": args.reader_top_k,
            "rrf_alpha": args.alpha,
            "rrf_k": args.rrf_k,
            "scope": "canonical_cat1_4",
        }

        append_prediction(
            args.output,
            output_row,
        )

        new_calls += 1

        if new_calls % 20 == 0:
            total_done = (
                len(completed_query_ids)
                + new_calls
            )

            print(
                f"Generated {new_calls} new predictions; "
                f"total completed {total_done}/{EXPECTED_N}"
            )

        sleep_seconds = (
            args.sleep_min
            + random.random()
            * max(
                0.0,
                args.sleep_max - args.sleep_min,
            )
        )

        time.sleep(sleep_seconds)

    final_completed = load_completed_predictions(
        args.output
    )

    print()
    print(f"New API calls: {new_calls}")
    print(
        f"Final completed predictions: "
        f"{len(final_completed)}/{EXPECTED_N}"
    )
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
