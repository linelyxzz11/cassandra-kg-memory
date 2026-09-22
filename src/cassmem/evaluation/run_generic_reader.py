#!/usr/bin/env python3
"""Generic frozen Reader for any retrieval method. Takes ranking CSV, calls deepseek-chat, resumes."""
from __future__ import annotations
import argparse, csv, json, os, random, re, time
from collections import defaultdict
from pathlib import Path
from openai import OpenAI

EXPECTED_N = 1540

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_questions(path: Path) -> dict[str, dict[str, str]]:
    questions = {}
    for row in list(csv.DictReader(path.open(encoding="utf-8-sig", newline=""))):
        qid = row.get("qa_id", row.get("query_id", "")).strip()
        cat = row.get("category", "").strip()
        if not qid or cat not in {"1", "2", "3", "4"}:
            continue
        questions[qid] = {
            "query_id": qid,
            "category": cat,
            "question": row.get("question", "").strip(),
            "gold_answer": row.get("answer", row.get("gold_answer", "")).strip(),
        }
    if len(questions) != EXPECTED_N:
        raise RuntimeError(f"Expected {EXPECTED_N} questions, got {len(questions)}")
    return questions

def load_memory_text(path: Path) -> dict[str, str]:
    out = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig", newline="")):
        mid = row.get("memory_id", "").strip()
        text = row.get("text", "").strip()
        if mid and text:
            out[mid] = text
    return out

def build_prompt(question: str, evidence_items: list[str]) -> str:
    lines = [f"[{i}] {t}" for i, t in enumerate(evidence_items, 1)]
    return "\n".join([
        "Answer the question using only the evidence below.",
        "If the evidence does not contain the answer, respond exactly with 'Cannot answer'.",
        "Return only the shortest answer. Do not explain.",
        "",
        "Evidence:",
        "\n\n".join(lines),
        "",
        f"Question: {question}",
        "Answer:",
    ])

def call_deepseek(client: OpenAI, prompt: str, model: str, max_tokens: int) -> str:
    for wait in [1, 2, 4, 8, 16, 30, 45, 60]:
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=max_tokens, timeout=60)
            ans = r.choices[0].message.content.strip()
            if ans:
                return ans
            raise RuntimeError("Empty answer")
        except Exception as e:
            print(f"  API fail (wait={wait}s): {e}")
            time.sleep(wait)
    raise RuntimeError("DeepSeek failed after all retries")

def load_ranking(path: Path, top_k: int) -> dict[str, list[str]]:
    """Load ranking CSV with query_id,memory_id,rank (and optionally score) columns."""
    out = defaultdict(list)
    for row in csv.DictReader(path.open(encoding="utf-8-sig", newline="")):
        qid = row.get("query_id", row.get("qa_id", "")).strip()
        mid = row.get("memory_id", "").strip()
        if not qid or not mid:
            continue
        out[qid].append(mid)
    # Deduplicate preserving order and trim to top_k
    final = {}
    for qid, mids in out.items():
        seen = set()
        unique = []
        for m in mids:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        final[qid] = unique[:top_k]
    return final

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--memories", required=True, type=Path)
    ap.add_argument("--ranking", required=True, type=Path)
    ap.add_argument("--method", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--reader-top-k", type=int, default=10)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-new-calls", type=int, default=0)
    ap.add_argument("--sleep-min", type=float, default=0.3)
    ap.add_argument("--sleep-max", type=float, default=0.5)
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    memory_text = load_memory_text(args.memories)
    ranking = load_ranking(args.ranking, args.reader_top_k)
    method = args.method

    # Validate
    missing = [qid for qid in questions if qid not in ranking]
    if missing:
        raise RuntimeError(f"Ranking missing for {len(missing)} queries: {missing[:5]}")

    # Resume
    completed = set()
    if args.output.exists():
        for row in load_jsonl(args.output):
            qid = row.get("query_id", row.get("qa_id", ""))
            pred = row.get("prediction", row.get("predicted_answer", "")).strip()
            if qid and pred:
                completed.add(qid)

    pending = sorted(q for q in questions if q not in completed)
    print(f"Questions: {len(questions)}")
    print(f"Completed: {len(completed)}")
    print(f"Pending: {len(pending)}")

    if not pending:
        print("All done.")
        return

    api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("DEEPSEEK_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    new_calls = 0
    for qid in pending:
        if args.max_new_calls > 0 and new_calls >= args.max_new_calls:
            break

        q = questions[qid]
        mem_ids = ranking[qid]
        evidence = [memory_text.get(mid, "") for mid in mem_ids]
        prompt = build_prompt(q["question"], evidence)
        pred = call_deepseek(client, prompt, args.model, args.max_tokens)

        row = {
            "query_id": qid, "qa_id": qid, "category": q["category"],
            "method": method, "question": q["question"],
            "gold_answer": q["gold_answer"],
            "prediction": pred, "predicted_answer": pred,
            "top10_memory_ids": ";".join(mem_ids),
            "model": args.model, "temperature": 0,
            "top_k": args.reader_top_k, "scope": "canonical_cat1_4",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        new_calls += 1
        if new_calls % 20 == 0:
            print(f"  {len(completed)+new_calls}/{EXPECTED_N}")
        time.sleep(args.sleep_min + random.random() * (args.sleep_max - args.sleep_min))

    final = len(load_jsonl(args.output)) if args.output.exists() else 0
    print(f"New calls: {new_calls}, Total: {final}/{EXPECTED_N}")

if __name__ == "__main__":
    main()
