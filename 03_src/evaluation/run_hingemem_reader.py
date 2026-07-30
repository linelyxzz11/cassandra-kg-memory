#!/usr/bin/env python3
"""HingeMem-compatible Reader with unified + category-format prompts. GPT-4o-mini."""
from __future__ import annotations
import argparse, csv, json, os, random, re, time
from collections import defaultdict
from pathlib import Path
from openai import OpenAI

EXPECTED_CAT14 = 1540
EXPECTED_CAT5 = 446
EXPECTED_FULL5 = 1986

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.uiuihao.com/v1"

# ============================================================
# Data loading
# ============================================================

def load_records(path: Path):
    rows = []
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
    return rows

def load_questions(path: Path, filter_cats: set[str] | None = None):
    questions = {}
    for row in load_records(path):
        qid = (row.get("qa_id") or row.get("query_id") or "").strip()
        cat = (row.get("category") or "").strip()
        if not qid or not cat:
            continue
        if filter_cats and cat not in filter_cats:
            continue
        questions[qid] = {
            "query_id": qid,
            "category": cat,
            "question": (row.get("question") or "").strip(),
            "answer": (row.get("answer") or "").strip(),
        }
    return questions

def load_memory_text(path: Path):
    out = {}
    for row in load_records(path):
        mid = (row.get("memory_id") or "").strip()
        if not mid:
            continue
        out[mid] = {
            "text": (row.get("text") or "").strip(),
            "speaker": (row.get("speaker") or "").strip(),
            "timestamp": (row.get("timestamp") or "").strip(),
        }
    return out

def load_ranking(path: Path, top_k: int, **kw):
    out = defaultdict(list)
    for row in load_records(path):
        if kw:
            match = all(row.get(k, "") == v for k, v in kw.items())
            if not match:
                continue
        qid = (row.get("query_id") or row.get("qa_id") or "").strip()
        mid = (row.get("memory_id") or "").strip()
        if qid and mid:
            out[qid].append(mid)
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

def build_sample_scoped_dense(
    dense_path: Path, mem: dict, questions: set[str], top_k: int = 50
):
    """Build conversation-scoped dense ranking from frozen scores."""
    conv = defaultdict(set)
    for mid, info in mem.items():
        sample = mid.split("_session_")[0] if "_session_" in mid else ""
        if sample:
            conv[sample].add(mid)
    scores = defaultdict(list)
    for row in load_records(dense_path):
        qid = (row.get("query_id") or row.get("qa_id") or "").strip()
        if qid not in questions:
            continue
        cid = qid.split("_qa_")[0]
        mid = (row.get("memory_id") or "").strip()
        if mid not in conv.get(cid, set()):
            continue
        scores[qid].append((mid, float(row.get("score", 0))))
    out = {}
    for qid in sorted(scores):
        scored = sorted(scores[qid], key=lambda x: -x[1])
        out[qid] = [mid for mid, _ in scored[:top_k]]
    return out

# ============================================================
# Context rendering
# ============================================================

def render_context(memory_ids: list[str], mem: dict, question: dict) -> str:
    """Render memory evidence with speaker/timestamp. Cat2 adds dates."""
    lines = []
    for i, mid in enumerate(memory_ids, 1):
        info = mem.get(mid, {})
        text = info.get("text", "")
        speaker = info.get("speaker", "")
        timestamp = info.get("timestamp", "")
        lines.append(f"[{i}]")
        if timestamp:
            lines.append(f"Date: {timestamp}")
        if speaker:
            lines.append(f"Speaker: {speaker}")
        lines.append(f"Text: {text}")
        lines.append("")
    return "\n".join(lines)

# ============================================================
# Prompt builders
# ============================================================

UNIFIED_PROMPT = (
    "You are given a conversation context and a question.\n\n"
    "Based on the context, answer the question with a short phrase.\n"
    "Use the exact wording from the context whenever possible.\n"
    "Do not provide an explanation or reasoning process.\n\n"
    "Conversation context:\n{CONTEXT}\n\n"
    "Question:\n{QUESTION}\n\n"
    "Short answer:"
)

TEMPORAL_PROMPT = (
    "You are given a conversation context containing dated dialogue memories.\n\n"
    "Based on the context, answer the temporal question with a short phrase.\n"
    "Use the conversation dates and time expressions to infer an approximate\n"
    "date or time when necessary.\n"
    "Use the exact wording from the context whenever possible.\n"
    "Do not provide an explanation or reasoning process.\n\n"
    "Conversation context:\n{CONTEXT}\n\n"
    "Question:\n{QUESTION}\n\n"
    "Short answer:"
)

CAT5_PROMPT = (
    "You are given a conversation context and an adversarial question.\n\n"
    "Based on the context, select the correct answer.\n\n"
    "Conversation context:\n{CONTEXT}\n\n"
    "Question:\n{QUESTION}\n\n"
    "Options:\n"
    "(a) {ADVERSARIAL_ANSWER}\n"
    "(b) Not mentioned in the conversation\n\n"
    "Return only \"(a)\" or \"(b)\".\n"
    "Answer:"
)

def build_prompt(
    question: dict, context: str, use_category_format: bool
) -> str:
    cat = question["category"]
    q_text = question["question"]
    if not use_category_format:
        return UNIFIED_PROMPT.format(CONTEXT=context, QUESTION=q_text)
    if cat == "2":
        return TEMPORAL_PROMPT.format(CONTEXT=context, QUESTION=q_text)
    if cat == "5":
        aa = question["answer"]
        return CAT5_PROMPT.format(
            CONTEXT=context, QUESTION=q_text, ADVERSARIAL_ANSWER=aa
        )
    return UNIFIED_PROMPT.format(CONTEXT=context, QUESTION=q_text)

# ============================================================
# API call
# ============================================================

def call_api(client: OpenAI, prompt: str, model: str, max_tokens: int) -> str:
    for wait in [1, 2, 4, 8, 16, 30, 45, 60]:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                timeout=120,
            )
            ans = r.choices[0].message.content
            if ans is None:
                raise RuntimeError("None answer")
            ans = ans.strip()
            if ans:
                return ans
            raise RuntimeError("Empty answer")
        except Exception as e:
            print(f"  API fail (wait={wait}s): {e}")
            time.sleep(wait)
    raise RuntimeError("Failed after all retries")

# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--memories", required=True, type=Path)
    ap.add_argument("--ranking", required=True, type=Path)
    ap.add_argument("--dense-scores", type=Path)
    ap.add_argument("--method", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--reader-top-k", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-new-calls", type=int, default=0)
    ap.add_argument("--sleep-min", type=float, default=0.1)
    ap.add_argument("--sleep-max", type=float, default=0.3)
    ap.add_argument("--category-format", action="store_true",
                    help="Use category-specific prompts (Cat2 temporal, Cat5 adversarial)")
    ap.add_argument("--filter-cats", default="",
                    help="Only process these categories, e.g. '1,2,3,4' or '5'")
    args = ap.parse_args()

    filter_cats = set(args.filter_cats.split(",")) if args.filter_cats else None
    questions = load_questions(args.questions, filter_cats)
    if len(questions) not in {EXPECTED_CAT14, EXPECTED_CAT5, EXPECTED_FULL5}:
        raise RuntimeError(f"Expected {EXPECTED_CAT14}/{EXPECTED_CAT5}/{EXPECTED_FULL5} questions, got {len(questions)}")

    memory_data = load_memory_text(args.memories)

    if args.dense_scores:
        ranking = build_sample_scoped_dense(
            args.dense_scores, memory_data, set(questions), args.reader_top_k
        )
    else:
        ranking = load_ranking(args.ranking, args.reader_top_k)

    missing = [q for q in questions if q not in ranking]
    if missing:
        raise RuntimeError(f"Ranking missing for {len(missing)} queries: {missing[:5]}")

    completed = set()
    if args.output.exists():
        for row in load_records(args.output):
            qid = (row.get("query_id") or row.get("qa_id") or "").strip()
            pred = (row.get("prediction") or "").strip()
            if qid and pred:
                completed.add(qid)

    pending = sorted(q for q in questions if q not in completed)
    print(f"Questions: {len(questions)}")
    print(f"Completed: {len(completed)}")
    print(f"Pending: {len(pending)}")
    print(f"Category format: {args.category_format}")

    if not pending:
        print("All done.")
        return

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    new_calls = 0
    for qid in pending:
        if args.max_new_calls > 0 and new_calls >= args.max_new_calls:
            break

        q = questions[qid]
        mem_ids = ranking.get(qid, [])[:args.reader_top_k]
        context = render_context(mem_ids, memory_data, q)
        prompt = build_prompt(q, context, args.category_format)
        pred = call_api(client, prompt, args.model, args.max_tokens)

        row = {
            "query_id": qid,
            "qa_id": qid,
            "category": q["category"],
            "method": args.method,
            "question": q["question"],
            "gold_answer": q["answer"],
            "prediction": pred,
            "predicted_answer": pred,
            "top10_memory_ids": ";".join(mem_ids),
            "model": args.model,
            "temperature": 0,
            "top_k": args.reader_top_k,
            "category_format": args.category_format,
            "scope": "canonical",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        new_calls += 1
        if new_calls % 20 == 0:
            print(f"  {len(completed) + new_calls}/{len(questions)}")
        time.sleep(args.sleep_min + random.random() * (args.sleep_max - args.sleep_min))

    final = len(load_records(args.output)) if args.output.exists() else 0
    print(f"New calls: {new_calls}, Total: {final}/{len(questions)}")

if __name__ == "__main__":
    main()
