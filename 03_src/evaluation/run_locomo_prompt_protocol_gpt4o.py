#!/usr/bin/env python3
"""
LoCoMo GPT-4o Dual Prompt Protocol
Setting A (Cat.x): identical short-answer prompt for Cat1-5
Setting B (Cat.v): category-format per LoCoMo official spec
Full 1986 queries, GPT-4o-2024-08-06, temperature=0, max_tokens=64
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, re, time
from collections import defaultdict, Counter
from pathlib import Path
from openai import OpenAI

EXPECTED_FULL5 = 1986
CAT5_N = 446
MODEL = "gpt-4o-2024-08-06"
BASE_URL = "https://api.uiuihao.com/v1"
MAX_TOKENS = 64

CATEGORY_DIST = {"1": 282, "2": 321, "3": 96, "4": 841, "5": 446}

# ============================================================
# Data loading
# ============================================================

def load_records(path: Path) -> list[dict]:
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

def load_questions(path: Path) -> dict[str, dict]:
    questions = {}
    for row in load_records(path):
        qid = (row.get("qa_id") or row.get("query_id") or "").strip()
        cat = (row.get("category") or "").strip()
        if not qid or cat not in CATEGORY_DIST:
            continue
        answer = (row.get("answer") or "").strip()
        questions[qid] = {
            "query_id": qid,
            "category": cat,
            "question": (row.get("question") or "").strip(),
            "answer": answer,
            "adversarial_answer": answer if cat == "5" else "",
        }
    if len(questions) != EXPECTED_FULL5:
        raise RuntimeError(f"Expected {EXPECTED_FULL5}, got {len(questions)}")
    cats = Counter(q["category"] for q in questions.values())
    if dict(cats) != CATEGORY_DIST:
        raise RuntimeError(f"Category mismatch: {dict(cats)} != {CATEGORY_DIST}")
    return questions

def load_memory_text(path: Path) -> dict[str, dict]:
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

def load_ranking(path: Path, top_k: int, **filter_kw) -> dict[str, list[str]]:
    out = defaultdict(list)
    for row in load_records(path):
        if filter_kw:
            if not all(row.get(k, "") == v for k, v in filter_kw.items()):
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

def build_sample_scoped_dense(dense_path: Path, mem: dict, qids: set[str], top_k: int) -> dict[str, list[str]]:
    conv = defaultdict(set)
    for mid in mem:
        sid = mid.split("_session_")[0] if "_session_" in mid else ""
        if sid:
            conv[sid].add(mid)
    scores = defaultdict(list)
    for row in load_records(dense_path):
        qid = (row.get("query_id") or row.get("qa_id") or "").strip()
        if qid not in qids:
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
# Full5 ranking assembly (cat1-4 sample + cat5 legacy fallback)
# ============================================================

def build_full5_rankings(questions: dict[str, dict], mem: dict,
                         method: str) -> dict[str, list[str]]:
    BASE = Path("D:/memorytable/cassandra-kg-memory")
    cat14_qids = {qid for qid, q in questions.items() if q["category"] != "5"}
    cat5_qids = {qid for qid, q in questions.items() if q["category"] == "5"}
    all_qids = set(questions)

    legacy = load_records(
        BASE / "results/final/reader_f1_memory_only_full_scoped_bm25_predictions.csv"
    )

    # Map method to legacy name and sample-scoped ranking file
    if method == "BM25":
        legacy_name = "BM25"
        sample_path = BASE / "05_reports/official_eval/bm25_raw_ranking_canonical1540.csv"
        ranking = load_ranking(sample_path, 50)
    elif method == "Dense-bge":
        legacy_name = "Dense-bge"
        ranking = build_sample_scoped_dense(
            BASE / "scripts/experiments/artifacts/frozen_dense_scores_long.csv",
            mem, cat14_qids, 50)
    elif method == "Dense+GlobalKG":
        legacy_name = "Dense-bge+GlobalKG"
        ranking = build_sample_scoped_dense(
            BASE / "scripts/experiments/artifacts/frozen_dense_scores_long.csv",
            mem, cat14_qids, 50)
    elif method == "ZScore-Raw":
        legacy_name = "Dense-bge"
        sample_path = BASE / "05_reports/official_eval/zscore_raw_ranking_canonical1540.csv"
        ranking = load_ranking(sample_path, 50)
    elif method == "RRF_compact":
        legacy_name = "Dense-bge"
        sample_path = BASE / "05_reports/official_eval/rrf_compact_canonical1540/rrf_compact_top10.csv"
        ranking = load_ranking(sample_path, 50)
    elif method == "ZScore-RawERK":
        legacy_name = "Dense-bge"
        sample_path = BASE / "05_reports/p1_compact_component_ablation/p1c_zscore_rankings_cat1_4_1540.csv"
        ranking = load_ranking(sample_path, 50, variant="RawERK")
    else:
        raise ValueError(f"Unknown method: {method}")

    # Cat5 fallback: use legacy global top10
    cat5_legacy = [r for r in legacy if r["category"] == "5" and r["method"] == legacy_name]
    for r in cat5_legacy:
        qid = r["qa_id"]
        top10 = [m.strip() for m in r.get("top10_memory_ids", "").split(";") if m.strip()]
        if qid in cat5_qids:
            ranking[qid] = top10[:50]

    # Verify and trim to 10
    final = {}
    for qid in sorted(all_qids):
        if qid not in ranking:
            raise RuntimeError(f"No ranking for {qid} (method={method})")
        final[qid] = ranking[qid][:10]
    return final

# ============================================================
# Context rendering
# ============================================================

def render_context(memory_ids: list[str], mem: dict) -> str:
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

CAT5_UNIFIED_PROMPT = UNIFIED_PROMPT

CAT5_CATEGORY_PROMPT = (
    "You are given a conversation context and an adversarial question.\n\n"
    "Based on the context, select the correct answer.\n\n"
    "Conversation context:\n{CONTEXT}\n\n"
    "Question:\n{QUESTION}\n\n"
    "Options:\n"
    "(a) {OPTION_A}\n"
    "(b) {OPTION_B}\n\n"
    "Return only \"(a)\" or \"(b)\".\n"
    "Answer:"
)

def deterministic_swap(qid: str) -> bool:
    h = int(hashlib.shake_128(qid.encode()).hexdigest(4), 16)
    return h % 2 == 1

def build_prompt(q: dict, context: str, use_category_fmt: bool) -> str:
    cat = q["category"]
    if not use_category_fmt:
        return UNIFIED_PROMPT.format(CONTEXT=context, QUESTION=q["question"])
    if cat == "2":
        return TEMPORAL_PROMPT.format(CONTEXT=context, QUESTION=q["question"])
    if cat == "5":
        aa = q.get("adversarial_answer", q.get("answer", ""))
        if deterministic_swap(q["query_id"]):
            option_a = aa
            option_b = "Not mentioned in the conversation"
        else:
            option_a = "Not mentioned in the conversation"
            option_b = aa
        return CAT5_CATEGORY_PROMPT.format(
            CONTEXT=context, QUESTION=q["question"],
            OPTION_A=option_a, OPTION_B=option_b)
    return UNIFIED_PROMPT.format(CONTEXT=context, QUESTION=q["question"])

# ============================================================
# Pre-flight audit
# ============================================================

def preflight(questions: dict, mem: dict, ranking: dict[str, list[str]],
              method: str, output_dir: Path) -> dict:
    qids = sorted(questions)
    cats = Counter(q["category"] for q in questions.values())
    n_mem = len(mem)
    ts_ok = sum(1 for v in mem.values() if v.get("timestamp", "").strip())

    all_covered = all(qid in ranking for qid in qids)
    rank_lens = Counter(len(ranking.get(qid, [])) for qid in qids)
    missing = [qid for qid in qids if qid not in ranking]

    memory_ids_used = set()
    unresolvable = []
    for qid in qids:
        for mid in ranking.get(qid, []):
            if mid not in mem:
                unresolvable.append((qid, mid))
            else:
                memory_ids_used.add(mid)

    audit = {
        "method": method,
        "total_questions": len(qids),
        "category_counts": dict(sorted(cats.items())),
        "cat_dist_ok": dict(cats) == CATEGORY_DIST,
        "memory_records": n_mem,
        "memory_with_timestamp": ts_ok,
        "all_timestamps_ok": ts_ok == n_mem,
        "ranking_coverage": len(qids) - len(missing),
        "ranking_missing": missing[:10],
        "ranking_coverage_ok": all_covered,
        "rank_lengths": dict(sorted(rank_lens.items())),
        "unresolvable_memory_ids": len(unresolvable),
        "unresolvable_examples": [(q, m) for q, m in unresolvable[:5]],
        "preflight_pass": (dict(cats) == CATEGORY_DIST and all_covered and
                          len(unresolvable) == 0 and all(v == 10 for v in rank_lens.keys())),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preflight_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    return audit

# ============================================================
# API
# ============================================================

def call_api(client: OpenAI, prompt: str) -> str:
    for wait in [1, 2, 4, 8, 16, 30, 45, 60]:
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=MAX_TOKENS, timeout=120)
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
    ap.add_argument("--method", required=True)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--setting", required=True,
                    choices=["a_unified", "b_category"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", type=int, default=0,
                    help="Run N smoke test queries")
    ap.add_argument("--sleep-min", type=float, default=0.1)
    ap.add_argument("--sleep-max", type=float, default=0.3)
    args = ap.parse_args()

    use_category_fmt = args.setting == "b_category"
    setting_label = "Cat.x" if not use_category_fmt else "Cat.v"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    questions = load_questions(args.questions)
    memory_data = load_memory_text(args.memories)
    ranking = build_full5_rankings(questions, memory_data, args.method)

    # Pre-flight
    audit = preflight(questions, memory_data, ranking, args.method, output_dir)
    print(f"Method: {args.method}, Setting: {setting_label}")
    print(f"Questions: {len(questions)}, Cat counts: {audit['category_counts']}")
    print(f"Ranking coverage: {audit['ranking_coverage']}/{len(questions)}")
    print(f"Unresolvable memory IDs: {audit['unresolvable_memory_ids']}")
    print(f"Pre-flight: {'PASS' if audit['preflight_pass'] else 'FAIL'}")
    if not audit["preflight_pass"]:
        raise RuntimeError("Pre-flight failed")

    if args.dry_run:
        print("Dry-run only, no API calls.")
        return

    # Prompt samples
    sample_qids = sorted(questions)[:20]
    samples = []
    for qid in sample_qids:
        q = questions[qid]
        ctx = render_context(ranking[qid], memory_data)
        prompt = build_prompt(q, ctx, use_category_fmt)
        samples.append({"query_id": qid, "category": q["category"],
                       "prompt": prompt})
    (output_dir / "prompt_samples.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples), encoding="utf-8")

    # Resume
    pred_file = output_dir / "reader_predictions.jsonl"
    completed = set()
    if pred_file.exists():
        for row in load_records(pred_file):
            qid = (row.get("query_id") or row.get("qa_id") or "").strip()
            if qid and (row.get("prediction") or "").strip():
                completed.add(qid)

    pending = sorted(q for q in questions if q not in completed)
    print(f"Completed: {len(completed)}, Pending: {len(pending)}")

    if not pending:
        print("All done.")
        run_manifest = {"method": args.method, "setting": args.setting,
                        "setting_label": setting_label, "model": MODEL,
                        "temperature": 0, "max_tokens": MAX_TOKENS,
                        "total": len(questions), "api_calls": 0}
        (output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    max_calls = args.smoke if args.smoke > 0 else len(pending)
    new_calls = 0
    for qid in pending:
        if new_calls >= max_calls:
            break

        q = questions[qid]
        ctx = render_context(ranking[qid], memory_data)
        prompt = build_prompt(q, ctx, use_category_fmt)
        pred = call_api(client, prompt)

        row = {
            "query_id": qid, "qa_id": qid, "category": q["category"],
            "method": args.method, "question": q["question"],
            "gold_answer": q["answer"], "prediction": pred,
            "top10_memory_ids": ";".join(ranking[qid]),
            "model": MODEL, "temperature": 0, "max_tokens": MAX_TOKENS,
            "setting": args.setting, "setting_label": setting_label,
        }
        with pred_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        new_calls += 1
        if new_calls % 20 == 0:
            print(f"  {len(completed) + new_calls}/{len(questions)}")
        time.sleep(args.sleep_min + random.random() * (args.sleep_max - args.sleep_min))

    final_n = len(load_records(pred_file)) if pred_file.exists() else 0
    print(f"New calls: {new_calls}, Total: {final_n}/{len(questions)}")

    run_manifest = {"method": args.method, "setting": args.setting,
                    "setting_label": setting_label, "model": MODEL,
                    "temperature": 0, "max_tokens": MAX_TOKENS,
                    "total": len(questions), "new_api_calls": new_calls,
                    "total_completed": final_n}
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
