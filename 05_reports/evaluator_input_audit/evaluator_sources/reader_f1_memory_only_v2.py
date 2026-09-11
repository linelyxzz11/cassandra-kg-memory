import argparse
import csv
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

NUMBER_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}

DEFAULT_METHODS = [
    ("BM25", "results/sample_scoped/locomo_bm25_sample_scoped_results.csv"),
    ("Dense-bge", "results/sample_scoped/locomo_dense_bge_sample_scoped_results.csv"),
    ("Dense-bge+GlobalKG", "results/sample_scoped/locomo_dense_global_kg_sample_scoped_results.csv"),
    ("Dense-bge+QueryKG", "results/sample_scoped/locomo_dense_query_kg_sample_scoped_results.csv"),
]

PRED_FIELDS = [
    "qa_id", "category", "method", "question", "gold_answer", "adversarial_answer",
    "top10_memory_ids", "resolved_memory_ids", "missing_memory_ids",
    "gold_memory_ids", "gold_in_top1", "gold_in_top5", "gold_in_top10",
    "predicted_answer", "strict_em", "strict_f1", "relaxed_em", "relaxed_f1", "is_cannot_answer",
]


def infer_base_dir():
    here = Path(__file__).resolve()
    candidates = []
    try:
        candidates.append(here.parents[2])
    except IndexError:
        pass
    candidates.append(Path.cwd())
    for c in candidates:
        if (c / "results/locomo_qa_records.csv").exists() and (c / "results/locomo_memory_records.csv").exists():
            return c
    return candidates[0]


def load_csv(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append_csv(path, row, fieldnames):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if p.exists() else "w"
    with p.open(mode, "a" if False else "", encoding="utf-8-sig"):
        pass


def append_csv(path, row, fieldnames):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if p.exists() else "w"
    with p.open(mode, encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        w.writerow(row)


def write_csv(path, rows, fieldnames):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def normalize_strict(text):
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_relaxed(text):
    text = normalize_strict(text)
    tokens = text.split()
    out = [NUMBER_MAP.get(tok, tok) for tok in tokens]
    return " ".join(out)


def is_cannot_answer(pred):
    s = normalize_relaxed(pred)
    patterns = [
        "cannot answer", "can not answer", "cannot determine", "can not determine",
        "not enough information", "insufficient information", "not mentioned",
        "no information", "not provided", "unknown", "no evidence",
    ]
    return 1 if any(p in s for p in patterns) else 0


def compute_em(pred, gold, normalizer):
    return 1 if normalizer(pred) == normalizer(gold) else 0


def compute_token_f1(pred, gold, normalizer):
    pred_tokens = normalizer(pred).split()
    gold_tokens = normalizer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def split_ids(raw):
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    return [x.strip() for x in re.split(r"[;,|]", raw) if x.strip()]


def normalize_session_id(session_id):
    s = str(session_id or "").strip()
    if not s:
        return ""
    if s.startswith("session_"):
        return s
    m = re.search(r"\d+", s)
    if m:
        return f"session_{int(m.group(0))}"
    return s


def dia_colon_to_underscore(dia_id):
    return str(dia_id or "").strip().replace(":", "_")


def dia_underscore_to_colon(dia_id):
    s = str(dia_id or "").strip()
    m = re.match(r"^(D\d+)_(\d+)$", s)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return s


def render_memory(row):
    mid = str(row.get("memory_id", "")).strip()
    sample_id = str(row.get("sample_id", "")).strip()
    session_id = str(row.get("session_id", "")).strip()
    dia_id = str(row.get("dia_id", "")).strip()
    speaker = str(row.get("speaker", "")).strip()
    timestamp = str(row.get("timestamp", "")).strip()
    text = str(row.get("text", "")).strip()

    meta = []
    if mid:
        meta.append(f"memory_id={mid}")
    if sample_id:
        meta.append(f"sample={sample_id}")
    if session_id:
        meta.append(f"session={session_id}")
    if dia_id:
        meta.append(f"turn={dia_id}")
    if timestamp:
        meta.append(f"time={timestamp}")
    if speaker:
        meta.append(f"speaker={speaker}")
    if meta:
        return " | ".join(meta) + f"\nText: {text}"
    return text


def memory_aliases_from_row(row):
    aliases = set()
    mid = str(row.get("memory_id", "")).strip()
    sample_id = str(row.get("sample_id", "")).strip()
    session_id = normalize_session_id(row.get("session_id", ""))
    dia_id = str(row.get("dia_id", "")).strip()

    if mid:
        aliases.add(mid)
        aliases.add(mid.replace(":", "_"))

    if sample_id and dia_id:
        dia_us = dia_colon_to_underscore(dia_id)
        dia_colon = dia_underscore_to_colon(dia_id)
        aliases.add(f"{sample_id}_{dia_us}")
        aliases.add(f"{sample_id}_{dia_colon}")
        if session_id:
            aliases.add(f"{sample_id}_{session_id}_{dia_colon}")
            aliases.add(f"{sample_id}_{session_id}_{dia_us}")

    if mid:
        m = re.match(r"^(.*)_(D\d+)_(\d+)$", mid)
        if m:
            sample = m.group(1)
            dia_colon = f"{m.group(2)}:{m.group(3)}"
            dia_us = f"{m.group(2)}_{m.group(3)}"
            session = f"session_{int(m.group(2)[1:])}"
            aliases.add(f"{sample}_{dia_colon}")
            aliases.add(f"{sample}_{dia_us}")
            aliases.add(f"{sample}_{session}_{dia_colon}")
            aliases.add(f"{sample}_{session}_{dia_us}")

    return {a for a in aliases if a}


def load_memory_records(path, evidence_map_path=None):
    memory_render = {}
    alias_to_mid = {}
    raw_rows = {}

    for row in load_csv(path):
        mid = str(row.get("memory_id", "")).strip()
        if not mid:
            continue
        raw_rows[mid] = row
        memory_render[mid] = render_memory(row)
        for alias in memory_aliases_from_row(row):
            alias_to_mid[alias] = mid

    if evidence_map_path and Path(evidence_map_path).exists():
        for row in load_csv(evidence_map_path):
            mid = str(row.get("memory_id", "")).strip()
            evidence_id = str(row.get("evidence_id", "")).strip()
            sample_id = str(row.get("sample_id", "")).strip()
            if mid and mid in memory_render:
                if evidence_id:
                    alias_to_mid[evidence_id] = mid
                    alias_to_mid[evidence_id.replace(":", "_")] = mid
                if sample_id and evidence_id:
                    alias_to_mid[f"{sample_id}_{evidence_id}"] = mid
                    alias_to_mid[f"{sample_id}_{evidence_id.replace(':', '_')}"] = mid
                    m = re.match(r"^D(\d+):", evidence_id)
                    if m:
                        alias_to_mid[f"{sample_id}_session_{int(m.group(1))}_{evidence_id}"] = mid
                        alias_to_mid[f"{sample_id}_session_{int(m.group(1))}_{evidence_id.replace(':', '_')}"] = mid

    return memory_render, alias_to_mid, raw_rows


def resolve_memory_id(raw_mid, memory_render, alias_to_mid):
    raw_mid = str(raw_mid or "").strip()
    if not raw_mid:
        return ""
    if raw_mid in memory_render:
        return raw_mid
    if raw_mid in alias_to_mid:
        return alias_to_mid[raw_mid]

    candidates = [
        raw_mid.replace(":", "_"),
        raw_mid.replace("_", ":"),
    ]

    m = re.match(r"^(.*)_session_(\d+)_(D\d+):(\d+)$", raw_mid)
    if m:
        candidates.append(f"{m.group(1)}_{m.group(3)}_{m.group(4)}")
        candidates.append(f"{m.group(1)}_{m.group(3)}:{m.group(4)}")

    m = re.match(r"^(.*)_session_(\d+)_(D\d+)_(\d+)$", raw_mid)
    if m:
        candidates.append(f"{m.group(1)}_{m.group(3)}_{m.group(4)}")
        candidates.append(f"{m.group(1)}_{m.group(3)}:{m.group(4)}")

    for c in candidates:
        if c in memory_render:
            return c
        if c in alias_to_mid:
            return alias_to_mid[c]
    return ""


def load_qa_data(path):
    qa_info = {}
    for row in load_csv(path):
        qid = str(row.get("qa_id", "")).strip()
        if not qid:
            continue
        ans = str(row.get("answer", "")).strip()
        adv = str(row.get("adversarial_answer", "")).strip()
        qa_info[qid] = {
            "category": str(row.get("category", "")).strip(),
            "question": str(row.get("question", "")).strip(),
            "gold_answer": ans,
            "adversarial_answer": adv,
        }
    return qa_info



def load_gold_map(evidence_map_path, memory_render=None, alias_to_mid=None):
    """Load QA -> canonical gold memory ids.

    The evidence map may contain canonical memory_id or aliases such as D1:3.
    We resolve every id through the same alias layer used for retrieved ids so
    retrieval-hit diagnostics are comparable across file formats.
    """
    gold_map = defaultdict(set)
    for row in load_csv(evidence_map_path):
        qid = str(row.get("qa_id", "")).strip()
        raw_mid = str(row.get("memory_id", "")).strip()
        evidence_id = str(row.get("evidence_id", "")).strip()
        sample_id = str(row.get("sample_id", "")).strip()

        candidates = []
        if raw_mid:
            candidates.append(raw_mid)
        if evidence_id:
            candidates.append(evidence_id)
            if sample_id:
                candidates.extend([
                    f"{sample_id}_{evidence_id}",
                    f"{sample_id}_{evidence_id.replace(':', '_')}",
                ])

        resolved = ""
        if memory_render is not None and alias_to_mid is not None:
            for cand in candidates:
                resolved = resolve_memory_id(cand, memory_render, alias_to_mid)
                if resolved:
                    break
        else:
            resolved = raw_mid or evidence_id

        if qid and resolved:
            gold_map[qid].add(resolved)
    return dict(gold_map)


def load_retrieval_rankings(csv_path):
    rows = load_csv(csv_path)
    if not rows:
        return {}
    candidate_cols = ["retrieved_memory_ids", "top10_memory_ids", "memory_ids", "top_memory_ids"]
    available = set(rows[0].keys())
    col = next((c for c in candidate_cols if c in available), None)
    if col is None:
        raise ValueError(f"No retrieval id column found in {csv_path}. Available columns: {sorted(available)}")

    out = {}
    for row in rows:
        qid = str(row.get("qa_id", "")).strip()
        if not qid:
            continue
        out[qid] = split_ids(row.get(col, ""))[:10]
    return out


def build_prompt(question, evidence_items):
    evidence_lines = []
    for i, text in enumerate(evidence_items, 1):
        evidence_lines.append(f"[{i}] {text}")
    return "\n".join([
        "Answer the question using only the evidence below.",
        "If the evidence does not contain the answer, respond exactly with 'Cannot answer'.",
        "Return only the shortest answer. Do not explain.",
        "",
        "Evidence:",
        "\n\n".join(evidence_lines),
        "",
        f"Question: {question}",
        "Answer:",
    ])


def call_deepseek(client, prompt):
    last_error = None
    sleep_times = [1, 2, 4, 8, 16, 30, 45, 60]
    for attempt, wait_s in enumerate(sleep_times, start=1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=128,
                timeout=60,
            )
            ans = resp.choices[0].message.content.strip()
            if ans.startswith("ERROR:"):
                raise RuntimeError(ans)
            return ans
        except Exception as e:
            last_error = e
            print(f"  DeepSeek call failed attempt {attempt}/{len(sleep_times)}: {e}")
            if attempt < len(sleep_times):
                time.sleep(wait_s)
    raise RuntimeError(str(last_error))


def dedup_prediction_rows(rows):
    by_key = {}
    for row in rows:
        key = (str(row.get("qa_id", "")).strip(), str(row.get("method", "")).strip())
        if key[0] and key[1] and str(row.get("predicted_answer", "")).strip() and not str(row.get("predicted_answer", "")).startswith("ERROR:"):
            by_key[key] = row
    return list(by_key.values())


def coerce_rows(rows):
    out = []
    for r in rows:
        rr = dict(r)
        rr["category"] = int(rr["category"])
        rr["strict_em"] = int(float(rr["strict_em"]))
        rr["strict_f1"] = float(rr["strict_f1"])
        rr["relaxed_em"] = int(float(rr["relaxed_em"]))
        rr["relaxed_f1"] = float(rr["relaxed_f1"])
        rr["is_cannot_answer"] = int(float(rr["is_cannot_answer"]))
        for k in ("gold_in_top1", "gold_in_top5", "gold_in_top10"):
            try:
                rr[k] = int(float(rr.get(k, 0) or 0))
            except ValueError:
                rr[k] = 0
        out.append(rr)
    return out


def compute_summary(rows, cat_range):
    cats = set(cat_range)
    filtered = [r for r in rows if int(r["category"]) in cats]
    n = len(filtered)
    if n == 0:
        return {
            "n": 0, "rEM": 0, "rF1": 0, "sEM": 0, "sF1": 0,
            "wrong_abstention": 0, "hit1": 0, "hit5": 0, "hit10": 0,
        }
    s_em = sum(r["strict_em"] for r in filtered) / n
    s_f1 = sum(r["strict_f1"] for r in filtered) / n
    r_em = sum(r["relaxed_em"] for r in filtered) / n
    r_f1 = sum(r["relaxed_f1"] for r in filtered) / n
    wrong_abst = sum(
        1 for r in filtered
        if r["is_cannot_answer"] and r["relaxed_f1"] == 0 and r.get("gold_answer", "")
    ) / n
    hit1 = sum(int(r.get("gold_in_top1", 0)) for r in filtered) / n
    hit5 = sum(int(r.get("gold_in_top5", 0)) for r in filtered) / n
    hit10 = sum(int(r.get("gold_in_top10", 0)) for r in filtered) / n
    return {
        "n": n, "rEM": r_em, "rF1": r_f1, "sEM": s_em, "sF1": s_f1,
        "wrong_abstention": wrong_abst, "hit1": hit1, "hit5": hit5, "hit10": hit10,
    }

def compute_abstention(rows):
    cat5_rows = [r for r in rows if int(r["category"]) == 5]
    n = len(cat5_rows)
    if n == 0:
        return {"n": 0, "abstention_accuracy": 0, "non_abstention_rate": 0}
    ca = sum(1 for r in cat5_rows if r["is_cannot_answer"]) / n
    non_ca = sum(1 for r in cat5_rows if not r["is_cannot_answer"]) / n
    return {"n": n, "abstention_accuracy": ca, "non_abstention_rate": non_ca}


def select_qa_ids(qa_info, categories, sample_size, seed, sample_strategy="random"):
    ids = sorted(qa_info.keys())
    if categories:
        cat_set = {str(c) for c in categories}
        ids = [qid for qid in ids if qa_info[qid]["category"] in cat_set]

    if not sample_size or sample_size <= 0 or sample_size >= len(ids):
        return ids

    rng = random.Random(seed)

    if sample_strategy == "balanced-category":
        by_cat = defaultdict(list)
        for qid in ids:
            by_cat[qa_info[qid]["category"]].append(qid)
        cats = sorted(by_cat.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        per_cat = sample_size // max(1, len(cats))
        remainder = sample_size % max(1, len(cats))
        selected = []
        for i, cat in enumerate(cats):
            quota = per_cat + (1 if i < remainder else 0)
            pool = sorted(by_cat[cat])
            if quota >= len(pool):
                selected.extend(pool)
            else:
                selected.extend(rng.sample(pool, quota))
        return sorted(selected)

    return sorted(rng.sample(ids, sample_size))


def validate_rankings(qa_ids, methods, rankings, memory_render, alias_to_mid, max_examples=10):
    stats = []
    examples = []
    for method_name, _ in methods:
        total_ids = 0
        missing = 0
        missing_q = 0
        for qid in qa_ids:
            ids = rankings.get(method_name, {}).get(qid, [])[:10]
            q_missing = []
            for mid in ids:
                total_ids += 1
                if not resolve_memory_id(mid, memory_render, alias_to_mid):
                    missing += 1
                    q_missing.append(mid)
            if q_missing:
                missing_q += 1
                if len(examples) < max_examples:
                    examples.append((method_name, qid, q_missing[:5]))
        rate = missing / total_ids if total_ids else 0
        stats.append((method_name, total_ids, missing, rate, missing_q))
    return stats, examples


def summarize_and_write(prediction_csv, summary_csv, category_csv, cat5_csv, method_order=None):
    rows = dedup_prediction_rows(load_csv(prediction_csv)) if Path(prediction_csv).exists() else []
    rows = coerce_rows(rows)

    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    ordered_methods = []
    if method_order:
        ordered_methods.extend([m for m in method_order if m in by_method])
    ordered_methods.extend([m for m in sorted(by_method.keys()) if m not in ordered_methods])

    print("\n" + "=" * 70)
    print("RESULTS: Answerable QA (cat1-4), deduplicated by qa_id+method")
    print("=" * 70)
    print(f"{'Method':25s} {'n':>5s} {'rEM':>7s} {'rF1':>7s} {'WrongAbst':>9s} {'Hit@10':>7s}")
    print("-" * 70)

    summary_rows = []
    for method_name in ordered_methods:
        method_rows = by_method[method_name]
        s = compute_summary(method_rows, range(1, 5))
        summary_rows.append({"Method": method_name, **{k: round(v, 4) for k, v in s.items()}})
        print(
            f"{method_name:25s} {s['n']:>5d} {s['rEM']:>7.4f} {s['rF1']:>7.4f} "
            f"{s['wrong_abstention']:>9.4f} {s['hit10']:>7.4f}"
        )
    write_csv(
        summary_csv,
        summary_rows,
        ["Method", "n", "rEM", "rF1", "sEM", "sF1", "wrong_abstention", "hit1", "hit5", "hit10"],
    )

    print("\n" + "=" * 70)
    print("RESULTS: Adversarial QA (cat5), deduplicated by qa_id+method")
    print("=" * 70)
    print(f"{'Method':25s} {'n':>5s} {'AbstAcc':>8s} {'NonAbst':>8s}")
    print("-" * 55)

    cat5_rows = []
    for method_name in ordered_methods:
        method_rows = by_method[method_name]
        a = compute_abstention(method_rows)
        cat5_rows.append({"Method": method_name, **{k: round(v, 4) for k, v in a.items()}})
        print(f"{method_name:25s} {a['n']:>5d} {a['abstention_accuracy']:>8.4f} {a['non_abstention_rate']:>8.4f}")
    write_csv(cat5_csv, cat5_rows, ["Method", "n", "abstention_accuracy", "non_abstention_rate"])

    print("\n" + "=" * 70)
    print("RESULTS: Category-wise F1 / n / retrieval Hit@10 (cat1-4)")
    print("=" * 70)
    cats = [1, 2, 3, 4]
    cat_names = {1: "multi-hop", 2: "temporal", 3: "commonsense", 4: "single-hop"}
    print(f"{'Method':25s} ", end="")
    for c in cats:
        print(f"cat{c}({cat_names[c]}) n/F1/H10      ", end="")
    print()
    print("-" * 140)

    cat_rows = []
    for method_name in ordered_methods:
        method_rows = by_method[method_name]
        row_out = {"Method": method_name}
        print(f"{method_name:25s} ", end="")
        for c in cats:
            s = compute_summary(method_rows, [c])
            print(f"{s['n']:>3d}/{s['rF1']:.4f}/{s['hit10']:.4f}          ", end="")
            row_out[f"cat{c}_n"] = s["n"]
            row_out[f"cat{c}_rF1"] = round(s["rF1"], 4)
            row_out[f"cat{c}_wrong_abstention"] = round(s["wrong_abstention"], 4)
            row_out[f"cat{c}_hit1"] = round(s["hit1"], 4)
            row_out[f"cat{c}_hit5"] = round(s["hit5"], 4)
            row_out[f"cat{c}_hit10"] = round(s["hit10"], 4)
        print()
        cat_rows.append(row_out)

    cat_fields = ["Method"]
    for c in cats:
        cat_fields.extend([
            f"cat{c}_n", f"cat{c}_rF1", f"cat{c}_wrong_abstention",
            f"cat{c}_hit1", f"cat{c}_hit5", f"cat{c}_hit10",
        ])
    write_csv(category_csv, cat_rows, cat_fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(infer_base_dir()))
    parser.add_argument("--output-suffix", default="v2")
    parser.add_argument("--sample-size", type=int, default=0, help="0 means all QA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-strategy",
        choices=["random", "balanced-category"],
        default="random",
        help="random keeps the original behavior; balanced-category samples roughly equally from selected categories.",
    )
    parser.add_argument("--categories", default="", help="Comma-separated categories, e.g. 1,2,3,4")
    parser.add_argument("--methods", default="", help="Comma-separated method names; default runs all")
    parser.add_argument("--max-new-calls", type=int, default=0, help="0 means no cap")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-prompts", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--sleep-min", type=float, default=0.3)
    parser.add_argument("--sleep-max", type=float, default=0.5)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    output_dir = base_dir / "results/final"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    predictions_csv = output_dir / f"reader_f1_memory_only{suffix}_predictions.csv"
    summary_csv = output_dir / f"reader_f1_memory_only{suffix}_summary.csv"
    category_csv = output_dir / f"reader_f1_memory_only{suffix}_by_category.csv"
    cat5_csv = output_dir / f"reader_f1_cat5_abstention{suffix}.csv"

    qa_csv = base_dir / "results/locomo_qa_records.csv"
    memory_csv = base_dir / "results/locomo_memory_records.csv"
    evidence_map_csv = base_dir / "results/locomo_evidence_map.csv"

    methods = [(name, str(base_dir / rel)) for name, rel in DEFAULT_METHODS]
    if args.methods.strip():
        wanted = {x.strip() for x in args.methods.split(",") if x.strip()}
        methods = [m for m in methods if m[0] in wanted]
        if not methods:
            raise ValueError(f"No methods matched --methods={args.methods}")

    categories = [int(x.strip()) for x in args.categories.split(",") if x.strip()] if args.categories.strip() else []

    if args.overwrite:
        for p in [predictions_csv, summary_csv, category_csv, cat5_csv]:
            if p.exists():
                p.unlink()

    print("Loading data...")
    print(f"  Base dir: {base_dir}")
    memory_render, alias_to_mid, _ = load_memory_records(memory_csv, evidence_map_csv)
    gold_map = load_gold_map(evidence_map_csv, memory_render, alias_to_mid)
    qa_info = load_qa_data(qa_csv)
    rankings = {}
    for method_name, csv_path in methods:
        rankings[method_name] = load_retrieval_rankings(csv_path)
        print(f"  {method_name}: {len(rankings[method_name])} rankings loaded")

    qa_ids = select_qa_ids(qa_info, categories, args.sample_size, args.seed, args.sample_strategy)
    print(f"  QA selected: {len(qa_ids)}")
    selected_cat_counts = Counter(qa_info[qid]["category"] for qid in qa_ids)
    print(f"  QA category counts: {dict(sorted(selected_cat_counts.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0])))}")
    print(f"  Memory records: {len(memory_render)}")
    print(f"  Memory aliases: {len(alias_to_mid)}")

    stats, examples = validate_rankings(qa_ids, methods, rankings, memory_render, alias_to_mid)
    print("\nMapping validation:")
    has_missing = False
    for method_name, total_ids, missing, rate, missing_q in stats:
        if missing:
            has_missing = True
        print(f"  {method_name:25s} total_top_ids={total_ids:5d} missing={missing:5d} missing_rate={rate:.4f} affected_q={missing_q}")
    if examples:
        print("\nMissing examples:")
        for method_name, qid, mids in examples:
            print(f"  {method_name} {qid}: {mids}")

    if args.validate_only:
        print("\nValidate-only mode. No API calls made.")
        return

    if has_missing and not args.allow_missing:
        raise RuntimeError("Some retrieved memory IDs cannot be mapped to memory text. Fix mapping or rerun with --allow-missing for debugging only.")

    if args.print_prompts > 0:
        printed = 0
        for qid in qa_ids:
            info = qa_info[qid]
            for method_name, _ in methods:
                raw_ids = rankings.get(method_name, {}).get(qid, [])[:10]
                evidence_items = []
                resolved_ids = []
                missing_ids = []
                for raw_mid in raw_ids:
                    rid = resolve_memory_id(raw_mid, memory_render, alias_to_mid)
                    if rid:
                        resolved_ids.append(rid)
                        evidence_items.append(memory_render[rid])
                    else:
                        missing_ids.append(raw_mid)
                        if args.allow_missing:
                            evidence_items.append(f"UNMAPPED_MEMORY_ID={raw_mid}")
                prompt = build_prompt(info["question"], evidence_items)
                print("\n" + "=" * 90)
                print(f"PROMPT qa_id={qid} method={method_name} category={info['category']}")
                print(f"raw_ids={raw_ids}")
                print(f"resolved_ids={resolved_ids}")
                print(f"missing_ids={missing_ids}")
                print("-" * 90)
                print(prompt[:4000])
                printed += 1
                if printed >= args.print_prompts:
                    print("\nPrompt print mode. No API calls made.")
                    return

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY environment variable")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    done = set()
    if predictions_csv.exists():
        existing = dedup_prediction_rows(load_csv(predictions_csv))
        for row in existing:
            done.add((row["qa_id"], row["method"]))
        print(f"\nExisting deduplicated predictions: {len(done)} completed")

    total_calls = len(qa_ids) * len(methods)
    remaining = sum(1 for qid in qa_ids for method_name, _ in methods if (qid, method_name) not in done)
    print(f"Total target calls for this run: {total_calls}")
    print(f"Remaining calls for this run: {remaining}")

    calls = 0
    for qa_id in qa_ids:
        info = qa_info[qa_id]
        category = info["category"]
        question = info["question"]
        gold_answer = info["gold_answer"]
        adversarial_answer = info["adversarial_answer"]

        for method_name, _ in methods:
            if (qa_id, method_name) in done:
                continue
            if args.max_new_calls and calls >= args.max_new_calls:
                print("Max new calls reached.")
                summarize_and_write(predictions_csv, summary_csv, category_csv, cat5_csv, method_order=[m[0] for m in methods])
                return

            raw_ids = rankings.get(method_name, {}).get(qa_id, [])[:10]
            evidence_items = []
            resolved_ids = []
            missing_ids = []
            for raw_mid in raw_ids:
                rid = resolve_memory_id(raw_mid, memory_render, alias_to_mid)
                if rid:
                    resolved_ids.append(rid)
                    evidence_items.append(memory_render[rid])
                else:
                    missing_ids.append(raw_mid)
                    if args.allow_missing:
                        evidence_items.append(f"UNMAPPED_MEMORY_ID={raw_mid}")

            if missing_ids and not args.allow_missing:
                raise RuntimeError(f"Unmapped memory IDs for {qa_id} {method_name}: {missing_ids[:10]}")

            prompt = build_prompt(question, evidence_items)
            predicted = call_deepseek(client, prompt).strip()

            eval_gold = gold_answer
            strict_em = compute_em(predicted, eval_gold, normalize_strict) if eval_gold else 0
            strict_f1 = compute_token_f1(predicted, eval_gold, normalize_strict) if eval_gold else 0.0
            relaxed_em = compute_em(predicted, eval_gold, normalize_relaxed) if eval_gold else 0
            relaxed_f1 = compute_token_f1(predicted, eval_gold, normalize_relaxed) if eval_gold else 0.0
            ca_flag = is_cannot_answer(predicted)

            gold_ids = sorted(gold_map.get(qa_id, set()))
            gold_set = set(gold_ids)
            resolved_set_1 = set(resolved_ids[:1])
            resolved_set_5 = set(resolved_ids[:5])
            resolved_set_10 = set(resolved_ids[:10])
            gold_in_top1 = int(bool(gold_set & resolved_set_1))
            gold_in_top5 = int(bool(gold_set & resolved_set_5))
            gold_in_top10 = int(bool(gold_set & resolved_set_10))

            row = {
                "qa_id": qa_id,
                "category": category,
                "method": method_name,
                "question": question,
                "gold_answer": gold_answer,
                "adversarial_answer": adversarial_answer,
                "top10_memory_ids": ";".join(raw_ids),
                "resolved_memory_ids": ";".join(resolved_ids),
                "missing_memory_ids": ";".join(missing_ids),
                "gold_memory_ids": ";".join(gold_ids),
                "gold_in_top1": gold_in_top1,
                "gold_in_top5": gold_in_top5,
                "gold_in_top10": gold_in_top10,
                "predicted_answer": predicted,
                "strict_em": strict_em,
                "strict_f1": strict_f1,
                "relaxed_em": relaxed_em,
                "relaxed_f1": relaxed_f1,
                "is_cannot_answer": ca_flag,
            }
            append_csv(predictions_csv, row, PRED_FIELDS)
            done.add((qa_id, method_name))
            calls += 1

            if calls % 20 == 0:
                print(f"  [{calls}] {qa_id} {method_name} done...")

            time.sleep(args.sleep_min + random.random() * max(0.0, args.sleep_max - args.sleep_min))

    print(f"\nTotal new API calls: {calls}")
    summarize_and_write(predictions_csv, summary_csv, category_csv, cat5_csv, method_order=[m[0] for m in methods])
    print(f"\nPredictions: {predictions_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
