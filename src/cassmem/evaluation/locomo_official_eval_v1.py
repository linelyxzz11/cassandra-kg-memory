#!/usr/bin/env python3
"""
Offline LoCoMo QA evaluator.

Implements the official LoCoMo category-aware QA score:
- Cat1: multi-answer F1 (comma-split prediction/gold; average best match per gold item)
- Cat2/3/4: stemmed token F1
- Cat3: gold answer truncated before the first semicolon
- Cat5: 1 iff prediction contains "no information available" or "not mentioned"; else 0

It also reports clearly-labeled project diagnostics:
- rEM_cat14_project: category-aware normalized set exact match
- WrongAbstention_cat14_project: answerable query where a broad abstention phrase is used and F1=0
- Cat5 broad abstention and leakage

No API calls are made.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from nltk.stem import PorterStemmer
except ImportError as exc:
    raise SystemExit("nltk is required: pip install nltk") from exc

PS = PorterStemmer()

QID_CANDIDATES = ("qa_id", "query_id", "question_id", "id")
PRED_CANDIDATES = ("prediction", "predicted_answer", "response", "output", "generated_answer", "model_answer", "answer_prediction")
METHOD_CANDIDATES = ("method", "retriever", "retrieval_method", "experiment_name")
CATEGORY_CANDIDATES = ("category", "cat", "question_type")
GOLD_CANDIDATES = ("answer", "gold_answer", "ground_truth", "reference_answer")

OFFICIAL_CAT5_PHRASES = ("no information available", "not mentioned")
BROAD_ABSTENTION_PHRASES = (
    "cannot answer", "can not answer", "cannot determine", "can not determine",
    "not enough information", "insufficient information", "not mentioned",
    "no information", "not provided", "unknown", "no evidence", "unable to",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{line_no}: JSONL row is not an object")
                rows.append(obj)
        return rows
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(obj, list) or not all(isinstance(x, dict) for x in obj):
            raise ValueError(f"{path}: JSON input must be a list of objects")
        return obj
    raise ValueError(f"Unsupported file type: {path}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def pick_column(columns: Iterable[str], candidates: Iterable[str], explicit: str | None, role: str, required: bool = True) -> str | None:
    cols = set(columns)
    if explicit:
        if explicit not in cols:
            raise ValueError(f"Requested {role} column '{explicit}' not found. Available: {sorted(cols)}")
        return explicit
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise ValueError(f"Could not infer {role} column. Available: {sorted(cols)}")
    return None


def normalize_answer(text: Any) -> str:
    # Exact implementation order follows the official LoCoMo evaluator.
    s = str(text or "").replace(",", "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def official_exact_match(prediction: Any, ground_truth: Any) -> float:
    p = normalize_answer(prediction)
    g = normalize_answer(ground_truth)
    return float(set(p.split()) == set(g.split()))


def official_token_f1(prediction: Any, ground_truth: Any) -> float:
    pred_tokens = [PS.stem(w) for w in normalize_answer(prediction).split()]
    gold_tokens = [PS.stem(w) for w in normalize_answer(ground_truth).split()]
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def split_multi(text: Any) -> list[str]:
    return [x.strip() for x in str(text or "").split(",")]


def official_multi_answer_f1(prediction: Any, ground_truth: Any) -> float:
    predictions = split_multi(prediction)
    golds = split_multi(ground_truth)
    if not golds:
        return 0.0
    return sum(max(official_token_f1(p, g) for p in predictions) for g in golds) / len(golds)


def project_multi_answer_em(prediction: Any, ground_truth: Any) -> float:
    predictions = split_multi(prediction)
    golds = split_multi(ground_truth)
    if not golds:
        return 0.0
    return sum(max(official_exact_match(p, g) for p in predictions) for g in golds) / len(golds)


def official_cat5_correct(prediction: Any) -> float:
    p = str(prediction or "").lower()
    return float(any(phrase in p for phrase in OFFICIAL_CAT5_PHRASES))


def broad_is_abstention(prediction: Any) -> float:
    p = normalize_answer(prediction)
    return float(any(phrase in p for phrase in BROAD_ABSTENTION_PHRASES))


def score_one(category: int, prediction: Any, gold_answer: Any) -> dict[str, float | None]:
    gold = str(gold_answer or "")
    pred = str(prediction or "")
    if category == 1:
        f1 = official_multi_answer_f1(pred, gold)
        em = project_multi_answer_em(pred, gold)
        cat5 = None
    elif category in (2, 3, 4):
        if category == 3:
            gold = gold.split(";")[0].strip()
        f1 = official_token_f1(pred, gold)
        em = official_exact_match(pred, gold)
        cat5 = None
    elif category == 5:
        cat5 = official_cat5_correct(pred)
        f1 = None
        em = None
    else:
        raise ValueError(f"Unsupported category: {category}")
    broad_abst = broad_is_abstention(pred)
    wrong_abst = float(category in (1, 2, 3, 4) and broad_abst == 1.0 and (f1 or 0.0) == 0.0)
    official_score = cat5 if category == 5 else f1
    return {
        "official_score": float(official_score),
        "rF1_cat14": None if category == 5 else float(f1),
        "rEM_cat14_project": None if category == 5 else float(em),
        "official_cat5_score": None if category != 5 else float(cat5),
        "broad_abstention": broad_abst,
        "WrongAbstention_cat14_project": wrong_abst,
    }


def mean_or_none(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else None


def fmt_value(v: float | None) -> str | float:
    return "" if v is None else round(float(v), 6)


def summarize(scope: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    cats = Counter(int(r["category"]) for r in rows)
    answerable = [r for r in rows if int(r["category"]) in (1, 2, 3, 4)]
    cat5 = [r for r in rows if int(r["category"]) == 5]
    return {
        "scope": scope,
        "n": len(rows),
        "category_counts": json.dumps(dict(sorted(cats.items())), ensure_ascii=False),
        "official_score": fmt_value(mean_or_none(r["official_score"] for r in rows)),
        "rF1_cat14": fmt_value(mean_or_none(r["rF1_cat14"] for r in answerable)),
        "rEM_cat14_project": fmt_value(mean_or_none(r["rEM_cat14_project"] for r in answerable)),
        "WrongAbstention_cat14_project": fmt_value(mean_or_none(r["WrongAbstention_cat14_project"] for r in answerable)),
        "cat5_official_abstention": fmt_value(mean_or_none(r["official_cat5_score"] for r in cat5)),
        "cat5_broad_abstention": fmt_value(mean_or_none(r["broad_abstention"] for r in cat5)),
        "cat5_leakage_official": fmt_value(
            None if not cat5 else 1.0 - float(mean_or_none(r["official_cat5_score"] for r in cat5) or 0.0)
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline official LoCoMo category-aware QA evaluator")
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--default-method", default="")
    ap.add_argument("--qid-column", default="")
    ap.add_argument("--prediction-column", default="")
    ap.add_argument("--method-column", default="")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--expected-full5", type=int, default=1986)
    ap.add_argument("--expected-cat14", type=int, default=1540)
    ap.add_argument("--expected-cat5", type=int, default=446)
    args = ap.parse_args()

    q_rows = read_records(args.questions)
    p_rows = read_records(args.predictions)
    if not q_rows:
        raise ValueError("Questions file is empty")
    if not p_rows:
        raise ValueError("Predictions file is empty")

    qcols = q_rows[0].keys()
    q_qid = pick_column(qcols, QID_CANDIDATES, args.qid_column or None, "question ID")
    q_cat = pick_column(qcols, CATEGORY_CANDIDATES, None, "question category")
    q_gold = pick_column(qcols, GOLD_CANDIDATES, None, "gold answer")

    questions: dict[str, dict[str, Any]] = {}
    q_duplicates: list[str] = []
    for row in q_rows:
        qid = str(row.get(q_qid, "")).strip()
        if not qid:
            raise ValueError("Blank question ID encountered")
        if qid in questions:
            q_duplicates.append(qid)
        questions[qid] = row
    if q_duplicates:
        raise ValueError(f"Duplicate question IDs: {q_duplicates[:10]}")

    expected_q_counts = Counter(int(str(r[q_cat]).strip()) for r in q_rows)
    if len(questions) == args.expected_full5:
        required = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
        if dict(expected_q_counts) != required:
            raise ValueError(f"Unexpected canonical category counts: {dict(expected_q_counts)} != {required}")

    pcols = p_rows[0].keys()
    p_qid = pick_column(pcols, QID_CANDIDATES, args.qid_column or None, "prediction ID")
    p_pred = pick_column(pcols, PRED_CANDIDATES, args.prediction_column or None, "prediction")
    p_method = pick_column(pcols, METHOD_CANDIDATES, args.method_column or None, "method", required=False)
    if not p_method and not args.default_method:
        raise ValueError("Prediction file has no method column; provide --default-method")

    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    exact_duplicate_count = 0
    conflict_examples: list[dict[str, str]] = []
    for row in p_rows:
        qid = str(row.get(p_qid, "")).strip()
        pred = str(row.get(p_pred, "") or "").strip()
        method = str(row.get(p_method, "") if p_method else args.default_method).strip()
        if not qid or not method:
            raise ValueError("Blank qid or method in prediction file")
        key = (method, qid)
        if key in dedup:
            old_pred = str(dedup[key].get(p_pred, "") or "").strip()
            if old_pred == pred:
                exact_duplicate_count += 1
                continue
            conflict_examples.append({"method": method, "qa_id": qid, "old": old_pred, "new": pred})
            continue
        dedup[key] = row
    if conflict_examples:
        raise ValueError(f"Conflicting duplicate predictions found: {conflict_examples[:3]}")

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_qids: dict[str, list[str]] = defaultdict(list)
    unknown_qids: dict[str, list[str]] = defaultdict(list)

    for (method, qid), prow in dedup.items():
        if qid not in questions:
            unknown_qids[method].append(qid)
            continue
        qrow = questions[qid]
        category = int(str(qrow[q_cat]).strip())
        gold = qrow[q_gold]
        pred = prow[p_pred]
        scores = score_one(category, pred, gold)
        by_method[method].append({
            "method": method,
            "qa_id": qid,
            "category": category,
            "question": qrow.get("question", ""),
            "gold_answer": gold,
            "prediction": pred,
            **scores,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_per_query: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    method_audits: dict[str, Any] = {}

    for method, rows in sorted(by_method.items()):
        rows.sort(key=lambda r: r["qa_id"])
        seen = {r["qa_id"] for r in rows}
        missing = sorted(set(questions) - seen)
        missing_qids[method] = missing
        cat_counts = Counter(int(r["category"]) for r in rows)

        if not args.allow_partial and len(rows) not in {args.expected_full5, args.expected_cat14, args.expected_cat5}:
            raise ValueError(
                f"{method}: coverage n={len(rows)} is not one of expected "
                f"{args.expected_full5}/{args.expected_cat14}/{args.expected_cat5}; use --allow-partial only for audit"
            )
        canonical_cat14_counts = {1: 282, 2: 321, 3: 96, 4: 841}
        if len(rows) == args.expected_cat14 and dict(cat_counts) != canonical_cat14_counts:
            raise ValueError(
                f"{method}: n=1540 but category counts are {dict(cat_counts)} "
                f"instead of {canonical_cat14_counts}"
            )
        if len(rows) == args.expected_cat5 and dict(cat_counts) != {5: 446}:
            raise ValueError(f"{method}: n=446 but category counts are {dict(cat_counts)} instead of {{5: 446}}")
        if len(rows) == args.expected_full5:
            canonical_full5_counts = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
            if dict(cat_counts) != canonical_full5_counts:
                raise ValueError(
                    f"{method}: n=1986 but category counts are {dict(cat_counts)} "
                    f"instead of {canonical_full5_counts}"
                )

        all_per_query.extend(rows)
        for cat in (1, 2, 3, 4, 5):
            crows = [r for r in rows if int(r["category"]) == cat]
            if crows:
                s = summarize(f"cat{cat}", crows)
                category_rows.append({"method": method, "category": cat, **s})

        cat14 = [r for r in rows if int(r["category"]) in (1, 2, 3, 4)]
        cat5 = [r for r in rows if int(r["category"]) == 5]
        if cat14:
            overall_rows.append({"method": method, **summarize("canonical_cat1_4", cat14)})
        if cat5:
            overall_rows.append({"method": method, **summarize("cat5", cat5)})
        if len(rows) == args.expected_full5 and set(cat_counts) == {1, 2, 3, 4, 5}:
            overall_rows.append({"method": method, **summarize("full_cat1_5", rows)})

        method_audits[method] = {
            "n_predictions": len(rows),
            "category_counts": dict(sorted(cat_counts.items())),
            "missing_question_count": len(missing),
            "missing_question_examples": missing[:20],
            "unknown_prediction_id_count": len(unknown_qids.get(method, [])),
            "unknown_prediction_id_examples": unknown_qids.get(method, [])[:20],
        }

    per_query_fields = [
        "method", "qa_id", "category", "question", "gold_answer", "prediction",
        "official_score", "rF1_cat14", "rEM_cat14_project", "official_cat5_score",
        "broad_abstention", "WrongAbstention_cat14_project",
    ]
    summary_fields = [
        "method", "scope", "n", "category_counts", "official_score", "rF1_cat14",
        "rEM_cat14_project", "WrongAbstention_cat14_project", "cat5_official_abstention",
        "cat5_broad_abstention", "cat5_leakage_official",
    ]
    category_fields = ["method", "category"] + [x for x in summary_fields if x not in {"method"}]

    write_csv(args.output_dir / "per_query_scores.csv", all_per_query, per_query_fields)
    write_csv(args.output_dir / "scores_overall.csv", overall_rows, summary_fields)
    write_csv(args.output_dir / "scores_by_category.csv", category_rows, category_fields)

    audit = {
        "api_called": False,
        "evaluator": "official_locomo_category_score_plus_labeled_project_diagnostics",
        "questions_path": str(args.questions.resolve()),
        "questions_sha256": sha256_file(args.questions),
        "predictions_path": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "question_count": len(questions),
        "question_category_counts": dict(sorted(expected_q_counts.items())),
        "raw_prediction_rows": len(p_rows),
        "deduplicated_prediction_rows": len(dedup),
        "exact_duplicate_rows_ignored": exact_duplicate_count,
        "methods": method_audits,
        "definitions": {
            "official_score": "Cat1 multi-answer F1; Cat2-4 stemmed token F1; Cat5 strict binary refusal; micro-average per query",
            "rF1_cat14": "Official LoCoMo answer F1 on Cat1-4 only",
            "rEM_cat14_project": "Project diagnostic, not emitted by official LoCoMo eval_question_answering",
            "WrongAbstention_cat14_project": "Broad abstention on Cat1-4 with official F1 == 0",
            "cat5_official_abstention": "Strict official phrases only: no information available / not mentioned",
        },
    }
    (args.output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Wrote: {args.output_dir / 'per_query_scores.csv'}")
    print(f"Wrote: {args.output_dir / 'scores_by_category.csv'}")
    print(f"Wrote: {args.output_dir / 'scores_overall.csv'}")
    print(f"Wrote: {args.output_dir / 'evaluation_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
