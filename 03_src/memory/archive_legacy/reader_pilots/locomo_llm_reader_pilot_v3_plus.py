import csv
import json
import os
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

BASE_V3_RESULTS = "results/llm_reader_pilot_v3_results.csv"
PLUS_RESULTS = "results/llm_reader_pilot_v3_plus_results.csv"
PLUS_SUMMARY = "results/llm_reader_pilot_v3_plus_summary.csv"

SAMPLE_CSV = "results/llm_reader_pilot_qa.csv"
QA_CSV = "results/locomo_qa_records.csv"
MEMORY_CSV = "results/locomo_memory_records.csv"
EVIDENCE_CSV = "results/locomo_evidence_map.csv"
LOCOMO_JSON = "external_data/locomo10.json"
KG_AWARE_CSV = "results/locomo_kg_aware_fusion_best_results.csv"

METHODS_ORDER = [
    "BM25",
    "Dense-ONNX-MiniLM",
    "BM25-Dense-RRF",
    "KG-boost",
    "KG-aware-RRF",
]

NUMBER_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

YES_WORDS = {"yes", "yeah", "yep", "correct", "true"}
NO_WORDS = {"no", "nope", "false"}
CA_VARIANTS = {
    "cannot answer",
    "cannot determine",
    "cannot be determined",
    "not enough information",
    "insufficient information",
    "unknown",
}

RESULT_FIELDS = [
    "qa_id",
    "category",
    "method",
    "question",
    "gold_answer",
    "top10_evidence_ids",
    "resolved_memory_ids",
    "evidence_texts",
    "predicted_answer",
    "strict_em",
    "strict_f1",
    "relaxed_em",
    "relaxed_f1",
    "retrieval_hit10",
    "answer_string_in_evidence",
    "prompt",
]

SUMMARY_FIELDS = [
    "method",
    "category",
    "n",
    "strict_em",
    "strict_f1",
    "relaxed_em",
    "relaxed_f1",
    "retrieval_hit10",
    "answer_string_in_evidence_rate",
    "cannot_answer_rate",
    "relaxed_f1_when_hit10",
    "relaxed_f1_when_miss10",
]

def load_csv(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def append_csv(path, row, fieldnames):
    p = Path(path)
    mode = "a" if p.exists() else "w"
    with p.open(mode, encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        w.writerow(row)

def make_ascii_safe(text):
    if text is None:
        return ""
    text = str(text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\ufeff": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")

def normalize_strict(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_relaxed(text):
    text = normalize_strict(text)
    tokens = text.split()
    out = []
    for tok in tokens:
        if tok in NUMBER_MAP:
            out.append(NUMBER_MAP[tok])
        elif tok in YES_WORDS:
            out.append("yes")
        elif tok in NO_WORDS:
            out.append("no")
        else:
            out.append(tok)
    text = " ".join(out)
    if text in CA_VARIANTS:
        return "cannot answer"
    return text

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
    if raw.startswith("[") and raw.endswith("]"):
        try:
            vals = json.loads(raw)
            return [str(x).strip() for x in vals if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in re.split(r"[;,\|]", raw) if x.strip()]

def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def load_memory_index(path):
    mem_by_id = {}
    mem_by_dia = {}
    for row in load_csv(path):
        mid = str(row.get("memory_id", "")).strip()
        sample_id = str(row.get("sample_id", "")).strip()
        dia_id = str(row.get("dia_id", "")).strip()
        if mid:
            mem_by_id[mid] = row
        if sample_id and dia_id:
            mem_by_dia[(sample_id, dia_id)] = row
    return mem_by_id, mem_by_dia

def load_qa_answer_map(path):
    out = {}
    for row in load_csv(path):
        qid = str(row.get("qa_id", "")).strip()
        ans = str(row.get("answer", "")).strip()
        adv = str(row.get("adversarial_answer", "")).strip()
        if qid:
            out[qid] = ans if ans else adv
    return out

def load_gold_map(path):
    out = defaultdict(set)
    for row in load_csv(path):
        qid = str(row.get("qa_id", "")).strip()
        eid = str(row.get("evidence_id", "")).strip()
        mid = str(row.get("memory_id", "")).strip()
        if qid and eid:
            out[qid].add(eid)
        if qid and mid:
            out[qid].add(mid)
    return out

def load_locomo_context(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    obs_lookup = {}
    summary_lookup = {}
    for sample in data:
        sample_id = str(sample.get("sample_id", "")).strip()
        observation = sample.get("observation", {})
        session_summary = sample.get("session_summary", {})
        if isinstance(observation, dict):
            for key, val in observation.items():
                m = re.search(r"session_(\d+)_observation", str(key))
                if m:
                    obs_lookup[(sample_id, f"session_{m.group(1)}")] = val
        if isinstance(session_summary, dict):
            for key, val in session_summary.items():
                m = re.search(r"session_(\d+)_summary", str(key))
                if m:
                    summary_lookup[(sample_id, f"session_{m.group(1)}")] = str(val)
    return obs_lookup, summary_lookup

def resolve_memory_id_with_sample(raw_id, sample_id, mem_by_id, mem_by_dia):
    raw_id = str(raw_id).strip()
    sample_id = str(sample_id).strip()
    if not raw_id:
        return ""
    if raw_id in mem_by_id:
        return raw_id
    m = re.match(r"D(\d+):(\d+)$", raw_id)
    if m:
        candidate = f"{sample_id}_session_{m.group(1)}_{raw_id}"
        if candidate in mem_by_id:
            return candidate
        row = mem_by_dia.get((sample_id, raw_id))
        if row:
            return row.get("memory_id", "")
    if raw_id.startswith("session_"):
        candidate = f"{sample_id}_{raw_id}"
        if candidate in mem_by_id:
            return candidate
    candidate = f"{sample_id}_{raw_id}"
    if candidate in mem_by_id:
        return candidate
    return ""

def resolve_memory_id(raw_id, qa_id, mem_by_id, mem_by_dia):
    sample_id = qa_id.split("_qa_")[0] if "_qa_" in qa_id else qa_id
    return resolve_memory_id_with_sample(raw_id, sample_id, mem_by_id, mem_by_dia)

def get_neighbor_turns(memory_id, mem_by_id, mem_by_dia):
    row = mem_by_id.get(memory_id)
    if not row:
        return None, None, None
    sample_id = str(row.get("sample_id", "")).strip()
    dia_id = str(row.get("dia_id", "")).strip()
    m = re.match(r"D(\d+):(\d+)", dia_id)
    if not m:
        return None, row, None
    d = int(m.group(1))
    t = int(m.group(2))
    prev_row = mem_by_dia.get((sample_id, f"D{d}:{t - 1}")) if t > 1 else None
    next_row = mem_by_dia.get((sample_id, f"D{d}:{t + 1}"))
    return prev_row, row, next_row

def flatten_observations(obs_obj, max_items=8):
    lines = []
    if isinstance(obs_obj, dict):
        for speaker, facts in obs_obj.items():
            if isinstance(facts, list):
                for fact in facts:
                    if isinstance(fact, list) and fact:
                        text = str(fact[0]).strip()
                    else:
                        text = str(fact).strip()
                    if text:
                        lines.append(f"- {speaker}: {text}")
                    if len(lines) >= max_items:
                        return lines
    elif isinstance(obs_obj, list):
        for fact in obs_obj:
            text = str(fact[0] if isinstance(fact, list) and fact else fact).strip()
            if text:
                lines.append(f"- {text}")
            if len(lines) >= max_items:
                return lines
    return lines

def truncate_text(text, limit):
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."

def get_session_context(cur_row, obs_lookup, summary_lookup):
    sample_id = str(cur_row.get("sample_id", "")).strip()
    session_id = str(cur_row.get("session_id", "")).strip()
    obs = obs_lookup.get((sample_id, session_id), {})
    summary = summary_lookup.get((sample_id, session_id), "")
    return flatten_observations(obs, 8), truncate_text(summary, 700)

def format_turn(prefix, row):
    if not row:
        return ""
    dia = str(row.get("dia_id", "")).strip()
    speaker = str(row.get("speaker", "")).strip()
    text = truncate_text(str(row.get("text", "")).strip(), 500)
    return f"{prefix}: {dia} | {speaker} | {text}"

def format_evidence_block(idx, prev_row, cur_row, next_row, obs_lines, summary):
    lines = []
    lines.append(f"[{idx}]")
    lines.append(f"retrieved_memory_id: {cur_row.get('memory_id', '')}")
    lines.append(f"sample_id: {cur_row.get('sample_id', '')}")
    lines.append(f"session_id: {cur_row.get('session_id', '')}")
    lines.append(f"dia_id: {cur_row.get('dia_id', '')}")
    lines.append(f"timestamp: {cur_row.get('timestamp', '')}")
    lines.append("")
    lines.append("Nearby conversation:")
    if prev_row:
        lines.append(format_turn("Previous", prev_row))
    lines.append(format_turn("Retrieved", cur_row))
    if next_row:
        lines.append(format_turn("Next", next_row))
    if obs_lines:
        lines.append("")
        lines.append("Session observations:")
        lines.extend(obs_lines)
    if summary:
        lines.append("")
        lines.append("Session summary:")
        lines.append(summary)
    return "\n".join(lines)

def build_prompt(question, evidence_blocks):
    return "\n".join([
        "You are answering a memory-based question using retrieved memory evidence.",
        "",
        "Each evidence item includes timestamp metadata, nearby conversation context, and optional session-level memory facts.",
        "Use timestamps to resolve relative time expressions such as yesterday, last week, last Saturday, next month, and similar phrases.",
        "",
        "Use only the provided evidence and metadata.",
        "If the answer can be inferred from any retrieved evidence, answer it.",
        "Use Cannot answer only when none of the evidence is relevant.",
        "",
        "Return only the shortest correct answer. Do not explain.",
        "",
        "Evidence:",
        evidence_blocks,
        "",
        f"Question: {question}",
        "Answer:",
    ])

def answer_in_evidence(gold_answer, evidence_text):
    g = normalize_strict(gold_answer)
    e = normalize_strict(evidence_text)
    return 1 if g and g in e else 0
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
            print(f"DeepSeek call failed attempt {attempt}/{len(sleep_times)}: {e}")
            if attempt < len(sleep_times):
                time.sleep(wait_s)
    raise RuntimeError(str(last_error))



def load_kg_aware_rankings():
    rows = load_csv(KG_AWARE_CSV)
    out = {}
    for row in rows:
        qid = str(row.get("qa_id", "")).strip()
        if not qid:
            continue
        raw = row.get("retrieved_memory_ids", "")
        if not raw:
            raw = row.get("kg_aware_top10_memory_ids", "")
        out[qid] = unique_keep_order(split_ids(raw))[:10]
    return out

def build_evidence_for_ids(qa_id, raw_ids, mem_by_id, mem_by_dia, obs_lookup, summary_lookup):
    blocks = []
    texts = []
    resolved = []
    seen = set()
    for raw in raw_ids:
        mid = resolve_memory_id(raw, qa_id, mem_by_id, mem_by_dia)
        if not mid or mid in seen:
            continue
        seen.add(mid)
        prev_row, cur_row, next_row = get_neighbor_turns(mid, mem_by_id, mem_by_dia)
        if cur_row is None:
            continue
        obs_lines, summary = get_session_context(cur_row, obs_lookup, summary_lookup)
        blocks.append(format_evidence_block(len(blocks) + 1, prev_row, cur_row, next_row, obs_lines, summary))
        resolved.append(mid)
        text_parts = []
        if prev_row:
            text_parts.append(str(prev_row.get("text", "")))
        text_parts.append(str(cur_row.get("text", "")))
        if next_row:
            text_parts.append(str(next_row.get("text", "")))
        text_parts.extend(obs_lines)
        if summary:
            text_parts.append(summary)
        texts.append("\n".join(text_parts))
        if len(blocks) >= 10:
            break
    return resolved, texts, "\n\n".join(blocks)

def is_retrieval_hit(qa_id, raw_ids, resolved_ids, gold_map):
    gold = gold_map.get(qa_id, set())
    if not gold:
        return 0
    return 1 if (set(raw_ids) | set(resolved_ids)) & gold else 0

def ensure_plus_base():
    plus = Path(PLUS_RESULTS)
    if plus.exists():
        rows = load_csv(PLUS_RESULTS)
        bad = [r for r in rows if str(r.get("predicted_answer", "")).startswith("ERROR:")]
        if bad:
            raise RuntimeError(f"{PLUS_RESULTS} contains {len(bad)} ERROR rows")
        return rows
    base = load_csv(BASE_V3_RESULTS)
    bad = [r for r in base if str(r.get("predicted_answer", "")).startswith("ERROR:")]
    if bad:
        raise RuntimeError(f"{BASE_V3_RESULTS} contains {len(bad)} ERROR rows")
    write_csv(PLUS_RESULTS, base, RESULT_FIELDS)
    return load_csv(PLUS_RESULTS)

def summarize_group(method, category, rows):
    n = len(rows)
    if n == 0:
        return {
            "method": method,
            "category": category,
            "n": 0,
            "strict_em": "0.0000",
            "strict_f1": "0.0000",
            "relaxed_em": "0.0000",
            "relaxed_f1": "0.0000",
            "retrieval_hit10": "0.0000",
            "answer_string_in_evidence_rate": "0.0000",
            "cannot_answer_rate": "0.0000",
            "relaxed_f1_when_hit10": "0.0000",
            "relaxed_f1_when_miss10": "0.0000",
        }
    s_em = sum(int(r.get("strict_em", 0)) for r in rows) / n
    s_f1 = sum(float(r.get("strict_f1", 0)) for r in rows) / n
    r_em = sum(int(r.get("relaxed_em", 0)) for r in rows) / n
    r_f1 = sum(float(r.get("relaxed_f1", 0)) for r in rows) / n
    hit = sum(int(r.get("retrieval_hit10", 0)) for r in rows) / n
    ans_ev = sum(int(r.get("answer_string_in_evidence", 0)) for r in rows) / n
    ca = sum(1 for r in rows if normalize_relaxed(r.get("predicted_answer", "")) == "cannot answer") / n
    f1_hit = [float(r.get("relaxed_f1", 0)) for r in rows if int(r.get("retrieval_hit10", 0)) == 1]
    f1_miss = [float(r.get("relaxed_f1", 0)) for r in rows if int(r.get("retrieval_hit10", 0)) == 0]
    return {
        "method": method,
        "category": category,
        "n": n,
        "strict_em": f"{s_em:.4f}",
        "strict_f1": f"{s_f1:.4f}",
        "relaxed_em": f"{r_em:.4f}",
        "relaxed_f1": f"{r_f1:.4f}",
        "retrieval_hit10": f"{hit:.4f}",
        "answer_string_in_evidence_rate": f"{ans_ev:.4f}",
        "cannot_answer_rate": f"{ca:.4f}",
        "relaxed_f1_when_hit10": f"{sum(f1_hit) / len(f1_hit) if f1_hit else 0:.4f}",
        "relaxed_f1_when_miss10": f"{sum(f1_miss) / len(f1_miss) if f1_miss else 0:.4f}",
    }

def write_summary(rows):
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    summary = []
    print("")
    print("=== LLM Reader v3 Plus ===")
    print(f"Total rows: {len(rows)}")
    print("")
    print(f"{'Method':20s} {'rEM':>7s} {'rF1':>7s} {'hit10':>7s} {'ansInEv':>7s} {'CA%':>6s} {'F1_hit':>7s} {'F1_miss':>7s}")
    print("-" * 75)

    for method in METHODS_ORDER:
        item = summarize_group(method, "ALL", by_method.get(method, []))
        summary.append(item)
        print(
            f"{method:20s} "
            f"{float(item['relaxed_em']):7.4f} "
            f"{float(item['relaxed_f1']):7.4f} "
            f"{float(item['retrieval_hit10']):7.4f} "
            f"{float(item['answer_string_in_evidence_rate']):7.4f} "
            f"{float(item['cannot_answer_rate']):6.4f} "
            f"{float(item['relaxed_f1_when_hit10']):7.4f} "
            f"{float(item['relaxed_f1_when_miss10']):7.4f}"
        )

    for method in METHODS_ORDER:
        by_cat = defaultdict(list)
        for row in by_method.get(method, []):
            by_cat[str(row.get("category", ""))].append(row)
        for cat in sorted(by_cat.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
            summary.append(summarize_group(method, f"cat_{cat}", by_cat[cat]))

    write_csv(PLUS_SUMMARY, summary, SUMMARY_FIELDS)

    print("")
    print("Category-level KG-aware-RRF:")
    kg_aware = by_method.get("KG-aware-RRF", [])
    by_cat = defaultdict(list)
    for row in kg_aware:
        by_cat[str(row.get("category", ""))].append(row)
    for cat in sorted(by_cat.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
        item = summarize_group("KG-aware-RRF", f"cat_{cat}", by_cat[cat])
        ca_num = sum(1 for r in by_cat[cat] if normalize_relaxed(r.get("predicted_answer", "")) == "cannot answer")
        print(f"  cat {cat}: EM={float(item['relaxed_em']):.4f} F1={float(item['relaxed_f1']):.4f} CA={ca_num}/{len(by_cat[cat])}")

def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY environment variable")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    sampled = load_csv(SAMPLE_CSV)
    answer_map = load_qa_answer_map(QA_CSV)
    mem_by_id, mem_by_dia = load_memory_index(MEMORY_CSV)
    obs_lookup, summary_lookup = load_locomo_context(LOCOMO_JSON)
    gold_map = load_gold_map(EVIDENCE_CSV)
    kg_aware_rankings = load_kg_aware_rankings()

    existing = ensure_plus_base()
    done = set()
    for row in existing:
        pred = str(row.get("predicted_answer", ""))
        if pred and not pred.startswith("ERROR:"):
            done.add((row.get("qa_id", ""), row.get("method", "")))

    print(f"Loaded {len(sampled)} sampled QA pairs")
    print(f"Loaded {len(obs_lookup)} observation entries, {len(summary_lookup)} session summaries")
    print(f"Existing plus rows: {len(existing)}")
    print("Running missing KG-aware-RRF rows only")

    calls = 0

    for qa in sampled:
        qa_id = str(qa.get("qa_id", "")).strip()
        if (qa_id, "KG-aware-RRF") in done:
            continue

        category = str(qa.get("category", "")).strip()
        question = str(qa.get("question", "")).strip()
        gold_answer = str(qa.get("answer", "")).strip() or answer_map.get(qa_id, "")

        raw_ids = kg_aware_rankings.get(qa_id, [])[:10]
        resolved_ids, evidence_texts, evidence_blocks = build_evidence_for_ids(
            qa_id, raw_ids, mem_by_id, mem_by_dia, obs_lookup, summary_lookup
        )

        prompt = build_prompt(question, evidence_blocks)
        prompt = make_ascii_safe(prompt)

        predicted = call_deepseek(client, prompt).strip()

        strict_em = compute_em(predicted, gold_answer, normalize_strict)
        strict_f1 = compute_token_f1(predicted, gold_answer, normalize_strict)
        relaxed_em = compute_em(predicted, gold_answer, normalize_relaxed)
        relaxed_f1 = compute_token_f1(predicted, gold_answer, normalize_relaxed)

        evidence_blob = "\n".join(evidence_texts)
        retrieval_hit10 = is_retrieval_hit(qa_id, raw_ids, resolved_ids, gold_map)
        ans_in_ev = answer_in_evidence(gold_answer, evidence_blob)

        row = {
            "qa_id": qa_id,
            "category": category,
            "method": "KG-aware-RRF",
            "question": question,
            "gold_answer": gold_answer,
            "top10_evidence_ids": ";".join(raw_ids),
            "resolved_memory_ids": ";".join(resolved_ids),
            "evidence_texts": json.dumps(evidence_texts, ensure_ascii=False),
            "predicted_answer": predicted,
            "strict_em": strict_em,
            "strict_f1": f"{strict_f1:.4f}",
            "relaxed_em": relaxed_em,
            "relaxed_f1": f"{relaxed_f1:.4f}",
            "retrieval_hit10": retrieval_hit10,
            "answer_string_in_evidence": ans_in_ev,
            "prompt": json.dumps(prompt, ensure_ascii=False),
        }

        append_csv(PLUS_RESULTS, row, RESULT_FIELDS)
        done.add((qa_id, "KG-aware-RRF"))
        calls += 1

        if calls % 20 == 0:
            print(f"  {calls} KG-aware-RRF calls done...")

        time.sleep(0.3 + random.random() * 0.2)

    final_rows = load_csv(PLUS_RESULTS)
    error_rows = [r for r in final_rows if str(r.get("predicted_answer", "")).startswith("ERROR:")]
    if error_rows:
        raise RuntimeError(f"Found {len(error_rows)} ERROR rows in {PLUS_RESULTS}")

    write_summary(final_rows)

    print("")
    print(f"New KG-aware-RRF calls: {calls}")
    print(f"Output: {PLUS_RESULTS}")
    print(f"Summary: {PLUS_SUMMARY}")

if __name__ == "__main__":
    main()