import csv
import json
import os
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

METHODS = {
    "BM25": ("results/locomo_bm25_results.csv", "retrieved_memory_ids"),
    "Dense-ONNX-MiniLM": ("results/locomo_dense_onnx_results.csv", "retrieved_memory_ids"),
    "BM25-Dense-RRF": ("results/locomo_fusion_bm25_dense_results.csv", "fusion_top10_memory_ids"),
    "KG-boost": ("results/locomo_cassandra_kg_results.csv", "retrieved_memory_ids"),
}

NUMBER_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

YES_WORDS = {"yes", "yeah", "yep", "correct", "true"}
NO_WORDS = {"no", "nope", "false"}
CA_VARIANTS = {"cannot answer", "cannot determine", "not enough information", "insufficient information", "unknown"}

def load_csv(file_path):
    with Path(file_path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(file_path, rows, fieldnames):
    with Path(file_path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def append_csv(file_path, row, fieldnames):
    path = Path(file_path)
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8-sig", newline="") as f:
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
    text = text.encode("ascii", "ignore").decode("ascii")
    return text

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
    parts = re.split(r"[;,\|]", raw)
    return [p.strip() for p in parts if p.strip()]

def load_memory_index(file_path):
    mem_by_id = {}
    mem_by_dia = {}
    ordered_by_session = defaultdict(list)
    for row in load_csv(file_path):
        mid = str(row.get("memory_id", "")).strip()
        if not mid:
            continue
        mem_by_id[mid] = row
        sample_id = str(row.get("sample_id", "")).strip()
        dia_id = str(row.get("dia_id", "")).strip()
        session_id = str(row.get("session_id", "")).strip()
        if sample_id and dia_id:
            mem_by_dia[(sample_id, dia_id)] = row
        if sample_id and session_id:
            ordered_by_session[(sample_id, session_id)].append(row)
    for key in ordered_by_session:
        ordered_by_session[key].sort(key=lambda r: dia_sort_key(str(r.get("dia_id", ""))))
    return mem_by_id, mem_by_dia, ordered_by_session

def dia_sort_key(dia_id):
    m = re.match(r"D(\d+):(\d+)", str(dia_id))
    if not m:
        return (999999, 999999)
    return (int(m.group(1)), int(m.group(2)))

def load_qa_answer_map(file_path):
    answer_map = {}
    for row in load_csv(file_path):
        qid = str(row.get("qa_id", "")).strip()
        ans = str(row.get("answer", "")).strip()
        adv = str(row.get("adversarial_answer", "")).strip()
        answer_map[qid] = ans if ans else adv
    return answer_map

def load_gold_map(file_path):
    gold = defaultdict(set)
    for row in load_csv(file_path):
        qid = str(row.get("qa_id", "")).strip()
        evidence_id = str(row.get("evidence_id", "")).strip()
        memory_id = str(row.get("memory_id", "")).strip()
        if evidence_id:
            gold[qid].add(evidence_id)
        if memory_id:
            gold[qid].add(memory_id)
    return gold

def load_locomo_context(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
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

def resolve_memory_id(raw_id, qa_id, mem_by_id, mem_by_dia):
    raw_id = str(raw_id).strip()
    if not raw_id:
        return ""
    if raw_id in mem_by_id:
        return raw_id
    graph_id = qa_id.split("_qa_")[0]
    m = re.match(r"D(\d+):(\d+)", raw_id)
    if m:
        candidate = f"{graph_id}_session_{m.group(1)}_{raw_id}"
        if candidate in mem_by_id:
            return candidate
        row = mem_by_dia.get((graph_id, raw_id))
        if row:
            return row.get("memory_id", "")
    if raw_id.startswith("session_"):
        candidate = f"{graph_id}_{raw_id}"
        if candidate in mem_by_id:
            return candidate
    candidate = f"{graph_id}_{raw_id}"
    if candidate in mem_by_id:
        return candidate
    return raw_id if raw_id in mem_by_id else ""

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
    obs_obj = obs_lookup.get((sample_id, session_id), {})
    summary = summary_lookup.get((sample_id, session_id), "")
    obs_lines = flatten_observations(obs_obj, max_items=8)
    summary = truncate_text(summary, 700)
    return obs_lines, summary

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
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=128,
            )
            ans = resp.choices[0].message.content.strip()
            if ans.startswith("ERROR:"):
                raise RuntimeError(ans)
            return ans
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(str(e))

def load_method_retrieval():
    out = defaultdict(dict)
    for method, (file_path, col_name) in METHODS.items():
        rows = load_csv(file_path)
        for row in rows:
            qid = str(row.get("qa_id", "")).strip()
            if not qid:
                continue
            raw = row.get(col_name, "")
            ids = split_ids(raw)
            out[qid][method] = ids
    return out

def build_evidence_for_method(qa_id, raw_ids, mem_by_id, mem_by_dia, obs_lookup, summary_lookup):
    blocks = []
    texts = []
    resolved_ids = []
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
        block = format_evidence_block(len(blocks) + 1, prev_row, cur_row, next_row, obs_lines, summary)
        blocks.append(block)
        resolved_ids.append(mid)
        turn_texts = []
        if prev_row:
            turn_texts.append(str(prev_row.get("text", "")))
        turn_texts.append(str(cur_row.get("text", "")))
        if next_row:
            turn_texts.append(str(next_row.get("text", "")))
        if obs_lines:
            turn_texts.extend(obs_lines)
        if summary:
            turn_texts.append(summary)
        texts.append("\n".join(turn_texts))
        if len(blocks) >= 10:
            break
    return resolved_ids, texts, "\n\n".join(blocks)

def is_retrieval_hit(qa_id, raw_ids, resolved_ids, gold_map):
    gold = gold_map.get(qa_id, set())
    if not gold:
        return 0
    candidates = set(raw_ids) | set(resolved_ids)
    return 1 if candidates & gold else 0

def summarize(rows, gold_map):
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    summary_rows = []
    print(f"{'Method':20s} {'rEM':>7s} {'rF1':>7s} {'hit10':>7s} {'ansInEv':>7s} {'CA%':>6s} {'F1_hit':>7s} {'F1_miss':>7s}")
    print("-" * 75)
    for method in METHODS:
        group = by_method.get(method, [])
        row = summarize_group(method, "ALL", group, gold_map)
        summary_rows.append(row)
        print(f"{method:20s} {float(row['relaxed_em']):7.4f} {float(row['relaxed_f1']):7.4f} {float(row['retrieval_hit10']):7.4f} {float(row['answer_string_in_evidence_rate']):7.4f} {float(row['cannot_answer_rate']):6.4f} {float(row['relaxed_f1_when_hit10']):7.4f} {float(row['relaxed_f1_when_miss10']):7.4f}")
    for method in METHODS:
        group = by_method.get(method, [])
        by_cat = defaultdict(list)
        for row in group:
            by_cat[str(row.get("category", ""))].append(row)
        for cat in sorted(by_cat.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            summary_rows.append(summarize_group(method, f"cat_{cat}", by_cat[cat], gold_map))
    return summary_rows, by_method

def summarize_group(method, category, group, gold_map):
    n = len(group)
    if n == 0:
        return {
            "method": method, "category": category, "n": 0,
            "strict_em": "0.0000", "strict_f1": "0.0000",
            "relaxed_em": "0.0000", "relaxed_f1": "0.0000",
            "retrieval_hit10": "0.0000", "answer_string_in_evidence_rate": "0.0000",
            "cannot_answer_rate": "0.0000", "relaxed_f1_when_hit10": "0.0000",
            "relaxed_f1_when_miss10": "0.0000",
        }
    s_em = sum(int(r["strict_em"]) for r in group) / n
    s_f1 = sum(float(r["strict_f1"]) for r in group) / n
    r_em = sum(int(r["relaxed_em"]) for r in group) / n
    r_f1 = sum(float(r["relaxed_f1"]) for r in group) / n
    hit_vals = [int(r.get("retrieval_hit10", 0)) for r in group]
    hit10 = sum(hit_vals) / n
    ans_ev = sum(int(r.get("answer_string_in_evidence", 0)) for r in group) / n
    ca = sum(1 for r in group if normalize_relaxed(r.get("predicted_answer", "")) == "cannot answer") / n
    f1_hit = [float(r["relaxed_f1"]) for r in group if int(r.get("retrieval_hit10", 0)) == 1]
    f1_miss = [float(r["relaxed_f1"]) for r in group if int(r.get("retrieval_hit10", 0)) == 0]
    return {
        "method": method,
        "category": category,
        "n": n,
        "strict_em": f"{s_em:.4f}",
        "strict_f1": f"{s_f1:.4f}",
        "relaxed_em": f"{r_em:.4f}",
        "relaxed_f1": f"{r_f1:.4f}",
        "retrieval_hit10": f"{hit10:.4f}",
        "answer_string_in_evidence_rate": f"{ans_ev:.4f}",
        "cannot_answer_rate": f"{ca:.4f}",
        "relaxed_f1_when_hit10": f"{sum(f1_hit)/len(f1_hit) if f1_hit else 0:.4f}",
        "relaxed_f1_when_miss10": f"{sum(f1_miss)/len(f1_miss) if f1_miss else 0:.4f}",
    }

def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY environment variable")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    sampled = load_csv("results/llm_reader_pilot_qa.csv")
    answer_map = load_qa_answer_map("results/locomo_qa_records.csv")
    mem_by_id, mem_by_dia, ordered_by_session = load_memory_index("results/locomo_memory_records.csv")
    obs_lookup, summary_lookup = load_locomo_context("external_data/locomo10.json")
    method_retrieval = load_method_retrieval()
    gold_map = load_gold_map("results/locomo_evidence_map.csv")

    result_path = Path("results/llm_reader_pilot_v3_results.csv")
    summary_path = Path("results/llm_reader_pilot_v3_summary.csv")

    result_fields = [
        "qa_id", "category", "method", "question", "gold_answer",
        "top10_evidence_ids", "resolved_memory_ids", "evidence_texts",
        "predicted_answer", "strict_em", "strict_f1", "relaxed_em", "relaxed_f1",
        "retrieval_hit10", "answer_string_in_evidence", "prompt"
    ]

    done = set()
    if result_path.exists():
        for row in load_csv(str(result_path)):
            pred = str(row.get("predicted_answer", ""))
            if pred and not pred.startswith("ERROR:"):
                done.add((row.get("qa_id", ""), row.get("method", "")))

    print(f"Loaded {len(sampled)} sampled QA pairs")
    print(f"Loaded {len(obs_lookup)} observation entries, {len(summary_lookup)} session summaries")

    calls = 0
    for qa in sampled:
        qa_id = str(qa.get("qa_id", "")).strip()
        category = str(qa.get("category", "")).strip()
        question = str(qa.get("question", "")).strip()
        gold_answer = str(qa.get("answer", "")).strip() or answer_map.get(qa_id, "")

        for method in METHODS:
            if (qa_id, method) in done:
                continue

            raw_ids = method_retrieval.get(qa_id, {}).get(method, [])[:10]
            resolved_ids, evidence_texts, evidence_blocks = build_evidence_for_method(
                qa_id, raw_ids, mem_by_id, mem_by_dia, obs_lookup, summary_lookup
            )

            prompt = build_prompt(question, evidence_blocks)
            prompt = make_ascii_safe(prompt)

            predicted = call_deepseek(client, prompt)
            predicted = predicted.strip()

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
                "method": method,
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

            append_csv(str(result_path), row, result_fields)
            done.add((qa_id, method))
            calls += 1

            if calls % 20 == 0:
                print(f"  {calls} calls done...")

            time.sleep(0.3 + random.random() * 0.2)

    final_rows = load_csv(str(result_path))
    error_rows = [r for r in final_rows if str(r.get("predicted_answer", "")).startswith("ERROR:")]
    if error_rows:
        raise RuntimeError(f"Found {len(error_rows)} ERROR rows in v3 results")

    print("")
    print("=== LLM Reader v3 Pilot ===")
    print(f"QA sample: {len(sampled)}, Total rows: {len(final_rows)}, New calls: {calls}")

    ca_count = sum(1 for r in final_rows if normalize_relaxed(r.get("predicted_answer", "")) == "cannot answer")
    print(f"CannotAnswer: {ca_count}/{len(final_rows)} = {ca_count / len(final_rows) * 100:.1f}%")
    print("")

    summary_rows, by_method = summarize(final_rows, gold_map)

    print("")
    print("Category-level KG-boost:")
    kg_rows = by_method.get("KG-boost", [])
    kg_by_cat = defaultdict(list)
    for row in kg_rows:
        kg_by_cat[str(row.get("category", ""))].append(row)
    for cat in sorted(kg_by_cat.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        sub = kg_by_cat[cat]
        n = len(sub)
        em = sum(int(r["relaxed_em"]) for r in sub) / n if n else 0
        f1 = sum(float(r["relaxed_f1"]) for r in sub) / n if n else 0
        ca = sum(1 for r in sub if normalize_relaxed(r.get("predicted_answer", "")) == "cannot answer")
        print(f"  cat {cat}: EM={em:.4f} F1={f1:.4f} CA={ca}/{n}")

    summary_fields = [
        "method", "category", "n", "strict_em", "strict_f1",
        "relaxed_em", "relaxed_f1", "retrieval_hit10",
        "answer_string_in_evidence_rate", "cannot_answer_rate",
        "relaxed_f1_when_hit10", "relaxed_f1_when_miss10"
    ]
    write_csv(str(summary_path), summary_rows, summary_fields)

    print("")
    print(f"Output: {result_path}")
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    main()