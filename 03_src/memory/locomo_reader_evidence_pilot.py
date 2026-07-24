"""Experiment 2: Reader Evidence Packaging. text_only vs text_plus_triples. DeepSeek API.
Pilot: 80 queries (20 per cat1-4), 2 variants = 160 API calls."""
import csv, json, os, re, statistics, time, random
from collections import Counter, defaultdict
from pathlib import Path
from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    raise ValueError("Set DEEPSEEK_API_KEY environment variable")
BASE = Path("D:/memorytable/cassandra-kg-memory/results")
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_reader_evidence_packaging")
OUT.mkdir(parents=True, exist_ok=True)
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

STOP = set("i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between through during before after above below to from up down in out on off over under again further then once here there when where why how all both each few more most other some such no nor not only own same so than too very s t can will just don should now d ll m o re ve y".split())
NUMBER_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10","eleven":"11","twelve":"12"}

# ===================== DATA LOADING =====================
print("Loading data...")
t0 = time.time()

memories = {}
with (BASE/"locomo_memory_records.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): memories[r["memory_id"]] = r

qas = {}
with (BASE/"locomo_qa_records.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): qas[r["qa_id"]] = r

gold_map = defaultdict(set)
with (BASE/"locomo_evidence_map.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): gold_map[r["qa_id"]].add(r["memory_id"])

cat14 = [q for q in qas if qas[q]["category"] != "5"]
print(f"  cat1-4: {len(cat14)} queries")

# Dense+GlobalKG canonical top5
dense_kg_ranks = {}
with (BASE/"locomo_dense_kg_boost_best_results.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        retrieved = r.get("retrieved_memory_ids","")
        if retrieved: dense_kg_ranks[r["qa_id"]] = [x.strip() for x in retrieved.split(";") if x.strip()]

# KG edges for enrichment
mem_kg = defaultdict(list)
with (BASE/"locomo_kg_edges_spacy.csv").open(encoding="utf-8-sig") as f:
    for e in csv.DictReader(f):
        ev, gid = e.get("evidence",""), e["graph_id"]
        for mid, m in memories.items():
            if m["sample_id"] == gid and (m["dia_id"] == ev or ev in mid):
                mem_kg[mid].append(e)
                break

# ===================== EVIDENCE PACKAGING =====================
def build_context_text_only(top5_mids):
    blocks = []
    for mid in top5_mids:
        m = memories.get(mid)
        if not m: continue
        blocks.append(f"[{mid}] Speaker: {m.get('speaker','?')}, Time: {m.get('timestamp','?')}\n{m.get('text','')}")
    return "\n\n".join(blocks)

def build_context_with_triples(top5_mids):
    blocks = []
    for mid in top5_mids:
        m = memories.get(mid)
        if not m: continue
        edges = mem_kg.get(mid, [])
        triples_str = ""
        if edges:
            triples_str = "KG triples: " + "; ".join(
                f"({e.get('src_id','')}, {e.get('relation','')}, {e.get('dst_id','')})"
                for e in edges[:5])
        blocks.append(
            f"[{mid}] Speaker: {m.get('speaker','?')}, Time: {m.get('timestamp','?')}\n"
            f"{m.get('text','')}\n"
            f"{triples_str}"
        )
    return "\n\n".join(blocks)

# ===================== DEEPSEEK CALL =====================
def ask_deepseek(question, context):
    prompt = f"""You are answering a question based on a person's conversation memories.
Use ONLY the provided memories below. If the answer is not in the memories, say "Cannot answer based on provided memories."
Answer concisely in 1-3 sentences.

CONVERSATION MEMORIES:
{context}

QUESTION: {question}

ANSWER:"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0, max_tokens=200
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR: {e}]"

# ===================== EVALUATION =====================
def normalize(text):
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def relaxed_normalize(text):
    text = normalize(text)
    tokens = text.split()
    return " ".join(NUMBER_MAP.get(t, t) for t in tokens)

def is_abstain(pred):
    p = relaxed_normalize(pred)
    patterns = ["cannot answer","can not answer","cannot determine","not enough information","insufficient information","not mentioned","no information","not provided","unknown","no evidence","unable to"]
    return any(pt in p for pt in patterns)

def compute_f1(pred, gold):
    p_toks = relaxed_normalize(pred).split()
    g_toks = relaxed_normalize(gold).split()
    if not p_toks and not g_toks: return 1.0
    if not p_toks or not g_toks: return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    same = sum(common.values())
    if same == 0: return 0.0
    p, r = same/len(p_toks), same/len(g_toks)
    return 2*p*r/(p+r)

def compute_em(pred, gold):
    return 1.0 if relaxed_normalize(pred) == relaxed_normalize(gold) else 0.0

# ===================== RUN =====================
print("\n=== Sampling queries ===")
rng = random.Random(42)
sampled = []
for cat in ["1","2","3","4"]:
    cqids = [q for q in cat14 if qas[q]["category"] == cat and q in dense_kg_ranks]
    rng.shuffle(cqids)
    sampled.extend(cqids[:20])
print(f"  Sampled {len(sampled)} queries ({sum(1 for q in sampled if qas[q]['category']=='1')} cat1, {sum(1 for q in sampled if qas[q]['category']=='2')} cat2, {sum(1 for q in sampled if qas[q]['category']=='3')} cat3, {sum(1 for q in sampled if qas[q]['category']=='4')} cat4)")

results = []
for variant, build_fn in [("text_only", build_context_text_only), ("text_plus_triples", build_context_with_triples)]:
    print(f"\n=== Variant: {variant} ===")
    for i, qid in enumerate(sampled):
        cat = qas[qid]["category"]
        question = qas[qid]["question"]
        gold_answer = qas[qid].get("answer","") or qas[qid].get("adversarial_answer","")
        if not gold_answer: gold_answer = ""
        top5 = dense_kg_ranks.get(qid, [])[:5]
        context = build_fn(top5)
        
        print(f"  [{i+1}/{len(sampled)}] {qid} cat{cat} ...", end=" ", flush=True)
        t_start = time.time()
        pred = ask_deepseek(question, context)
        elapsed = time.time() - t_start
        
        r = {
            "qa_id": qid, "category": cat, "variant": variant, "question": question,
            "gold_answer": gold_answer, "predicted": pred, "elapsed_sec": round(elapsed, 2),
            "n_memories": len(top5), "n_triples": sum(1 for m in top5 if m in mem_kg),
            "is_abstain": is_abstain(pred),
            "f1": compute_f1(pred, gold_answer) if gold_answer else 0.0,
            "em": compute_em(pred, gold_answer) if gold_answer else 0.0,
        }
        results.append(r)
        print(f"f1={r['f1']:.3f} abstain={r['is_abstain']} ({elapsed:.1f}s)")
        time.sleep(0.3)  # rate limit

# Save detailed results
with (OUT/"reader_packaging_pilot_results.jsonl").open("w") as f:
    for r in results: f.write(json.dumps(r,ensure_ascii=False)+"\n")

# ===================== SUMMARY =====================
print("\n=== Summary ===")
summary_rows = []
for variant in ["text_only", "text_plus_triples"]:
    vr = [r for r in results if r["variant"] == variant]
    n = len(vr)
    mean_f1 = statistics.mean([r["f1"] for r in vr])
    mean_em = statistics.mean([r["em"] for r in vr])
    abstain_rate = statistics.mean([r["is_abstain"] for r in vr])
    mean_elapsed = statistics.mean([r["elapsed_sec"] for r in vr])
    summary_rows.append({
        "variant": variant, "n": n,
        "mean_f1": round(mean_f1, 4), "mean_em": round(mean_em, 4),
        "abstain_rate": round(abstain_rate, 4),
        "avg_elapsed_sec": round(mean_elapsed, 2),
        "avg_triples": round(statistics.mean([r["n_triples"] for r in vr]), 1),
    })
    print(f"  {variant}: f1={mean_f1:.4f} em={mean_em:.4f} abstain={abstain_rate:.2%} elapsed={mean_elapsed:.1f}s")

# By category
cat_rows = []
for variant in ["text_only", "text_plus_triples"]:
    for cat in ["1","2","3","4"]:
        vr = [r for r in results if r["variant"]==variant and r["category"]==cat]
        if not vr: continue
        cat_rows.append({
            "variant": variant, "category": f"cat{cat}", "n": len(vr),
            "mean_f1": round(statistics.mean([r["f1"] for r in vr]), 4),
            "abstain_rate": round(statistics.mean([r["is_abstain"] for r in vr]), 4),
        })

# CSV outputs
flds = ["variant","n","mean_f1","mean_em","abstain_rate","avg_elapsed_sec","avg_triples"]
with (OUT/"reader_packaging_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=flds); w.writeheader(); w.writerows(summary_rows)

cflds = ["variant","category","n","mean_f1","abstain_rate"]
with (OUT/"reader_packaging_by_category.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cflds); w.writeheader(); w.writerows(cat_rows)

# Method comparison summary
tx = [r for r in results if r["variant"]=="text_only"]
tp = [r for r in results if r["variant"]=="text_plus_triples"]
tx_f1 = statistics.mean([r["f1"] for r in tx])
tp_f1 = statistics.mean([r["f1"] for r in tp])
delta = tp_f1 - tx_f1

summary_md = f"""# Reader Evidence Packaging — Pilot Results

## Overall (80 queries, 20 per cat1-4, Dense+GlobalKG top5)
- text_only: mean F1={tx_f1:.4f}, EM={statistics.mean([r['em'] for r in tx]):.4f}, abstain={statistics.mean([r['is_abstain'] for r in tx]):.1%}
- text_plus_triples: mean F1={tp_f1:.4f}, EM={statistics.mean([r['em'] for r in tp]):.4f}, abstain={statistics.mean([r['is_abstain'] for r in tp]):.1%}
- Delta F1: {delta:+.4f} ({'+' if delta>0 else ''}{delta*100:.1f}%)

## Verdict
- {'text+triples significantly boosts reader F1 (+{:.1f}%). KG evidence structuring matters for reader utilization.' if delta>0.02 else 'Minimal gain from triples. Reader already extracts evidence from text efficiently.' if delta>0 else 'text+triples HURTS reader. KG noise may confuse the LLM.'}

## Next step
- If delta positive, run full 1540 queries
- If delta neutral/negative, stop evidence packaging direction

## Runtime
- {time.time()-t0:.1f}s
"""
with (OUT/"method_comparison_summary.md").open("w") as f: f.write(summary_md)
print(f"\n{summary_md}")
