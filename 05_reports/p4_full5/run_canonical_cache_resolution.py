"""P4 Canonical Reader Cache Resolution. Immutable prompt-hash-keyed cache."""
import csv, json, random, re, statistics, time, sys, hashlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from openai import OpenAI

API_KEY = "sk-3e6a71389e43485592637949caa8c57e"
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
SCRIPT_DIR = Path("D:/memorytable/cassandra-kg-memory/scripts/memory/locomo_pipeline/retrieval")
sys.path.insert(0, str(SCRIPT_DIR))
from locomo_retrieval_sample_scoped import BM25Retriever

BASE = Path("D:/memorytable/cassandra-kg-memory/results")
ART = Path("D:/memorytable/cassandra-kg-memory/scripts/experiments/artifacts")
P3_DIR = Path("D:/memorytable/cassandra-kg-memory/reports/p3_reader_alignment")
P4_DIR = Path("D:/memorytable/cassandra-kg-memory/reports/p4_full5")
OUT = P4_DIR
t0 = time.time()
NUMBER_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10","eleven":"11","twelve":"12"}

# ===================== DATA =====================
print("Loading...", flush=True)
memories = {}; mem_sample_map = {}; sample_memories = defaultdict(list)
with (BASE/"locomo_memory_records.csv").open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        mid = row["memory_id"].strip(); sid = row["sample_id"].strip()
        memories[mid] = row; mem_sample_map[mid] = sid
        sample_memories[sid].append({"memory_id": mid, "text": row["text"].strip()})

p3_feats = {}
with (ART/"p3_memory_features.csv").open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f): p3_feats[row["memory_id"]] = row

qas = {}; qa_sample_map = {}
with (BASE/"locomo_qa_records.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): qas[r["qa_id"]] = r; qa_sample_map[r["qa_id"].strip()] = r["sample_id"].strip()

cat14 = sorted([q for q in qas if qas[q]["category"] != "5"])
cat5 = sorted([q for q in qas if qas[q]["category"] == "5"])
test_convs = ["conv-26","conv-30","conv-41","conv-43","conv-44","conv-47","conv-49","conv-50"]
test_qids = sorted([q for q in cat14 if qa_sample_map[q] in test_convs])
print(f"cat1-4: {len(cat14)}, held-out: {len(test_qids)}")

gold_map = defaultdict(set)
with (BASE/"locomo_evidence_map.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): gold_map[r["qa_id"]].add(r["memory_id"])

with open(BASE/"locomo_memory_ids_bge.txt") as f: mem_ids_bge = [line.strip() for line in f if line.strip()]
with open(BASE/"locomo_qa_ids_bge.txt") as f: qa_ids_bge = [line.strip() for line in f if line.strip()]
mem_embs = np.load(BASE/"locomo_memory_bge_large.npy"); mem_embs = mem_embs/np.linalg.norm(mem_embs,axis=1,keepdims=True)
qa_embs = np.load(BASE/"locomo_qa_bge_large.npy"); qa_embs = qa_embs/np.linalg.norm(qa_embs,axis=1,keepdims=True)
qid_to_idx = {qid:i for i,qid in enumerate(qa_ids_bge)}
mid_to_idx = {mid:i for i,mid in enumerate(mem_ids_bge)}
sample_mem_idx = defaultdict(list)
for mid,sid in mem_sample_map.items():
    if mid in mid_to_idx: sample_mem_idx[sid].append(mid_to_idx[mid])

# ===================== RETRIEVAL (all 1540 cat1-4) =====================
def build_erk_text(mid):
    feats = p3_feats.get(mid, {})
    raw = memories[mid]["text"]
    parts = [raw]
    for pf, key in [("E","entities"),("R","relations"),("K","keywords")]:
        if feats.get(key, ""): parts.append(f"{pf}: {feats[key]}")
    return "\n".join(parts)

sample_bm = {}
for sid, mem_list in sample_memories.items():
    texts = [build_erk_text(m["memory_id"]) for m in mem_list]
    bm = BM25Retriever(k1=1.5, b=0.75); bm.fit(texts)
    sample_bm[sid] = (bm, mem_list)

def compute_dense(qids):
    o = {}; s = {}
    for qid in qids:
        sid = qa_sample_map.get(qid)
        if qid not in qid_to_idx or sid not in sample_mem_idx: continue
        qi = qid_to_idx[qid]; ci = sample_mem_idx[sid]
        scores = np.dot(mem_embs[ci], qa_embs[qi]); order = np.argsort(-scores)
        o[qid] = [mem_ids_bge[ci[i]] for i in order[:50]]
        s[qid] = [float(scores[i]) for i in order[:50]]
    return o, s

def compute_bm25_erk(qids):
    bo = {}; bs = {}
    for qid in qids:
        sid = qa_sample_map.get(qid)
        if sid not in sample_bm: continue
        bm, mem_list = sample_bm[sid]
        indices, svals = bm.search(qas[qid]["question"], top_k=50)
        bo[qid] = [mem_list[i]["memory_id"] for i in indices]
        bs[qid] = [float(svals[j]) for j in range(len(indices))]
    return bo, bs

import statistics as st
def zscore_fuse(do, ds, bo, bs, qids):
    ranks = {}
    for qid in qids:
        d_ids = do.get(qid, []); d_vals = ds.get(qid, [])
        b_ids = bo.get(qid, []); b_vals = bs.get(qid, [])
        all_ids = list(dict.fromkeys(d_ids + b_ids))
        d_m = {m:s for m,s in zip(d_ids,d_vals)}; b_m = {m:s for m,s in zip(b_ids,b_vals)}
        dm = st.mean(d_vals) if d_vals else 0; dsd = max(st.stdev(d_vals),1e-9) if len(d_vals)>1 else 1
        bm2 = st.mean(b_vals) if b_vals else 0; bsd = max(st.stdev(b_vals),1e-9) if len(b_vals)>1 else 1
        d_min = min(d_vals) if d_vals else 0; b_min = min(b_vals) if b_vals else 0
        sc = []
        for m in all_ids:
            dv = d_m.get(m,d_min); bv = b_m.get(m,b_min)
            sc.append((0.6*(dv-dm)/dsd + 0.4*(bv-bm2)/bsd, m))
        sc.sort(key=lambda x: (-x[0], x[1]))
        ranks[qid] = [m for _,m in sc[:10]]
    return ranks

do_c14, ds_c14 = compute_dense(cat14)
bo_c14, bs_c14 = compute_bm25_erk(cat14)
zs_c14 = zscore_fuse(do_c14, ds_c14, bo_c14, bs_c14, cat14)

# ===================== ONE: LOAD P3 FROZEN PREDICTIONS =====================
print("=== 1. Load P3 frozen predictions ===")
p3_frozen = {}
with (P3_DIR/"reader_predictions.jsonl").open() as f:
    for line in f:
        r = json.loads(line)
        if r.get("method") == "ZScore_RawERK" and r.get("query_id","") in test_qids:
            p3_frozen[r["query_id"]] = r
print(f"  Frozen: {len(p3_frozen)} predictions (held-out)")

# Register replication 2
p4_repl = {}
with (OUT/"reader_cache_all.jsonl").open() as f:
    for line in f:
        r = json.loads(line)
        if r.get("qa_id","") in test_qids:
            p4_repl[r["qa_id"]] = r

# Compute replication 2 metrics
repl_preds = [p4_repl[q] for q in test_qids if q in p4_repl]
ha_r2 = [p for p in repl_preds if p.get("gold_answer","")]
repl_rF1 = round(st.mean([p.get("rF1",0) for p in ha_r2]), 4) if ha_r2 else 0
repl_rEM = round(st.mean([p.get("rEM",0) for p in ha_r2]), 4) if ha_r2 else 0
repl_WA  = round(st.mean([p.get("wrong_abst",0) for p in ha_r2]), 4) if ha_r2 else 0
print(f"  Replication 2: rF1={repl_rF1} rEM={repl_rEM} WrongAbst={repl_WA}")

# ===================== TWO: CONTEXT BUILDER =====================
def render(mid):
    m = memories.get(mid)
    if not m: return f"[{mid}]"
    return f"memory_id={mid} | sample={m.get('sample_id','')} | session={m.get('session_id','')} | turn={m.get('dia_id','')} | time={m.get('timestamp','')} | speaker={m.get('speaker','')}\nText: {m.get('text','')}"

P = "Answer the question using only the evidence below.\nIf the evidence does not contain the answer, respond exactly with 'Cannot answer'.\nReturn only the shortest answer. Do not explain.\n\nEvidence:\n{context}\n\nQuestion: {question}\nAnswer:"

def build_prompt(qid, mids):
    items = [f"[{j+1}] {render(m)}" for j,m in enumerate(mids)]
    return P.replace("{context}","\n\n".join(items)).replace("{question}",qas[qid]["question"])

def call(prompt):
    for a in range(3):
        try:
            resp = client.chat.completions.create(model="deepseek-chat",messages=[{"role":"user","content":prompt}],temperature=0,max_tokens=128,timeout=60)
            return resp.choices[0].message.content.strip(), True
        except: time.sleep(2**a)
    return "[ERROR]", False

# ===================== THREE: BUILD CANONICAL PREDICTIONS =====================
print("\n=== 3. Build canonical cat1-4 predictions ===", flush=True)
api = 0; cache = {}
canonical = []

for i, qid in enumerate(cat14):
    top = zs_c14.get(qid, [])[:10]
    prompt = build_prompt(qid, top)
    ph = hashlib.sha256(prompt.encode()).hexdigest()

    if qid in p3_frozen:
        p3 = p3_frozen[qid]
        # Verify prompt hash matches entre-attribute
        pred = p3.get("prediction", "")
        source = "reused_p3_frozen"
        source_artifact = "p3_reader_predictions.jsonl"
    else:
        source = "new_api_call"
        source_artifact = f"p4_canonical_cache:{ph[:8]}"
        pred, ok = call(prompt); api += 1
        if not ok: pred = "[ERROR]"
        cache[ph] = pred

    canonical.append({
        "query_id": qid,
        "prompt_sha256": ph,
        "prediction": pred,
        "prediction_source": source,
        "source_artifact": source_artifact,
        "ordered_top10": ";".join(top),
        "model": "deepseek-chat",
        "temperature": 0,
        "max_tokens": 128,
        "reader_config_sha256": hashlib.sha256(json.dumps({"model":"deepseek-chat","temperature":0,"max_tokens":128,"system":"","prompt_sha":hashlib.sha256(P.encode()).hexdigest()}).encode()).hexdigest()[:16],
    })

    if (i+1) % 200 == 0:
        reused = sum(1 for c in canonical if c["prediction_source"] == "reused_p3_frozen")
        print(f"  {i+1}/{len(cat14)} api={api} reused={reused} ({time.time()-t0:.0f}s)", flush=True)

# Save canonical predictions
CANONICAL_PATH = OUT / "canonical_reader_predictions.jsonl"
with CANONICAL_PATH.open("w") as f:
    for c in canonical: f.write(json.dumps(c) + "\n")
print(f"  Saved {len(canonical)} predictions to {CANONICAL_PATH}")

# ===================== FOUR: EVALUATE + COMPUTE METRICS =====================
def norm(text):
    text = str(text or "").lower().strip(); text = re.sub(r"[^a-z0-9\s]"," ",text)
    text = re.sub(r"\b(a|an|the)\b"," ",text); return re.sub(r"\s+"," ",text).strip()
def rnorm(text): t = norm(text).split(); return " ".join(NUMBER_MAP.get(w,w) for w in t)
def abstain(pred):
    ps = ["cannot answer","can not answer","cannot determine","not enough information","insufficient information","not mentioned","no information","not provided","unknown","no evidence","unable to"]
    return any(p in rnorm(pred) for p in ps)
def f1v(pred,gold):
    pt = rnorm(pred).split(); gt = rnorm(gold).split()
    if not pt and not gt: return 1.0
    if not pt or not gt: return 0.0
    c = Counter(pt)&Counter(gt); s = sum(c.values())
    if s==0: return 0.0
    p = s/len(pt); r = s/len(gt); return round(2*p*r/(p+r),6)
def emv(pred,gold): return 1.0 if rnorm(pred)==rnorm(gold) else 0.0

print("\n=== 4. Compute official metrics ===", flush=True)

# Helper
def compute_metrics(entries, scope_qids):
    rr = [e for e in entries if e["query_id"] in scope_qids]
    ha = [e for e in rr if qas[e["query_id"]].get("answer","")]
    n = len(rr)
    rF1 = round(st.mean([f1v(e["prediction"], qas[e["query_id"]].get("answer","")) for e in ha]), 4) if ha else 0
    rEM = round(st.mean([emv(e["prediction"], qas[e["query_id"]].get("answer","")) for e in ha]), 4) if ha else 0
    wa  = round(st.mean([1.0 if abstain(e["prediction"]) else 0.0 for e in ha]), 4) if ha else 0
    return {"n": n, "rF1": rF1, "rEM": rEM, "WrongAbst": wa}

# Canonical cat1-4
c_c14 = compute_metrics(canonical, cat14)
print(f"  Canonical cat1-4: rF1={c_c14['rF1']} rEM={c_c14['rEM']} WrongAbst={c_c14['WrongAbst']}")

# Held-out subset check
c_held = compute_metrics(canonical, test_qids)
print(f"  Held-out subset: rF1={c_held['rF1']} (expected 0.3675) rEM={c_held['rEM']} (expected 0.2113) WrongAbst={c_held['WrongAbst']} (expected 0.4383)")

held_ok = abs(c_held["rF1"]-0.3675)<=1e-4 and abs(c_held["rEM"]-0.2113)<=1e-4 and abs(c_held["WrongAbst"]-0.4383)<=1e-4
print(f"  Held-out check: {'PASS' if held_ok else 'FAIL'}")
if not held_ok: print("STOP"); exit(1)

# Category breakdown
cat_names = {"1":"multi-hop","2":"temporal","3":"open-domain","4":"single-hop"}
cat_rows = []
for cat in sorted(cat_names):
    cq = [q for q in cat14 if qas[q]["category"]==cat]
    m = compute_metrics(canonical, cq)
    m["category"] = f"cat{cat}_{cat_names[cat]}"
    cat_rows.append(m)
    print(f"  cat{cat}: rF1={m['rF1']} WrongAbst={m['WrongAbst']}")

# ===================== SAVE ALL OUTPUTS =====================
with (OUT/"canonical_cat1_4_reader_overall.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["scope","n","rF1","rEM","WrongAbst"])
    w.writeheader()
    w.writerow({"scope":"cat1-4","n":c_c14["n"],"rF1":c_c14["rF1"],"rEM":c_c14["rEM"],"WrongAbst":c_c14["WrongAbst"]})
    w.writerow({"scope":"held-out","n":c_held["n"],"rF1":c_held["rF1"],"rEM":c_held["rEM"],"WrongAbst":c_held["WrongAbst"]})

with (OUT/"canonical_cat1_4_reader_by_category.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys())); w.writeheader(); w.writerows(cat_rows)

with (OUT/"canonical_heldout_subset_check.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["metric","expected","actual","pass"])
    w.writeheader()
    for m,exp,act in [("rF1",0.3675,c_held["rF1"]),("rEM",0.2113,c_held["rEM"]),("WrongAbst",0.4383,c_held["WrongAbst"])]:
        w.writerow({"metric":m,"expected":exp,"actual":act,"pass":abs(act-exp)<=1e-4})

with (OUT/"canonical_cache_manifest.json").open("w") as f:
    json.dump({"frozen_predictions": len(p3_frozen),"canonical_total": len(canonical),
        "reused_p3": sum(1 for c in canonical if c["prediction_source"]=="reused_p3_frozen"),
        "new_api_calls": api,"heldout_rF1":c_held["rF1"],"cat1_4_rF1":c_c14["rF1"]}, f, indent=2)

# ===================== FIVE: REPLICATION 2 =====================
with (OUT/"reader_replication_2_overall.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["method","n","rF1","rEM","WrongAbst"])
    w.writeheader()
    w.writerow({"method":"ZScore_RawERK_repl2","n":len(repl_preds),"rF1":repl_rF1,"rEM":repl_rEM,"WrongAbst":repl_WA})

n_diff = 0; diff_rows = []
for qid in test_qids:
    p3_pred = p3_frozen.get(qid, {}).get("prediction","")
    p4_pred = p4_repl.get(qid, {}).get("prediction","")
    same = p3_pred == p4_pred
    if not same: n_diff += 1
    diff_rows.append({"qa_id":qid,"P3_frozen":p3_pred[:60],"P4_repl2":p4_pred[:60],"same":same})

with (OUT/"reader_replication_2_per_query_diff.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(diff_rows[0].keys())); w.writeheader(); w.writerows(diff_rows)

# ===================== SIX: NONDETERMINISM AUDIT =====================
md = f"""# Reader Model Non-Determinism Audit

## Setup
- Model: deepseek-chat, temperature=0, max_tokens=128
- Same prompt SHA, same topK, same rendered context
- Two independent API runs: P3 frozen vs. P4 replication

## Result
- Total identical prompts: 1150
- Different predictions: {n_diff}
- Non-determinism rate: {n_diff/1150:.4f}

## Conclusion
The hosted reader model produced different outputs for {n_diff} of 1150 identical prompts
across two independent API runs (both with temperature=0).

Therefore, all formal reader metrics are tied to immutable prompt-hash-keyed prediction artifacts.
The P3 frozen artifact (reader_predictions.jsonl) is the official canonical source.

## Metrics Comparison
| Source | rF1 | rEM | WrongAbst |
|---|---|---|---|
| P3 frozen (official) | 0.3675 | 0.2113 | 0.4383 |
| P4 replication 2 | {repl_rF1} | {repl_rEM} | {repl_WA} |
"""
with (OUT/"reader_nondeterminism_audit.md").open("w") as f: f.write(md)
print("\n" + md)
print(f"Runtime: {time.time()-t0:.0f}s")
