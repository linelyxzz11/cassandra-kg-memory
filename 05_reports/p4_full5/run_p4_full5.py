"""P4: ZScore-RawERK Full-5 Benchmark. Cat1-4 + Cat5 + held-out. Single script."""
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
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/p4_full5")
OUT.mkdir(parents=True, exist_ok=True)
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

all_qids = sorted(qas.keys())
cat14 = sorted([q for q in qas if qas[q]["category"] != "5"])
cat5 = sorted([q for q in qas if qas[q]["category"] == "5"])
test_convs = ["conv-26","conv-30","conv-41","conv-43","conv-44","conv-47","conv-49","conv-50"]
test_qids = sorted([q for q in cat14 if qa_sample_map[q] in test_convs])

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

# ===================== DENSE + BM25_ERK =====================
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

print(f"cat1-4: {len(cat14)}, cat5: {len(cat5)}, held-out: {len(test_qids)}")

# ===================== DENSE COMPUTE (for all needed scopes) =====================
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

def zscore_fuse(do, ds, bo, bs, qids, alpha=0.6):
    ranks = {}
    for qid in qids:
        d_ids = do.get(qid, []); d_vals = ds.get(qid, [])
        b_ids = bo.get(qid, []); b_vals = bs.get(qid, [])
        all_ids = list(dict.fromkeys(d_ids + b_ids))
        d_m = {m:s for m,s in zip(d_ids,d_vals)}; b_m = {m:s for m,s in zip(b_ids,b_vals)}
        dm = statistics.mean(d_vals) if d_vals else 0; dsd = max(statistics.stdev(d_vals),1e-9) if len(d_vals)>1 else 1
        bm2 = statistics.mean(b_vals) if b_vals else 0; bsd = max(statistics.stdev(b_vals),1e-9) if len(b_vals)>1 else 1
        d_min = min(d_vals) if d_vals else 0; b_min = min(b_vals) if b_vals else 0
        sc = []
        for m in all_ids:
            dv = d_m.get(m,d_min); bv = b_m.get(m,b_min)
            sc.append((alpha*(dv-dm)/dsd + (1-alpha)*(bv-bm2)/bsd, m))
        sc.sort(key=lambda x: (-x[0], x[1]))
        ranks[qid] = [m for _,m in sc[:10]]
    return ranks

def evaluate_retrieval(ranks, qids):
    h1 = h10 = 0; rrs = []
    for qid in qids:
        top = ranks.get(qid, [])[:10]; gold = gold_map[qid]
        h1 += any(m in gold for m in top[:1]); h10 += any(m in gold for m in top)
        for rk, m in enumerate(top, 1):
            if m in gold: rrs.append(1.0/rk); break
        else: rrs.append(0)
    n = len(qids)
    return {"R@1":round(h1/n,4),"R@10":round(h10/n,4),"MRR":round(statistics.mean(rrs),4),"Hit@10":round(h10/n,4),"n":n}

# ===================== A. GATE =====================
print("\n=== A. Gate ===")
do_held, ds_held = compute_dense(test_qids)
bo_held, bs_held = compute_bm25_erk(test_qids)
zs_held = zscore_fuse(do_held, ds_held, bo_held, bs_held, test_qids)
ret_held = evaluate_retrieval(zs_held, test_qids)
print(f"  Retrieval: MRR={ret_held['MRR']:.4f} (exp 0.5462) R@10={ret_held['R@10']:.4f} (exp 0.8070)")
mrr_ok = abs(ret_held["MRR"]-0.5462)<=1e-4; r10_ok = abs(ret_held["R@10"]-0.8070)<=1e-4

# Reader gate: load P3 held-out predictions
held_cache = {}
pred_path = Path("D:/memorytable/cassandra-kg-memory/reports/p3_reader_alignment/reader_predictions.jsonl")
if pred_path.exists():
    for line in pred_path.read_text().strip().split("\n"):
        if line.strip():
            r = json.loads(line)
            if r.get("method","") == "ZScore_RawERK" and r.get("query_id","") in test_qids:
                held_cache[r["query_id"]] = r

held_preds = [held_cache[qid] for qid in test_qids if qid in held_cache]
held_rF1 = round(statistics.mean([p.get("rF1",0) for p in held_preds if p.get("category","")!="5"]),4) if held_preds else 0
held_rEM = round(statistics.mean([p.get("rEM",0) for p in held_preds if p.get("category","")!="5"]),4) if held_preds else 0
held_WA = round(statistics.mean([p.get("WrongAbst",0) for p in held_preds if p.get("category","")!="5"]),4) if held_preds else 0
rF1_ok = abs(held_rF1-0.3675)<=0.001; rEM_ok = abs(held_rEM-0.2113)<=0.001; WA_ok = abs(held_WA-0.4383)<=0.001
print(f"  Reader:    rF1={held_rF1:.4f} (exp 0.3675) rEM={held_rEM:.4f} (exp 0.2113) WrongAbst={held_WA:.4f} (exp 0.4383)")
gate_ok = mrr_ok and r10_ok and rF1_ok and rEM_ok and WA_ok
print(f"  Gate: {'PASS' if gate_ok else 'FAIL'}")
if not gate_ok: print("STOP"); exit(1)

with (OUT/"final_method_gate.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f,fieldnames=["metric","expected","actual","pass"])
    w.writeheader()
    for m,exp,act,ok in [("MRR",0.5462,ret_held["MRR"],mrr_ok),("R@10",0.8070,ret_held["R@10"],r10_ok),
        ("rF1",0.3675,held_rF1,rF1_ok),("rEM",0.2113,held_rEM,rEM_ok),("WrongAbst",0.4383,held_WA,WA_ok)]:
        w.writerow({"metric":m,"expected":exp,"actual":act,"pass":ok})

# ===================== RETRIEVAL: CAT1-4 + FULL-5 =====================
print("\n=== Retrieval: 1540 cat1-4 + 1986 full-5 ===")
do_c14, ds_c14 = compute_dense(cat14)
bo_c14, bs_c14 = compute_bm25_erk(cat14)
zs_c14 = zscore_fuse(do_c14, ds_c14, bo_c14, bs_c14, cat14)
ret_c14 = evaluate_retrieval(zs_c14, cat14)
print(f"  cat1-4: MRR={ret_c14['MRR']:.4f} R@10={ret_c14['R@10']:.4f} R@1={ret_c14['R@1']:.4f}")

do_f5, ds_f5 = compute_dense(all_qids)
bo_f5, bs_f5 = compute_bm25_erk(all_qids)
zs_f5 = zscore_fuse(do_f5, ds_f5, bo_f5, bs_f5, all_qids)
ret_f5 = evaluate_retrieval(zs_f5, cat14)  # cat1-4 only for retrieval (cat5 no gold evidence)
print(f"  full-5:  MRR={ret_f5['MRR']:.4f} R@10={ret_f5['R@10']:.4f} R@1={ret_f5['R@1']:.4f}")

# ===================== READER: CAT1-4 1540 =====================
print(f"\n=== Reader: cat1-4 {len(cat14)} queries ===", flush=True)
def render(mid):
    m = memories.get(mid)
    if not m: return f"[{mid}]"
    return f"memory_id={mid} | sample={m.get('sample_id','')} | session={m.get('session_id','')} | turn={m.get('dia_id','')} | time={m.get('timestamp','')} | speaker={m.get('speaker','')}\nText: {m.get('text','')}"

P = "Answer the question using only the evidence below.\nIf the evidence does not contain the answer, respond exactly with 'Cannot answer'.\nReturn only the shortest answer. Do not explain.\n\nEvidence:\n{context}\n\nQuestion: {question}\nAnswer:"

def call(prompt):
    for a in range(3):
        try:
            resp = client.chat.completions.create(model="deepseek-chat",messages=[{"role":"user","content":prompt}],temperature=0,max_tokens=128,timeout=60)
            return resp.choices[0].message.content.strip(),True
        except: time.sleep(2**a)
    return "[ERROR]",False

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

api = 0; ch = 0; fl = 0; cache = {**held_cache}; per = []
READER_CACHE_PATH = OUT / "reader_cache_all.jsonl"

for qid in cat14:
    top = zs_c14.get(qid, [])[:10]
    gold_ans = qas[qid].get("answer","") or ""
    items = [f"[{j+1}] {render(mid)}" for j,mid in enumerate(top)]
    prompt = P.replace("{context}","\n\n".join(items)).replace("{question}",qas[qid]["question"])
    ck = hashlib.sha256(f"{qid}|{'|'.join(top)}".encode()).hexdigest()

    if qid in cache:
        cached = cache[qid]
        if cached.get("prompt_sha","") == hashlib.sha256(prompt.encode()).hexdigest()[:16]:
            pred = cached.get("prediction",""); ok = True; ch += 1
        else:
            pred, ok = call(prompt); api += 1; cache[qid] = {}
    else:
        pred, ok = call(prompt); api += 1
    if not ok: fl += 1

    gold = gold_ans if gold_ans else ""
    f = f1v(pred,gold) if gold else 0; e = emv(pred,gold) if gold else 0
    wa = 1 if (gold and abstain(pred)) else 0
    entry = {"qa_id":qid,"category":qas[qid]["category"],"question":qas[qid]["question"],
        "gold_answer":gold,"top10":";".join(top),"prediction":pred,
        "rF1":f,"rEM":e,"is_abstain":abstain(pred),"wrong_abst":wa,
        "prompt_sha":hashlib.sha256(prompt.encode()).hexdigest()[:16]}
    cache[qid] = entry; per.append(entry)

    if (len(per)-ch)%200==0:
        rr = [r for r in per if r["gold_answer"]]
        af1 = statistics.mean([r["rF1"] for r in rr]) if rr else 0
        print(f"  {len(per)}/{len(cat14)} api={api} cache={ch} fail={fl} F1={af1:.3f} ({time.time()-t0:.0f}s)", flush=True)
    time.sleep(0.08)

# Save cache
with READER_CACHE_PATH.open("w") as f:
    for qid, entry in cache.items(): f.write(json.dumps(entry)+"\n")

# Reader metrics cat1-4
reader_rows = []
for scope_qids, scope_name in [(cat14,"cat1-4"),(test_qids,"held-out")]:
    rr = [r for r in per if r["qa_id"] in scope_qids]
    ha = [r for r in rr if r["gold_answer"]]
    n = len(rr)
    rF1 = round(statistics.mean([r["rF1"] for r in ha]),4) if ha else 0
    rEM = round(statistics.mean([r["rEM"] for r in ha]),4) if ha else 0
    wa = round(statistics.mean([r["wrong_abst"] for r in ha]),4) if ha else 0
    ret = evaluate_retrieval(zs_c14, scope_qids)
    reader_rows.append({"scope":scope_name,"n":n,"R@1":ret["R@1"],"R@10":ret["R@10"],"MRR":ret["MRR"],
        "Hit@10":ret["Hit@10"],"rF1":rF1,"rEM":rEM,"WrongAbst":wa})
    print(f"  {scope_name}: R@1={ret['R@1']:.4f} MRR={ret['MRR']:.4f} rF1={rF1:.4f} WrongAbst={wa:.4f}")

# ===================== READER: CAT5 446 =====================
print(f"\n=== Reader: cat5 {len(cat5)} queries ===", flush=True)
cat5_preds = []
for qid in cat5:
    top = zs_f5.get(qid, [])[:10]
    items = [f"[{j+1}] {render(mid)}" for j,mid in enumerate(top)]
    prompt = P.replace("{context}","\n\n".join(items)).replace("{question}",qas[qid]["question"])
    ck = hashlib.sha256(f"{qid}|{'|'.join(top)}".encode()).hexdigest()

    if qid in cache:
        pred = cache[qid].get("prediction",""); ch += 1
    else:
        pred, ok = call(prompt); api += 1
        if not ok: fl += 1

    adv_ans = qas[qid].get("adversarial_answer","") or ""
    is_ab = abstain(pred)
    cat5_preds.append({"qa_id":qid,"question":qas[qid]["question"],
        "adversarial_answer":adv_ans,"prediction":pred,"is_abstain":is_ab})
    cache[qid] = {"qa_id":qid,"prediction":pred,"is_abstain":is_ab}

    if (len(cat5_preds))%100==0:
        n_ab = sum(1 for p in cat5_preds if p["is_abstain"])
        print(f"  {len(cat5_preds)}/{len(cat5)} abstain={n_ab} api={api} ({time.time()-t0:.0f}s)", flush=True)
    time.sleep(0.08)

# Save updated cache
with READER_CACHE_PATH.open("w") as f:
    for qid, entry in cache.items(): f.write(json.dumps(entry)+"\n")

# ===================== CAT5 EVALUATION =====================
n_c5 = len(cat5_preds)
n_correct_ab = sum(1 for p in cat5_preds if p["is_abstain"])
n_non_abstain = n_c5 - n_correct_ab
adv_leak = sum(1 for p in cat5_preds if not p["is_abstain"] and rnorm(p["adversarial_answer"]) in rnorm(p["prediction"]))
halluc = sum(1 for p in cat5_preds if not p["is_abstain"] and not (rnorm(p["adversarial_answer"]) in rnorm(p["prediction"])))
print(f"\n  Cat5: n={n_c5} CorrectAbst={n_correct_ab} ({n_correct_ab/n_c5:.4f})")
print(f"  NonAbstain={n_non_abstain} AdvLeak={adv_leak} Halluc={halluc}")

cat5_metrics = [{"scope":"cat5","n":n_c5,"CorrectAbstention":round(n_correct_ab/n_c5,4),
    "NonAbstentionRate":round(n_non_abstain/n_c5,4),
    "AdversarialAnswerLeakage":round(adv_leak/n_c5,4),
    "HallucinationRate":round(halluc/n_c5,4)}]

with (OUT/"cat5_adversarial_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f,fieldnames=cat5_metrics[0].keys()); w.writeheader(); w.writerows(cat5_metrics)
with (OUT/"cat5_adversarial_per_query.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f,fieldnames=["qa_id","question","adversarial_answer","prediction","is_abstain","correct_abst"])
    w.writeheader()
    for p in cat5_preds: w.writerow({**p, "correct_abst": int(p["is_abstain"])})

# Cat5 error classification
cat5_error_rows = []
for p in cat5_preds:
    if p["is_abstain"]: continue
    err_type = "7_other"
    adv = rnorm(p["adversarial_answer"]); pr = rnorm(p["prediction"])
    if adv in pr: err_type = "1_direct_adversarial"
    elif not any(w in pr for w in adv.split()[:3]): err_type = "2_new_fact"
    else: err_type = "3_wrong_citation"
    cat5_error_rows.append({"qa_id":p["qa_id"],"error_type":err_type,"adversarial_answer":p["adversarial_answer"][:100],"prediction":p["prediction"][:100]})

with (OUT/"cat5_error_types.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f,fieldnames=["qa_id","error_type","adversarial_answer","prediction"]); w.writeheader(); w.writerows(cat5_error_rows)
with (OUT/"cat5_error_analysis.jsonl").open("w") as f:
    for r in cat5_error_rows: f.write(json.dumps(r)+"\n")

error_counts = Counter(r["error_type"] for r in cat5_error_rows)
with (OUT/"cat5_error_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f,fieldnames=["error_type","count","pct"]); w.writeheader()
    for et,c in error_counts.most_common(): w.writerow({"error_type":et,"count":c,"pct":round(c/len(cat5_preds),4)})

# ===================== CATEGORY BREAKDOWN =====================
cat_map = {"1":"multi-hop","2":"temporal","3":"open-domain","4":"single-hop"}
cats = {}
for cat in sorted(cat_map):
    cq = [q for q in cat14 if qas[q]["category"]==cat]
    ret = evaluate_retrieval(zs_c14, cq)
    rr = [r for r in per if r["qa_id"] in cq and r["gold_answer"]]
    rF1 = round(statistics.mean([r["rF1"] for r in rr]),4) if rr else 0
    rEM = round(statistics.mean([r["rEM"] for r in rr]),4) if rr else 0
    wa = round(statistics.mean([r["wrong_abst"] for r in rr]),4) if rr else 0
    cats[cat] = {"category":f"cat{cat}_{cat_map[cat]}","n":len(cq),
        "R@1":ret["R@1"],"R@10":ret["R@10"],"MRR":ret["MRR"],"Hit@10":ret["Hit@10"],
        "rF1":rF1,"rEM":rEM,"WrongAbst":wa}
    print(f"  cat{cat}: n={len(cq)} MRR={ret['MRR']:.4f} rF1={rF1:.4f}")

# ===================== ALL OUTPUTS =====================
with (OUT/"cat1_4_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f,fieldnames=["scope","n","R@1","R@10","MRR","Hit@10","rF1","rEM","WrongAbst"]); w.writeheader(); w.writerows(reader_rows)
with (OUT/"cat1_4_by_category.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f,fieldnames=list(cats[list(cats.keys())[0]].keys())); w.writeheader(); w.writerows(cats.values())
with (OUT/"scope_manifest.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w = csv.DictWriter(f,fieldnames=["scope_name","query_count","included_categories","retrieval_evaluated","reader_evaluated","adversarial_evaluated"])
    w.writeheader()
    w.writerows([{"scope_name":"canonical_cat1_4","query_count":1540,"included_categories":"1,2,3,4","retrieval_evaluated":True,"reader_evaluated":True,"adversarial_evaluated":False},
        {"scope_name":"full5","query_count":1986,"included_categories":"1,2,3,4,5","retrieval_evaluated":True,"reader_evaluated":True,"adversarial_evaluated":True},
        {"scope_name":"heldout_grouped_test","query_count":1150,"included_categories":"1,2,3,4","retrieval_evaluated":True,"reader_evaluated":True,"adversarial_evaluated":False}])

print(f"\nAPI calls: {api}, cache: {ch}, fail: {fl}, runtime: {time.time()-t0:.0f}s")
print(f"\nFinal Tables:")
print(f"  cat1-4 overall:  R@1={ret_c14['R@1']:.4f} R@10={ret_c14['R@10']:.4f} MRR={ret_c14['MRR']:.4f} rF1={reader_rows[0]['rF1']:.4f} WrongAbst={reader_rows[0]['WrongAbst']:.4f}")
print(f"  cat5:            CorrectAbstention={cat5_metrics[0]['CorrectAbstention']:.4f} AdvLeak={cat5_metrics[0]['AdversarialAnswerLeakage']:.4f}")
print(f"  held-out:        MRR={ret_held['MRR']:.4f} rF1={held_rF1:.4f}")
