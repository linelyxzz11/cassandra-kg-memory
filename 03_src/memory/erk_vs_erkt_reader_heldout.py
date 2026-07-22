"""ERK vs ERKT held-out reader A/B. 1150 test queries, 2 methods, DeepSeek API."""
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
ENR = Path("D:/memorytable/cassandra-kg-memory/reports/locomo_retrieval_representation")
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/erk_vs_erkt_reader_heldout")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
NUMBER_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10","eleven":"11","twelve":"12"}

# ===================== DATA =====================
print("Loading data...", flush=True)
memories = {}; mem_sample_map = {}; sample_memories = defaultdict(list)
with (BASE/"locomo_memory_records.csv").open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        mid = row["memory_id"].strip(); sid = row["sample_id"].strip()
        memories[mid] = row; mem_sample_map[mid] = sid
        sample_memories[sid].append({"memory_id": mid, "text": row["text"].strip(), "dia_id": row["dia_id"].strip()})

qas = {}; qa_sample_map = {}
with (BASE/"locomo_qa_records.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        qas[r["qa_id"]] = r; qa_sample_map[r["qa_id"].strip()] = r["sample_id"].strip()
cat14 = sorted([q for q in qas if qas[q]["category"] != "5"])

gold_map = defaultdict(set)
with (BASE/"locomo_evidence_map.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): gold_map[r["qa_id"]].add(r["memory_id"])

with open(BASE/"locomo_memory_ids_bge.txt") as f: mem_ids_bge = [line.strip() for line in f if line.strip()]
with open(BASE/"locomo_qa_ids_bge.txt") as f: qa_ids_bge = [line.strip() for line in f if line.strip()]
mem_embs = np.load(BASE/"locomo_memory_bge_large.npy"); mem_embs = mem_embs / np.linalg.norm(mem_embs, axis=1, keepdims=True)
qa_embs = np.load(BASE/"locomo_qa_bge_large.npy"); qa_embs = qa_embs / np.linalg.norm(qa_embs, axis=1, keepdims=True)
qid_to_idx = {qid: i for i, qid in enumerate(qa_ids_bge)}
mid_to_idx = {mid: i for i, mid in enumerate(mem_ids_bge)}
sample_mem_idx = defaultdict(list)
for mid, sid in mem_sample_map.items():
    if mid in mid_to_idx: sample_mem_idx[sid].append(mid_to_idx[mid])

enriched = list(csv.DictReader((ENR/"enriched_memory_records.csv").open(encoding="utf-8-sig")))
def ef(text, field): return text.split(f"{field}:")[1].split("\n")[0].strip() if f"{field}:" in text else ""
enrich = {"E":{},"R":{},"K":{},"T":{}}
for e in enriched:
    mid = e["memory_id"]
    enrich["E"][mid]=ef(e["enriched_text"],"Entities"); enrich["R"][mid]=ef(e["enriched_text"],"Relations"); enrich["K"][mid]=ef(e["enriched_text"],"Keywords")
for mid in memories:
    ts=str(memories[mid].get("timestamp","")).strip()
    m=re.search(r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})",ts,re.IGNORECASE)
    months={"january":"01","february":"02","march":"03","april":"04","may":"05","june":"06","july":"07","august":"08","september":"09","october":"10","november":"11","december":"12"}
    enrich["T"][mid]=f"{m.group(3)}-{months.get(m.group(2).lower(),'??')}-{int(m.group(1)):02d}" if m else (m2.group(1) if (m2:=re.search(r"(\d{4})",ts)) else "")

# ===================== HELD-OUT SPLIT =====================
test_convs = ["conv-26","conv-30","conv-41","conv-43","conv-44","conv-47","conv-49","conv-50"]
test_qids = sorted([q for q in cat14 if qa_sample_map[q] in test_convs])
dev_convs = ["conv-42","conv-48"]
print(f"  Test: {len(test_qids)} queries on {test_convs}")

# ===================== BM25 + DENSE + RRF =====================
def build_variant_texts(fields, mem_list):
    texts=[]
    for m in mem_list:
        mid=m["memory_id"]; parts=[m["text"]]
        for f in fields:
            if enrich[f].get(mid): parts.append(f"{f}: {enrich[f][mid]}")
        texts.append("\n".join(parts))
    return texts

def run_bm25(fields):
    sample_bm={}
    for sid, mem_list in sample_memories.items():
        texts=build_variant_texts(fields,mem_list)
        bm=BM25Retriever(k1=1.5,b=0.75); bm.fit(texts)
        sample_bm[sid]=(bm,mem_list)
    ranks={}
    for qid in test_qids:
        sid=qa_sample_map[qid]
        if sid not in sample_bm: continue
        bm,mem_list=sample_bm[sid]
        indices,_=bm.search(qas[qid]["question"],top_k=10)
        ranks[qid]=[mem_list[i]["memory_id"] for i in indices]
    return ranks

def dense_ranks():
    ranks={}
    for qid in test_qids:
        sid=qa_sample_map.get(qid)
        if qid not in qid_to_idx or sid not in sample_mem_idx: continue
        qi=qid_to_idx[qid]; q_emb=qa_embs[qi]
        ci=sample_mem_idx[sid]
        if not ci: continue
        scores=np.dot(mem_embs[ci],q_emb)
        sl=np.argsort(-scores); tk=min(10,len(sl))
        ranks[qid]=[mem_ids_bge[ci[i]] for i in sl[:tk]]
    return ranks

print("BM25+Dense+RRF...", flush=True)
dense_t10 = dense_ranks()
bm25_erk = run_bm25(["E","R","K"])
bm25_erkt = run_bm25(["E","R","K","T"])

A,K=0.5,10
def rrf_fuse(d_ranks,b_ranks):
    out={}
    for qid in test_qids:
        dl=d_ranks.get(qid,[])[:10]; bl=b_ranks.get(qid,[])[:10]
        dr={m:i+1 for i,m in enumerate(dl)}; br={m:i+1 for i,m in enumerate(bl)}
        am=list(dict.fromkeys(dl+bl))
        sc=[(A/(K+dr.get(m,999))+(1-A)/(K+br.get(m,999)),m) for m in am]
        sc.sort(key=lambda x:-x[0])
        out[qid]=[m for _,m in sc[:10]]
    return out

bm25_raw = run_bm25([])
rrf_raw = rrf_fuse(dense_t10, bm25_raw)
rrf_erk = rrf_fuse(dense_t10, bm25_erk)
rrf_erkt = rrf_fuse(dense_t10, bm25_erkt)

# ===================== A. RETRIEVAL CONSISTENCY AUDIT =====================
print("=== A. Retrieval Audit ===", flush=True)
def evaluate(ranks, qids):
    h1=h10=0; rrs=[]
    for qid in qids:
        top=ranks.get(qid,[])[:10]; gold=gold_map[qid]
        h1+=any(m in gold for m in top[:1]); h10+=any(m in gold for m in top)
        for rk,m in enumerate(top,1):
            if m in gold: rrs.append(1.0/rk); break
        else: rrs.append(0)
    n=len(qids)
    return {"R@1":round(h1/n,4),"R@10":round(h10/n,4),"MRR":round(statistics.mean(rrs),4),"Hit@10":round(h10/n,4),"n":n}

expected = {"RRF_Raw":0.4774,"RRF_ERK":0.5189,"RRF_ERKT":0.5210,"dMRR":0.0021}
actual_raw=evaluate(rrf_raw,test_qids); actual_erk=evaluate(rrf_erk,test_qids); actual_erkt=evaluate(rrf_erkt,test_qids)
audit = {
    "query_count": len(test_qids), "expected_1150": len(test_qids)==1150,
    "split_match": test_convs==["conv-26","conv-30","conv-41","conv-43","conv-44","conv-47","conv-49","conv-50"],
    "RRF_Raw_MRR": {"expected":expected["RRF_Raw"],"actual":actual_raw["MRR"],"pass":abs(actual_raw["MRR"]-expected["RRF_Raw"])<=1e-4},
    "RRF_ERK_MRR": {"expected":expected["RRF_ERK"],"actual":actual_erk["MRR"],"pass":abs(actual_erk["MRR"]-expected["RRF_ERK"])<=1e-4},
    "RRF_ERKT_MRR": {"expected":expected["RRF_ERKT"],"actual":actual_erkt["MRR"],"pass":abs(actual_erkt["MRR"]-expected["RRF_ERKT"])<=1e-4},
    "delta_MRR": {"expected":expected["dMRR"],"actual":round(actual_erkt["MRR"]-actual_erk["MRR"],4),"pass":abs(round(actual_erkt["MRR"]-actual_erk["MRR"],4)-expected["dMRR"])<=2e-4},
}
all_ok = all(audit[k]["pass"] for k in ["RRF_Raw_MRR","RRF_ERK_MRR","RRF_ERKT_MRR","delta_MRR"]) and audit["expected_1150"] and audit["split_match"]
with (OUT/"00_retrieval_consistency_audit.json").open("w") as f: json.dump(audit,f,indent=2)
per_query_audit = []
for qid in test_qids:
    erk_rr = sum(1.0/rk for rk,m in enumerate(rrf_erk.get(qid,[])[:10],1) if m in gold_map[qid])
    erkt_rr = sum(1.0/rk for rk,m in enumerate(rrf_erkt.get(qid,[])[:10],1) if m in gold_map[qid])
    per_query_audit.append({"qa_id":qid,"ERK_rr":erk_rr,"ERKT_rr":erkt_rr,"delta":round(erkt_rr-erk_rr,6)})
delta_mean = round(statistics.mean([r["delta"] for r in per_query_audit]),4)
with (OUT/"00_retrieval_per_query_delta.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=["qa_id","ERK_rr","ERKT_rr","delta"]); w.writeheader(); w.writerows(per_query_audit)
audit_summary = f"# Retrieval Consistency Audit\n- queries: {len(test_qids)} (expected 1150: {audit['expected_1150']})\n- split: {audit['split_match']}\n- RRF_Raw MRR: {actual_raw['MRR']:.4f} (expected 0.4774, {'PASS' if audit['RRF_Raw_MRR']['pass'] else 'FAIL'})\n- RRF_ERK MRR: {actual_erk['MRR']:.4f} (expected 0.5189, {'PASS' if audit['RRF_ERK_MRR']['pass'] else 'FAIL'})\n- RRF_ERKT MRR: {actual_erkt['MRR']:.4f} (expected 0.5210, {'PASS' if audit['RRF_ERKT_MRR']['pass'] else 'FAIL'})\n- dMRR: {actual_erkt['MRR']-actual_erk['MRR']:+.4f} (expected +0.0021)\n- per-query delta mean: {delta_mean}\n- OVERALL: {'ALL PASS' if all_ok else 'FAIL'}"
with (OUT/"00_retrieval_audit_summary.md").open("w") as f: f.write(audit_summary)
print(f"  {audit_summary}")
if not all_ok: print("STOP: Audit FAILED"); exit(1)

# ===================== B. READER CONFIG =====================
reader_prompt = """Answer the question using only the evidence below.
If the evidence does not contain the answer, respond exactly with 'Cannot answer'.
Return only the shortest answer. Do not explain.

Evidence:
{EVIDENCE}

Question: {QUESTION}
Answer:"""
config_sha = hashlib.sha256(f"deepseek-chat_temp0_maxtokens128_{reader_prompt}".encode()).hexdigest()[:16]
with (OUT/"01_reader_run_config.json").open("w") as f: json.dump({"model":"deepseek-chat","temperature":0,"max_tokens":128,"prompt_format":"legacy_v2"},f,indent=2)
with (OUT/"01_reader_prompt.txt").open("w") as f: f.write(reader_prompt)
with (OUT/"01_reader_config_sha256.txt").open("w") as f: f.write(config_sha)

# ===================== C. READER GENERATION =====================
print("=== C. Reader Generation ===", flush=True)

def render_memory(mid):
    m=memories.get(mid)
    if not m: return f"[{mid}]"
    return f"memory_id={mid} | sample={m.get('sample_id','')} | session={m.get('session_id','')} | turn={m.get('dia_id','')} | time={m.get('timestamp','')} | speaker={m.get('speaker','')}\nText: {m.get('text','')}"

def build_prompt(question, mids):
    items = [f"[{i+1}] {render_memory(mid)}" for i, mid in enumerate(mids[:10])]
    return reader_prompt.replace("{EVIDENCE}", "\n\n".join(items)).replace("{QUESTION}", question)

def call_model(prompt):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(model="deepseek-chat", messages=[{"role":"user","content":prompt}], temperature=0, max_tokens=128, timeout=60)
            return resp.choices[0].message.content.strip(), True
        except: time.sleep(2**attempt)
    return "[ERROR]", False

def normalize(text):
    text=str(text or "").lower().strip(); text=re.sub(r"[^a-z0-9\s]"," ",text)
    text=re.sub(r"\b(a|an|the)\b"," ",text); return re.sub(r"\s+"," ",text).strip()
def relaxed_norm(text):
    t=normalize(text).split(); return " ".join(NUMBER_MAP.get(w,w) for w in t)
def is_abstain(pred):
    s=relaxed_norm(pred); patterns=["cannot answer","can not answer","cannot determine","not enough information","insufficient information","not mentioned","no information","not provided","unknown","no evidence","unable to"]
    return any(p in s for p in patterns)
def compute_f1(pred,gold):
    p_toks=relaxed_norm(pred).split(); g_toks=relaxed_norm(gold).split()
    if not p_toks and not g_toks: return 1.0
    if not p_toks or not g_toks: return 0.0
    c=Counter(p_toks)&Counter(g_toks); same=sum(c.values())
    if same==0: return 0.0
    p=same/len(p_toks); r=same/len(g_toks); return round(2*p*r/(p+r),6)
def compute_em(pred,gold): return 1.0 if relaxed_norm(pred)==relaxed_norm(gold) else 0.0

api_calls = 0; cache_hits = 0; failures = 0
per_query_rows = []
cache = {}

for method, ranks in [("RRF_ERK", rrf_erk), ("RRF_ERKT", rrf_erkt)]:
    print(f"\n  [{method}] {len(test_qids)} queries...", flush=True)
    for i, qid in enumerate(test_qids):
        top = ranks.get(qid, [])[:10]
        gold_answer = qas[qid].get("answer","") or ""
        question = qas[qid]["question"]
        prompt = build_prompt(question, top)
        cache_key = hashlib.sha256(f"{qid}|{'|'.join(top)}|{config_sha}".encode()).hexdigest()

        if cache_key in cache:
            pred, ok = cache[cache_key]; cache_hits += 1
        else:
            pred, ok = call_model(prompt); api_calls += 1
            if not ok: failures += 1
            cache[cache_key] = (pred, ok)

        gold = gold_answer if gold_answer else ""
        em = compute_em(pred, gold) if gold else 0
        f1 = compute_f1(pred, gold) if gold else 0
        wa = 1 if (gold and is_abstain(pred)) else 0
        ab = 1 if is_abstain(pred) else 0

        per_query_rows.append({"method":method,"qa_id":qid,"conversation":qa_sample_map[qid],"category":qas[qid]["category"],
            "question":question,"gold_answer":gold,"ordered_top10_memory_ids":";".join(top),
            "prediction":pred,"rF1":f1,"rEM":em,"is_abstain":ab,"wrong_abst":wa,
            "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()[:16],"reader_model":"deepseek-chat"})

        if (i+1)%200==0:
            done=[r for r in per_query_rows if r["method"]==method and r["gold_answer"]]
            avg_f1=statistics.mean([r["rF1"] for r in done]) if done else 0
            print(f"    {i+1}/{len(test_qids)} F1={avg_f1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.15)

print(f"\n  API calls: {api_calls}, cache hits: {cache_hits}, failures: {failures}")

with (OUT/"05_prediction_change_cases.jsonl").open("w") as f:
    for r in per_query_rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

# ===================== D-F. SUMMARY =====================
print("=== Summary ===", flush=True)
def summarize(method, rows):
    has_ans=[r for r in rows if r["gold_answer"]]
    n=len(rows)
    rF1=round(statistics.mean([r["rF1"] for r in has_ans]),4) if has_ans else 0
    rEM=round(statistics.mean([r["rEM"] for r in has_ans]),4) if has_ans else 0
    wa=round(statistics.mean([r["wrong_abst"] for r in has_ans]),4) if has_ans else 0
    return {"Method":method,"n":n,"rF1":rF1,"rEM":rEM,"WrongAbst":wa}

erk_sum = summarize("RRF_ERK", [r for r in per_query_rows if r["method"]=="RRF_ERK"])
erkt_sum = summarize("RRF_ERKT", [r for r in per_query_rows if r["method"]=="RRF_ERKT"])
delta_sum = {"Method":"ERKT-ERK","n":"-","drF1":round(erkt_sum["rF1"]-erk_sum["rF1"],4),"drEM":round(erkt_sum["rEM"]-erk_sum["rEM"],4),"dWrongAbst":round(erk_sum["WrongAbst"]-erkt_sum["WrongAbst"],4)}

overall_rows = [
    {"Method":m,"R@1":e(m)["R@1"],"R@10":e(m)["R@10"],"MRR":e(m)["MRR"],"Hit@10":e(m)["Hit@10"],"rF1":sum["rF1"],"rEM":sum["rEM"],"WrongAbst":sum["WrongAbst"],"n":sum["n"]}
    for m, e, sum in [("RRF_ERK",lambda m_:evaluate(rrf_erk,test_qids),erk_sum),("RRF_ERKT",lambda m_:evaluate(rrf_erkt,test_qids),erkt_sum)]
]
overall_rows.append({"Method":"ERKT-ERK","R@1":round(erkt_sum["rF1"]-erk_sum["rF1"],4),"R@10":"-","MRR":"-","Hit@10":"-",
    "rF1":delta_sum["drF1"],"rEM":delta_sum["drEM"],"WrongAbst":delta_sum["dWrongAbst"],"n":"-"})

with (OUT/"02_reader_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=["Method","R@1","R@10","MRR","Hit@10","rF1","rEM","WrongAbst","n"],extrasaction="ignore")
    w.writeheader(); w.writerows(overall_rows)

# Category
cat_map = {"1":"multi-hop","2":"temporal","3":"open-domain","4":"single-hop"}
cat_out=[]
for method in ["RRF_ERK","RRF_ERKT"]:
    for cat in sorted(cat_map):
        cq=[q for q in test_qids if qas[q]["category"]==cat]
        n=len(cq); rows=[r for r in per_query_rows if r["method"]==method and r["category"]==cat]
        has_ans=[r for r in rows if r["gold_answer"]]
        rF1=round(statistics.mean([r["rF1"] for r in has_ans]),4) if has_ans else 0
        rEM=round(statistics.mean([r["rEM"] for r in has_ans]),4) if has_ans else 0
        wa=round(statistics.mean([r["wrong_abst"] for r in has_ans]),4) if has_ans else 0
        ranks=rrf_erk if method=="RRF_ERK" else rrf_erkt
        rm=evaluate(ranks,cq)
        cat_out.append({"method":method,"category":f"cat{cat}_{cat_map[cat]}","n":n,"rF1":rF1,"rEM":rEM,"WrongAbst":wa,"retrieval_MRR":rm["MRR"],"retrieval_R@10":rm["R@10"]})

with (OUT/"03_reader_by_category.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(cat_out[0].keys())); w.writeheader(); w.writerows(cat_out)

# Win/Tie/Loss
wtl_rows=[]
for qid in test_qids:
    erk_rows=[r for r in per_query_rows if r["method"]=="RRF_ERK" and r["qa_id"]==qid]
    erkt_rows=[r for r in per_query_rows if r["method"]=="RRF_ERKT" and r["qa_id"]==qid]
    if not erk_rows or not erkt_rows: continue
    ek, et = erk_rows[0], erkt_rows[0]
    gold=ek["gold_answer"]
    if gold:
        wtl_rows.append({"qa_id":qid,"category":ek["category"],
            "ERK_F1":ek["rF1"],"ERKT_F1":et["rF1"],"f1_outcome":"erkt_better" if et["rF1"]>ek["rF1"] else ("tied" if et["rF1"]==ek["rF1"] else "erk_better"),
            "ERK_EM":ek["rEM"],"ERKT_EM":et["rEM"],"em_outcome":"erkt_newly_correct" if et["rEM"]==1 and ek["rEM"]==0 else ("erk_correct_erkt_incorrect" if et["rEM"]==0 and ek["rEM"]==1 else ("both_correct" if ek["rEM"]==1 else "both_incorrect")),
            "ERK_abstain":ek["is_abstain"],"ERKT_abstain":et["is_abstain"],
            "abstain_outcome":"erkt_fixed" if not et["is_abstain"] and ek["is_abstain"] else ("erkt_new_error" if et["is_abstain"] and not ek["is_abstain"] else ("both_abstain" if et["is_abstain"] else "both_answered"))})
with (OUT/"05_reader_win_tie_loss.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(wtl_rows[0].keys())); w.writeheader(); w.writerows(wtl_rows)

# Statistics
def bs_paired(m1,m2,metric_fn,n_reps=10000):
    rng=random.Random(42); n=len(test_qids)
    diffs=[]
    for _ in range(n_reps):
        idx=[rng.randint(0,n-1) for __ in range(n)]
        diffs.append(metric_fn([m1.get(test_qids[i],{}) for i in idx])-metric_fn([m2.get(test_qids[i],{}) for i in idx]))
    diffs.sort(); return round(statistics.mean(diffs),4),round(diffs[250],4),round(diffs[9750],4)

def bs_cluster(m1,m2,metric_fn,n_reps=10000):
    conv_qids=defaultdict(list)
    for q in test_qids: conv_qids[qa_sample_map[q]].append(q)
    conv_list=list(conv_qids.keys())
    rng=random.Random(42); diffs=[]
    for _ in range(n_reps):
        sampled=[]
        for __ in range(len(conv_list)): sampled.extend(conv_qids[rng.choice(conv_list)])
        diffs.append(metric_fn([m1.get(q,{}) for q in sampled])-metric_fn([m2.get(q,{}) for q in sampled]))
    diffs.sort(); return round(statistics.mean(diffs),4),round(diffs[250],4),round(diffs[9750],4)

erk_map={r["qa_id"]:r for r in per_query_rows if r["method"]=="RRF_ERK"}
erkt_map={r["qa_id"]:r for r in per_query_rows if r["method"]=="RRF_ERKT"}

def f1_fn(items):
    vals=[i.get("rF1",0) for i in items if i.get("gold_answer")]
    return statistics.mean(vals) if vals else 0
def em_fn(items):
    vals=[i.get("rEM",0) for i in items if i.get("gold_answer")]
    return statistics.mean(vals) if vals else 0
def wa_fn(items):
    vals=[i.get("wrong_abst",0) for i in items if i.get("gold_answer")]
    return statistics.mean(vals) if vals else 0

bs_rows=[]
for metric,fn,label in [("rF1",f1_fn,"rF1"),("rEM",em_fn,"rEM"),("WrongAbst",wa_fn,"WrongAbst")]:
    mq,loq,hiq=bs_paired(erkt_map,erk_map,fn)
    mc,loc,hic=bs_cluster(erkt_map,erk_map,fn)
    bs_rows.append({"metric":label,"point_estimate":mq,"query_ci_lo":loq,"query_ci_hi":hiq,"cluster_ci_lo":loc,"cluster_ci_hi":hic})
    print(f"  {label}: pt={mq:+.4f} query_CI=[{loq:+.4f},{hiq:+.4f}] cluster_CI=[{loc:+.4f},{hic:+.4f}]")

with (OUT/"04_query_bootstrap.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(bs_rows[0].keys())); w.writeheader(); w.writerows(bs_rows)

# ===================== FINAL DECISION =====================
print("=== Final Decision ===", flush=True)
cluster_lo = bs_rows[0]["cluster_ci_lo"]
drF1 = delta_sum["drF1"]
drEM = delta_sum["drEM"]
dWA = delta_sum["dWrongAbst"]

if drF1 > 0 and cluster_lo > 0 and drEM >= -0.001 and dWA >= -0.01:
    decision = "UPGRADE to ERKT"
    reason = f"rF1 positive (+{drF1:+.4f}), cluster CI > 0, rEM/WrongAbst stable"
elif drF1 > 0 and cluster_lo <= 0:
    decision = "KEEP ERK as default, ERKT as extension"
    reason = f"rF1 point positive (+{drF1:+.4f}) but cluster CI crosses zero"
else:
    decision = "KEEP ERK, reject ERKT"
    reason = f"rF1 negative ({drF1:+.4f}) or no reader gain"

final_md = f"""# ERK vs ERKT Final Reader Decision

## Default representation: {decision}

- Time retrieval effect (held-out RRF): dMRR=+0.0021
- Time reader effect (held-out): drF1={drF1:+.4f}, drEM={drEM:+.4f}, dWrongAbst={dWA:+.4f}
- Query-level F1 CI: [{bs_rows[0]['query_ci_lo']:+.4f}, {bs_rows[0]['query_ci_hi']:+.4f}]
- Cluster-level F1 CI: [{bs_rows[0]['cluster_ci_lo']:+.4f}, {bs_rows[0]['cluster_ci_hi']:+.4f}]
- Best category: see 03_reader_by_category.csv
- Decision reason: {reason}
- API calls: {api_calls}, cache hits: {cache_hits}, failures: {failures}, runtime: {time.time()-t0:.0f}s
"""
with (OUT/"06_final_erk_erkt_reader_decision.md").open("w") as f: f.write(final_md)
print(final_md)
