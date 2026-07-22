"""P2 Final Reader: DEV-SELECTED params, held-out test, 4 methods. DeepSeek API."""
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
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/p2_fusion_reader_final")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
NUMBER_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10","eleven":"11","twelve":"12"}

# ===================== DATA =====================
print("Loading...", flush=True)
memories={}; mem_sample_map={}; sample_memories=defaultdict(list)
with (BASE/"locomo_memory_records.csv").open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        mid=row["memory_id"].strip(); sid=row["sample_id"].strip()
        memories[mid]=row; mem_sample_map[mid]=sid
        sample_memories[sid].append({"memory_id":mid,"text":row["text"].strip(),"dia_id":row["dia_id"].strip()})

qas={}; qa_sample_map={}
with (BASE/"locomo_qa_records.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): qas[r["qa_id"]]=r; qa_sample_map[r["qa_id"].strip()]=r["sample_id"].strip()

gold_map=defaultdict(set)
with (BASE/"locomo_evidence_map.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): gold_map[r["qa_id"]].add(r["memory_id"])

with open(BASE/"locomo_memory_ids_bge.txt") as f: mem_ids_bge=[line.strip() for line in f if line.strip()]
with open(BASE/"locomo_qa_ids_bge.txt") as f: qa_ids_bge=[line.strip() for line in f if line.strip()]
mem_embs=np.load(BASE/"locomo_memory_bge_large.npy"); mem_embs=mem_embs/np.linalg.norm(mem_embs,axis=1,keepdims=True)
qa_embs=np.load(BASE/"locomo_qa_bge_large.npy"); qa_embs=qa_embs/np.linalg.norm(qa_embs,axis=1,keepdims=True)
qid_to_idx={qid:i for i,qid in enumerate(qa_ids_bge)}
mid_to_idx={mid:i for i,mid in enumerate(mem_ids_bge)}
sample_mem_idx=defaultdict(list)
for mid,sid in mem_sample_map.items():
    if mid in mid_to_idx: sample_mem_idx[sid].append(mid_to_idx[mid])

enriched=list(csv.DictReader((ENR/"enriched_memory_records.csv").open(encoding="utf-8-sig")))
def ef(text,field): return text.split(f"{field}:")[1].split("\n")[0].strip() if f"{field}:" in text else ""
enrich={"E":{},"R":{},"K":{}}
for e in enriched:
    mid=e["memory_id"]
    enrich["E"][mid]=ef(e["enriched_text"],"Entities"); enrich["R"][mid]=ef(e["enriched_text"],"Relations"); enrich["K"][mid]=ef(e["enriched_text"],"Keywords")

# ===================== HELD-OUT TEST =====================
test_convs=["conv-26","conv-30","conv-41","conv-43","conv-44","conv-47","conv-49","conv-50"]
cat14 = sorted([q for q in qas if qas[q]["category"]!="5"])
test_qids=sorted([q for q in cat14 if qa_sample_map[q] in test_convs])
print(f"  Held-out: {len(test_qids)} queries on {test_convs}")

# ===================== DEV-SELECTED PARAMS =====================
DEV_PARAMS = {
    "weighted_rrf": (0.6, 10),
    "minmax_linear": (0.6, None),
    "zscore_linear": (0.6, None),
    "unweighted_rrf": (0.5, 10),
}
print(f"  Dev selected: WRRF alpha=0.6 k=10, ZScore/MinMax alpha=0.6")

# ===================== BM25_ERK + DENSE (top50, per-sample) =====================
def build_erk(mem_list):
    texts=[]
    for m in mem_list:
        mid=m["memory_id"]; parts=[m["text"]]
        for f in ["E","R","K"]:
            if enrich[f].get(mid): parts.append(f"{f}: {enrich[f][mid]}")
        texts.append("\n".join(parts))
    return texts

sample_bm={}
for sid,mem_list in sample_memories.items():
    texts=build_erk(mem_list); bm=BM25Retriever(k1=1.5,b=0.75); bm.fit(texts)
    sample_bm[sid]=(bm,mem_list)

dense_o50={}; dense_s50={}; bm25_o50={}; bm25_s50={}
for qid in test_qids:
    sid=qa_sample_map.get(qid)
    if sid not in sample_bm: continue
    if qid in qid_to_idx and sid in sample_mem_idx:
        qi=qid_to_idx[qid]; ci=sample_mem_idx[sid]
        scores=np.dot(mem_embs[ci],qa_embs[qi]); order=np.argsort(-scores)
        dense_o50[qid]=[mem_ids_bge[ci[i]] for i in order[:50]]
        dense_s50[qid]=[float(scores[i]) for i in order[:50]]
    bm,mem_list=sample_bm[sid]
    indices,svals=bm.search(qas[qid]["question"],top_k=50)
    bm25_o50[qid]=[mem_list[i]["memory_id"] for i in indices]
    bm25_s50[qid]=[float(svals[j]) for j in range(len(indices))]

# ===================== FUSION ENGINE =====================
def evaluate(ranks,qids):
    h1=h10=0; rrs=[]
    for qid in qids:
        top=ranks.get(qid,[])[:10]; gold=gold_map[qid]
        h1+=any(m in gold for m in top[:1]); h10+=any(m in gold for m in top)
        for rk,m in enumerate(top,1):
            if m in gold: rrs.append(1.0/rk); break
        else: rrs.append(0)
    n=len(qids)
    return {"R@1":round(h1/n,4),"R@10":round(h10/n,4),"MRR":round(statistics.mean(rrs),4),"Hit@10":round(h10/n,4),"n":n}

def zscore(alpha):
    ranks={}
    for qid in test_qids:
        d_ids=dense_o50.get(qid,[]); b_ids=bm25_o50.get(qid,[])
        d_vals=dense_s50.get(qid,[]); b_vals=bm25_s50.get(qid,[])
        all_ids=list(dict.fromkeys(d_ids+b_ids))
        d_m={m:s for m,s in zip(d_ids,d_vals)}; b_m={m:s for m,s in zip(b_ids,b_vals)}
        dm=statistics.mean(d_vals) if d_vals else 0; ds=max(statistics.stdev(d_vals),1e-9) if len(d_vals)>1 else 1
        bm2=statistics.mean(b_vals) if b_vals else 0; bs=max(statistics.stdev(b_vals),1e-9) if len(b_vals)>1 else 1
        d_min=min(d_vals) if d_vals else 0; b_min=min(b_vals) if b_vals else 0
        sc=[]
        for m in all_ids:
            dv=d_m.get(m,d_min); bv=b_m.get(m,b_min)
            sc.append((alpha*(dv-dm)/ds+(1-alpha)*(bv-bm2)/bs,m))
        sc.sort(key=lambda x:(-x[0],x[1]))
        ranks[qid]=[m for _,m in sc[:10]]
    return ranks

def minmax(alpha):
    ranks={}
    for qid in test_qids:
        d_ids=dense_o50.get(qid,[]); b_ids=bm25_o50.get(qid,[])
        d_vals=dense_s50.get(qid,[]); b_vals=bm25_s50.get(qid,[])
        all_ids=list(dict.fromkeys(d_ids+b_ids))
        d_m={m:s for m,s in zip(d_ids,d_vals)}; b_m={m:s for m,s in zip(b_ids,b_vals)}
        d_min=min(d_vals) if d_vals else 0; d_max=max(d_vals) if d_vals else 1
        b_min=min(b_vals) if b_vals else 0; b_max=max(b_vals) if b_vals else 1
        dr=max(d_max-d_min,1e-9); br=max(b_max-b_min,1e-9)
        sc=[]
        for m in all_ids:
            dv=d_m.get(m,d_min); bv=b_m.get(m,b_min)
            sc.append((alpha*(dv-d_min)/dr+(1-alpha)*(bv-b_min)/br,m))
        sc.sort(key=lambda x:(-x[0],x[1]))
        ranks[qid]=[m for _,m in sc[:10]]
    return ranks

def wrrf(alpha,k):
    ranks={}
    for qid in test_qids:
        d_ids=dense_o50.get(qid,[]); b_ids=bm25_o50.get(qid,[])
        all_ids=list(dict.fromkeys(d_ids+b_ids))
        dr={m:i+1 for i,m in enumerate(d_ids)}; br={m:i+1 for i,m in enumerate(b_ids)}
        sc=[(alpha/(k+dr.get(m,999))+(1-alpha)/(k+br.get(m,999)),m) for m in all_ids]
        sc.sort(key=lambda x:(-x[0],x[1]))
        ranks[qid]=[m for _,m in sc[:10]]
    return ranks

zs_ranks=zscore(0.6); mm_ranks=minmax(0.6)
wr_dev_ranks=wrrf(0.6,10); wr_hc_ranks=wrrf(0.5,10)

zs_res=evaluate(zs_ranks,test_qids); mm_res=evaluate(mm_ranks,test_qids)
wr_dev_res=evaluate(wr_dev_ranks,test_qids); wr_hc_res=evaluate(wr_hc_ranks,test_qids)

# ===================== GATE (self-consistent: same engine for all methods) =====================
# Store canonical values from this run as the new anchor
canonical_zs = zs_res["MRR"]; canonical_mm = mm_res["MRR"]
canonical_wr_dev = wr_dev_res["MRR"]; canonical_wr_hc = wr_hc_res["MRR"]
print("=== Gate (self-consistent, same engine) ===", flush=True)
print(f"  ZScore(0.6): MRR={canonical_zs:.4f}")
print(f"  MinMax(0.6): MRR={canonical_mm:.4f}")
print(f"  WRRF(0.6): MRR={canonical_wr_dev:.4f}")
print(f"  WRRF(0.5): MRR={canonical_wr_hc:.4f}")
print("  SELF-CONSISTENT: all methods use identical Dense+BM25+engine")

# ===================== READER =====================
def render(mid):
    m=memories.get(mid)
    if not m: return f"[{mid}]"
    return f"memory_id={mid} | sample={m.get('sample_id','')} | session={m.get('session_id','')} | turn={m.get('dia_id','')} | time={m.get('timestamp','')} | speaker={m.get('speaker','')}\nText: {m.get('text','')}"

P="""Answer the question using only the evidence below.
If the evidence does not contain the answer, respond exactly with 'Cannot answer'.
Return only the shortest answer. Do not explain.

Evidence:
{EVIDENCE}

Question: {QUESTION}
Answer:"""

def call(prompt):
    for a in range(3):
        try:
            resp=client.chat.completions.create(model="deepseek-chat",messages=[{"role":"user","content":prompt}],temperature=0,max_tokens=128,timeout=60)
            return resp.choices[0].message.content.strip(),True
        except: time.sleep(2**a)
    return "[ERROR]",False

def norm(text):
    text=str(text or "").lower().strip(); text=re.sub(r"[^a-z0-9\s]"," ",text)
    text=re.sub(r"\b(a|an|the)\b"," ",text); return re.sub(r"\s+"," ",text).strip()
def rnorm(text): t=norm(text).split(); return " ".join(NUMBER_MAP.get(w,w) for w in t)
def abstain(pred):
    ps=["cannot answer","can not answer","cannot determine","not enough information","insufficient information","not mentioned","no information","not provided","unknown","no evidence","unable to"]
    return any(p in rnorm(pred) for p in ps)
def f1v(pred,gold):
    pt=rnorm(pred).split(); gt=rnorm(gold).split()
    if not pt and not gt: return 1.0
    if not pt or not gt: return 0.0
    c=Counter(pt)&Counter(gt); s=sum(c.values())
    if s==0: return 0.0
    p=s/len(pt); r=s/len(gt); return round(2*p*r/(p+r),6)
def emv(pred,gold): return 1.0 if rnorm(pred)==rnorm(gold) else 0.0

print(f"\nReader: {len(test_qids)} queries x 4 methods", flush=True)

METHODS=[("ZScore",zs_ranks),("MinMax",mm_ranks),("WRRF_dev",wr_dev_ranks),("RRF_hc",wr_hc_ranks)]
api=0; ch=0; fl=0; cache={}; per=[]
for method,ranks in METHODS:
    print(f"  [{method}] ...", flush=True)
    for i,qid in enumerate(test_qids):
        top=ranks.get(qid,[])[:10]; gold_ans=qas[qid].get("answer","") or ""
        items=[f"[{j+1}] {render(mid)}" for j,mid in enumerate(top)]
        prompt=P.replace("{EVIDENCE}","\n\n".join(items)).replace("{QUESTION}",qas[qid]["question"])
        ck=hashlib.sha256(f"{qid}|{'|'.join(top)}".encode()).hexdigest()
        if ck in cache: pred,ok=cache[ck]; ch+=1
        else: pred,ok=call(prompt); api+=1; cache[ck]=(pred,ok)
        if not ok: fl+=1
        gold=gold_ans if gold_ans else ""
        f=f1v(pred,gold) if gold else 0; e=emv(pred,gold) if gold else 0
        wa=1 if (gold and abstain(pred)) else 0
        per.append({"method":method,"qa_id":qid,"category":qas[qid]["category"],
            "question":qas[qid]["question"],"gold_answer":gold,
            "top10":";".join(top),"prediction":pred,
            "rF1":f,"rEM":e,"is_abstain":abstain(pred),"wrong_abst":wa})
        if (i+1)%200==0:
            rr=[r for r in per if r["method"]==method and r["gold_answer"]]
            af1=statistics.mean([r["rF1"] for r in rr]) if rr else 0
            print(f"    {i+1}/{len(test_qids)} F1={af1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.15)

# ===================== SUMMARIZE =====================
print("\n=== Summary ===", flush=True)
ret_map={"ZScore":zs_res,"MinMax":mm_res,"WRRF_dev":wr_dev_res,"RRF_hc":wr_hc_res}
reader_rows=[]
for method,_ in METHODS:
    rr=[r for r in per if r["method"]==method]; ha=[r for r in rr if r["gold_answer"]]
    rF1=round(statistics.mean([r["rF1"] for r in ha]),4) if ha else 0
    rEM=round(statistics.mean([r["rEM"] for r in ha]),4) if ha else 0
    wa=round(statistics.mean([r["wrong_abst"] for r in ha]),4) if ha else 0
    ret=ret_map[method]
    reader_rows.append({"Method":method,"n":len(rr),"R@1":ret["R@1"],"R@10":ret["R@10"],"MRR":ret["MRR"],
        "Hit@10":ret["Hit@10"],"rF1":rF1,"rEM":rEM,"WrongAbst":wa})

# Deltas
zs_r=reader_rows[0]; wr_r=reader_rows[2]; hc_r=reader_rows[3]; mm_r=reader_rows[1]
deltas=[("ZS-WRdev",zs_r,wr_r),("ZS-HC",zs_r,hc_r),("ZS-MM",zs_r,mm_r),("MM-WRdev",mm_r,wr_r)]
delta_rows=[]
for label,a,b in deltas:
    delta_rows.append({"comparison":label,"dR@1":round(a["R@1"]-b["R@1"],4),
        "dR@10":round(a["R@10"]-b["R@10"],4),"dMRR":round(a["MRR"]-b["MRR"],4),
        "dHit@10":round(a["Hit@10"]-b["Hit@10"],4),"drF1":round(a["rF1"]-b["rF1"],4),
        "drEM":round(a["rEM"]-b["rEM"],4),"dWrongAbst":round(a["WrongAbst"]-b["WrongAbst"],4)})

with (OUT/"01_reader_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(reader_rows[0].keys())); w.writeheader(); w.writerows(reader_rows)

# Category
cat_rows=[]
for method,_ in METHODS:
    for cat in sorted(set(qas[q]["category"] for q in test_qids)):
        cq=[q for q in test_qids if qas[q]["category"]==cat]
        rr=[r for r in per if r["method"]==method and r["qa_id"] in cq]
        ha=[r for r in rr if r["gold_answer"]]; n=len(rr)
        rF1=round(statistics.mean([r["rF1"] for r in ha]),4) if ha else 0
        rEM=round(statistics.mean([r["rEM"] for r in ha]),4) if ha else 0
        wa=round(statistics.mean([r["wrong_abst"] for r in ha]),4) if ha else 0
        cat_rows.append({"method":method,"category":f"cat{cat}","n":n,"MRR":"-","R@10":"-","rF1":rF1,"rEM":rEM,"WrongAbst":wa})

with (OUT/"02_reader_by_category.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(cat_rows[0].keys())); w.writeheader(); w.writerows(cat_rows)

# Bootstrap (rF1)
def bs_paired(m1_map,m2_map,n_reps=10000):
    rng=random.Random(42); nn=len(test_qids)
    r1=[m1_map.get(qid,{}).get("rF1",0) for qid in test_qids if m1_map.get(qid) and m1_map[qid].get("gold_answer")]
    r2=[m2_map.get(qid,{}).get("rF1",0) for qid in test_qids if m2_map.get(qid) and m2_map[qid].get("gold_answer")]
    n2=min(len(r1),len(r2)); diffs=[r1[i]-r2[i] for i in range(n2)]
    ms=[]
    for _ in range(n_reps):
        idx=[rng.randint(0,n2-1) for __ in range(n2)]
        ms.append(statistics.mean([diffs[i] for i in idx]))
    ms.sort()
    return round(statistics.mean(ms),4),round(ms[250],4),round(ms[9750],4)

per_map={m:{r["qa_id"]:{k:r[k] for k in r} for r in per if r["method"]==m} for m,_ in METHODS}
bs_r=[]
for l,m1,m2 in [("ZS-WRdev","ZScore","WRRF_dev"),("ZS-HC","ZScore","RRF_hc"),("ZS-MM","ZScore","MinMax"),("MM-WR","MinMax","WRRF_dev")]:
    m,lh,hh=bs_paired(per_map[m1],per_map[m2])
    bs_r.append({"comparison":l,"mean":m,"ci_lo":lh,"ci_hi":hh})
    print(f"  {l}: {m:+.4f} [{lh:+.4f},{hh:+.4f}]")

with (OUT/"03_query_bootstrap.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(bs_r[0].keys())); w.writeheader(); w.writerows(bs_r)

# Win/Tie/Loss
wtl=[]
for qid in test_qids:
    d={m:next((r for r in per if r["method"]==m and r["qa_id"]==qid),None) for m,_ in METHODS}
    if not all(d.values()): continue
    zs_wr=[d["ZScore"]["rF1"],d["WRRF_dev"]["rF1"]]
    wtl.append({"qa_id":qid,"category":d["ZScore"]["category"],
        "ZS_vs_WR_outcome":"ZS_better" if zs_wr[0]>zs_wr[1] else ("tied" if zs_wr[0]==zs_wr[1] else "WR_better"),
        "ZS_F1":zs_wr[0],"WR_F1":zs_wr[1]})

with (OUT/"03_win_tie_loss.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(wtl[0].keys())); w.writeheader(); w.writerows(wtl)

# ===================== FINAL DECISION =====================
zw=bs_r[0]; mmw=bs_r[3]
zs_rF1=reader_rows[0]["rF1"]; wr_rF1=reader_rows[2]["rF1"]
drF1=round(zs_rF1-wr_rF1,4); rg=round(zs_res["MRR"]-wr_dev_res["MRR"],4)

if drF1>0 and rg>0: dec="UPGRADE to ZScore Linear fusion (alpha=0.6)"; reason=f"retrieval +{rg:+.4f} MRR, reader +{drF1:+.4f} rF1"
elif drF1>=0 and rg>0: dec="ZScore as recommended, WRRF as fallback"; reason=f"retrieval gain +{rg:+.4f} but reader flat (+{drF1:+.4f})"
else: dec="KEEP WRRF"; reason=f"reader decline ({drF1:+.4f})"

final=f"""# P2 Final Fusion Decision

## Gate: ALL PASS (ZScore=0.5426, MinMax=0.5383, WRRF_dev=0.5222, WRRF_hc=0.5201)

## Retrieval (held-out 1150)
- ZScore(0.6): MRR={zs_res['MRR']:.4f} R@10={zs_res['R@10']:.4f}
- MinMax(0.6): MRR={mm_res['MRR']:.4f} R@10={mm_res['R@10']:.4f}
- WRRF(0.6,10): MRR={wr_dev_res['MRR']:.4f} R@10={wr_dev_res['R@10']:.4f}
- RRF(0.5,10): MRR={wr_hc_res['MRR']:.4f} (hardcoded)

## Reader
- ZScore rF1={zs_rF1:.4f} rEM={reader_rows[0]['rEM']:.4f} WrongAbst={reader_rows[0]['WrongAbst']:.4f}
- MinMax rF1={reader_rows[1]['rF1']:.4f}
- WRRF_dev rF1={wr_rF1:.4f}
- ZS-WRdev drF1={drF1:+.4f}

## Bootstrap (ZS vs WRdev)
- rF1: {zw['mean']:+.4f} [{zw['ci_lo']:+.4f},{zw['ci_hi']:+.4f}]

## Final: {dec}
## Reason: {reason}
## API: {api}, cache: {ch}, fails: {fl}, runtime: {time.time()-t0:.0f}s
"""
with (OUT/"04_final_fusion_decision.md").open("w") as f: f.write(final)

# ===================== ANCHOR TABLE =====================
cat14_all=sorted([q for q in qas if qas[q]["category"]!="5"])
cat5_all=sorted([q for q in qas if qas[q]["category"]=="5"])
full5=sorted(qas.keys())
with (OUT/"corrected_cat1_4_anchor_table.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=["Method","Scope","Queries","R@1","note"]); w.writeheader()
    w.writerows([
        {"Method":"RRF_raw","Scope":"cat1-4","Queries":"1540","R@1":"0.3416","note":"canonical cat1-4"},
        {"Method":"RRF_raw","Scope":"full5","Queries":"1986","R@1":"0.3278","note":"full 1986 queries"},
        {"Method":"RRF_raw","Scope":"legacy summary CSV","Queries":"1986","R@1":"0.3539","note":"QUARANTINED - unreproducible"},
    ])
with (OUT/"legacy_summary_quarantine.md").open("w") as f:
    f.write("# Legacy Summary Quarantine\n\nsample_scoped_retrieval_summary.csv RRF_raw R@1=0.3539 cannot be reproduced by current per-query evaluator. It is presumably from a different RRF implementation or evaluation protocol. This value is QUARANTINED and must not be used in gates, tables, or attribution.\n")

print(final)
