"""P2 Final v4: canonical Dense+BM25 from sample_scoped CSVs for gate. Reader on held-out 1150."""
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
OUT = Path("D:/memorytable/cassandra-kg-memory/reports/p2_fusion_final_validation")
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
cat14=sorted([q for q in qas if qas[q]["category"]!="5"])
print(f"  cat1-4: {len(cat14)} queries")

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

# ===================== CANONICAL DENSE+BM25 (from sample_scoped CSVs) =====================
dense_t10_c = {}
with (BASE/"sample_scoped/locomo_dense_bge_sample_scoped_results.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        dense_t10_c[r["qa_id"]]=[x.strip() for x in r.get("top10_memory_ids","").split(";") if x.strip()][:10]
bm25_t10_c = {}
with (BASE/"sample_scoped/locomo_bm25_sample_scoped_results.csv").open(encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        bm25_t10_c[r["qa_id"]]=[x.strip() for x in r.get("top10_memory_ids","").split(";") if x.strip()][:10]

# ===================== BM25_ERK per-sample =====================
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

# Get top50 scores for fusion
dense_o50={}; dense_s50={}; bm25_o50={}; bm25_s50={}
for qid in cat14:
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
    bm25_s50[qid]=[float(svals[i]) for i in range(len(indices))]

# BM25_ERK top10
bm25_erk_t10={qid:[m for m,_ in list(zip(bm25_o50[qid],bm25_s50[qid]))[:10]] for qid in cat14 if qid in bm25_o50}

def evaluate(ranks):
    h1=h10=0; rrs=[]
    for qid in cat14:
        top=ranks.get(qid,[])[:10]; gold=gold_map[qid]
        h1+=any(m in gold for m in top[:1]); h10+=any(m in gold for m in top)
        for rk,m in enumerate(top,1):
            if m in gold: rrs.append(1.0/rk); break
        else: rrs.append(0)
    n=len(cat14)
    return {"R@1":round(h1/n,4),"R@10":round(h10/n,4),"MRR":round(statistics.mean(rrs),4),"Hit@10":round(h10/n,4),"n":n}

def rrf_top10(d_ranks,b_ranks,a=0.5,k=10):
    out={}
    for qid in cat14:
        dl=d_ranks.get(qid,[])[:10]; bl=b_ranks.get(qid,[])[:10]
        dr={m:i+1 for i,m in enumerate(dl)}; br={m:i+1 for i,m in enumerate(bl)}
        am=list(dict.fromkeys(dl+bl))
        sc=[(a/(k+dr.get(m,999))+(1-a)/(k+br.get(m,999)),m) for m in am]
        sc.sort(key=lambda x:(-x[0],x[1]))
        out[qid]=[m for _,m in sc[:10]]
    return out

# GATE
print("=== Gate ===", flush=True)
rrf_raw_c = rrf_top10(dense_t10_c, bm25_t10_c)
rrf_erk_c = rrf_top10(dense_t10_c, bm25_erk_t10)

gate = {
    "BM25_raw": (evaluate(bm25_t10_c),{"R@1":0.2552,"R@10":0.5487,"MRR":0.3488}),
    "Dense_raw": (evaluate(dense_t10_c),{"R@1":0.3740,"R@10":0.7299,"MRR":0.4859}),
    "RRF_raw": (evaluate(rrf_raw_c),{"R@1":0.3539,"R@10":0.7409,"MRR":0.4763}),
}
all_ok=True
for label,(act,exp) in gate.items():
    for m in ["R@1","R@10","MRR"]:
        d=abs(act[m]-exp[m])
        if d>1e-4: all_ok=False
        print(f"  {label} {m}: {act[m]} vs {exp[m]} {'PASS' if d<=1e-4 else 'FAIL'}")
if not all_ok: print("STOP: Gate FAILED"); exit(1)
print("  ALL PASS")

# FUSION
def zscore50(alpha=0.6):
    ranks={}
    for qid in cat14:
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

def minmax50(alpha=0.6):
    ranks={}
    for qid in cat14:
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

def wrrf50(alpha=0.5,k=10):
    ranks={}
    for qid in cat14:
        d_ids=dense_o50.get(qid,[]); b_ids=bm25_o50.get(qid,[])
        all_ids=list(dict.fromkeys(d_ids+b_ids))
        dr={m:i+1 for i,m in enumerate(d_ids)}; br={m:i+1 for i,m in enumerate(b_ids)}
        sc=[(alpha/(k+dr.get(m,999))+(1-alpha)/(k+br.get(m,999)),m) for m in all_ids]
        sc.sort(key=lambda x:(-x[0],x[1]))
        ranks[qid]=[m for _,m in sc[:10]]
    return ranks

zs_ranks=zscore50(); mm_ranks=minmax50(); wr_ranks=wrrf50()
zs_res=evaluate(zs_ranks); mm_res=evaluate(mm_ranks); wr_res=evaluate(wr_ranks)
print(f"\nFull 1540: WRRF={wr_res['MRR']:.4f} ZScore={zs_res['MRR']:.4f} MinMax={mm_res['MRR']:.4f}")

# ===================== READER =====================
test_convs=["conv-26","conv-30","conv-41","conv-43","conv-44","conv-47","conv-49","conv-50"]
test_qids=sorted([q for q in cat14 if qa_sample_map[q] in test_convs])

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

print(f"\nReader: {len(test_qids)} queries", flush=True)
api=0; ch=0; fl=0; cache={}; per=[]
for method,ranks in [("WRRF",wr_ranks),("MinMax",mm_ranks),("ZScore",zs_ranks)]:
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
            "question":qas[qid]["question"],"gold_answer":gold,"top10":";".join(top),
            "prediction":pred,"rF1":f,"rEM":e,"is_abstain":abstain(pred),"wrong_abst":wa})
        if (i+1)%200==0:
            rr=[r for r in per if r["method"]==method and r["gold_answer"]]
            af1=statistics.mean([r["rF1"] for r in rr]) if rr else 0
            print(f"    {i+1}/{len(test_qids)} F1={af1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.15)

reader_rows=[]
for method in ["WRRF","MinMax","ZScore"]:
    rr=[r for r in per if r["method"]==method]; ha=[r for r in rr if r["gold_answer"]]
    rF1=round(statistics.mean([r["rF1"] for r in ha]),4) if ha else 0
    rEM=round(statistics.mean([r["rEM"] for r in ha]),4) if ha else 0
    wa=round(statistics.mean([r["wrong_abst"] for r in ha]),4) if ha else 0
    ret={"WRRF":wr_res,"MinMax":mm_res,"ZScore":zs_res}[method]
    reader_rows.append({"Method":method,"R@1":ret["R@1"],"R@10":ret["R@10"],"MRR":ret["MRR"],
        "Hit@10":ret["Hit@10"],"rF1":rF1,"rEM":rEM,"WrongAbst":wa,"n":len(rr)})

with (OUT/"03_reader_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(reader_rows[0].keys())); w.writeheader(); w.writerows(reader_rows)

zs_rF1=reader_rows[2]["rF1"]; wr_rF1=reader_rows[0]["rF1"]
drF1=round(zs_rF1-wr_rF1,4); rg=round(zs_res["MRR"]-wr_res["MRR"],4)
dec="UPGRADE to ZScore Linear (alpha=0.6)" if drF1>0 and rg>0 else ("ZScore recommended" if rg>0 else "KEEP WRRF")

final=f"""# P2 Final Fusion Decision v4

## Gate: ALL PASS (canonical 1540 cat1-4: BM25=0.2552, Dense=0.3740, RRF=0.3539)

## Retrieval (1540 cat1-4, top50 candidate depth)
- ZScore(0.6): MRR={zs_res['MRR']:.4f} R@10={zs_res['R@10']:.4f}
- MinMax(0.6): MRR={mm_res['MRR']:.4f} R@10={mm_res['R@10']:.4f}
- WRRF(0.5,10): MRR={wr_res['MRR']:.4f} R@10={wr_res['R@10']:.4f}

## Reader (held-out 1150)
- ZScore rF1={zs_rF1:.4f} rEM={reader_rows[2]['rEM']:.4f} WrongAbst={reader_rows[2]['WrongAbst']:.4f}
- MinMax rF1={reader_rows[1]['rF1']:.4f}
- WRRF rF1={wr_rF1:.4f}
- ZS-WR drF1={drF1:+.4f}  retrieval dMRR={rg:+.4f}

## Final fusion: {dec}
## API: {api}, cache: {ch}, fails: {fl}, runtime: {time.time()-t0:.0f}s
"""
with (OUT/"06_final_fusion_decision.md").open("w") as f: f.write(final)
print(final)
