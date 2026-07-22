"""P2 Final Fusion: audit + ZScore/MinMax/WRRF reproduction + reader A/B/C + robustness. DeepSeek API."""
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
P2 = Path("D:/memorytable/cassandra-kg-memory/reports/p2_fusion_comparison")
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

test_convs=["conv-26","conv-30","conv-41","conv-43","conv-44","conv-47","conv-49","conv-50"]
test_qids=sorted([q for q in cat14 if qa_sample_map[q] in test_convs])
print(f"  Test: {len(test_qids)} queries")

# ===================== BM25_ERK + DENSE (top50 with scores) =====================
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
    texts=build_erk(mem_list)
    bm=BM25Retriever(k1=1.5,b=0.75); bm.fit(texts)
    sample_bm[sid]=(bm,mem_list)

dense_scores={}; bm25_scores={}
for qid in test_qids:
    sid=qa_sample_map.get(qid)
    if sid not in sample_bm: continue
    if qid in qid_to_idx and sid in sample_mem_idx:
        qi=qid_to_idx[qid]; ci=sample_mem_idx[sid]
        scores=np.dot(mem_embs[ci],qa_embs[qi])
        order=np.argsort(-scores)
        dense_scores[qid]={mem_ids_bge[ci[i]]:float(scores[i]) for i in order[:50]}
    bm,mem_list=sample_bm[sid]
    indices,scores_vals=bm.search(qas[qid]["question"],top_k=50)
    bm25_scores[qid]={mem_list[i]["memory_id"]:float(scores_vals[j]) for j,i in enumerate(indices)}

# ===================== A. AUDIT RRF ANCHOR =====================
print("=== A. Audit ===", flush=True)
# WRRF(0.5,10) with topN=50 candidate union
def wrrf_50(qids):
    ranks={}
    for qid in qids:
        ds=dense_scores.get(qid,{}); bs=bm25_scores.get(qid,{})
        all_ids=list(dict.fromkeys(list(ds.keys())+list(bs.keys())))
        dr={m:i+1 for i,m in enumerate(ds.keys())}; br={m:i+1 for i,m in enumerate(bs.keys())}
        sc=[(0.5/(10+dr.get(m,999))+(0.5/(10+br.get(m,999))),m) for m in all_ids]
        sc.sort(key=lambda x:(-x[0],x[1]))
        ranks[qid]=[m for _,m in sc[:10]]
    return ranks

# WRRF(0.5,10) with only top10 (old method — from previous heldout)
def wrrf_10(qids):
    ranks={}
    for qid in qids:
        ds={m:(i+1) for i,m in enumerate(list(dense_scores.get(qid,{}).keys())[:10])}
        bs={m:(i+1) for i,m in enumerate(list(bm25_scores.get(qid,{}).keys())[:10])}
        all_ids=list(dict.fromkeys(list(ds.keys())+list(bs.keys())))
        sc=[(0.5/(10+ds.get(m,999))+(0.5/(10+bs.get(m,999))),m) for m in all_ids]
        sc.sort(key=lambda x:(-x[0],x[1]))
        ranks[qid]=[m for _,m in sc[:10]]
    return ranks

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

wrrf50_res=evaluate(wrrf_50(test_qids),test_qids)
wrrf10_res=evaluate(wrrf_10(test_qids),test_qids)
print(f"  WRRF(0.5,10) topN=50:  MRR={wrrf50_res['MRR']:.4f}")
print(f"  WRRF(0.5,10) topN=10:  MRR={wrrf10_res['MRR']:.4f}")
print(f"  EXPECTED (old heldout): 0.5189")

audit_md=f"""# RRF Anchor Audit
- Old held-out (topN=10, Dense+BM25_ERK top10 only): MRR=0.5189
- P2 WRRF(0.5,10) topN=50: MRR={wrrf50_res['MRR']:.4f}
- P2 WRRF(0.5,10) topN=10: MRR={wrrf10_res['MRR']:.4f}
- Root cause: P2 used candidate depth=50 (union of top50 from each ranker),
  adding more candidates beyond the original top10 pool.
- This changes RRF scores for memories that appear in top50 but not top10.
{'' if abs(wrrf10_res['MRR']-0.5189)<=1e-4 else 'WARNING: top10-only WRRF does not match old heldout either.'}
"""
with (OUT/"00_rrf_anchor_audit.md").open("w") as f: f.write(audit_md)

# ===================== B-D. ZSCORE/MINMAX/WRRF =====================
print("=== B-D: ZScore/MinMax/WRRF ===", flush=True)
TOP_N=50

def zscore_fuse(qids,alpha=0.6):
    ranks={}
    for qid in qids:
        ds=dense_scores.get(qid,{}); bs=bm25_scores.get(qid,{})
        all_ids=list(dict.fromkeys(list(ds.keys())+list(bs.keys())))
        d_vals=list(ds.values()); b_vals=list(bs.values())
        d_mean=statistics.mean(d_vals) if d_vals else 0; d_std=max(statistics.stdev(d_vals),1e-9) if len(d_vals)>1 else 1
        b_mean=statistics.mean(b_vals) if b_vals else 0; b_std=max(statistics.stdev(b_vals),1e-9) if len(b_vals)>1 else 1
        d_min=min(d_vals) if d_vals else 0; b_min=min(b_vals) if b_vals else 0
        sc=[]
        for m in all_ids:
            ds_val=ds.get(m,d_min); bs_val=bs.get(m,b_min)
            dn=(ds_val-d_mean)/d_std; bn=(bs_val-b_mean)/b_std
            sc.append((alpha*dn+(1-alpha)*bn,m))
        sc.sort(key=lambda x:(-x[0],x[1]))
        ranks[qid]=[m for _,m in sc[:10]]
    return ranks

def minmax_fuse(qids,alpha=0.6):
    ranks={}
    for qid in qids:
        ds=dense_scores.get(qid,{}); bs=bm25_scores.get(qid,{})
        all_ids=list(dict.fromkeys(list(ds.keys())+list(bs.keys())))
        d_vals=list(ds.values()); b_vals=list(bs.values())
        d_min=min(d_vals) if d_vals else 0; d_max=max(d_vals) if d_vals else 1
        b_min=min(b_vals) if b_vals else 0; b_max=max(b_vals) if b_vals else 1
        dr=max(d_max-d_min,1e-9); br=max(b_max-b_min,1e-9)
        sc=[]
        for m in all_ids:
            ds_val=ds.get(m,d_min); bs_val=bs.get(m,b_min)
            dn=(ds_val-d_min)/dr; bn=(bs_val-b_min)/br
            sc.append((alpha*dn+(1-alpha)*bn,m))
        sc.sort(key=lambda x:(-x[0],x[1]))
        ranks[qid]=[m for _,m in sc[:10]]
    return ranks

zs_ranks=zscore_fuse(test_qids); mm_ranks=minmax_fuse(test_qids); wr_ranks=wrrf_50(test_qids)

zs_res=evaluate(zs_ranks,test_qids); mm_res=evaluate(mm_ranks,test_qids); wr_res=evaluate(wr_ranks,test_qids)
print(f"  ZScore(0.6):   MRR={zs_res['MRR']:.4f}  (expected 0.5426)")
print(f"  MinMax(0.6):   MRR={mm_res['MRR']:.4f}  (expected 0.5383)")
print(f"  WRRF(0.5,10):  MRR={wr_res['MRR']:.4f}")

# Fusion definitions
with (OUT/"01_fusion_definitions.md").open("w") as f:
    f.write("# Fusion Definitions\n\n## ZScore\nz = (score - mean) / std\nMean/std computed within top50 candidates for each ranker.\nstd=0: set to 1e-9\nMissing candidate: uses min score for that ranker\n\n## MinMax\nx' = (x - min) / (max - min)\nmax=min: set range to 1e-9\nMissing candidate: uses min score\n\n## WRRF\nscore = 0.5/(10+rank_dense) + 0.5/(10+rank_bm25)\nMissing: rank=999\nTie-break: canonical memory_id ascending\nAll methods: topN=50, candidate union, 1-based ranking\n")

# Reproduction gate
gates={"ZScore":(zs_res["MRR"],0.5426),"MinMax":(mm_res["MRR"],0.5383)}
for k,(act,exp) in gates.items():
    ok=abs(act-exp)<=1e-4
    print(f"  Gate {k}: {act:.4f} vs {exp} {'PASS' if ok else 'FAIL'}")

# ===================== STATISTICAL COMPARISON =====================
print("=== Statistical Comparison ===", flush=True)
def bs_paired(m1,m2,qids,metric="MRR",n_reps=10000):
    rng=random.Random(42); n=len(qids)
    rrs1=[]; rrs2=[]
    for qid in qids:
        top1=m1.get(qid,[])[:10]; top2=m2.get(qid,[])[:10]; gold=gold_map[qid]
        rr1=0; rr2=0
        for rk,m in enumerate(top1,1):
            if m in gold: rr1=1.0/rk; break
        for rk,m in enumerate(top2,1):
            if m in gold: rr2=1.0/rk; break
        rrs1.append(rr1); rrs2.append(rr2)
    diffs=[rrs1[i]-rrs2[i] for i in range(n)]
    means=[]
    for _ in range(n_reps):
        idx=[rng.randint(0,n-1) for __ in range(n)]
        means.append(statistics.mean([diffs[i] for i in idx]))
    means.sort()
    return round(statistics.mean(means),4),round(means[250],4),round(means[9750],4)

bs=[]
for label,m1,m2 in [("ZS_vs_MM",zs_ranks,mm_ranks),("ZS_vs_WR",zs_ranks,wr_ranks),("MM_vs_WR",mm_ranks,wr_ranks)]:
    m,lo,hi=bs_paired(m1,m2,test_qids)
    bs.append({"comparison":label,"mean":m,"ci_lo":lo,"ci_hi":hi})
    print(f"  {label}: {m:+.4f} [{lo:+.4f},{hi:+.4f}]")

with (OUT/"02_normalized_fusion_query_bootstrap.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(bs[0].keys())); w.writeheader(); w.writerows(bs)

# ===================== READER A/B/C =====================
print("=== Reader ===", flush=True)
def render(mid):
    m=memories.get(mid)
    if not m: return f"[{mid}]"
    return f"memory_id={mid} | sample={m.get('sample_id','')} | session={m.get('session_id','')} | turn={m.get('dia_id','')} | time={m.get('timestamp','')} | speaker={m.get('speaker','')}\nText: {m.get('text','')}"

reader_prompt="""Answer the question using only the evidence below.
If the evidence does not contain the answer, respond exactly with 'Cannot answer'.
Return only the shortest answer. Do not explain.

Evidence:
{EVIDENCE}

Question: {QUESTION}
Answer:"""

def call_model(prompt):
    for a in range(3):
        try:
            resp=client.chat.completions.create(model="deepseek-chat",messages=[{"role":"user","content":prompt}],temperature=0,max_tokens=128,timeout=60)
            return resp.choices[0].message.content.strip(),True
        except: time.sleep(2**a)
    return "[ERROR]",False

def norm(text):
    text=str(text or "").lower().strip(); text=re.sub(r"[^a-z0-9\s]"," ",text)
    text=re.sub(r"\b(a|an|the)\b"," ",text); return re.sub(r"\s+"," ",text).strip()
def rnorm(text):
    t=norm(text).split(); return " ".join(NUMBER_MAP.get(w,w) for w in t)
def is_abstain(pred):
    s=rnorm(pred); patterns=["cannot answer","can not answer","cannot determine","not enough information","insufficient information","not mentioned","no information","not provided","unknown","no evidence","unable to"]
    return any(p in s for p in patterns)
def f1(pred,gold):
    p_toks=rnorm(pred).split(); g_toks=rnorm(gold).split()
    if not p_toks and not g_toks: return 1.0
    if not p_toks or not g_toks: return 0.0
    c=Counter(p_toks)&Counter(g_toks); s=sum(c.values())
    if s==0: return 0.0
    p=s/len(p_toks); r=s/len(g_toks); return round(2*p*r/(p+r),6)
def em(pred,gold): return 1.0 if rnorm(pred)==rnorm(gold) else 0.0

api_calls=0; cache_hits=0; failures=0; cache={}; per_rows=[]

for method,ranks in [("WRRF",wr_ranks),("MinMax",mm_ranks),("ZScore",zs_ranks)]:
    print(f"  [{method}] {len(test_qids)} queries...", flush=True)
    for i,qid in enumerate(test_qids):
        top=ranks.get(qid,[])[:10]
        gold_ans=qas[qid].get("answer","") or ""
        q=qas[qid]["question"]
        items=[f"[{j+1}] {render(mid)}" for j,mid in enumerate(top)]
        prompt=reader_prompt.replace("{EVIDENCE}","\n\n".join(items)).replace("{QUESTION}",q)
        ck=hashlib.sha256(f"{qid}|{'|'.join(top)}".encode()).hexdigest()

        if ck in cache: pred,ok=cache[ck]; cache_hits+=1
        else: pred,ok=call_model(prompt); api_calls+=1; cache[ck]=(pred,ok)
        if not ok: failures+=1

        gold=gold_ans if gold_ans else ""
        f1_val=f1(pred,gold) if gold else 0; em_val=em(pred,gold) if gold else 0
        wa=1 if (gold and is_abstain(pred)) else 0
        per_rows.append({"method":method,"qa_id":qid,"category":qas[qid]["category"],
            "question":q,"gold_answer":gold,"top10":";".join(top),
            "prediction":pred,"rF1":f1_val,"rEM":em_val,"is_abstain":is_abstain(pred),"wrong_abst":wa})

        if (i+1)%200==0:
            rr=[r for r in per_rows if r["method"]==method and r["gold_answer"]]
            avg_f1=statistics.mean([r["rF1"] for r in rr]) if rr else 0
            print(f"    {i+1}/{len(test_qids)} F1={avg_f1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.15)

# ===================== SUMMARIZE =====================
print("=== Summary ===", flush=True)
reader_rows=[]
for method in ["WRRF","MinMax","ZScore"]:
    rr=[r for r in per_rows if r["method"]==method]
    has_ans=[r for r in rr if r["gold_answer"]]
    n=len(rr)
    rF1=round(statistics.mean([r["rF1"] for r in has_ans]),4) if has_ans else 0
    rEM=round(statistics.mean([r["rEM"] for r in has_ans]),4) if has_ans else 0
    wa=round(statistics.mean([r["wrong_abst"] for r in has_ans]),4) if has_ans else 0
    ret={"WRRF":wr_res,"MinMax":mm_res,"ZScore":zs_res}[method]
    reader_rows.append({"Method":method,"R@1":ret["R@1"],"R@10":ret["R@10"],"MRR":ret["MRR"],
        "Hit@10":ret["Hit@10"],"rF1":rF1,"rEM":rEM,"WrongAbst":wa,"n":n})
    print(f"  {method}: rF1={rF1:.4f} rEM={rEM:.4f} WrongAbst={wa:.4f}")

with (OUT/"03_reader_overall.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(reader_rows[0].keys())); w.writeheader(); w.writerows(reader_rows)

# Reader bootstrap
er_map={r["qa_id"]:r for r in per_rows if r["method"]=="WRRF"}
mm_map={r["qa_id"]:r for r in per_rows if r["method"]=="MinMax"}
zs_map={r["qa_id"]:r for r in per_rows if r["method"]=="ZScore"}

def reader_f1_fn(items):
    vals=[i.get("rF1",0) for i in items if i.get("gold_answer")]
    return statistics.mean(vals) if vals else 0

rbs=[]
for label,m1,m2 in [("ZS_vs_WR",zs_map,er_map),("ZS_vs_MM",zs_map,mm_map),("MM_vs_WR",mm_map,er_map)]:
    rng=random.Random(42); n=len(test_qids)
    rrs1=[m1.get(qid,{}).get("rF1",0) for qid in test_qids if m1.get(qid) and m1[qid].get("gold_answer")]
    rrs2=[m2.get(qid,{}).get("rF1",0) for qid in test_qids if m2.get(qid) and m2[qid].get("gold_answer")]
    n2=min(len(rrs1),len(rrs2)); diffs=[rrs1[i]-rrs2[i] for i in range(n2)]
    means=[]
    for _ in range(10000):
        idx=[rng.randint(0,n2-1) for __ in range(n2)]
        means.append(statistics.mean([diffs[i] for i in idx]))
    means.sort()
    rbs.append({"comparison":label,"mean":round(statistics.mean(means),4),"ci_lo":round(means[250],4),"ci_hi":round(means[9750],4)})
    print(f"  Reader {label}: {rbs[-1]['mean']:+.4f} [{rbs[-1]['ci_lo']:+.4f},{rbs[-1]['ci_hi']:+.4f}]")

with (OUT/"04_reader_query_bootstrap.csv").open("w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(rbs[0].keys())); w.writeheader(); w.writerows(rbs)

# ===================== FINAL DECISION =====================
print("=== Final Decision ===", flush=True)
zs_rF1=reader_rows[2]["rF1"]; wr_rF1=reader_rows[0]["rF1"]
drF1=round(zs_rF1-wr_rF1,4)
ret_gain=round(zs_res["MRR"]-wr_res["MRR"],4)
zs_vs_wr_bs=rbs[0]

if drF1>0 and ret_gain>0:
    dec="UPGRADE to ZScore Linear fusion (alpha=0.6)"
    reason=f"Retrieval +{ret_gain:+.4f} MRR, reader +{drF1:+.4f} rF1"
elif drF1>=0 and ret_gain>0:
    dec="ZScore as recommended, WRRF as fallback"
    reason=f"Retrieval gain but reader flat (drF1={drF1:+.4f})"
else:
    dec="KEEP WRRF"
    reason=f"Reader decline (drF1={drF1:+.4f})"

final=f"""# P2 Final Fusion Decision

## Default fusion: {dec}

## Retrieval
- ZScore(0.6) MRR={zs_res['MRR']:.4f}
- MinMax(0.6) MRR={mm_res['MRR']:.4f}
- WRRF(0.5,10) MRR={wr_res['MRR']:.4f}

## Reader
- ZScore rF1={zs_rF1:.4f} rEM={reader_rows[2]['rEM']:.4f} WrongAbst={reader_rows[2]['WrongAbst']:.4f}
- WRRF rF1={wr_rF1:.4f}
- drF1={drF1:+.4f}

## Statistical Support
- Retrieval ZS_vs_WR: {bs[1]['mean']:+.4f} [{bs[1]['ci_lo']:+.4f},{bs[1]['ci_hi']:+.4f}]
- Reader ZS_vs_WR: {zs_vs_wr_bs['mean']:+.4f} [{zs_vs_wr_bs['ci_lo']:+.4f},{zs_vs_wr_bs['ci_hi']:+.4f}]

## Final reason: {reason}
## API calls: {api_calls}, cache: {cache_hits}, failures: {failures}, runtime: {time.time()-t0:.0f}s
"""
with (OUT/"06_final_fusion_decision.md").open("w") as f: f.write(final)
print(final)
