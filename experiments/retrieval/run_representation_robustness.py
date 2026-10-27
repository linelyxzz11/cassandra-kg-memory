"""Offline 15-arm representation experiment. Exact-text cache, shared nested masks.

Run all: baseline first, memory dropout, then field dropout. Resume by artifacts.
No historical outputs are modified; no network/Reader/Judge calls.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json, time, argparse, itertools
from pathlib import Path
import numpy as np
import run_raw_preserving_stage1 as old

ROOT=old.ROOT
OUT=ROOT/'results/retrieval/representation_robustness_20261012'
REPS=['Raw','ERK','RawERK']
METHODS=[f'Dense_{r}' for r in REPS]+[f'BM25_{r}' for r in REPS]+[f'H_{d}_{b}' for d in REPS for b in REPS]
RATES=[0.,.1,.25,.5,.75,1.]
SEEDS=list(range(20261012,20261022))
METRICS=old.METRICS

def save(name,obj):
    dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def log(s):
    old.log(s)
    save('status.json',{'time':time.strftime('%Y-%m-%d %H:%M:%S'),'message':s,'pid':os.getpid()})
def csvout(name,rows):old.base.write_csv(OUT/name,rows,list(rows[0]))

class Experiment:
    def __init__(self):
        OUT.mkdir(parents=True,exist_ok=True)
        self.p=old.base.load_p1_module()
        self.config=self.p.load_json(old.base.P1_CONFIG)
        self.mem=old.load('memory_inputs.json');self.ids=[r['memory_id'] for r in self.mem]
        self.mi={m:i for i,m in enumerate(self.ids)}
        self.order=old.load('scope_memory_order.json');self.docs=old.load('documents.json')
        features={r['memory_id']:r for r in old.readcsv(ROOT/self.config['paths']['memory_features'])}
        self.fields=[[self.p.nonempty(features[m].get(k)) for k in ['entities','relations','keywords']] for m in self.ids]
        self.full=np.array([sum((1<<j) for j,f in enumerate(fs) if f) for fs in self.fields],dtype=np.uint8)
        self.raw=np.load(old.OUT/'local_raw_vectors.npy');self.erk=np.load(old.OUT/'local_erk_vectors.npy')
        gold={r['query_id']:r for r in old.readcsv(old.base.GOLD_PATH) if r['gold_status']=='mapped' and r['gold_memory_ids']}
        self.qs=[{'query_id':q.query_id,'scope':q.conversation_id,'category':q.category,'question':q.question,'split':gold[q.query_id]['split'],'gold':gold[q.query_id]['gold_memory_ids'].split(';')} for q in self.p.load_queries(ROOT/self.config['paths']['queries']) if q.query_id in gold]
        assert len(self.qs)==1536 and sum(q['split']=='test' for q in self.qs)==1146
        qids=(ROOT/'data/locomo_qa_ids_bge.txt').read_text(encoding='utf-8-sig').splitlines();qi={q:i for i,q in enumerate(qids)}
        qv=np.load(ROOT/'data/locomo_qa_bge_large.npy');qv=qv/np.linalg.norm(qv,axis=1,keepdims=True)
        self.qv=np.stack([qv[qi[q['query_id']]] for q in self.qs])
        self.si={s:np.array([self.mi[m] for m in ids]) for s,ids in self.order.items()}
        self.sq={s:[i for i,q in enumerate(self.qs) if q['scope']==s] for s in self.order}
        self.cache={'':np.zeros(1024,dtype=np.float32)};self.model=None
        for i,r in enumerate(self.mem):
            self.cache[r['raw_dense']]=self.raw[i];self.cache[r['erk_dense']]=self.erk[i]
            assert self.struct(i,int(self.full[i]))==self.docs['ERK'][self.ids[i]]
        cache_dir=OUT/'embedding_cache';cache_dir.mkdir(exist_ok=True)
        for meta in sorted(cache_dir.glob('*.json')):
            arr=np.load(meta.with_suffix('.npy'));texts=json.loads(meta.read_text(encoding='utf-8'))
            assert len(texts)==len(arr)
            self.cache.update(zip(texts,arr))
        source_files=[old.OUT/'local_raw_vectors.npy',old.OUT/'local_erk_vectors.npy',old.OUT/'memory_inputs.json',old.OUT/'documents.json',old.base.GOLD_PATH,ROOT/self.config['paths']['queries'],ROOT/self.config['paths']['memory_features'],ROOT/'data/locomo_qa_bge_large.npy',old.base.P1_CONFIG,old.base.P1_SCRIPT,Path(__file__)]
        contract={'methods':METHODS,'rates':RATES,'seeds':SEEDS,'modes':['memory','field'],'primary_split':'test1146','supplementary_splits':['dev390','all1536'],'alpha':.6,'candidate_depth':50,'final_k':10,'bm25':self.config['bm25'],'encoding':{'model':str(old.MODEL),'dtype':'float32','pooling':'CLS L2','char_limit':1500,'token_limit':512,'query':'same frozen normalized vectors','raw':'speaker: text','rawerk':'same raw prefix + labelled nonempty E/R/K; truncate concatenation to1500','empty_erk':'zero vector, all IDs retained','cache':'exact text; no stale uncorrupted embeddings'},'mask':'nested uniform threshold U<rate, scope-specific fixed seeds; memory: drop all available fields together; field: independent E/R/K blocks; queries and gold not used','statistics':'10000 paired conversation cluster resamples; across seeds mean and spread, joint seed/conversation resampling for robustness contrasts; marginal unadjusted95% intervals','sources':{str(p):old.sha(p) for p in source_files},'reader_calls':0,'judge_calls':0}
        if (OUT/'contract.json').exists():assert load('contract.json')==contract,'contract/source changed; use new run directory'
        else:save('contract.json',contract)
        save('queries.json',self.qs);save('memory_order.json',self.ids);save('methods.json',METHODS)

    def struct(self,i,bits):return '\n'.join(f'{lab}: {self.fields[i][j]}' for j,lab in enumerate(['E','R','K']) if bits&(1<<j) and self.fields[i][j])
    def text(self,i,bits,rep):
        if rep=='Raw':return self.mem[i]['raw_dense']
        st=self.struct(i,bits)
        return (st if rep=='ERK' else self.mem[i]['raw_dense']+('\n'+st if st else ''))[:1500]

    def encode(self,texts):
        todo=sorted(set(texts)-self.cache.keys(),key=lambda s:(len(s),s))
        if not todo:return
        import torch
        from transformers import AutoTokenizer,AutoModel
        torch.set_num_threads(8)
        if self.model is None:
            self.tok=AutoTokenizer.from_pretrained(old.MODEL,local_files_only=True)
            self.model=AutoModel.from_pretrained(old.MODEL,local_files_only=True).eval()
        log(f'Encoding {len(todo)} new exact texts')
        for block in range(0,len(todo),128):
            tt=todo[block:block+128];arrays=[]
            for a in range(0,len(tt),16):
                batch=self.tok(tt[a:a+16],padding=True,truncation=True,max_length=512,return_tensors='pt')
                with torch.inference_mode():arrays.append(torch.nn.functional.normalize(self.model(**batch).last_hidden_state[:,0],p=2,dim=1).numpy())
            vectors=np.concatenate(arrays).astype(np.float32)
            key=__import__('hashlib').sha256(json.dumps(tt,ensure_ascii=False).encode()).hexdigest()
            np.save(OUT/'embedding_cache'/f'{key}.npy',vectors)
            save(f'embedding_cache/{key}.json',tt)
            self.cache.update(zip(tt,vectors));log(f'Encoded {min(block+128,len(todo))}/{len(todo)}')

    def mask(self,mode,rate,seed):
        bits=self.full.copy()
        for n,s in enumerate(sorted(self.si)):
            ix=self.si[s];rng=np.random.default_rng(np.random.SeedSequence([seed,n,0 if mode=='memory' else 1]))
            if mode=='memory':bits[ix[rng.random(len(ix))<rate]]=0
            else:
                u=rng.random((len(ix),3))
                for j in range(3):bits[ix[u[:,j]<rate]] &= np.uint8(7^(1<<j))
        assert np.all((bits & self.full)==bits)
        return bits

    def branches(self,bits):
        texts={r:[self.text(i,int(bits[i]),r) for i in range(len(self.ids))] for r in REPS}
        self.encode(t for r in REPS for t in texts[r])
        dense={r:[] for r in REPS};lex={r:[None]*len(self.qs) for r in REPS}
        for r in REPS:
            v=np.stack([self.cache[t] for t in texts[r]])
            for k,q in enumerate(self.qs):
                ix=self.si[q['scope']];scores=v[ix]@self.qv[k];rank=np.argsort(-scores,kind='stable')[:50]
                dense[r].append([(self.ids[ix[j]],float(scores[j])) for j in rank])
            for s,ix in self.si.items():
                dd=[]
                for i in ix:
                    st=self.struct(i,int(bits[i]));raw=self.docs['Raw'][self.ids[i]]
                    dd.append(raw if r=='Raw' else st if r=='ERK' else raw+('\n'+st if st else ''))
                retriever=self.p.BM25Retriever(**self.config['bm25'])
                empty=not any(d.strip() for d in dd)
                if not empty:retriever.fit(dd)
                for k in self.sq[s]:
                    jj,ss=(list(range(min(50,len(ix)))),[0.]*min(50,len(ix))) if empty else retriever.search(self.qs[k]['question'],50)
                    lex[r][k]=[(self.ids[ix[j]],float(sc)) for j,sc in zip(jj,ss)]
        return dense,lex

    def run_condition(self,mode,rate,seed):
        name=f'{mode}_r{int(rate*100):03d}_s{seed}';path=OUT/'conditions'/f'{name}.npz'
        if path.exists():return
        log('Running '+name);bits=self.mask(mode,rate,seed)
        dense,lex=self.branches(bits)
        metrics=np.zeros((15,len(self.qs),len(METRICS)));tops=np.zeros((15,len(self.qs),10),dtype=np.int32)
        for k,q in enumerate(self.qs):
            ranks=[dense[r][k] for r in REPS]+[lex[r][k] for r in REPS]+[self.p.zscore_fuse(dense[d][k],lex[b][k],.6,'branch_min',50) for d in REPS for b in REPS]
            for a,rr in enumerate(ranks):
                top=[m for m,_ in rr[:10]]
                assert len(top)==len(set(top))==10 and all(m in self.order[q['scope']] for m in top)
                tops[a,k]=[self.mi[m] for m in top];sc=old.base.score(q['gold'],top)
                metrics[a,k]=[sc[m] for m in METRICS]
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,metrics=metrics,top10=tops,field_bits=bits)
        save(f'conditions/{name}.json',{'mode':mode,'rate':rate,'seed':seed,'records':len(bits),'initial_nonempty':int(sum(self.full!=0)),'final_nonempty':int(sum(bits!=0)),'sha256':old.sha(path)})
        self.sanity(name,metrics,tops,bits)
        self.summarize();log('Completed '+name)

    def sanity(self,name,metrics,tops,bits):
        if np.array_equal(bits,self.full):
            baseline=old.readcsv(old.OUT/'per_query.csv');lookup={(r['method'],r['query_id']):r for r in baseline}
            check=0
            for a,m in enumerate(METHODS):
                mapped=m.replace('Dense_Raw','Dense_LocalRaw').replace('Dense_ERK','Dense_LocalERK') if m.startswith('Dense_') else m.replace('H_Raw_','H_LocalRaw_').replace('H_ERK_','H_LocalERK_')
                for k,q in enumerate(self.qs):
                    row=lookup.get((mapped,q['query_id']))
                    if row:
                        assert [self.ids[i] for i in tops[a,k]]==row['top10_memory_ids'].split(';'),(m,q['query_id'])
                        assert np.allclose(metrics[a,k],[float(row[x]) for x in METRICS],atol=1e-14)
                        check+=1
            save('baseline_parity.json',{'matched_method_query_rows':check,'status':'PASS'})
        if not np.any(bits):
            for a,b in [('Dense_Raw','Dense_RawERK'),('BM25_Raw','BM25_RawERK'),('H_Raw_Raw','H_RawERK_RawERK')]:
                assert np.array_equal(tops[METHODS.index(a)],tops[METHODS.index(b)])
            save('endpoint_sanity.json',{'status':'PASS','rawerk_equals_raw_at100percent':True})

    def summarize(self):
        rows=[];cats=[]
        for path in sorted((OUT/'conditions').glob('*.npz')):
            meta=load(f'conditions/{path.stem}.json');values=np.load(path)['metrics']
            for split in ['test','dev','all']:
                ix=[i for i,q in enumerate(self.qs) if split=='all' or q['split']==split]
                for a,m in enumerate(METHODS):
                    prefix={k:meta[k] for k in ['mode','rate','seed']}
                    rows.append({**prefix,'split':split,'method':m,'n':len(ix),**dict(zip(METRICS,values[a,ix].mean(0)))})
                    for c in ['1','2','3','4']:
                        jj=[i for i in ix if self.qs[i]['category']==c]
                        cats.append({**prefix,'split':split,'category':c,'method':m,'n':len(jj),**dict(zip(METRICS,values[a,jj].mean(0)))})
        csvout('overall.csv',rows);csvout('by_category.csv',cats)
        clean=[r for r in rows if r['mode']=='memory' and r['rate']==0 and r['split']=='test']
        if clean:
            lines=['# Representation robustness — running results','','15 arms; fixed alpha=.6, BGE-large-en-v1.5 CLS/L2 float32, Top50/Top10. Held-out1146 main; dev390/all1536 separately in CSV. No API calls. Local matched encoding; historical frozen result remains separate.','','## Clean baseline (held-out)','','|Method|MRR@10|Recall@10|','|---|---:|---:|']
            lines += [f"|{r['method']}|{r['MRR@10']:.6f}|{r['Recall@10']:.6f}|" for r in clean]
            lines+=['','All available conditions and all six metrics: overall.csv; by_category.csv. Condition NPZ contains per-query metrics, ordered Top10 corpus indices, shared field masks. Query and corpus order saved alongside. Progress is in status.json. Completion is confirmed only by completion.json.']
            (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['baseline','memory','field','all'],default='all');args=ap.parse_args()
    e=Experiment();e.run_condition('memory',0.,SEEDS[0])
    if args.stage in ['memory','all']:
        for rate in RATES[1:]:
            for seed in (SEEDS[:1] if rate==1 else SEEDS):e.run_condition('memory',rate,seed)
    if args.stage in ['field','all']:
        for rate in RATES[1:]:
            for seed in (SEEDS[:1] if rate==1 else SEEDS):e.run_condition('field',rate,seed)
    log('Requested stages finished; statistical report pending')

if __name__=='__main__':main()
