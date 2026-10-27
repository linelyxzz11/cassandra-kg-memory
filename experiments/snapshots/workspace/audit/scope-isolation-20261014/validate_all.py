"""Independent, offline validation; preserves original experiment artifacts."""
import json,csv,math,hashlib,itertools
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parent
REPO=Path('D:/memorytable/cassandra-kg-memory')
errors=[];checks=Counter();hashchecks=[]
def require(ok,label):
    checks[label]+=1
    if not ok:errors.append(label)
def read(n):return json.loads((ROOT/n).read_text(encoding='utf-8'))
def stream(n):
    with (ROOT/n).open(encoding='utf-8') as f:
        for line in f:yield json.loads(line)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def metric(g,ids):
    rr=[i for i,x in enumerate(ids[:10],1) if x in g];first=rr[0] if rr else None
    return {'MRR@10':1/first if first else 0.,'Hit@1':float(first==1),'Hit@5':float(first is not None and first<=5),'Hit@10':float(bool(rr)),
        'Recall@10':len(set(ids[:10])&g)/len(g),'nDCG@10':sum(1/math.log2(i+1) for i in rr)/sum(1/math.log2(i+1) for i in range(1,min(10,len(g))+1))}
for name,key in [('initial_protocol.json','source_hashes'),('stress_protocol.json','hashes')]:
    for p,h in read(name)[key].items():
        match=sha(Path(p))==h;hashchecks.append(dict(path=p,match=match));require(match,'source_hash')
bprotocol=read('backend_retry1/protocol.json')
require(sha(REPO/'src/cassmem/backend/live_cells_graph.py')==bprotocol['source_hash'],'backend_source_hash')
with (REPO/'data/retrieval_gold/locomo_cat1_4_gold_memory.csv').open(encoding='utf-8-sig') as f:
    gold={r['query_id']:set(r['gold_memory_ids'].split(';')) for r in csv.DictReader(f) if r['split']=='test' and r['gold_status']=='mapped' and r['gold_memory_ids']}
require(len(gold)==1146,'gold_count')
baseline={};initialkeys=set();counts=Counter();metricmax=0.;scoremax=0.;totalcandidates=0;foreign=0
for r in stream('initial_observations.jsonl'):
    key=(r['condition'],r['order'],r['query_id']);require(key not in initialkeys,'initial_unique');initialkeys.add(key)
    if r['condition']=='S0':baseline[r['query_id']]=r
require(set(baseline)==set(gold),'baseline_coverage')
scopes=sorted({r['scope'] for r in baseline.values()});scopeqs={s:{q for q,r in baseline.items() if r['scope']==s} for s in scopes}
expected={(cond,order,q) for q in gold for cond,order in [('S0','target_only'),('S1','target_first'),('S1','external_first'),('S4','target_first'),('S4','external_first')]}
require(initialkeys==expected,'initial_grid')
stresskeys=Counter();readintervals=defaultdict(list)
def verify(r,is_stress):
    global metricmax,scoremax,totalcandidates,foreign
    counts[r['condition']]+=1;require(r.get('ok',False),'record_success')
    if 'result' not in r:return
    ref=baseline[r['query_id']]['result'];got=r['result']
    for branch in ('top10','dense_top50','sparse_top50'):
        ids=[m for m,v in got[branch]];require(len(ids)==len(set(ids)),'ranking_unique')
        require(ids==[m for m,v in ref[branch]],'ranking_exact')
        old=dict(ref[branch]);diff=max([abs(v-old.get(m,v)) for m,v in got[branch]] or [0.]);scoremax=max(scoremax,diff)
        require(diff<=1e-7,'score_tolerance')
        if branch!='top10':totalcandidates+=len(ids)
    expecttrace=[['sparse',r['scope']],['dense',r['scope']]]
    require(r.get('scope_trace' if is_stress else 'trace')==expecttrace,'actual_snapshot_provenance')
    require(r['content_version_ok'],'content_version')
    leak=r.get('foreign_candidates',0);foreign+=leak;require(leak==0,'foreign_candidates_zero')
    computed=metric(gold[r['query_id']],[m for m,v in got['top10']]);bm=baseline[r['query_id']]['metrics']
    for k,v in computed.items():
        require(abs(v-r['metrics'][k])<1e-12,'metric_recalculation');metricmax=max(metricmax,abs(v-bm[k]));require(abs(v-bm[k])<1e-12,'metric_invariance')
for r in stream('initial_observations.jsonl'):verify(r,False)
for r in stream('stress_observations.jsonl'):
    verify(r,True)
    if r['condition'] in ('S2','S3'):key=(r['condition'],r['seed'],r['multiplier'],r['order'],r['query_id'])
    else:
        key=(r['condition'],r['scope'],r['round'],r['query_id'])
        if r['condition']=='S5-concurrent':readintervals[(r['scope'],r['round'])].append((r['start_ns'],r['end_ns']))
    stresskeys[key]+=1
expectedstress=set()
for cond,seed,mult,order,q in itertools.product(('S2','S3'),(20261014,20261015,20261016),(1,5,10),('target_first','external_first'),gold):expectedstress.add((cond,seed,mult,order,q))
actualstress={k for k in stresskeys if k[0] in ('S2','S3')}
require(actualstress==expectedstress,'stress_grid');require(all(v==1 for v in stresskeys.values()),'stress_unique')
for cond in ('S5-interleaved','S5-concurrent'):
    for s in scopes:
        for rnd in range(10):
            require(sum(v for k,v in stresskeys.items() if k[0]==cond and k[1]==s and k[2]==rnd)==5,'S5_reads_coverage')
            require(sum(v for k,v in stresskeys.items() if k[0]==cond+'-post' and k[1]==s and k[2]==rnd)==1,'S5_post_coverage')
initialsummary=read('initial_summary.json');stresssummary=read('stress_summary.json')
require(len(initialsummary)==40,'initial_summary_count');require(len(stresssummary)==456,'stress_summary_count')
for r in initialsummary+stresssummary:
    require(r.get('state_unchanged',True),'state_invariance');require(r.get('failed_queries',r.get('failures',0))==0,'summary_zero_failure')
baseline_states={r['scope']:r['state'] for r in initialsummary if r['condition']=='S0'}
for r in initialsummary:require(r['state']==baseline_states[r['scope']],'initial_state_hashes_exact')
aoverlap=0
for r in stresssummary:
    if r['condition'].startswith('S5-'):
        require(r['update']['accepted'] and r['update']['visible'],'external_update_verified')
    if r['condition']=='S5-concurrent':
        u=r['update'];n=sum(a<u['end_ns'] and b>u['start_ns'] for a,b in readintervals[(r['scope'],r['round'])]);require(n==r['reads_overlapping_update'],'A_overlap_recomputed');aoverlap+=n
faults=[r for r in stresssummary if r['condition']=='fault_control']
require({r['scope'] for r in faults}==set(scopes) and all(r['detected'] and r['leaked_queries']>0 for r in faults),'fault_control_coverage')
# Backend: recompute expected target graph and relation results independently of saved flags.
def expectedgraph(scope,mid):
    i=int(mid[1:]);raw=f'Alice lives_in Paris. Record {i}, version 1.'
    rawerk=raw+'\nE: Alice, Paris\nR: lives_in\nK: city'
    return dict(memory=dict(scope_id=scope,memory_id=mid,version=1,raw_text=raw,entities='Alice, Paris',relations='lives_in',keywords='city',triples='(Alice, lives_in, Paris)',rawerk=rawerk,embedding_sha256=hashlib.sha256(raw.encode()).hexdigest()),mentions=['Alice','Paris'],graph_entities=['Alice','Paris'],edges=[dict(ordinal=0,src='Alice',relation='lives_in',dst='Paris')])
brows=list(stream('backend_retry1/observations.jsonl'));bcounts=Counter();bkeys=Counter();graphchecks=0
for r in brows:
    bcounts[r['cell']]+=1;bkeys[(r['cell'],r['scope'],r['phase'],r['round'])]+=1
    require(r.get('ok',False),'B_record_success');ids=[f'm{i:03d}' for i in range(24)]
    require(r['actual_ids']==ids,'B_scope_ids')
    require(r['relations']=={'lives_in':ids,'visited':[],'works_in':[]},'B_relation_isolation')
    require(len(r['graphs'])==24 and len(r['projections'])==24,'B_payload_coverage')
    for g in r['graphs']:
        require(g['actual']==expectedgraph(r['scope'],g['memory_id']),'B_graph_exact');graphchecks+=1
    for p in r['projections']:require(p['actual']==expectedgraph(r['scope'],p['memory_id'])['memory']['rawerk'],'B_projection_exact')
cells=bprotocol['cells'];runid=bprotocol['runid'];expectedb={}
for cell,order in itertools.product(cells,('A_first','B_first')):
    scope=runid+'_'+cell+'_'+order+'_A'
    for phase in ('after_A_insert','after_both_insert','after_B_queries'):expectedb[(cell,scope,phase,-1)]=1
    for mode,rnd in itertools.product(('interleaved','concurrent'),range(10)):
        expectedb[(cell,scope,mode,rnd)]=3;expectedb[(cell,scope,mode+'_post',rnd)]=1
require(dict(bkeys)==expectedb,'B_complete_grid')
require(len({(r['cell'],r['scope'],r['phase'],r['round'],r['start_ns']) for r in brows})==len(brows),'B_unique_observations')
events=read('backend_retry1/update_events.json');ekeys={(e['cell'],e['order'],e['mode'],e['round']) for e in events}
require(len(events)==120 and ekeys==set(itertools.product(cells,('A_first','B_first'),('interleaved','concurrent'),range(10))),'B_update_grid')
boverlap=0
for e in events:
    w=e['writer'];require(w['external_graph_equal'],'B_external_update_verified')
    external_scope=runid+'_'+e['cell']+'_'+e['order']+'_B'
    raw=f"Alice works in London, update {e['mode']} {e['round']}."
    mem=dict(scope_id=external_scope,memory_id=f"m{e['round']:03d}",version=8 if e['mode']=='interleaved' else 9,raw_text=raw,
        entities='Alice, London',relations='works_in',keywords='employment',triples='(Alice, works_in, London)',
        rawerk=raw+'\nE: Alice, London\nR: works_in\nK: employment',embedding_sha256=hashlib.sha256(raw.encode()).hexdigest())
    expected_external=dict(memory=mem,mentions=['Alice','London'],graph_entities=['Alice','London'],edges=[dict(ordinal=0,src='Alice',relation='works_in',dst='London')])
    require(w['actual']==expected_external,'B_external_graph_recomputed')
    scope=runid+'_'+e['cell']+'_'+e['order']+'_A'
    rr=[r for r in brows if r['cell']==e['cell'] and r['scope']==scope and r['phase']==e['mode'] and r['round']==e['round']]
    n=sum(r['start_ns']<w['end_ns'] and r['end_ns']>w['start_ns'] for r in rr);require(n==e['overlap_reads'],'B_overlap_recomputed');boverlap+=n
require(aoverlap==read('stress_complete.json')['overlapping_reads'],'A_overlap_total')
require(boverlap==read('backend_retry1/complete.json')['overlapping_reads'],'B_overlap_total')
result=dict(passed=not errors,errors=Counter(errors),check_counts=checks,source_hashes=hashchecks,
    stage_A=dict(unique_queries=1146,counts=counts,total_observations=sum(counts.values()),candidate_occurrences=totalcandidates,foreign_candidates=foreign,max_score_delta=scoremax,max_metric_delta=metricmax,overlapping_reads=aoverlap,fault_controls=len(faults)),
    stage_B=dict(counts=bcounts,observations=len(brows),graph_checks=graphchecks,projection_checks=graphchecks,update_events=len(events),overlapping_reads=boverlap),
    limitations=['A content/version and snapshot-state assertions have recorded flags/hashes; no independent replay of in-memory state after process exit.',
        'Fault control retains per-scope counts, not full fault-control rankings.',
        'B has no frozen run_backend.py hash in original protocol; current adapter hash matches, observations independently checked against target fixture.',
        'No authentication/authorization testing; supplied scope assumed correct; no scope leakage upper-bound inference from correlated repeated tests.'])
(ROOT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','check_counts')},ensure_ascii=False,indent=2))
if errors:raise SystemExit(1)
