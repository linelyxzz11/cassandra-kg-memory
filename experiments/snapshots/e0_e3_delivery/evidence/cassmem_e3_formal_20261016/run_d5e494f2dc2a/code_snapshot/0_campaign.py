import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'D:/memorytable/cassmem_e3_serialized_20261015')
import live
B=live.base
sys.path.insert(0,'D:/memorytable/cassmem_e0_preflight_20261015')
import importlib.util
_espec=importlib.util.spec_from_file_location('e3_isolated_environment','D:/memorytable/cassmem_e0_preflight_20261015/environment.py')
E=importlib.util.module_from_spec(_espec);_espec.loader.exec_module(E)
sys.path.insert(0,str(HERE))
import validator as V
import json,hashlib,random,inspect,textwrap,shutil,traceback,datetime,platform,subprocess,collections,csv
import psutil
BASE=Path('D:/memorytable/cassmem_e3_preflight_20261015/smoke_f0978e6a51a6')
CELLS=['cassandra-base','cassandra-materialized','neo4j-native','neo4j-materialized']
SCENARIOS={'control':None,'structured_delay':'structured','sparse_delay':'sparse','dense_delay':'dense'}
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):E.save(p,x)
def verify(m):
    for p,h in m.items():
        if sha(p)!=h:raise RuntimeError('Frozen source drift: '+p)
def start_source():
    return textwrap.dedent(inspect.getsource(E.start)).replace('password=secrets.token_urlsafe(24)',"password='E3_'+secrets.token_urlsafe(24)").replace("'cassmem-e0-preflight-'","'cassmem-e3-formal-'")
def repaired_run_source():
    return textwrap.dedent(inspect.getsource(live.SerializedCase.run)).replace("assert self.read('final_'+str(i))","self.read('final_'+str(i))")
def prepare():
    fixtures=load(BASE/'fixtures.json');assert len(fixtures)==50 and len({f['memory_id'] for f in fixtures})==50
    p=dict(experiment='E3-formal-paired-20261016',implementations=['baseline','serialized'],cells=CELLS,scenarios=SCENARIOS,seeds=[20261016,20261017,20261018],fixture_count=50,
       formal_cases=4800,formal_updates=9600,smoke_cases=160,smoke_fixture_ids=list(range(5)),
       reader_interval_ms=20,barrier_timeout_s=60,worker_timeout_s=120,post_final_probes=3,
       resources=dict(cpus=8,memory_gib=12,cassandra_heap='6G',neo4j_heap='5G',neo4j_pagecache='3G',engines_sequential=True),
       scope_size=5,fixture_semantics='Frozen real-memory-derived v1/v2/v3 payloads, not natural revision histories. One target plus four seeded distractors; identical across implementations/layouts.',
       seed_semantics='Seeds determine four background IDs per target and target order; three repeats are not independent datasets.',
       schedule_baseline='Pause v2 before designated stage; prove v3 ready then release old v2 stage. Preserve original guards.',
       schedule_serialized='Pause lock-owning v2; prove v3 waiting; release v2 and allow v3. No claim that overtaking occurred.',
       variant='Single-process per-ID serialized mitigation, not distributed fencing or atomic multi-index publication.',
       metrics='Independent component content/versions, post-v3-ready failure, final regression, sampled first correct and right-censoring, lock wait, query and extra audit. Unknown separately. Read target uses latest submitted at start.',
       gate='All160 smoke harness cases and negative tests must pass before ANY formal case. Baseline treatment violations do not fail harness. Repaired/control violations stop smoke launch.',
       failures='Keep every attempted case. Formal scientific violations retained; infrastructure exceptions halt campaign; no automatic replacement.',
       code={str(x):sha(x) for x in HERE.glob('*.py')})
    out=HERE/('run_'+hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:12]);out.mkdir(exist_ok=False)
    save(out/'protocol.json',p);shutil.copy2(BASE/'fixtures.json',out/'fixtures.json')
    schedules=[]
    for phase in ('smoke','formal'):
        seeds=p['seeds'][:1] if phase=='smoke' else p['seeds']
        targets=list(range(5 if phase=='smoke' else 50))
        for seed in seeds:
            rng=random.Random(seed);ids=targets[:];rng.shuffle(ids)
            bg={i:rng.sample([j for j in range(50) if j!=i],4) for i in ids}
            for cell in CELLS:
                for impl in p['implementations']:
                    for sc,stage in SCENARIOS.items():
                        for i in ids:
                            schedules.append(dict(case_id=f'{phase}_{cell}_{impl}_{sc}_{seed}_{i}',phase=phase,cell=cell,implementation=impl,scenario=sc,seed=seed,fixture_id=i,background_ids=bg[i],blocked_stage=stage))
    assert sum(c['phase']=='formal' for c in schedules)==4800 and sum(c['phase']=='smoke' for c in schedules)==160
    save(out/'schedules.json',schedules)
    (out/'engine_start_generated.py').write_text(start_source(),encoding='utf-8');(out/'serialized_run_generated.py').write_text(repaired_run_source(),encoding='utf-8')
    paths=[*HERE.glob('*.py'),Path(inspect.getfile(B)),Path(inspect.getfile(live)),Path(E.__file__),Path('D:/memorytable/cassmem_e3_preflight_20261015/validate.py'),
       Path('D:/memorytable/concurrent_update_consistency/preflight.py'),Path('D:/memorytable/cassandra-kg-memory/src/cassmem/serving/serialized_updates.py')]
    paths += [Path(x) for x in load(BASE/'source_manifest.json') if '/src/' in x.replace('\\','/')]
    paths += [out/n for n in ('protocol.json','fixtures.json','schedules.json','engine_start_generated.py','serialized_run_generated.py')]
    save(out/'source_manifest.json',{str(x):sha(x) for x in paths})
    (out/'code_snapshot').mkdir()
    for i,x in enumerate(dict.fromkeys(paths)):
        if x.suffix=='.py':shutil.copy2(x,out/'code_snapshot'/f'{i}_{x.name}')
    save(out/'environment.json',dict(python=sys.version,os=platform.platform(),logical_cpus=psutil.cpu_count(),ram=psutil.virtual_memory().total,pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),docker_version=E.docker('version','--format','{{json .Server}}')))
    import test_historical
    save(out/'historical_validator_tests.json',test_historical.main())
    print(out)
def main(out):
    manifest=load(out/'source_manifest.json');verify(manifest);p=load(out/'protocol.json');fs=load(out/'fixtures.json');schedule=load(out/'schedules.json')
    assert start_source()==(out/'engine_start_generated.py').read_text(encoding='utf-8')
    assert repaired_run_source()==(out/'serialized_run_generated.py').read_text(encoding='utf-8')
    envglobals=dict(vars(E));exec(compile(start_source(),'<frozen-environment>','exec'),envglobals)
    run_globals=dict(vars(live));exec(compile(repaired_run_source(),'<frozen-serialized-run>','exec'),run_globals)
    class Repair(live.SerializedCase):run=run_globals['run']
    state=E.suspend_old(out);results=[]
    def progress(stage,**kw):
        save(out/'progress.json',dict(stage=stage,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw));print(stage,kw,flush=True)
    try:
        for phase in ('smoke','formal'):
            selected=[c for c in schedule if c['phase']==phase];count=0
            for engine in ('cassandra','neo4j'):
                envout=out/(phase+'_'+engine);envout.mkdir()
                progress('starting_database',phase=phase,engine=engine)
                b,name=envglobals['start'](engine,envout)
                try:
                    for c in [c for c in selected if c['cell'].startswith(engine)]:
                        dest=out/phase/c['case_id'];dest.mkdir(parents=True,exist_ok=False)
                        casefs=[fs[c['fixture_id']]]+[fs[i] for i in c['background_ids']]
                        save(dest/'fixtures.json',casefs);save(dest/'case.json',c)
                        try:
                            cls=B.Case if c['implementation']=='baseline' else Repair
                            cls(b,c,casefs[0],dest,p).run()
                            res=V.validate_case(dest);save(dest/'validate.json',res)
                            if phase=='smoke':
                                assert res['harness_status']=='PASS',str(res['errors']+res['schedule_errors'])
                                if c['implementation']=='serialized' or c['scenario']=='control':assert not res['final_regression'] and res['post_ready_failures']==0,'smoke correctness gate'
                                if c['fixture_id']==0:save(dest/'negative_tests.json',dict(passed=V.negative_tests(dest)))
                        except Exception:
                            save(dest/'failure.json',dict(error=traceback.format_exc()));raise
                        count+=1;results.append({k:v for k,v in res.items() if k not in ('classified','query_ms','extra_audit_ms','id_lock_wait_ms')})
                        with (out/'case_summaries.jsonl').open('a',encoding='utf-8') as h:h.write(json.dumps(dict(phase=phase,**results[-1]))+'\n')
                        if count%10==0 or count==len(selected):progress('running',phase=phase,completed=count,total=len(selected),last_case=c['case_id'],last_harness=res['harness_status'])
                        if count%100==0:verify(manifest)
                finally:E.stop(b,name)
            verify(manifest);save(out/(phase+'_complete.json'),dict(cases=count,harness_counts=dict(collections.Counter(r['harness_status'] for r in results if r['case_id'].startswith(phase+'_')))))
            progress('phase_complete',phase=phase,cases=count)
        save(out/'complete.json',dict(formal_cases=4800,smoke_cases=160,validation='per-case raw-evidence checks complete',source_hashes_unchanged=True))
        progress('complete',formal_cases=4800)
    finally:E.restore(state,out)
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        out=Path(sys.argv[1])
        try:main(out)
        except Exception:save(out/'failure.json',dict(error=traceback.format_exc(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));raise
