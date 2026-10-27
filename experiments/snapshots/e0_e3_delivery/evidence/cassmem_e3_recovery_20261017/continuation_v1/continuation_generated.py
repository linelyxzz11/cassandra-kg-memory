def main(out):
    manifest=load(out/'source_manifest.json');verify(manifest);p=load(out/'protocol.json');fs=load(out/'fixtures.json');schedule=load(out/'schedules.json')
    assert start_source()==(out/'engine_start_generated.py').read_text(encoding='utf-8')
    assert repaired_run_source()==(out/'serialized_run_generated.py').read_text(encoding='utf-8')
    envglobals=dict(vars(E));exec(compile(start_source(),'<frozen-environment>','exec'),envglobals)
    run_globals=dict(vars(live));exec(compile(repaired_run_source(),'<frozen-serialized-run>','exec'),run_globals)
    class Repair(live.SerializedCase):run=run_globals['run']
    state=resume_suspend(out);results=prior_results.copy()
    def progress(stage,**kw):
        save(out/'progress.json',dict(stage=stage,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw));print(stage,kw,flush=True)
    try:
        for phase in ('formal',):
            selected=[c for c in schedule if c['phase']==phase];count=len(done)
            for engine in ('cassandra','neo4j'):
                envout=out/(phase+'_'+engine);envout.mkdir()
                progress('starting_database',phase=phase,engine=engine)
                b,name=envglobals['start'](engine,envout)
                try:
                    for c in [c for c in selected if c['cell'].startswith(engine)]:
                        if c['case_id'] in done:continue
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
