"""Final trace/environment attestation; raw protocols and measurements unchanged."""
import json,sys,hashlib,subprocess,copy,csv,importlib.util
from pathlib import Path
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def main(out):
    p=load(out/'protocol.json');verdict=load(out/'validate.json');assert verdict['passed']
    matrix=load(out/'formal_matrix.json');traces=load(out/'formal_update_traces.json');assert len(matrix)==36
    assert len({(x['layout'],x['rate'],x['seed']) for x in matrix})==36
    total=0
    for point in matrix:
        trace=traces[point['trace']];assert len({x['memory_id'] for x in trace})==len(trace)
        measured=[x for x in trace if x['phase']=='measured'];assert len(measured)==int(120*point['rate']);total+=len(measured)
    assert total==2520 and total*19==47880
    # Verify actual container configuration, selecting only non-secret settings.
    sealed={}
    for engine in ('cassandra','neo4j'):
        env=load(out/(engine+'_environment.json'));info=json.loads(subprocess.check_output(['docker','inspect',env['name']],text=True))[0]
        assert info['HostConfig']['Memory']==12*1024**3 and info['HostConfig']['NanoCpus']==8*10**9
        selected={v.split('=',1)[0]:v.split('=',1)[1] for v in info['Config']['Env'] if v.split('=',1)[0] in ('MAX_HEAP_SIZE','HEAP_NEWSIZE','NEO4J_server_memory_heap_initial__size','NEO4J_server_memory_heap_max__size','NEO4J_server_memory_pagecache_size')}
        sealed[engine]=dict(image=info['Image'],memory=info['HostConfig']['Memory'],cpus=info['HostConfig']['NanoCpus'],heap_cache=selected,stopped=not info['State']['Running'])
    save(out/'environment_seal.json',sealed)
    classifications=[];resources=[];gc_events=[]
    for cell in p['cells']:
        for phase in ('warmup','measured'):
            dest=out/cell/phase;ctx=load(dest/'context.json');rows=lines(dest/'events.jsonl')+lines(dest/'reads.jsonl');deadline=load(dest/'deadline.json');end=ctx['absolute_deadline_ns']
            assert len(deadline['states'])==ctx['operations']
            for r in rows:
                status='queued_not_started' if r['worker_start_ns']>end else ('started_unfinished' if r['response_finish_ns']>end else 'completed')
                classifications.append(dict(cell=cell,phase=phase,event_id=r['event_id'],state_at_absolute_deadline=status,eventual_status=r['final_status']))
                if deadline['states'][str(r['event_id'])]=='finished':assert deadline['finish_times'][str(r['event_id'])]==r['response_finish_ns']<=deadline['observed_ns']
        resources += [dict(cell=cell,**r) for r in lines(out/cell/'resource_usage.jsonl')]
        gc_events += [dict(cell=cell,**r) for r in lines(out/cell/'gc.jsonl')]
    save(out/'deadline_classifications.json',classifications);save(out/'resource_usage_all.json',resources);save(out/'gc_all.json',gc_events)
    with (out/'resource-usage.csv').open('w',newline='',encoding='utf-8-sig') as h:
        flat=[{k:(json.dumps(v) if isinstance(v,dict) else v) for k,v in r.items()} for r in resources];w=csv.DictWriter(h,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    # Re-run clock tests and persist evidence; classify late completions separately from load validity.
    root=Path(__file__).resolve().parent
    tests=subprocess.run([sys.executable,str(root/'test_clocks.py')],capture_output=True,text=True);assert tests.returncode==0
    (out/'clock_tests.log').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('e0validator',root/'validate_e0.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    dest=out/p['cells'][0]/'measured';ctx=load(dest/'context.json');rr=lines(dest/'events.jsonl')+lines(dest/'reads.jsonl');ss=lines(dest/'submissions.jsonl')
    rr=copy.deepcopy(rr);rr[-1]['response_finish_ns']=ctx['absolute_deadline_ns']+1
    _,late=v.phase_check(ctx,rr,ss,load(out/'fixtures.json'));assert late['offered_load_valid'] and not late['completed_before_absolute_drain']
    save(out/'acceptance.json',dict(passed=True,formal_matrix_points=36,formal_frozen_updates=2520,formal_frozen_reads=47880,
        smoke_layouts=4,measured_updates=24,measured_reads=456,warmup_updates=8,warmup_reads=152,resource_samples=len(resources),
        negative_tests=len(verdict['negative_tests'])+1,clock_unit_tests=2,formal_started=False,smoke_scale_only=10,
        observability_boundary='Driver pool wait and isolated cleanup unobserved; null, not zero. Complete enclosing client-call/structured timings retained.',
        environment_restored=load(out/'restored.json'),attestation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(load(out/'acceptance.json'),indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
