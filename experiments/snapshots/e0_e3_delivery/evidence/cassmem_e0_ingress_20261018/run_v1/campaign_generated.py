def campaign(out):
    manifest=load(out/'source_manifest.json');verify(manifest)
    p=load(out/'protocol.json');matrix=load(out/'formal_matrix.json');traces=load(out/'formal_update_traces.json')
    assert phase_source()==(out/'phase_generated.py').read_text(encoding='utf-8')
    def status(stage,**kw):
        save(out/'progress.json',dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),**kw));print(stage,kw,flush=True)
    source=build_graph_records();initial=expand(source);assert len(initial)==100000
    assert all(r.version==1 for r in initial)
    original={r.memory_id:r for r in initial};sourceby={r.memory_id:r for r in source};mv=vectors('memory');qv=vectors('qa');qs={q['query_id']:q for q in rows(ROOT/'data/frozen_retrieval/p3_queries.csv')}
    fixtures={}
    for key,trace in traces.items():
        fs=[]
        for t in trace:
            old=original[t['memory_id']];new=dataclasses.replace(sourceby[t['source_memory_id']],memory_id=old.memory_id,scope_id=old.scope_id,version=2)
            assert old.scope_id==t['scope'];q=qs[t['query_id']]
            fs.append(dict(memory_id=old.memory_id,scope=old.scope_id,versions=[version(old,mv[old.memory_id.split('::')[-1]]),version(new,mv[t['source_memory_id']])],query=dict(query_id=t['query_id'],question=q['question'],vector=qv[t['query_id']].tolist())))
        fixtures[key]=fs
    save(out/'resolved_fixtures.json',fixtures);save(out/'resolved_fixture_sha.json',{'sha256':sha(out/'resolved_fixtures.json')})
    state=suspend_old(out)
    try:
        for engine in ('cassandra',):
            b,name=start(engine,out)
            try:
                for cell in [c for c in p['cells'] if c.startswith(engine)]:
                    # The only reset target is this newly created isolated container.
                    b.reset();status('loading_100k',layout=cell,loaded=0)
                    if engine=='cassandra':
                        with ThreadPoolExecutor(32) as pool:
                            for n,_ in enumerate(pool.map(lambda r:b.insert(cell,r),initial),1):
                                if n%5000==0:status('loading_100k',layout=cell,loaded=n)
                    else:
                        for offset in range(0,len(initial),5000):
                            load_scope_safe(b,cell,initial[offset:offset+5000],out/(cell+'_loader_retries.jsonl'),workers=12);status('loading_100k',layout=cell,loaded=offset+5000)
                    grouped=collections.defaultdict(list)
                    for r in initial:grouped[r.scope_id].append(r.memory_id)
                    assert all(sorted(b.ids(cell,s))==sorted(ids) for s,ids in grouped.items())
                    save(out/(cell+'_load_gate.json'),dict(memories=100000,scopes=len(grouped),all_scope_ids_match=True))
                    # Formal adapter integration gate at full loaded scope sizes; excluded from 36 points.
                    gate=out/(cell+'_integration_smoke');gate.mkdir()
                    gatefs=fixtures['0.25_20261015'][:2]
                    gtrial=AuditTrial(b,cell,initial,mv,gatefs,gate,{'active':{'rate':1,'drain_s':10}},cell+'-integration')
                    save(gate/'fixtures.json',gatefs);status('integration_smoke',layout=cell)
                    gtrial.phase('measured',2,0)
                    from validator import audit_phase,lines
                    gd=gate/'measured';gcheck=audit_phase(load(gd/'context.json'),lines(gd/'events.jsonl')+lines(gd/'reads.jsonl'),lines(gd/'submissions.jsonl'),gatefs)
                    gcheck['errors']+=ingress_check(load(gd/'context.json'),lines(gd/'events.jsonl')+lines(gd/'reads.jsonl'),lines(gd/'submissions.jsonl'),lines(gd/'receiver.jsonl'),lines(gd/'emitter.jsonl'))
                    save(gate/'validate.json',gcheck)
                    if gcheck['errors'] or gcheck['execution_errors'] or not gcheck['all_updates_eventually_ready']:raise RuntimeError('Formal adapter integration smoke failed; evidence retained')
                    for f in gatefs:
                        r=original[f['memory_id']];b.commit_raw(cell,r);b.write_structured(cell,r)
                        assert b.fetch_graph(cell,r.scope_id,r.memory_id)==r.graph_projection()
                    del gtrial;gc.collect()
                    offgate=out/(cell+'_integration_smoke_read_audit_off');offgate.mkdir()
                    offtrial=AuditTrial(b,cell,initial,mv,gatefs,offgate,{'active':{'rate':1,'drain_s':10}},cell+'-integration__read_audit_off')
                    save(offgate/'fixtures.json',gatefs)
                    offtrial.phase('measured',2,0)
                    od=offgate/'measured';oc=audit_phase(load(od/'context.json'),lines(od/'events.jsonl')+lines(od/'reads.jsonl'),lines(od/'submissions.jsonl'),gatefs)
                    assert all(r.get('read_audit_enabled') is False and 'audit' not in r for r in lines(od/'reads.jsonl'))
                    oc['errors']+=ingress_check(load(od/'context.json'),lines(od/'events.jsonl')+lines(od/'reads.jsonl'),lines(od/'submissions.jsonl'),lines(od/'receiver.jsonl'),lines(od/'emitter.jsonl'))
                    save(offgate/'validate.json',oc)
                    if oc['errors'] or oc['execution_errors'] or not oc['all_updates_eventually_ready']:raise RuntimeError('Off-arm integration smoke failed')
                    for f in gatefs:
                        rr=original[f['memory_id']];b.commit_raw(cell,rr);b.write_structured(cell,rr)
                        assert b.fetch_graph(cell,rr.scope_id,rr.memory_id)==rr.graph_projection()
                    del offtrial;gc.collect()
                    touched=[]
                    for point in [x for x in matrix if x['layout']==cell]:
                        verify(manifest)
                        for r in touched:
                            b.commit_raw(cell,r);b.write_structured(cell,r)
                            assert b.fetch_graph(cell,r.scope_id,r.memory_id)==r.graph_projection()
                            assert b.candidate_projection(cell,r.scope_id,r.memory_id)==r.rawerk
                        run_id=f"{cell}_r{point['rate']}_s{point['seed']}__{point['audit_arm']}";dest=out/run_id;dest.mkdir(exist_ok=False)
                        active={**p,'active':dict(rate=point['rate'],drain_s=120)};fs=fixtures[point['trace']]
                        save(dest/'point.json',point);save(dest/'fixtures.json',fs)
                        status('initializing_point',run_id=run_id)
                        trial=AuditTrial(b,cell,initial,mv,fs,dest,active,run_id)
                        touched=[original[f['memory_id']] for f in fs]
                        samples=[];gcs=[];gstart={};halt=threading.Event();proc=psutil.Process();proc.cpu_percent();psutil.cpu_percent()
                        def gccb(phase,info):
                            key=threading.get_ident()
                            if phase=='start':gstart[key]=time.perf_counter_ns()
                            else:gcs.append(dict(start_ns=gstart.pop(key,None),end_ns=time.perf_counter_ns(),generation=info['generation'],collected=info['collected']))
                        def sample():
                            while not halt.is_set():
                                t=time.perf_counter_ns()
                                try:container=json.loads(docker('stats','--no-stream','--format','{{json .}}',name))
                                except Exception:container=None
                                samples.append(dict(start_ns=t,end_ns=time.perf_counter_ns(),host_cpu_percent=psutil.cpu_percent(),host_ram_available=psutil.virtual_memory().available,client_cpu_percent=proc.cpu_percent(),client_rss=proc.memory_info().rss,container=container));halt.wait(1)
                        monitor=threading.Thread(target=sample);monitor.start();gc.callbacks.append(gccb)
                        try:
                            active['active']['drain_s']=0
                            status('warmup',run_id=run_id);trial.phase('warmup',60,0)
                            active['active']['drain_s']=120
                            status('measured',run_id=run_id);trial.phase('measured',120,int(60*point['rate']))
                            save(dest/'checkpoint.json',[dict(fixture_id=i,audit=trial.capture(i)) for i in range(len(fs))])
                        finally:
                            gc.callbacks.remove(gccb);halt.set();monitor.join();S.logrows(dest/'resource_usage.jsonl',samples);S.logrows(dest/'gc.jsonl',gcs)
                        from validator import validate_point
                        result=validate_point(dest);save(dest/'validate.json',result);status('point_complete',run_id=run_id,result=result['status'])
                        del trial;gc.collect()
            finally:stop(b,name)
        verify(manifest)
        from validator import summarize
        summarize(out);save(out/'complete.json',dict(matrix_executed=True,formal=True,points=12))
        status('complete')
    finally:restore(state,out)
