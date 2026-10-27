def phase(self,phase,duration,offset):
    rate=self.p['active']['rate'];count=int(duration*rate)*20;rows=[];subs=[];guard=threading.Lock();stages={};done_flags={}
    planned_start=time.perf_counter_ns()+2_000_000_000;injection_end=planned_start+int(duration*1e9);deadline=injection_end+int(self.p['active']['drain_s']*1e9)
    phaseout=self.out/phase;phaseout.mkdir()
    context=dict(phase=phase,planned_start_ns=planned_start,injection_end_ns=injection_end,absolute_deadline_ns=deadline,operations=count,updates=count//20,reads=count-count//20,
        utc_started=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=32,update_rate=rate)
    save(phaseout/'context.json',context)
    def execute(row):
        i=row['fixture_id'];r=self.records[i][2];mid=r.memory_id
        with guard:row['worker_start_ns']=time.perf_counter_ns();stages[row['event_id']]='started'
        LOCAL.row=row
        try:
            if row['op']=='update':
                row['service_hooks']=[]
                def hook(event,stage,ns):
                    row['service_hooks'].append(dict(event=event,stage=stage,ns=ns))
                    if stage=='lock':
                        row[{'waiting':'id_lock_wait_start_ns','acquired':'id_lock_acquired_ns','releasing':'id_lock_released_ns'}[event]]=ns
                    else:
                        row[stage+('_start_ns' if event=='before' else '_done_ns')]=ns
                result=self.service.apply(r,self.vectors[i][2],hook=hook)
                if result['status']!='applied':raise AssertionError('Unexpected revision result')
                row.update(raw_call_start_ns=row['raw_start_ns'],raw_ack_ns=row['raw_done_ns'],ready_check_start_ns=row['ready_start_ns'],ready_check_done_ns=row['ready_done_ns'],
                    relation_cleanup_start_ns=None,relation_cleanup_done_ns=None,structured_maintenance_start_ns=row['structured_start_ns'],structured_maintenance_done_ns=row['structured_done_ns'])
                row['post_ready_audit_start_ns']=time.perf_counter_ns();row['audit']=self.capture(i);row['post_ready_audit_done_ns']=time.perf_counter_ns()
                query=self.fixtures[i]['query'];row['post_update_query_start_ns']=time.perf_counter_ns()
                ranked=self.index.search(self.records[i][2].scope_id,query['question'],query['vector'],row['audit']['ids'])
                row['post_update_query_done_ns']=time.perf_counter_ns();row['post_update_top10']=[m for m,_ in ranked.top10];row['post_update_target_present']=mid in row['post_update_top10']
                row['ready']=True
            else:
                query=self.fixtures[i]['query'];ids=self.b.ids(self.cell,self.records[i][2].scope_id);row['candidate_ids_done_ns']=time.perf_counter_ns()
                with self.index._scope_locks[self.records[i][2].scope_id]:
                    sp,de=self.index._sparse[self.records[i][2].scope_id],self.index._dense[self.records[i][2].scope_id]
                    sv={m:self.index._sparse_versions[(self.records[i][2].scope_id,m)] for m in sp.memory_ids};dv={m:self.index._dense_versions[(self.records[i][2].scope_id,m)] for m in de.memory_ids}
                    row['snapshot_ns']=time.perf_counter_ns()
                t=time.perf_counter_ns();sr=_bm25_search(sp,query['question'],50);row['sparse_scoring_ns']=time.perf_counter_ns()-t
                t=time.perf_counter_ns();dr=_dense_search(de,query['vector'],50);row['dense_scoring_ns']=time.perf_counter_ns()-t
                t=time.perf_counter_ns();ranking=zscore_fuse([x for x in dr if x[0] in ids],[x for x in sr if x[0] in ids]);row['fusion_ns']=time.perf_counter_ns()-t
                row.update(query_id=query['query_id'],query_finish_ns=time.perf_counter_ns(),returned_memory_ids=[m for m,_ in ranking],
                    returned_versions=[dict(memory_id=m,sparse=sv[m],dense=dv[m]) for m,_ in ranking],target_present=mid in {m for m,_ in ranking})
                row['read_audit_enabled']=self.audit_arm=='full'
                if self.audit_arm=='full':
                    row['read_audit_start_ns']=time.perf_counter_ns();row['audit']=self.capture(i);row['read_audit_done_ns']=time.perf_counter_ns()
            row['final_status']='ok'
        except Exception as exc:row.update(final_status='error',exception_type=type(exc).__name__,exception_message=str(exc),traceback=traceback.format_exc())
        finally:
            row['driver_pool_wait_ms']=None
            row['target_content_match']=row.get('audit',{}).get('graph')==r.graph_projection()
            row['raw_version']=row.get('audit',{}).get('graph',{}).get('memory',{}).get('version',-1)
            row['relation_version']=None  # No persisted relation version exists in this adapter.
            if row.get('scope_locks'):
                row['scope_lock_wait_start_ns']=row['scope_locks'][0]['wait_start_ns'];row['scope_lock_acquired_ns']=row['scope_locks'][0]['acquired_ns'];row['scope_lock_released_ns']=row['scope_locks'][-1]['released_ns']
            with guard:row['response_finish_ns']=time.perf_counter_ns();stages[row['event_id']]='finished';done_flags[row['event_id']]=row['response_finish_ns']
            LOCAL.row=None
        return row
    def deadline_watch():
        delay=max(0,(deadline-time.perf_counter_ns())/1e9);time.sleep(delay)
        with guard:snapshot=dict(stages);finishes=dict(done_flags)
        save(phaseout/'deadline.json',dict(absolute_deadline_ns=deadline,observed_ns=time.perf_counter_ns(),states=snapshot,finish_times=finishes))
    watcher=threading.Thread(target=deadline_watch);watcher.start()
    with ThreadPoolExecutor(32) as pool:
        emitter=subprocess.Popen([sys.executable,'-B',EMITTER,'--start-ns',str(planned_start),'--count',str(count),'--ops-rate',str(rate*20),'--log',str(phaseout/'emitter.jsonl')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        futures=[]
        try:
            for line in emitter.stdout:
                ordinal,scheduled,emitted,enqueued=map(int,line.strip().split(','));assert ordinal==len(rows)
                dispatch=time.perf_counter_ns();i=offset+ordinal//20
                row=dict(run_id=self.run_id,layout=self.cell,phase=phase,event_id=ordinal,associated_update_id=(ordinal//20)*20,op='update' if ordinal%20==0 else 'read',fixture_id=i,
                    target_memory_id=self.records[i][2].memory_id,target_version=2,query_scope=self.records[i][2].scope_id,planned_arrival_ns=scheduled,emit_ns=emitted,actual_enqueue_ns=enqueued,execution_dispatch_ns=dispatch,
                    worker_start_ns=None,response_finish_ns=None,absolute_deadline_ns=deadline,ready=False,exception_type=None,exception_message=None)
                with guard:stages[ordinal]='queued'
                rows.append(row);subs.append(dict(event_id=ordinal,planned_arrival_ns=scheduled,emit_ns=emitted,actual_enqueue_ns=enqueued,execution_dispatch_ns=dispatch));futures.append(pool.submit(execute,row))
            if emitter.wait(timeout=15):raise RuntimeError(emitter.stderr.read())
            watcher.join();wait(futures)
            for f in futures:f.result()
        finally:
            if emitter.poll() is None:emitter.terminate();emitter.wait(timeout=10)
            # The finite diagnostic deadline watcher preserves already-offered states on failure.
            watcher.join()
            emitter.stdout.close();emitter.stderr.close()
            for row in rows:row['deadline_class']=classify(row,deadline)
            logrows(phaseout/'events.jsonl',[r for r in rows if r['op']=='update']);logrows(phaseout/'reads.jsonl',[r for r in rows if r['op']=='read']);logrows(phaseout/'submissions.jsonl',subs)
    return rows
