"""Offline validator; does not call the service's readiness verifier."""
import bisect
import support as S

def lines(path):
    return [S.json.loads(x) for x in S.Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def pct(values,p=95):
    return float(S.np.percentile(values,p)) if values else None

def clocks(ctx,rs,subs,recv,emitted):
    errors=[];count=ctx['operations']
    for label,items,key in [('requests',rs,'event_id'),('submissions',subs,'event_id'),('receiver',recv,'event_id'),('emitter',emitted,'ordinal')]:
        if sorted(x[key] for x in items)!=list(range(count)):errors.append(label+' missing/duplicate event IDs')
    maps=[{x[k]:x for x in items} for items,k in ((subs,'event_id'),(recv,'event_id'),(emitted,'ordinal'))]
    for row in rs:
        i=row['event_id'];sub=maps[0].get(i,{});rec=maps[1].get(i,{});em=maps[2].get(i,{})
        for k in ('planned_arrival_ns','emit_ns','actual_enqueue_ns'):
            if row.get(k)!=sub.get(k) or row.get(k)!=rec.get(k):errors.append(f'{i} independent ingress mismatch: {k}')
        if row.get('planned_arrival_ns')!=em.get('scheduled_ns') or row.get('emit_ns')!=em.get('emit_ns'):errors.append(f'{i} emitter mismatch')
        if row.get('execution_dispatch_ns')!=sub.get('execution_dispatch_ns'):errors.append(f'{i} dispatch mismatch')
        ts=[row.get(k) for k in ('planned_arrival_ns','emit_ns','actual_enqueue_ns','execution_dispatch_ns','worker_start_ns','response_finish_ns')]
        if not all(isinstance(t,int) for t in ts) or ts!=sorted(t for t in ts if isinstance(t,int)):errors.append(f'{i} impossible clock order')
        if row.get('absolute_deadline_ns')!=ctx['absolute_deadline_ns']:errors.append(f'{i} changed deadline')
        if row.get('response_finish_ns') is not None and row.get('completed_by_deadline')!=(row['response_finish_ns']<=ctx['absolute_deadline_ns']):errors.append(f'{i} wrong deadline label')
    return errors

def checkpoint_matches(obs,wanted):
    graph=obs.get('graph')
    fields={
        'graph':graph is not None and S.digest(graph)==wanted['graph_hash'],
        'projection':obs.get('projection') is not None and S.hashlib.sha256(obs['projection'].encode()).hexdigest()==wanted['projection_hash'],
        'sparse_text':S.hashlib.sha256(obs.get('sparse_document','').encode()).hexdigest()==wanted['projection_hash'],
        'sparse_version':obs.get('sparse_version')==wanted['version'],
        'dense_version':obs.get('dense_version')==wanted['version'],
        'dense_vector':obs.get('dense_vector_sha256')==wanted['dense_hash'],
        'candidate':obs.get('candidate_present') is True,
        'relations':all(present==(rel in wanted['relations']) for rel,present in obs.get('relation_memberships',{}).items()) and set(wanted['relations']).issubset(obs.get('relation_memberships',{})),
    }
    return fields

def overlap(updates,begin,end):
    total=0;pairs=0;max_global=0;active=0;events=[]
    groups=S.collections.defaultdict(list)
    for r in updates:
        a=r.get(begin);b=r.get(end)
        if a is None:continue
        if b is None:b=r['service_finish_ns']
        groups[(r['scope'],r['memory_id'])].append((a,b,r['update_index']))
        events.extend(((a,1),(b,-1)))
    for vs in groups.values():
        touched=set()
        for i,(a,b,idx) in enumerate(vs):
            for c,d,j in vs[i+1:]:
                if a<d and c<b:pairs+=1;touched.update((idx,j))
        total+=len(touched)
    for _,sign in sorted(events):active+=sign;max_global=max(max_global,active)
    return dict(updates_in_same_key_overlap=total,same_key_pairs=pairs,max_simultaneous_all_keys=max_global)

def validate_point(dest):
    dest=S.Path(dest);point=S.load(dest/'point.json');trace=S.load(dest/'trace.json')
    oracle=S.load(S.RUN/'oracle.json')[point['trace']]
    id_scopes={r.memory_id:r.scope_id for r in S.expand(S.build_graph_records())}
    errors=[];all_rows=[];phases={}
    for phase in (['measured'] if point['smoke'] else ['warmup','measured']):
        d=dest/phase;ctx=S.load(d/'context.json');rs=lines(d/'requests.jsonl');subs=lines(d/'submissions.jsonl')
        rec=lines(d/'receiver.jsonl');em=lines(d/'emitter.jsonl')
        errors.extend(f'{phase}: {x}' for x in clocks(ctx,rs,subs,rec,em))
        lag=[(r['actual_enqueue_ns']-r['planned_arrival_ns'])/1e6 for r in rec]
        load_ok=bool(lag) and pct(lag)<=50 and max(r['actual_enqueue_ns'] for r in rec)<=ctx['injection_end_ns']+1_000_000_000
        if not load_ok:errors.append(phase+': offered-load arrival invalid')
        for r in rs:
            i=ctx['update_offset']+r['event_id']//20
            if i>=len(trace):errors.append('trace index overflow');continue
            t=trace[i]
            for k in ('scope','memory_id','version','query_id'):
                if r.get(k)!=t[k]:errors.append(f'{phase}:{r["event_id"]} trace {k} mismatch')
            if r['op']!=('update' if r['event_id']%20==0 else 'read'):errors.append('operation-ratio mismatch')
            if r['ready']:
                if r['op']!='update' or r.get('outcome')!='applied':errors.append('false ready outcome')
                for stage in ('raw','structured','canonical_check','sparse','dense','ready'):
                    a,b=r.get(stage+'_start_ns'),r.get(stage+'_done_ns')
                    if not isinstance(a,int) or not isinstance(b,int) or a>b:errors.append('missing/reversed successful stage '+stage)
            hooks=r.get('service_hooks',[])
            ns=[h['ns'] for h in hooks]
            if ns!=sorted(ns):errors.append('stage hook order')
        phases[phase]=dict(receive_lag_p95_ms=pct(lag),offered_load_valid=load_ok,operations=len(rs),
                          receiver_to_dispatch_p95_ms=pct([(r['execution_dispatch_ns']-r['actual_enqueue_ns'])/1e6 for r in rs]),
                          completed_by_deadline=sum(r['completed_by_deadline'] for r in rs))
        all_rows.extend(rs)
    measured=[r for r in all_rows if r['phase']=='measured']
    us=[r for r in measured if r['op']=='update'];reads=[r for r in measured if r['op']=='read']
    all_updates=[r for r in all_rows if r['op']=='update']
    good=[r for r in us if r['ready']]
    ready_times=S.collections.defaultdict(list)
    for r in all_updates:
        if r['ready']:ready_times[(r['scope'],r['memory_id'])].append((r['ready_done_ns'],r['version']))
    frontier={}
    for key,items in ready_times.items():
        times=[];values=[];highest=1
        for ns,v in sorted(items):times.append(ns);highest=max(highest,v);values.append(highest)
        frontier[key]=(times,values)
    def latest(key,ns):
        if key not in frontier:return None
        ts,vs=frontier[key];i=bisect.bisect_right(ts,ns)-1
        return vs[i] if i>=0 else None
    content_mismatches=0;mixed=0;requested_lag=0;frontier_eligible=0;frontier_stale=0
    candidate_eligible=0;candidate_stale=0;leaks=0;candidate_leaks=0
    for r in reads:
        if r['final_status']!='ok':continue
        mid=r['memory_id'];sv=r['target_sparse_version'];dv=r['target_dense_version']
        os=oracle[mid].get(str(sv));od=oracle[mid].get(str(dv))
        if not os or not od or r['target_document_hash']!=os['projection_hash'] or r['target_vector_hash']!=od['dense_hash']:content_mismatches+=1
        mixed+=sv!=dv;requested_lag+=sv<r['version'] or dv<r['version']
        f=latest((r['scope'],mid),r['snapshot_ns'])
        if f is not None:frontier_eligible+=1;frontier_stale+=sv<f or dv<f
        for item in r['returned_versions']:
            m=item['memory_id'];leaks+=id_scopes.get(m)!=r['scope']
            v=latest((r['scope'],m),r['snapshot_ns'])
            if v is not None:candidate_eligible+=1;candidate_stale+=item['sparse']<v or item['dense']<v
        for m in set(r['dense_candidates']+r['sparse_candidates']):candidate_leaks+=id_scopes.get(m)!=r['scope']
        if len(r['returned_memory_ids'])!=len(set(r['returned_memory_ids'])):errors.append('duplicate returned ID')
        if set(r['returned_memory_ids'])-set(r['dense_candidates']+r['sparse_candidates']):errors.append('Top10 outside candidate union')
    checkpoints=S.load(dest/'checkpoint.json');final_failures=[]
    latest_offered={t['memory_id']:t['version'] for t in trace}
    if sorted(x['memory_id'] for x in checkpoints)!=sorted(latest_offered):errors.append('final checkpoint coverage mismatch')
    for x in checkpoints:
        m=x['memory_id'];v=latest_offered[m]
        required={r for k,values in oracle[m].items() if int(k)<=v for r in values['relations']}
        if set(x['observation']['relation_memberships'])!=required:errors.append('incomplete final relation-key audit')
        fields=checkpoint_matches(x['observation'],oracle[m][str(v)])
        if not all(fields.values()):final_failures.append(dict(memory_id=m,expected_version=v,fields=fields,raw_version=(x['observation'].get('graph') or {}).get('memory',{}).get('version')))
    summary=dict(point_id=point['id'],arm=point['arm'],pattern=point['pattern'],seed=point['seed'],smoke=point['smoke'],
                 status='FAIL' if errors else 'PASS',errors=errors,phases=phases,updates=len(us),reads=len(reads),ready_updates=len(good),
                 ready_by_deadline=sum(r['ready_done_ns']<=r['absolute_deadline_ns'] for r in good),
                 outcomes=dict(S.collections.Counter(r.get('outcome') for r in us)),execution_errors=sum(r['final_status']=='error' for r in all_rows),
                 final_keys=len(checkpoints),final_mismatches=len(final_failures),final_failures=final_failures,
                 snapshot_content_mismatches=content_mismatches,mixed_target_snapshots=mixed,requested_target_lag_snapshots=requested_lag,
                 target_post_ready_eligible=frontier_eligible,target_post_ready_stale=frontier_stale,
                 returned_post_ready_eligible=candidate_eligible,returned_post_ready_stale=candidate_stale,
                 returned_cross_scope_leaks=leaks,candidate_cross_scope_leaks=candidate_leaks,
                 service_overlap=overlap(us,'service_start_ns','service_finish_ns'),mutation_overlap=overlap(us,'raw_start_ns','dense_done_ns'))
    def spans(rows,a,b):return [(r[b]-r[a])/1e6 for r in rows if r.get(a) is not None and r.get(b) is not None]
    measurements={
        'arrival_to_ready':spans(good,'actual_enqueue_ns','ready_done_ns'),
        'planned_to_ready':spans(good,'planned_arrival_ns','ready_done_ns'),
        'read_arrival_to_query':spans(reads,'actual_enqueue_ns','query_finish_ns'),
        'read_arrival_to_response':spans(reads,'actual_enqueue_ns','response_finish_ns'),
        'read_queue':spans(reads,'actual_enqueue_ns','worker_start_ns'),
        'update_queue':spans(us,'actual_enqueue_ns','worker_start_ns'),
        'id_lock_wait':spans(us,'id_lock_wait_start_ns','id_lock_acquired_ns'),
        'id_lock_hold':spans(us,'id_lock_acquired_ns','id_lock_released_ns'),
        'raw_write':spans(us,'raw_start_ns','raw_done_ns'),
        'structured':spans(us,'structured_start_ns','structured_done_ns'),
        'sparse_update':spans(us,'sparse_start_ns','sparse_done_ns'),
        'dense_update':spans(us,'dense_start_ns','dense_done_ns'),
        'ready_checks':spans(us,'ready_start_ns','ready_done_ns')}
    summary['latencies_ms']={name:dict(n=len(xs),p50=pct(xs,50),p95=pct(xs,95),p99=pct(xs,99)) for name,xs in measurements.items()}
    ctx=S.load(dest/'measured/context.json');seconds=(ctx['injection_end_ns']-ctx['planned_start_ns'])/1e9
    summary['responses_during_injection_per_s']=sum(ctx['planned_start_ns']<=r['response_finish_ns']<=ctx['injection_end_ns'] for r in measured)/seconds
    summary['responses_by_deadline']=sum(r['completed_by_deadline'] for r in measured)
    summary['readiness_denominator_note']='Percentiles condition on a confirmed-ready target version; counts and all other outcomes retained. This is not saturation capacity.'
    return summary

def summarize():
    results=[]
    for p in S.load(S.RUN/'formal_matrix.json'):
        path=S.RUN/'points'/p['id']/'validate.json'
        if path.exists():results.append(S.load(path))
    S.save(S.RUN/'metrics_summary.json',results)
    flat=[]
    for r in results:
        row={k:r[k] for k in ('point_id','arm','pattern','seed','status','updates','reads','ready_updates','ready_by_deadline','final_keys','final_mismatches','target_post_ready_stale','responses_during_injection_per_s','responses_by_deadline')}
        row.update(same_key_service_overlap_updates=r['service_overlap']['updates_in_same_key_overlap'],same_key_mutation_overlap_updates=r['mutation_overlap']['updates_in_same_key_overlap'])
        for name,stats in r['latencies_ms'].items():
            for q,v in stats.items():row[f'{name}_{q}']=v
        flat.append(row)
    if flat:S.write_csv(S.RUN/'per_run_summary.csv',flat)
    text=['# Ordinary concurrent-load serialization experiment','',f'Completed validated point files: {len(results)}/18. See protocol.json for the frozen design.','',
          'Question: cost and observed safety of per-key serialization under a fixed ordinary open-loop load, without injected stalls.',
          '', 'Setup: Cassandra materialized, 100K memories / 171 scopes, 1 update/s + 19 reads/s, 32 workers; 60 s warmup, 120 s injection, 120 s absolute drain; two mechanisms, three key patterns, three seeds.',
          '', '|Point|Ready / offered|Final mismatches|Ready p95 ms|Read query p95 ms|Same-key service overlaps|', '|---|---:|---:|---:|---:|---:|']
    for r in results:
        a=r['latencies_ms']['arrival_to_ready']['p95'];b=r['latencies_ms']['read_arrival_to_query']['p95']
        text.append(f"|{r['point_id']}|{r['ready_updates']}/{r['updates']}|{r['final_mismatches']}|{a if a is None else round(a,3)}|{b if b is None else round(b,3)}|{r['service_overlap']['updates_in_same_key_overlap']}|")
    text.extend(['','Interpretation boundaries: no observed overlap is a workload observation, not proof of race freedom. Failed/rejected/superseded updates are not counted as ready. Read snapshot diagnostics do not prove atomic backend snapshots. No multi-machine or saturation-throughput claim. All attempts, raw rows, deadlines and source hashes must be retained.'])
    (S.RUN/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

if __name__=='__main__':summarize()
