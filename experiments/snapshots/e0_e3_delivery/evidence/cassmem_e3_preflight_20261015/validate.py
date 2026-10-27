"""Independent offline validation: no runner imports, no trust in PASS flags."""
import json,sys,hashlib,csv,copy,collections
from pathlib import Path
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def version(matches):return matches[0] if len(matches)==1 else -1
def assess(r,f,target):
    # Reconstruct expected content independent of live adapter/readiness implementation.
    vs=f['versions'];graphs=[]
    for v in vs:
        g=copy.deepcopy(v['graph']);g['memory']['scope_id']=r['scope'];graphs.append(g)
    g=r['graph'];m=g['memory'] if g else {}
    raw=version([i+1 for i,x in enumerate(graphs) if all(m.get(k)==x['memory'][k] for k in ('raw_text','version','embedding_sha256'))])
    structure=version([i+1 for i,x in enumerate(graphs) if g and all(m.get(k)==x['memory'][k] for k in ('entities','relations','keywords','triples')) and all(g.get(k)==x[k] for k in ('edges','mentions','graph_entities'))])
    sparse=version([i+1 for i,v in enumerate(vs) if r['sparse_document']==v['rawerk'] and r['sparse_version']==i+1])
    dense=version([i+1 for i,v in enumerate(vs) if r['dense_vector_sha256']==v['normalized_vector_sha256'] and r['dense_version']==i+1])
    projection=version([i+1 for i,v in enumerate(vs) if r['projection']==v['rawerk']])
    expected=set(vs[target-1]['required_relations']);rels=r['relation_memberships']
    relation_ok=all(value==(k in expected) for k,value in rels.items()) and expected<=set(rels)
    target_present=f['memory_id'] in r['ids']
    known=[v for v in (raw,structure,sparse,dense,projection) if v>=0]
    strict=g==graphs[target-1] and projection==target and sparse==target and dense==target and relation_ok and target_present
    candidates={x[0] for x in r['sparse_candidates']}|{x[0] for x in r['dense_candidates']}
    stale_candidate=f['memory_id'] in candidates and any(0<=v<target for v in (sparse,dense))
    return dict(strict=bool(strict),mixed=len(set(known))>1,unknown=any(v<0 for v in (raw,structure,sparse,dense,projection)),
       stale_candidate=stale_candidate,stale_relation=any(value and k not in expected for k,value in rels.items()),
       raw_version=raw,structure_content_version=structure,sparse_content_version=sparse,dense_content_version=dense,projection_content_version=projection)

def validate(out):
    errors=[];check=lambda ok,msg:errors.append(msg) if not ok else None
    p=load(out/'protocol.json');fixtures=load(out/'fixtures.json');cases=load(out/'schedules.json');events=lines(out/'timeline.jsonl');reads=lines(out/'reads.jsonl')
    manifest=load(out/'source_manifest.json')
    for path,h in manifest.items():check(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,'hash drift: '+path)
    check(len(fixtures)==50 and len({f['memory_id'] for f in fixtures})==50,'fixtures coverage')
    check(len(cases)==75 and len({c['case_id'] for c in cases})==75,'case grid')
    check({e['case_id'] for e in events}=={c['case_id'] for c in cases},'event coverage')
    results=[];violations=[];classified=[]
    for c in cases:
        cid=c['case_id'];es=sorted([e for e in events if e['case_id']==cid],key=lambda x:x['seq']);rs=[r for r in reads if r['case_id']==cid];f=fixtures[c['fixture_id']]
        check(sum(e['kind']=='case_complete' for e in es)==1,cid+': completion')
        check(not any(e['kind']=='execution_error' for e in es),cid+': execution error')
        check(all(a['ns']<=b['ns'] for a,b in zip(es,es[1:])),cid+': monotonic events')
        check(len({r['start_ns'] for r in rs})==len(rs),cid+': duplicate read')
        check(all(r['kind']=='read' for r in rs),cid+': read errors')
        rs=[r for r in rs if r['kind']=='read']
        submits=[e for e in es if e['kind']=='submit'];check([e['version'] for e in submits]==[2,3],cid+': offered updates')
        ready3=[e for e in es if e['kind']=='ready' and e['version']==3];check(len(ready3)==1,cid+': v3 ready evidence')
        ready_ns=ready3[0]['ns'] if ready3 else 10**30
        if c['blocked_stage']:
            hook=[e for e in es if e['kind']=='hook_reached'];release=[e for e in es if e['kind']=='hook_released'];starts=[e for e in es if e['kind']=='stage_start' and e['version']==2 and e['stage']==c['blocked_stage']]
            check(len(hook)==len(release)==len(starts)==1 and hook[0]['ns']<ready_ns<release[0]['ns']<=starts[0]['ns'],cid+': INVALID barrier ordering')
            check(any(r['tag']=='v3_before_release' and assess(r,f,3)['strict'] for r in rs),cid+': pre-release content')
        rows=[]
        for r in rs:
            target=max([1]+[e['version'] for e in submits if e['ns']<=r['start_ns']]);a=assess(r,f,target)
            check(target==r['target_at_start'],cid+': target clock')
            check(r['start_ns']<=r['snapshot_ns']<=r['query_end_ns']<=r['end_ns']<=r['ns'],cid+': read clock')
            check(a['strict']==r['strict_observed'],cid+': validator disagreement')
            post=r['start_ns']>=ready_ns;row=dict(case_id=cid,start_ns=r['start_ns'],end_ns=r['end_ns'],tag=r['tag'],target=target,post_final_ready=post,**a);rows.append(row);classified.append(row)
            if a['mixed'] or a['stale_relation'] or (post and not a['strict']):violations.append(row)
        final=[r for r in rows if r['tag'].startswith('final_')];check(len(final)==3,cid+': final probes')
        initial=[r for r in rows if r['tag']=='initial'];check(len(initial)==1 and initial[0]['strict'],cid+': initial content')
        # Control must remain correct after final readiness; treatment violations are observations, not a failed harness.
        if c['scenario']=='control':check(all(r['strict'] for r in final),cid+': control failed')
        for key in ('mixed','stale_candidate','stale_relation','unknown'):
            pass
        post=[r for r in rows if r['post_final_ready']];bad=[r for r in rows if not r['strict']];good=[r for r in rows if r['target']==3 and r['strict']]
        submit3=next(e['ns'] for e in submits if e['version']==3)
        results.append(dict(**c,reads=len(rows),post_ready_reads=len(post),mixed=sum(r['mixed'] for r in rows),unknown=sum(r['unknown'] for r in rows),
          stale_candidate=sum(r['stale_candidate'] for r in rows),stale_relation=sum(r['stale_relation'] for r in rows),post_ready_failures=sum(not r['strict'] for r in post),
          final_regression=any(not r['strict'] for r in final),guard_rejections=sum(e['kind']=='guard_rejection' for e in es),
          first_latest_correct_ms=(min(r['end_ns'] for r in good)-submit3)/1e6 if good else None,
          first_bad_start_ns=min((r['start_ns'] for r in bad),default=None),last_bad_end_ns=max((r['end_ns'] for r in bad),default=None),
          right_censored_bad=any(not r['strict'] for r in final),mean_query_ms=sum((r['query_end_ns']-r['start_ns'])/1e6 for r in rs)/len(rs),
          mean_extra_audit_ms=sum((r['end_ns']-r['query_end_ns'])/1e6 for r in rs)/len(rs)))
    for name,data in [('per_run_summary.csv',results)]:
        with (out/name).open('w',newline='',encoding='utf-8-sig') as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    groups=[]
    for cell in p['cells']:
        for scenario in p['scenarios']:
            selected=[r for r in results if r['cell']==cell and r['scenario']==scenario];n=sum(r['reads'] for r in selected);post=sum(r['post_ready_reads'] for r in selected)
            groups.append(dict(cell=cell,scenario=scenario,cases=len(selected),reads=n,post_ready_reads=post,
                mixed_version_rate=sum(r['mixed'] for r in selected)/n,unknown_rate=sum(r['unknown'] for r in selected)/n,
                stale_candidate_rate=sum(r['stale_candidate'] for r in selected)/n,stale_relation_rate=sum(r['stale_relation'] for r in selected)/n,
                post_final_ready_target_failure_rate=sum(r['post_ready_failures'] for r in selected)/post if post else None,
                final_state_regression_rate=sum(r['final_regression'] for r in selected)/len(selected),guard_rejections=sum(r['guard_rejections'] for r in selected)))
    with (out/'scenario_summary.csv').open('w',newline='',encoding='utf-8-sig') as h:w=csv.DictWriter(h,fieldnames=list(groups[0]));w.writeheader();w.writerows(groups)
    for name,data in [('violations.jsonl',violations),('classified_reads.jsonl',classified)]:
        (out/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data),encoding='utf-8')
    result=dict(harness_pass=not errors,errors=errors,cases=len(results),reads=len(reads),final_regression_cases=sum(r['final_regression'] for r in results),
        guard_rejections=sum(r['guard_rejections'] for r in results),scientific_zero_violation=not violations,source_hashes_unchanged=not any('hash drift' in e for e in errors),
        boundary='Deterministic adversarial schedule smoke; sampled rates not production probabilities; frozen relation/all schedules share hook. Unknown versions separate. One target plus four distractors per case.')
    save(out/'validate.json',result);print(json.dumps(result,indent=2));return result

def selftest(out):
    f=load(out/'fixtures.json')[0];v=f['versions'][2];g=copy.deepcopy(v['graph']);g['memory']['scope_id']='test'
    rels={r:r in v['required_relations'] for x in f['versions'] for r in x['required_relations']}
    r=dict(scope='test',graph=g,projection=v['rawerk'],sparse_version=3,dense_version=3,sparse_document=v['rawerk'],dense_vector_sha256=v['normalized_vector_sha256'],
           relation_memberships=rels,ids=[f['memory_id']],sparse_candidates=[[f['memory_id'],1]],dense_candidates=[[f['memory_id'],1]])
    assert assess(r,f,3)['strict'];tests={'valid_v3':True}
    mutations={
      'wrong_raw':lambda x:x['graph']['memory'].update(raw_text='wrong'),
      'wrong_sparse_content':lambda x:x.update(sparse_document='wrong'),
      'old_sparse':lambda x:x.update(sparse_document=f['versions'][1]['rawerk'],sparse_version=2),
      'wrong_dense':lambda x:x.update(dense_vector_sha256='wrong'),
      'wrong_projection':lambda x:x.update(projection='wrong'),
      'missing_candidate':lambda x:x.update(ids=[]),
      'stale_relation':lambda x:x['relation_memberships'].update({next(k for k in rels if not rels[k]):True}),
      'mixed_structure':lambda x:x['graph'].update(edges=f['versions'][1]['graph']['edges'])}
    for name,fn in mutations.items():
        x=copy.deepcopy(r);fn(x);assert not assess(x,f,3)['strict'];tests[name]=True
    x=copy.deepcopy(r);mutations['old_sparse'](x);assert assess(x,f,3)['mixed'] and assess(x,f,3)['stale_candidate']
    x=copy.deepcopy(r);mutations['wrong_dense'](x);assert assess(x,f,3)['unknown']
    save(out/'validator_unit_tests.json',dict(passed=True,tests=tests));print('validator tests passed',len(tests))
if __name__=='__main__':
    out=Path(sys.argv[1]);selftest(out) if '--selftest' in sys.argv else validate(out)
