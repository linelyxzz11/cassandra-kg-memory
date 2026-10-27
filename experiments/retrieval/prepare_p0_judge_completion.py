"""Offline-only missing-label inventory for a NEW, explicitly versioned J run.

Does not invoke a model. Never estimates historical missing labels from scores.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
if len(sys.argv)>1:sys.path.insert(0,sys.argv[1])
root=Path(__file__).resolve().parents[2]
out=root/'results/p0_audit_20261012'
hip=root.parent/'tmp/hipporag2-reader-full5-eval'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
v4=load('v4',out/'recovered/rebuild_reader_offline_metrics_v4.py')
reader=load('reader',root/'src/cassmem/evaluation/run_locomo_prompt_protocol_gpt4o.py')
qa=reader.load_questions(root/'data/locomo_qa_records.csv')
protocol=json.loads((root/'results/reader/llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected/protocol.json').read_text(encoding='utf-8'))
cache={r['judge_key']:r for r in v4.read_jsonl(hip/'judge_unique.jsonl')}
jm=json.loads((hip/'judge_manifest.json').read_text(encoding='utf-8'))
assert jm['model_requested']=='gpt-4o-2024-08-06' and jm['temperature']==0 and jm['max_tokens']==80
assert jm['prompt_template_sha256']==protocol['prompt_sha256']
assert jm['judge_unique_sha256']==hashlib.sha256((hip/'judge_unique.jsonl').read_bytes()).hexdigest()
assert all(r['model_returned']=='gpt-4o-2024-08-06' and r['label'] in {'CORRECT','WRONG'} for r in cache.values())
rows=[]; prompts={}
for method in ['CassMem','HippoRAG2']:
    for setting in ['a_unified','b_category']:
        path=out/f'recovered/cassmem_{setting}_predictions.jsonl' if method=='CassMem' else hip/setting/'reader_predictions.jsonl'
        for r in v4.read_jsonl(path):
            qid=r['query_id'];q=qa[qid]
            normalized=v4.normalize_prediction(dict(prediction=r['prediction'],category=q['category'],qa_id=qid,gold_answer=q['answer']),setting,reader)['prediction_for_eval']
            target='Not mentioned in the conversation' if q['category']=='5' else q['answer']
            prompt=protocol['prompt'].format(question=q['question'],gold_answer=target,generated_answer=normalized)
            key=hashlib.sha256(prompt.encode()).hexdigest()
            prompts[key]=prompt
            rows.append(dict(method=method,setting=setting,query_id=qid,heldout=qid.split('_qa_')[0] not in {'conv-42','conv-48'},judge_key=key))
summary={}
for population in ['full1986','heldout1487']:
    selected=[r for r in rows if population=='full1986' or r['heldout']]
    keys={r['judge_key'] for r in selected}
    summary[population]=dict(prediction_rows=len(selected),unique_prompts=len(keys),compatible_cache_hits=len(keys&cache.keys()),new_unique_calls_needed=len(keys-cache.keys()))
pending={k:p for k,p in prompts.items() if k not in cache}
record=dict(api_calls=0,status='NOT_RUN; awaiting authority for paid Judge completion or recovery of historical labels',
            evaluation='new shared v4 answer-normalization Judge evaluation, not restoration of historical aggregate values',
            reader_calls_needed=0,summary=summary,judge_model='gpt-4o-2024-08-06',temperature=0,max_tokens=80,
            response_format={'type':'json_object'},prompt_template_sha256=protocol['prompt_sha256'],
            cache_policy='Reuse only exact rendered prompt SHA-256 under matching Judge protocol/model; original caches read-only.')
(out/'judge_completion_preflight.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
(out/'judge_completion_pending_prompts.json').write_text(json.dumps(pending,ensure_ascii=False),encoding='utf-8')
(out/'judge_completion_row_map.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
print(json.dumps(record,indent=2))
