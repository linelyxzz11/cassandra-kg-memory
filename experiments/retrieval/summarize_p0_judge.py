"""Validate completed new Judge version and compute paired J uncertainty offline.

Original Reader/Judge outputs and the first offline audit are read-only.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import csv
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
if len(sys.argv)>1:sys.path.insert(0,sys.argv[1])
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/p0_audit_20261012'
OUT=BASE/'judge_completion_v2_uiuihao'
HIP=ROOT.parent/'tmp/hipporag2-reader-full5-eval'
SOURCES={}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    SOURCES[str(path)]=sha(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))
def jl(path):
    SOURCES[str(path)]=sha(path)
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def index(rows,key):
    result={key(r):r for r in rows}
    assert len(result)==len(rows),'duplicate record'
    return result

def main():
    contract=read(OUT/'contract.json')
    assert contract['endpoint']=='https://api.uiuihao.com/v1/chat/completions'
    assert contract['model']=='gpt-4o-2024-08-06' and contract['temperature']==0 and contract['max_tokens']==80
    for p,h in contract['inputs_sha256'].items():assert sha(Path(p))==h,'contract input changed'
    old=index(jl(HIP/'judge_unique.jsonl'),lambda r:r['judge_key'])
    new=index(jl(OUT/'judgments_new.jsonl'),lambda r:r['judge_key'])
    pending=read(BASE/'judge_completion_pending_prompts.json')
    mapping=read(BASE/'judge_completion_row_map.json')
    assert set(new)==set(pending),f'Judge completion incomplete: {len(new)}/{len(pending)}'
    assert not (set(new)&set(old)),'duplicate calls against existing exact cache'
    assert len(new)==1825
    for h,r in new.items():
        assert r['endpoint']==contract['endpoint'] and hashlib.sha256(pending[h].encode()).hexdigest()==h
    cache={**old,**new}
    assert all(r['model_returned']==contract['model'] and r['label'] in {'CORRECT','WRONG'} for r in cache.values())
    required={r['judge_key'] for r in mapping}
    assert len(required)==5114 and required<=cache.keys()
    score_rows=[dict(r,judge_label=cache[r['judge_key']]['label'],judge_score=int(cache[r['judge_key']]['label']=='CORRECT'),
                     judgment_source='new_uiuihao_completion' if r['judge_key'] in new else 'existing_exact_prompt_cache') for r in mapping]
    scores=index(score_rows,lambda r:(r['method'],r['setting'],r['query_id']))
    assert len(scores)==7944
    prior_pairs=read(BASE/'reader_paired_per_query.json')
    paired=[]
    for r in prior_pairs:
        a=scores[('CassMem',r['setting'],r['query_id'])];b=scores[('HippoRAG2',r['setting'],r['query_id'])]
        assert a['heldout']==b['heldout']==r['heldout']
        paired.append(dict(r,cassmem_j=a['judge_score'],hipporag2_j=b['judge_score'],
                           cassmem_judge_key=a['judge_key'],hipporag2_judge_key=b['judge_key']))
    assert len(paired)==3972 and len({(r['setting'],r['query_id']) for r in paired})==3972
    old_hip=index(jl(HIP/'judge_scored_predictions.jsonl'),lambda r:(r['setting'],r['query_id']))
    normalization_changes=[]
    for r in score_rows:
        if r['method']!='HippoRAG2':continue
        previous=old_hip[(r['setting'],r['query_id'])]
        if previous['judge_key']==r['judge_key']:
            assert previous['judge_label']==r['judge_label']
        else:normalization_changes.append(dict(query_id=r['query_id'],setting=r['setting'],old_label=previous['judge_label'],new_label=r['judge_label']))
    assert len(normalization_changes)==5
    spec=importlib.util.spec_from_file_location('p0_stats',Path(__file__).with_name('run_p0_paired_audit.py'))
    stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
    results=[];conversations=[];summaries=[]
    for setting in ['a_unified','b_category']:
        records=[r for r in paired if r['setting']==setting]
        assert len(records)==1986 and sum(r['heldout'] for r in records)==1487
        for pop in ['full1986','heldout1487']:
            chosen=sorted([r for r in records if pop=='full1986' or r['heldout']],key=lambda r:r['query_id'])
            a={r['query_id']:{'J':r['cassmem_j']} for r in chosen};b={r['query_id']:{'J':r['hipporag2_j']} for r in chosen}
            stats.summarize_pairs('reader',pop,setting,set(a),a,b,['J'],results,conversations)
            for method,prefix in [('CassMem','cassmem'),('HippoRAG2','hipporag2')]:
                summaries.append(dict(population=pop,setting=setting,method=method,n=len(chosen),
                                      f1=float(np.mean([r[prefix+'_f1'] for r in chosen])),
                                      b1=float(np.mean([r[prefix+'_b1'] for r in chosen])),
                                      j=float(np.mean([r[prefix+'_j'] for r in chosen]))))
            # Independent direct-concatenation implementation verifies cluster CIs.
            delta=np.array([r['cassmem_j']-r['hipporag2_j'] for r in chosen],float)
            scopes=[r['conversation'] for r in chosen];labels=sorted(set(scopes))
            indices=[np.array([i for i,s in enumerate(scopes) if s==label]) for label in labels]
            draws=np.random.default_rng(20261012).integers(0,len(labels),size=(10000,len(labels)))
            distribution=[delta[np.concatenate([indices[i] for i in draw])].mean() for draw in draws]
            assert np.allclose(np.quantile(distribution,[.025,.975]),results[-1]['cluster_ci95'],atol=1e-12,rtol=0)
    original_metrics=read(BASE/'paired_bootstrap.json')
    save('paired_j_bootstrap.json',results)
    save('paired_all_metrics.json',original_metrics+results)
    save('reader_paired_complete.json',paired)
    save('reader_scores_complete.json',score_rows)
    save('reader_population_summary.json',summaries)
    save('conversation_j_deltas.json',conversations)
    save('hipporag2_normalization_j_changes.json',normalization_changes)
    old_j_path=ROOT/'results/reader/llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected/j_scores_reader_main_wide.csv'
    SOURCES[str(old_j_path)]=sha(old_j_path)
    with old_j_path.open(encoding='utf-8-sig',newline='') as f:original_j=list(csv.DictReader(f))
    history=[]
    for r in original_j:
        if r['method']!='ZScore-RawERK':continue
        current=next(s for s in summaries if s['method']=='CassMem' and s['setting']==r['setting'] and s['population']=='full1986')
        history.append(dict(method='CassMem',setting=r['setting'],historical_j=float(r['overall_j'])/100,new_shared_j=current['j'],
                            delta_new_minus_historical=current['j']-float(r['overall_j'])/100))
    save('historical_j_comparison.json',history)
    old_audit=read(BASE/'manifest.json')
    assert all(sha(Path(p))==h for p,h in old_audit['source_sha256'].items()),'original audit input changed'
    assert all(sha(Path(p))==h for p,h in SOURCES.items()),'aggregation input changed'
    usage={k:sum(int(r['usage'].get(k,0) or 0) for r in new.values()) for k in ['prompt_tokens','completion_tokens','total_tokens']}
    manifest=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),new_successful_unique_calls=len(new),
                  reused_required_unique=len(required&old.keys()),new_usage=usage,reader_calls=0,
                  total_scored_rows=len(score_rows),paired_rows=len(paired),populations=[1986,1487],
                  model=contract['model'],endpoint=contract['endpoint'],prompt_template_sha256=contract['prompt_template_sha256'],
                  bootstrap_resamples=10000,cluster_seed=20261012,query_seed=20261013,
                  independent_ci_verification='PASS for all four paired J comparisons',
                  all_original_files_unchanged=True,input_sha256=SOURCES,script_sha256=sha(Path(__file__)),
                  version_note='New shared-v4 normalization Judge version; reuses exact compatible prompts. Not a reconstruction of original CassMem J labels. Historical aggregates retained.',
                  limitations='Eight held-out conversations; marginal percentile CIs, no multiple-comparison adjustment; CI resampling conditions on saved Judge labels and does not measure Judge sampling variability.')
    save('completion_manifest.json',manifest)
    lines=['# Judge 补齐完成：独立评分版本','',
           '本轮使用用户确认的 `https://api.uiuihao.com/v1/chat/completions`，模型返回名均为 `gpt-4o-2024-08-06`；temperature=0、max_tokens=80、固定 JSON CORRECT/WRONG 模板。',
           '',f'新增 {len(new):,} 条唯一 Judge 评分，复用 {len(required&old.keys()):,} 条完全一致的 prompt 缓存；覆盖两方法×两种 prompt×1,986 问，共 7,944 行。Reader 调用为 0。',
           '原 1,986 问答案、历史评分和 main.tex 全部保留。本报告是统一选项还原后的新 J 版本，不把新标签冒充丢失的历史标签。','',
           '## F1 / J / B1 汇总','',
           '| 问题集 | Prompt | 方法 | F1（%） | J（%） | B1 |','|---|---|---|---:|---:|---:|']
    for s in summaries:lines.append(f"| {s['population']} | {s['setting']} | {s['method']} | {100*s['f1']:.4f} | {100*s['j']:.4f} | {s['b1']:.6f} |")
    lines+=['','## Paired J bootstrap','',
            '差值方向为 CassMem−HippoRAG2，单位为百分点。每种方法 10,000 次配对重采样；聚类抽样保留 conversation 内所有问题并采用 query-weighted micro mean。','',
            '| 问题集 | Prompt | ΔJ（pp） | Conversation-cluster 95% CI | Query-level 95% CI |','|---|---|---:|---|---|']
    for r in results:lines.append(f"| {r['population']} | {r['setting']} | {100*r['delta']:+.4f} | [{100*r['cluster_ci95'][0]:+.4f}, {100*r['cluster_ci95'][1]:+.4f}] | [{100*r['query_ci95'][0]:+.4f}, {100*r['query_ci95'][1]:+.4f}] |")
    lines+=['','## 与历史版本的关系','',
            '| Prompt | CassMem 原 J（%） | CassMem 新 J（%） | 变化（pp） |','|---|---:|---:|---:|']
    for r in history:lines.append(f"| {r['setting']} | {100*r['historical_j']:.4f} | {100*r['new_shared_j']:.4f} | {100*r['delta_new_minus_historical']:+.4f} |")
    lines+=['','新旧差异可能涉及新一次 Judge 输出及统一规范化；由于旧 CassMem 逐题标签缺失，不能逐题分解原因。旧汇总未被覆盖。',
            f"HippoRAG2 共 5 个规范化后的 Judge prompt 重新评分，其中 {sum(r['old_label']!=r['new_label'] for r in normalization_changes)} 个标签改变；其余 3,967 行保持原缓存标签。",'',
            '## 验证与解释','',
            '- 完整覆盖 1,986 / 1,487 问，配对 ID 唯一、无缺失；没有以交集静默删题。',
            '- 四组聚类 CI 已用独立的逐 conversation 拼接实现核验一致。',
            '- 区间跨 0 时，不据此写统计显著提升；两种抽样结论不同应同时报告。',
            '- 仅 8 个 held-out conversations；区间为边际区间，未进行多重比较校正。Bootstrap 条件于已有 Judge 标签，不包括重复调用 Judge 的随机波动。',
            '- F1/B1 和检索的先前统计保留于上级目录；`paired_all_metrics.json` 合并 18 组指标。',
            '- 所有原文件哈希检查通过；本轮未修改论文。','',
            f"新增成功请求 token 使用量：prompt={usage['prompt_tokens']:,}，completion={usage['completion_tokens']:,}，total={usage['total_tokens']:,}。金额以服务商账单为准。"]
    (OUT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='complete',judge_results=results,new_usage=usage,report=str(OUT/'RESULTS_ZH.md')),indent=2))

if __name__=='__main__':main()
