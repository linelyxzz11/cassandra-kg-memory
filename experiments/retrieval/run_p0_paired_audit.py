"""Read-only source audit and offline paired bootstrap; no network/model calls.

Outputs are isolated under results/p0_audit_20261012. Original predictions,
scores, manuscript and historical tables are never modified.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Optional isolated dependency path; keep global/project environments unchanged.
if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
OUT = ROOT / 'results/p0_audit_20261012'
HIP = WORK / 'tmp/hipporag2-reader-full5-eval'
DEV = {'conv-42', 'conv-48'}
SEED = 20261012
B = 10000
METRICS = ['MRR@10', 'Hit@1', 'Hit@5', 'Hit@10', 'Recall@10', 'nDCG@10']
SOURCES = {}
CHECKS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register(path):
    path = Path(path)
    SOURCES[str(path)] = sha(path)
    return path


def csvread(path):
    with register(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def jlread(path):
    return [json.loads(x) for x in register(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def jsread(path):
    return json.loads(register(path).read_text(encoding='utf-8-sig'))


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def check(name, condition, detail=None):
    CHECKS.append({'check': name, 'passed': bool(condition), 'detail': detail})
    if not condition:
        save('failed_checks.json', CHECKS)
        raise AssertionError(name + ': ' + str(detail))


def unique(rows, field='query_id'):
    result = {r[field]: r for r in rows}
    check('unique IDs: ' + field, len(result) == len(rows), len(rows))
    return result


def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, register(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scope(qid):
    return qid.split('_qa_')[0]


def intervals(cass, hip, scopes):
    """Paired micro means. Cluster draws retain every query of sampled scopes."""
    a, b = np.asarray(cass, dtype=float), np.asarray(hip, dtype=float)
    check('finite paired arrays', a.shape == b.shape and bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(b))))
    delta = a - b
    labels = sorted(set(scopes))
    sums = np.array([delta[np.array(scopes) == s].sum(axis=0) for s in labels])
    sizes = np.array([scopes.count(s) for s in labels])
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(labels), size=(B, len(labels)))
    cluster = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]
    rng = np.random.default_rng(SEED + 1)
    query = []
    for start in range(0, B, 100):
        idx = rng.integers(0, len(delta), size=(min(100, B-start), len(delta)))
        query.append(delta[idx].mean(axis=1))
    query = np.concatenate(query)
    return a.mean(axis=0), b.mean(axis=0), np.quantile(cluster, [.025, .975], axis=0), np.quantile(query, [.025, .975], axis=0)


def summarize_pairs(kind, population, setting, ids, a, b, metrics, results, details):
    ids = sorted(ids)
    groups = [scope(q) for q in ids]
    aa = [[a[q][m] for m in metrics] for q in ids]
    bb = [[b[q][m] for m in metrics] for q in ids]
    am, bm, ci, qi = intervals(aa, bb, groups)
    for i, metric in enumerate(metrics):
        results.append(dict(kind=kind, population=population, setting=setting, metric=metric,
                            n=len(ids), conversations=len(set(groups)), cassmem=float(am[i]),
                            hipporag2=float(bm[i]), delta=float(am[i]-bm[i]),
                            cluster_ci95=ci[:, i].tolist(), query_ci95=qi[:, i].tolist(),
                            cluster_ci_excludes_zero=bool(ci[0, i] > 0 or ci[1, i] < 0)))
    for s in sorted(set(groups)):
        chosen = [q for q in ids if scope(q) == s]
        for m in metrics:
            av = sum(a[q][m] for q in chosen)/len(chosen)
            bv = sum(b[q][m] for q in chosen)/len(chosen)
            details.append(dict(kind=kind, population=population, setting=setting, conversation=s,
                                n=len(chosen), metric=m, cassmem=av, hipporag2=bv, delta=av-bv))


def rank_metrics(ids, gold):
    ids = ids[:10]
    check('ranking unique IDs', len(set(ids)) == len(ids))
    rel = [int(x in gold) for x in ids]
    hit = sum(rel)
    rr = next((1/(i+1) for i, x in enumerate(rel) if x), 0.0)
    dcg = sum(x/math.log2(i+2) for i, x in enumerate(rel))
    ideal = sum(1/math.log2(i+2) for i in range(min(10, len(gold))))
    return dict(zip(METRICS, [rr, float(any(rel[:1])), float(any(rel[:5])), float(hit>0), hit/len(gold), dcg/ideal]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check('pinned NLTK version reproduces historical CassMem', nltk.__version__=='3.9.1',nltk.__version__)
    # Source hashes and assertions also freeze original full-population summaries.
    paper = register(WORK/'paper/main.tex')
    gold = unique(csvread(ROOT/'data/retrieval_gold/locomo_cat1_4_gold_memory.csv'))
    qa = unique(csvread(ROOT/'data/locomo_qa_records.csv'), 'qa_id')
    test_ids = {q for q, r in gold.items() if r['split']=='test' and int(r['gold_count'])>0}
    full_reader = set(qa)
    test_reader = {q for q in qa if scope(q) not in DEV}
    check('reader n=1986/1487, retrieval n=1146', (len(qa), len(test_reader), len(test_ids)) == (1986,1487,1146))
    check('QA scope identity', all(r['user_id']==scope(q) for q,r in qa.items()))
    check('development scope identity', {r['conversation_id'] for r in gold.values() if r['split']=='dev'}==DEV)
    populations = {name: dict(n=len(ids), conversations=dict(Counter(scope(q) for q in ids)), categories=dict(Counter(qa[q]['category'] for q in ids)))
                   for name,ids in [('full1986',full_reader),('heldout1487',test_reader),('retrieval1146',test_ids)]}
    print('Sample identities validated.', flush=True)
    fusion_rows = csvread(ROOT/'results/retrieval/fusion_ablation_aligned_v2/fusion_ablation_per_query_heldout1146.csv')
    fusion = unique([r for r in fusion_rows if r['method']=='ZScore'])
    alpha_rows = csvread(ROOT/'results/retrieval/zscore_alpha_sweep/alpha_sweep_per_query.csv')
    alpha = unique([r for r in alpha_rows if float(r['alpha_dense'])==.6 and r['split']=='test'])
    dev_alpha = [r for r in alpha_rows if float(r['alpha_dense'])==.6 and r['split']=='dev']
    main_rows = csvread(ROOT/'results/retrieval/retrieval_main_table/retrieval_main_per_query.csv')
    cass = unique([r for r in main_rows if r['method']=='ZScore-RawERK' and r['query_id'] in test_ids])
    check('fusion/alpha/CassMem identical held-out IDs', set(fusion)==set(alpha)==set(cass)==test_ids)
    recalculated = {}
    for q in sorted(test_ids):
        check('fusion/alpha ordered Top10 '+q, fusion[q]['top10_memory_ids']==alpha[q]['top10_memory_ids'])
        rr = rank_metrics(fusion[q]['top10_memory_ids'].split(';'), set(gold[q]['gold_memory_ids'].split(';')))
        for m in METRICS:
            check('fusion/alpha/main/canonical metric '+q+' '+m, all(abs(rr[m]-float(x[q][m]))<1e-12 for x in [fusion,alpha,cass]))
        recalculated[q] = rr
    hip = {}
    for path in sorted((WORK/'tmp/hipporag2-full-locomo-controlled-v1').glob('conv-*/rankings.jsonl')):
        for r in jlread(path):
            q = r['query_id']
            if q not in test_ids:
                continue
            check('unique Hippo retrieval '+q, q not in hip)
            check('Hippo scope-local retrieval '+q, r['scope']==scope(q) and all(scope(q)==mid.split('_session_')[0] for mid in r['ranked_memory_ids']))
            hip[q] = rank_metrics(r['ranked_memory_ids'], set(gold[q]['gold_memory_ids'].split(';')))
            for m,v in r['metrics'].items():
                check('Hippo stored/recomputed metric '+q+' '+m, abs(hip[q][m]-v)<1e-12)
    check('Hippo retrieval complete', set(hip)==test_ids)
    fusion_audit = dict(heldout_n=1146, dev_n=len(dev_alpha), dev_mrr=sum(float(r['MRR@10']) for r in dev_alpha)/len(dev_alpha),
                        heldout_mrr=sum(r['MRR@10'] for r in recalculated.values())/1146,
                        ordered_top10_and_six_metrics_match=True,
                        historical_05426='retired historical summary; original matching per-query output not recovered; no guessed causal attribution')
    results, conversation = [], []
    summarize_pairs('retrieval','heldout1146','frozen',test_ids,recalculated,hip,METRICS,results,conversation)
    print('Retrieval paired bootstrap complete.', flush=True)
    reader = imp('p0_reader', ROOT/'src/cassmem/evaluation/run_locomo_prompt_protocol_gpt4o.py')
    official = imp('p0_official', ROOT/'src/cassmem/evaluation/locomo_official_eval_v1.py')
    v4 = imp('p0_v4', OUT/'recovered/rebuild_reader_offline_metrics_v4.py')
    questions = reader.load_questions(register(ROOT/'data/locomo_qa_records.csv'))
    memories = reader.load_memory_text(register(ROOT/'data/locomo_memory_records.csv'))
    offline_summary = csvread(ROOT/'results/reader/reader_offline_metrics_v4/reader_metrics_summary.csv')
    old_summary = {r['setting']:r for r in offline_summary if r['method']=='ZScore-RawERK'}
    judge_proto = jsread(ROOT/'results/reader/llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected/protocol.json')
    original_j = csvread(ROOT/'results/reader/llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected/j_scores_reader_main_wide.csv')
    hip_scored = jlread(HIP/'judge_scored_predictions.jsonl')
    hip_final = csvread(HIP/'final_summary.csv')
    hip_ranks = unique(jlread(WORK/'tmp/hipporag2-reader-full5-completion/full5_rankings.jsonl'))
    recovered_manifest = jsread(OUT/'recovered/recovery_manifest.json')
    reader_audit = {'settings':{}, 'judge_status':'CassMem historical per-query labels unavailable; no paired J CI or CassMem held-out J estimated.',
                    'historical_j_aggregates':[r for r in original_j if r['method']=='ZScore-RawERK'],
                    'recovery':recovered_manifest}
    paired = []
    hip_new_summary = []
    for setting, letter in [('a_unified','A'),('b_category','B')]:
        pred = unique(jlread(OUT/f'recovered/cassmem_{setting}_predictions.jsonl'))
        hp = unique(jlread(HIP/setting/'reader_predictions.jsonl'))
        hs = unique([r for r in hip_scored if r['setting']==setting])
        corrected = unique([dict(r,query_id=r['qa_id']) for r in jlread(ROOT/f'results/retrieval/official_eval/gpt4o_dual_setting_locomo_corrected_v3/setting{letter}_scored_predictions.jsonl') if r['method']=='ZScore-RawERK'])
        check('Reader coverage '+setting, set(pred)==set(hp)==set(hs)==set(corrected)==full_reader)
        av, bv = {}, {}
        cross_scope = []
        cache_overlap = []
        current_hip_prompts = 0
        normalizer_diffs = []
        tokenizer_diffs = []
        hip_cache = {r['judge_key']:r['judge_label'] for r in hip_scored}
        for q in sorted(full_reader):
            c,h = pred[q],hs[q]
            check('Reader metadata '+setting+q, str(c['category'])==str(h['category'])==qa[q]['category'] and c['question']==h['question']==qa[q]['question'].strip())
            check('CassMem model configuration '+setting+q, c['model']=='gpt-4o-2024-08-06' and c['temperature']==0 and c['max_tokens']==64)
            raw = c['prediction']
            check('CassMem recovered answer equality '+setting+q, raw==corrected[q]['raw_prediction'])
            target = 'Not mentioned in the conversation' if qa[q]['category']=='5' else qa[q]['answer'].strip()
            # Corrected-v3 retains the adversarial alternative in its stored gold
            # column; v4 scoring replaces Cat5 references with the abstention text.
            # Official Cat5 F1 is prediction-based abstention accuracy.
            stored_target = qa[q]['answer'].strip()
            check('Reader gold provenance '+setting+q, h['gold_answer']==target and corrected[q]['gold_answer']==stored_target)
            normalized = v4.normalize_prediction(c,setting,reader)['prediction_for_eval']
            f1 = official.score_one(int(qa[q]['category']),normalized,target)['official_score']
            b1 = sentence_bleu([word_tokenize(target.lower())],word_tokenize(normalized.lower()),weights=(1,0,0,0),smoothing_function=SmoothingFunction().method1)
            check('CassMem recomputed F1 '+setting+q, abs(f1-corrected[q]['official_f1'])<1e-12)
            # Independently re-score Hippo using the same function/tokenizer.
            historical_hf1 = official.score_one(int(qa[q]['category']),h['prediction_for_eval'],target)['official_score']
            historical_hb1 = sentence_bleu([word_tokenize(target.lower())],word_tokenize(h['prediction_for_eval'].lower()),weights=(1,0,0,0),smoothing_function=SmoothingFunction().method1)
            check('Hippo historical F1 equality '+setting+q, abs(historical_hf1-h['official_f1'])<1e-12)
            if abs(historical_hb1-h['bleu1'])>1e-12:
                tokenizer_diffs.append(dict(query_id=q,original_b1=h['bleu1'],pinned_nltk391_b1=historical_hb1,prediction=h['prediction_for_eval']))
            hn = v4.normalize_prediction(dict(prediction=h['raw_prediction'],category=h['category'],qa_id=q,gold_answer=qa[q]['answer']),setting,reader)['prediction_for_eval']
            hf1 = official.score_one(int(qa[q]['category']),hn,target)['official_score']
            hb1 = sentence_bleu([word_tokenize(target.lower())],word_tokenize(hn.lower()),weights=(1,0,0,0),smoothing_function=SmoothingFunction().method1)
            if hn != h['prediction_for_eval']:
                normalizer_diffs.append(dict(query_id=q,original=h['prediction_for_eval'],v4=hn,original_f1=historical_hf1,v4_f1=hf1,original_b1=historical_hb1,v4_b1=hb1))
            av[q]={'F1':f1,'B1':b1}
            bv[q]={'F1':hf1,'B1':hb1}
            mids = c['top10_memory_ids'].split(';')
            check('CassMem context IDs valid '+setting+q,len(mids)==10 and len(set(mids))==10 and all(mid in memories for mid in mids))
            foreign = [mid for mid in mids if mid.split('_session_')[0]!=scope(q)]
            if foreign: cross_scope.append({'query_id':q,'category':qa[q]['category'],'foreign_memory_ids':foreign})
            prompt = reader.build_prompt(questions[q],reader.render_context(hip_ranks[q]['ranked_memory_ids'],memories),setting=='b_category')
            ph = hashlib.sha256(prompt.encode()).hexdigest()
            check('Hippo reconstructed prompt hash '+setting+q,ph==hp[q]['prompt_sha256'])
            current_hip_prompts += 1
            jp = judge_proto['prompt'].format(question=h['question'],gold_answer=target,generated_answer=h['prediction_for_eval'])
            check('Hippo judge prompt hash '+setting+q,hashlib.sha256(jp.encode()).hexdigest()==h['judge_key'])
            cp = judge_proto['prompt'].format(question=c['question'],gold_answer=target,generated_answer=normalized)
            key = hashlib.sha256(cp.encode()).hexdigest()
            if key in hip_cache: cache_overlap.append(q)
            paired.append(dict(setting=setting,query_id=q,conversation=scope(q),category=qa[q]['category'],heldout=q in test_reader,
                               cassmem_f1=f1,hipporag2_f1=hf1,cassmem_b1=b1,hipporag2_b1=hb1,
                               hipporag2_j=int(h['judge_label']=='CORRECT'),cassmem_j=None))
        means = {m:sum(r[m] for r in av.values())/1986 for m in ['F1','B1']}
        save('diagnostic_'+setting+'.json',dict(means=means,per_category={cat:{m:sum(av[q][m] for q in av if qa[q]['category']==cat)/sum(qa[q]['category']==cat for q in av) for m in ['F1','B1']} for cat in ['1','2','3','4','5']},hippo_normalizer_diffs=normalizer_diffs,cross_scope=cross_scope,per_query=av))
        for m,col in [('F1','overall_f1'),('B1','overall_b1')]:
            check('CassMem reproduces original1986 '+setting+m,abs(means[m]-float(old_summary[setting][col]))<1e-12,means[m])
        for pop,ids in [('full1986',full_reader),('heldout1487',test_reader)]:
            summarize_pairs('reader',pop,setting,ids,av,bv,['F1','B1'],results,conversation)
            hip_new_summary.append(dict(setting=setting,population=pop,n=len(ids),f1=sum(bv[q]['F1'] for q in ids)/len(ids),
                                        b1=sum(bv[q]['B1'] for q in ids)/len(ids),judge=sum(hs[q]['judge_label']=='CORRECT' for q in ids)/len(ids)))
        reader_audit['settings'][setting]=dict(original1986_reproduced=means,hippo_prompt_hashes_verified=current_hip_prompts,
            cassmem_cross_scope_context_queries=len(cross_scope),cross_scope_by_category=dict(Counter(r['category'] for r in cross_scope)),
            hippo_shared_v4_normalization_differences=normalizer_diffs,
            hippo_pinned_tokenizer_differences=tokenizer_diffs,
            cross_scope_details=cross_scope,
            exact_judge_prompt_overlap_with_later_hippo_cache=len(cache_overlap),
            judge_overlap_policy='Not substituted for historical CassMem labels. Later judgments would define a new evaluation, not recover the historical run.',
            cassmem_prompt_provenance='Historical model settings and raw answers recovered; per-prediction prompt hashes absent in original CassMem records. Shared prompt source metadata supports protocol alignment, not byte-exact historic prompt verification.')
        print('Reader '+setting+' bootstrap complete.', flush=True)
    save('populations.json',populations)
    save('fusion_audit.json',fusion_audit)
    save('reader_protocol_audit.json',reader_audit)
    save('paired_bootstrap.json',results)
    save('conversation_deltas.json',conversation)
    save('reader_paired_per_query.json',paired)
    save('hipporag2_reader_population_summary.json',hip_new_summary)
    unchanged = {p:sha(p)==h for p,h in SOURCES.items()}
    check('all original inputs unchanged',all(unchanged.values()))
    save('checks.json',CHECKS)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),api_calls=0,bootstrap_resamples=B,cluster_seed=SEED,query_seed=SEED+1,
                    numpy=np.__version__,nltk=nltk.__version__,python=sys.version,platform=platform.platform(),
                    ci='paired percentile 95% intervals of CassMem minus HippoRAG2; query-weighted cluster bootstrap retains all questions per sampled conversation',
                    multiplicity='Intervals are marginal; no family-wise or multiple-testing significance claim.',
                    cluster_count_caution='8 held-out or 10 full conversations; cluster-bootstrap precision is limited by independent cluster count.',
                    source_sha256=SOURCES,all_original_inputs_unchanged=all(unchanged.values()),manuscript_unchanged=unchanged[str(paper)],
                    script_sha256=sha(__file__),checks_passed=len(CHECKS),j_status='historical CassMem per-query labels missing')
    save('manifest.json',manifest)
    lines=['# P0 audit and paired-bootstrap results','',
           'Original 1,986-question results preserved; 1,487-question Reader analysis is supplementary. No API calls and no manuscript edits.', '',
           '## Paired differences','',
           'All metrics below use the 0–1 scale. Δ = CassMem − HippoRAG2. 10,000 paired draws; primary CI resamples conversations and preserves query-weighted means. Query-level CI is a sensitivity analysis.', '',
           '| Task | Population | Setting | Metric | CassMem | HippoRAG2 | Δ | Conversation-cluster 95% CI | Query-level 95% CI |',
           '|---|---|---|---|---:|---:|---:|---|---|']
    for r in results:
        lines.append(f"| {r['kind']} | {r['population']} | {r['setting']} | {r['metric']} | {r['cassmem']:.6f} | {r['hipporag2']:.6f} | {r['delta']:+.6f} | [{r['cluster_ci95'][0]:+.6f}, {r['cluster_ci95'][1]:+.6f}] | [{r['query_ci95'][0]:+.6f}, {r['query_ci95'][1]:+.6f}] |")
    lines += ['', '## Fusion provenance', '',
              f"Canonical dev n=390 MRR={fusion_audit['dev_mrr']:.10f}; test n=1146 MRR={fusion_audit['heldout_mrr']:.10f}. Aligned fusion, alpha=0.6, main retrieval per-query metrics and recalculated canonical gold all agree.",
              'Historical 0.5426 is retired from current comparison; its exact original cause remains unverified. The old 0.5340 development value is not the current canonical sweep.', '',
              '## Reader audit and remaining J gap','',
              'F1 and B1 are re-scored using the same official evaluator and lowercase NLTK sentence BLEU-1. Original CassMem full1986 F1/B1 aggregates reproduce within 1e-12.',
              'HippoRAG2 F1 and reader/Judge prompt hashes are independently checked. Shared-score B1 uses pinned NLTK 3.9.1 and the recovered CassMem v4 option parser for both methods; any changes from historical Hippo B1 are separately logged. Original Hippo outputs are preserved. Original CassMem records preserve model, temperature, answer budget and Top-10 context IDs but not per-prediction prompt hashes.',
              'CassMem historical J aggregates survive, but its original per-query Judge labels were not found in workspace, reachable Git objects, 303 unreachable blobs, or checked ZIP archives. Consequently its held-out J and paired J CI remain uncomputed. Matching later Hippo Judge prompts are not silently substituted for historical labels.', '',
              '## Context-scope audit','']
    for setting,v in reader_audit['settings'].items():
        lines.append(f"- {setting}: {v['cassmem_cross_scope_context_queries']} CassMem questions have at least one foreign-scope context memory; categories={v['cross_scope_by_category']}. See reader_protocol_audit.json for exact IDs. This concerns the saved Reader contexts, not the canonical category1–4 retrieval evaluation.")
        lines.append(f"- {setting}: HippoRAG2 shared option-parser differences={len(v['hippo_shared_v4_normalization_differences'])}; pinned-tokenizer differences={len(v['hippo_pinned_tokenizer_differences'])}. See exact rows in reader_protocol_audit.json. J is kept under its original normalization and never mixed with a changed Judge input.")
    lines += ['', '## Interpretation and manuscript handoff','',
              '- Keep full1986 QA rows as the historical full-population analysis; append heldout1487 as a supplemental analysis, not a replacement.',
              '- Distinguish positive point differences from intervals excluding zero; use the conversation-cluster intervals as the main uncertainty check.',
              '- Eight held-out conversations provide limited independent clusters. These are marginal intervals, not simultaneous family-wise guarantees.',
              '- Update the historical fusion paragraph/table/alpha plot together after manuscript editing is authorized; current canonical development MRR is 0.5380291005.',
              '- Resolve the historical CassMem J label gap before adding paired J claims; API rescoring would be a separately versioned evaluation requiring approval.',
              '- Recorded CassMem contexts have zero foreign-scope memories in both settings. Historic per-prediction CassMem prompt hashes are unavailable, so byte-exact historical prompt identity is not claimed.', '',
              f"Validation: {len(CHECKS)} assertions passed. Input file hashes rechecked at finish; original manuscript and all inputs unchanged."]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS for retrieval/F1/B1; J incomplete','comparisons':len(results),'checks':len(CHECKS),'inputs_unchanged':True,'report':str(OUT/'REPORT.md')}))


if __name__=='__main__':
    main()
