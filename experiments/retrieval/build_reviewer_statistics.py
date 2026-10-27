"""Paired query bootstrap and conversation-cluster sensitivity from frozen scores.

Full evaluation includes development queries and is descriptive. The primary
inference family is three metrics by three comparators on held-out queries.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'results/retrieval/retrieval_main_table/retrieval_main_per_query.csv'
GOLD = ROOT / 'data/retrieval_gold/locomo_cat1_4_gold_memory.csv'
OUT = ROOT / 'results/retrieval/reviewer_statistics'
B = 10000
SEED = 20260922
METRICS = ('MRR@10', 'Recall@10', 'nDCG@10')
BASELINES = ('Dense-bge', 'RRF_compact', 'ZScore-Raw')


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = read(INPUT)
    scope = {r['query_id']: r['conversation_id'] for r in read(GOLD)}
    methods = sorted({r['method'] for r in rows})
    estimates, comparisons = [], []
    for split in ('test', 'full'):
        selected = [r for r in rows if split == 'full' or r['split'] == split]
        lookup = {m: {r['query_id']: r for r in selected if r['method'] == m} for m in methods}
        ids = sorted(lookup['ZScore-RawERK'])
        assert all(set(lookup[m]) == set(ids) for m in methods), 'Unequal paired query sets'
        assert len(ids) == (1146 if split == 'test' else 1536)
        values = np.asarray([[[float(lookup[m][q][metric]) for metric in METRICS] for m in methods] for q in ids])
        point = values.mean(axis=0)
        rng = np.random.default_rng(SEED)
        boot = np.empty((B, len(methods), len(METRICS)))
        for start in range(0, B, 32):
            indices = rng.integers(0, len(ids), size=(min(32, B-start), len(ids)))
            boot[start:start+len(indices)] = values[indices].mean(axis=1)
        clusters = sorted({scope[q] for q in ids})
        cluster_sums = np.asarray([values[[i for i,q in enumerate(ids) if scope[q] == c]].sum(axis=0) for c in clusters])
        cluster_sizes = np.asarray([sum(scope[q] == c for q in ids) for c in clusters])
        picks = np.random.default_rng(SEED+1).integers(0, len(clusters), size=(B,len(clusters)))
        cluster_boot = cluster_sums[picks].sum(axis=1) / cluster_sizes[picks].sum(axis=1)[:,None,None]
        for mi, method in enumerate(methods):
            for ki, metric in enumerate(METRICS):
                lo, hi = np.quantile(boot[:,mi,ki], [0.025,0.975])
                estimates.append(dict(split=split, method=method, metric=metric, n=len(ids), mean=point[mi,ki], ci95_low=lo, ci95_high=hi))
        ours = methods.index('ZScore-RawERK')
        for baseline in BASELINES:
            bi = methods.index(baseline)
            for ki, metric in enumerate(METRICS):
                diff = values[:,ours,ki]-values[:,bi,ki]
                draws = boot[:,ours,ki]-boot[:,bi,ki]
                lo,hi = np.quantile(draws,[0.025,0.975])
                clo,chi = np.quantile(cluster_boot[:,ours,ki]-cluster_boot[:,bi,ki],[0.025,0.975])
                # Paired randomization: null swaps method labels within each query.
                prng = np.random.default_rng(SEED+2)
                exceed = 0
                for start in range(0,B,64):
                    signs = prng.integers(0,2,size=(min(64,B-start),len(ids)))*2-1
                    exceed += int((np.abs((signs*diff).mean(axis=1)) >= abs(diff.mean())).sum())
                comparisons.append(dict(split=split, baseline=baseline, metric=metric, n=len(ids), delta=diff.mean(), ci95_low=lo, ci95_high=hi, cluster_ci95_low=clo, cluster_ci95_high=chi, clusters=len(clusters), paired_randomization_p=(exceed+1)/(B+1), holm_p=0.0, resamples=B, seed=SEED))
    for split in ('test','full'):
        family = sorted([r for r in comparisons if r['split']==split], key=lambda r:r['paired_randomization_p'])
        running=0.0
        for i,row in enumerate(family):
            running=max(running,min(1.0,(len(family)-i)*row['paired_randomization_p']))
            row['holm_p']=running
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'method_ci.csv',estimates)
    write(OUT/'paired_comparisons.csv',comparisons)
    lines=['# Retrieval uncertainty and paired comparisons','',
           'Primary evaluation: 1,146 held-out queries. Full evaluation: 1,536 queries, including 390 development queries; descriptive only.',
           '','95% percentile intervals use 10,000 paired query resamples. Conversation-cluster sensitivity resamples 10 conversations with replacement and retains query-weighted means. Only 10 clusters are available; these intervals do not establish generalization to unseen users.',
           '','Two-sided paired randomization p-values use 10,000 method-label swaps with a plus-one correction. Holm correction covers nine comparisons (three metrics × three baselines), separately by split. No retrieval settings or rankings are changed.',
           '','| Split | Baseline | Metric | Δ | Query 95% CI | Conversation 95% CI | Holm p |','|---|---|---|---:|---|---|---:|']
    for r in comparisons:
        lines.append(f"| {r['split']} | {r['baseline']} | {r['metric']} | {r['delta']:+.4f} | [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | [{r['cluster_ci95_low']:+.4f}, {r['cluster_ci95_high']:+.4f}] | {r['holm_p']:.4g} |")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest={'resamples':B,'seed':SEED,'query_ci':'paired percentile bootstrap','cluster_ci':'conversation resampling, query-weighted mean','test':'paired randomization with plus-one correction','holm_family_size':9,'full_scope':'descriptive; includes development queries','inputs':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (INPUT,GOLD)}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
