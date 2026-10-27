"""Build one complete Markdown handoff from verified P1-1 artifacts."""
from pathlib import Path
import csv,json,hashlib,statistics
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[2]
S=ROOT/'results/retrieval/raw_preserving_stage1_20261012'
R=ROOT/'results/retrieval/representation_robustness_20261012'
DEST=ROOT/'docs/P1_1_RAW_PRESERVING_COMPLETE_REPORT.md'
M=['MRR@10','Hit@1','Hit@5','Hit@10','Recall@10','nDCG@10']
sources={};lines=[];counts={}
def js(p):
    sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf-8-sig'))
def csvread(p):
    sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def add(t):lines.extend(t.strip().splitlines()+[''])
def link(p):return f'[{p.name}](<{p.as_posix()}>)'
def fmt(x):
    if isinstance(x,float):return f'{x:.6f}'
    return str(x).replace('|','\\|').replace('\n',' ')
def table(rows,keys=None):
    if not rows:return
    keys=keys or list(rows[0]);lines.append('| '+' | '.join(keys)+' |');lines.append('| '+' | '.join(['---']*len(keys))+' |')
    for row in rows:lines.append('| '+' | '.join(fmt(row.get(k,'')) for k in keys)+' |')
    lines.append('')
def numeric(rows,keys):return [{k:(float(v) if k in keys else v) for k,v in row.items()} for row in rows]
def section_csv(title,p,collapse=False):
    add('### '+title)
    if collapse:add('<details>\n<summary>展开完整数据表</summary>')
    rows=csvread(p);table(numeric(rows,M+['rate','mean','seed_sd','ci_low','ci_high','delta','delta_from_clean','auc_0_100','delta_auc']))
    counts[title]=len(rows);add('机器可读原始表：'+link(p))
    if collapse:add('</details>')
assert js(R/'completion.json')['status']=='complete'
assert js(R/'verification.json')['complete']
qs=js(R/'queries.json');methods=js(R/'methods.json');overall=csvread(R/'overall.csv');cat=csvread(R/'by_category.csv')
contract=js(R/'contract.json');auc=js(R/'auc_protocol.json')
add('''# P1-1：原文保留必要性、表示组合与结构缺失鲁棒性完整实验报告

实验批次：2026-10-12。用途：共同作者实验交接、论文修改及 P1-1 问题回应。本文汇总已完成实验，不修改 main.tex。

## 阅读导航

1. 研究问题与实验演进
2. 完整实验协议与指标定义
3. 第一阶段：26 组原文保留与结构组件结果
4. 扩展阶段：15 组完整矩阵与两种缺失模式
5. 鲁棒性 AUC、配对区间与核验
6. 数据解析与结论
7. 完整附表：分类、逐种子、全部配对统计
8. 如何回答 P1-1

数值展示保留六位小数；机器可读文件保留原始精度。所有已运行方法如实列出。正文结论使用 held-out 1,146 问，development 390 问和合计 1,536 问作为单独补充。合计结果含 dev，不能替代 held-out 主结果。逐题排名及六项指标通过文件链接提供，不把数百万逐题单元格复制为 Markdown 表。

## 1. 研究问题与实验演进

P1-1 要求：比较 Raw-only、ERK-only（不附加原文）、RawERK，以及 RawE/RawER 等，检验原文保留的贡献；报告 MRR@10、Recall@10 与问题类型结果，若 ERK-only 很好亦如实呈现。

三个不同问题需要分开回答：

- 在检索表示中完全去掉原文，会怎样？
- 原文是否需要同时出现在 Dense 和 BM25 两路？
- 结构信息缺失时，各种表示组合是否仍然稳定？

第一阶段以历史冻结 Dense 复现原论文，并在匹配的本地编码下比较 Raw/ERK Dense 分支及多种词法表示，共 26 组。第二阶段补齐此前遗漏的 Dense(RawERK)，形成 6 个单路与完整 3×3 混合矩阵，共 15 组，再做两类缺失实验。鲁棒性实验是在观察到 ERK-only 词法变体 clean MRR 较高后设计的补充诊断；缺失率和种子在本轮运行前固定，不把它写成最初论文预注册的实验。

## 2. 实验设置

### 2.1 数据、切分与结构字段来源

- LoCoMo category 1–4，5,882 条 memory，保留全部 memory ID。
- 使用 canonical gold mapping，保留 mapped 且具有非空 gold 的问题。原 1,540 条候选问题中 4 条无有效 gold，最终 1,536 条。
- 沿用既有按 conversation/scope 的 dev/test 划分，不因本轮结果重新切分或调参。
- 使用既有 session-level observation facts 对齐到来源 turn 后形成的冻结 E/R/K 字段。本轮不重新抽取，不调用抽取 LLM。
- ERK-only 的含义是“不拼接完整原文”，不是结构字段与原文毫无来源联系；K 等字段可以来自原文。抽取上下文与 per-memory 投影是不同层次。
- 原始 ERK 为空的记录有 3,529/5,882。0% 缺失意味着不额外删除字段，并非所有 memory 原本都有结构。
- held-out 1,146 问中，839 问全部 gold memory 有 ERK；307 问至少一条 gold 无 ERK，其中 174 问所有 gold 均无 ERK。该覆盖差异用于解释结果，不据此筛掉不利问题。
''')
splitrows=[]
for split in ['dev','test']:
    sub=[q for q in qs if q['split']==split];c=Counter(q['category'] for q in sub)
    splitrows.append({'split':split,'n':len(sub),'scopes':', '.join(sorted({q['scope'] for q in sub})),**{f'Cat{k}':c[k] for k in ['1','2','3','4']}})
table(splitrows)
add('''类别映射：Cat1=Multi-Hop；Cat2=Temporal；Cat3=Open-Domain；Cat4=Single-Hop。聚合采用逐题 micro mean。

### 2.2 输入表示与编码

| 通道 | Raw | ERK-only | RawERK |
|---|---|---|---|
| Dense | speaker: text | 非空 E/R/K 标签与字段 | 同一 raw 前缀，再附加非空 E/R/K |
| BM25 | 原始 text | 非空 E/R/K 标签与字段 | 原始 text，再附加非空 E/R/K |

- Dense：BAAI/bge-large-en-v1.5，1,024 维；本地 revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`。
- float32 CPU、CLS pooling、L2 归一化；输入最多 1,500 字符，再按 512 tokens 截断；batch 16、8 CPU threads。RawERK 截断作用于拼接后的文本，因此结论对应这一编码和拼接协议。
- 所有组合共享相同冻结 query 向量并归一化，按 cosine 排序。
- Dense(Raw)、Dense(ERK-only) 来自已核验的本地编码；补编码 Dense(RawERK) 及受损字段文本。缓存键为实际输入完整字符串，仅复用完全相同文本的向量。
- 空 ERK 保留 memory，Dense 为零向量，无 Raw 回填。BM25 空文本保留在语料中；全空 scope 采用零分稳定顺序。
- 纯 ERK 完全缺失时出现非零 Hit/MRR，是零分稳定排序偶然命中，并非仍然检索到了有效结构语义。

### 2.3 BM25 与混合检索

- 项目自实现 BM25，CountVectorizer；scope 内独立建统计量。k1=1.5，b=0.75，lowercase，English stopwords，1–2 gram，max_features=50,000。
- 不同表示/缺失条件重新建立相应词表、词频、文档频率与长度统计，所有 memory 均计入对应 scope 语料。Dense 和 BM25 同时消费相同受损结构状态。
- 各分支 Top-50，最终 Top-10。查询在 scope 内执行。
- per-query Z-score，population std（ddof=0），退化标准差按既有实现处理；缺失分支候选使用 branch-min。
- 固定 Dense 权重 alpha=0.6，词法权重 0.4；不按表示或缺失率调 test 参数。
- 分支同分遵循语料稳定顺序；融合候选使用 Dense 优先、再补词法独有候选的稳定并集顺序打破并列。
- 因此本轮比较的是统一固定协议下的表示选择，不是分别调优后各方法的性能上限。

### 2.4 缺失实验

memory：每条 memory 采样一个随机值，低于缺失率时一起移除其可用 E/R/K。

field：每条 memory 的 E、R、K 各采样一个随机值，独立移除对应字段块，不进行字段内 token 删除。

两种模式都保持 Raw 不变。每个 seed/scope 的随机值固定，较高缺失率的掩码包含较低缺失率的掩码。掩码生成不使用 query/gold，全部方法共享。缺失率是删除概率，非精确固定数量；原本为空的字段继续为空，实际剩余覆盖数另表报告。

缺失率：0、0.10、0.25、0.50、0.75、1.00。随机种子：20261012–20261021。中间四档各 10 次；0% 和 100% 为确定性端点，分别只运行一次。

共 83 个独立条件：共享 clean 1 个，memory 41 个，field 41 个；每条件 15 组，1,536 问均记录，test/dev 分开汇总。本实验模拟可用结构信息减少后的重建索引，不模拟并发更新、索引不同步、错误事实或对抗污染。

### 2.5 指标与统计

设 G 为该问题 gold memory 集合，L 为去重后的 Top-10：

- MRR@10：首条正确证据排名的倒数，无命中为 0。
- Hit@1/5/10：前 k 条是否至少有一个 gold，逐题取 0/1。
- Recall@10：|G∩L| / |G|，不是历史二值 R@10。
- nDCG@10：二元相关性，DCG 按 1/log2(rank+1) 折扣；IDCG 使用 min(|G|,10) 个理想命中。
- 六指标均对问题做 micro mean；多 seed 条件再对 seed 等权平均。
- 第一阶段：10,000 次 conversation-cluster bootstrap，seed=20261012；query-level 区间为辅助。
- 缺失实验：10,000 次配对 conversation 与 seed 联合重采样，seed=20261022；方法共享抽样，conversation 按重复抽取后问题数加权。主结果仅 8 个 test conversations。
- AUC：对 0–100% 全范围做梯形积分（区间长度为 1），不是 ROC-AUC，不是等权平均六个点，也不是相对 clean 的比值。权重依次 0.05、0.125、0.20、0.25、0.25、0.125。
- AUC 先对每个 seed 的完整曲线积分，再聚合；端点复用同一确定性结果。10,000 次配对 seed/conversation 重采样，seed=20261023，所有缺失率共享抽样。
- 区间均为边际 95% percentile CI，未作多重检验校正；最高点估计不等于显著优于全部对手。AUC 只对预定五项指标 MRR、Recall、nDCG、Hit1、Hit10 补算；Hit5 的完整曲线保留。

## 3. 第一阶段完整结果

FrozenRaw 为历史冻结 Dense；LocalRaw/LocalERK 为匹配本地编码。H_A_B 表示 Dense(A)+BM25(B)。RawE/RawR/RawK/RawER/RawEK/RawRK/RawERK 均为保留原文并附加指定字段的词法表示。

历史冻结当前方案 MRR=0.549822；本地匹配当前方案 MRR=0.549564。两者不是逐位相同的向量运行：后续 15 组公平比较使用本地匹配版本，不将 0.5498 与 0.5496 混用。
''')
section_csv('3.1 全部 26 组：held-out 六指标',S/'overall.csv')
add('### 3.2 第一阶段完整配对统计')
boot=js(S/'paired_bootstrap.json');table([{**{k:r[k] for k in ['a','b','metric','n','delta']},'cluster95':str(r['cluster_ci95']),'query95':str(r['query_ci95'])} for r in boot])
add('## 4. 扩展 15 组：完整矩阵与全部缺失条件\n\nDense_X、BM25_X 为单路；H_X_Y 为 Dense(X)+BM25(Y)，ERK 表示 ERK-only。当前方案为 H_Raw_RawERK。下列各表列出全部方法和六指标，不筛选领先方法。')
def mean_tables(rows,category=False):
    grouped=defaultdict(list)
    for r in rows:
        key=(r['split'],r['mode'],float(r['rate']),r.get('category',''),r['method']);grouped[key].append(r)
    for split in ['test','dev','all']:
        add(f'### {"分类" if category else "总体"}结果：{split}')
        for mode in ['memory','field']:
            for rate in [0,.1,.25,.5,.75,1.]:
                for c in (['1','2','3','4'] if category else ['']):
                    actual='memory' if rate==0 else mode
                    tt=[]
                    for m in methods:
                        rr=grouped[(split,actual,rate,c,m)];assert len(rr)==(1 if rate in [0,1] else 10)
                        tt.append({'method':m,'n':rr[0]['n'],'seeds':len(rr),**{metric:statistics.mean(float(r[metric]) for r in rr) for metric in M}})
                    add(f'#### {mode} / {rate:.0%}'+(f' / Cat{c}' if c else ''));table(tt)
mean_tables(overall)
add('## 5. AUC、置信区间与一致性核验')
section_csv('5.1 全部 15 组 × 两种模式 × 五项指标 AUC',R/'robustness_auc.csv')
for p in [R/'verification.json',R/'baseline_parity.json',R/'endpoint_sanity.json',R/'mask_preflight.json',S/'local_vs_frozen_dense.json']:
    add('### '+p.stem+'\n\n```json\n'+json.dumps(js(p),ensure_ascii=False,indent=2)+'\n```')
add('''算法审计独立重写 BM25 分数与融合计算，最终 Top-10 全部复现。曾发现一个 BM25 近同分记录受约 1e-15 运算顺序影响交换前两名，但候选集合与最终融合 Top-10 不变，不能描述为所有中间名次逐位一致。审计详见：''')
add(link(S/'algorithm_audit/REPORT_ZH.md'))
add('''## 6. 数据解析与实验结论

### 6.1 完全去掉原文：存在实质性能损失，但须区分覆盖与压缩

本地当前方案 Raw+RawERK 的 MRR/Recall 为 0.549564/0.738593；纯 ERK+ERK 为 0.479973/0.610329。MRR 差值 +0.069591，cluster95%CI [0.050499,0.089779]；Recall 差值 +0.128263，CI [0.102292,0.150781]。

这直接回答“检索表示完全不保留原文”的代价。覆盖诊断显示，该差距部分来自已有结构对 gold memory 覆盖不全。对于全部 gold 都有 ERK 的 839 问，差距明显缩小。因而可以主张原文提供覆盖保障，但不将全体差距都归因于结构化摘要的语义压缩损失。

### 6.2 Dense(Raw) 的选择有直接对照支持

无额外缺失时 Dense(Raw) MRR=0.496475，Dense(RawERK)=0.423442，Dense(ERK)=0.422591。固定 BM25(RawERK)，Dense(Raw) 的融合 MRR=0.549564，高于 Dense(RawERK) 的 0.521085 和 Dense(ERK) 的 0.517652。

当前方案相较两路 RawERK 的 MRR 差值 +0.028479，95%CI [0.014424,0.041893]。单路 Dense(Raw) 在全部非100%档位保持最高 MRR 点估计，且结构缺失不改变它的输入。结论是当前固定 BGE 拼接/截断协议下 Raw 更合适；“结构干扰语义”是可能解释，不是已独立验证的因果机制。

### 6.3 BM25(RawERK)：结构带来收益，Raw 提供缺失后的保底

固定 Dense(Raw)，Raw+Raw 的 MRR=0.510777，Raw+RawERK=0.549564；增加结构的收益为 +0.038787，95%CI [0.029175,0.049544]。

但 Raw+ERK 的 clean MRR=0.559727、nDCG=0.5800、Hit1=0.4433，确实高于当前方案的对应点估计。两者 clean MRR 差值 CI 跨零。单路 BM25(RawERK) 优于 BM25(ERK) 也不能保证融合后仍按同一顺序排列，因为候选集合与分支校准共同影响结果。

在两种模式的25%、50%、75%档位，当前方案 MRR 点估计为全15组最高；25%分字段缺失对 Raw+ERK 的 MRR 差值 CI 仍跨零，不能把所有点估计领先写成显著领先。0%与10%仍为 Raw+ERK 的 MRR 点估计更高。

当前方案在两种模式0%–75%的全部档位取得最高 Recall 点估计，100%与若干保留 Raw 的组合并列。整条结构50%缺失时，对 Raw+ERK 的 MRR优势为 +0.042049，95%CI [0.030233,0.055768]；分字段50%时为 +0.028706，[0.013746,0.042101]。

### 6.4 全范围 AUC：支持质量与鲁棒性的综合选择

两种模式下，当前方案五项 AUC 的点估计均为全15组最高。整条/分字段分别为：MRR 0.531310/0.533949（精确值以表为准）、Recall约0.7258/0.7275、nDCG约0.5584/0.5608、Hit1约0.4019/0.4053、Hit10约0.7970/0.7992。

对 Raw+ERK 的 AUC 比较，整条缺失下五项指标的差值区间均高于零；分字段缺失下 MRR、Recall、nDCG、Hit10 区间高于零，Hit1 区间跨零。没有据此声称显著优于所有14个对手。

AUC衡量基础得分与缺失后的综合表现，不等于纯粹抗下降能力，也不代表真实系统服从均匀缺失率。原始曲线、clean结果和全部方法均保留，避免用AUC掩盖低缺失时另一变体的优势。

### 6.5 端点与非单调现象

100%结构缺失时，RawERK 精确退化为 Raw。Raw+ERK 的词法分支变为零分，融合排序退化为 Dense-only，因此可能从75%回升。这不是越缺失越好，而是固定融合协议在部分受损分支和完全空分支下行为不同。

### 6.6 可用于论文的简短总结

完整表示组合实验支持非对称的通道设计：Dense 使用原文，BM25 使用原文与结构字段。相比原文双路，该设计利用结构线索改善检索；相比纯结构词法输入，它在结构缺失时保持更稳定的证据覆盖，并取得两种缺失模式下最高的五项全范围 AUC 点估计。clean 条件下 ERK-only 词法变体的部分排名指标更高，体现了前列排序与结构缺失鲁棒性之间的取舍。

此结论对应固定alpha、当前编码与随机结构缺失协议，不是各组合分别调优后的最优性证明，也不直接验证 Reader答案保真、错误事实鲁棒性或跨领域泛化。

## 7. 完整附表与原始记录

以下保留全部已生成聚合数据、逐种子表和配对区间。分类汇总与总体汇总均包括test/dev/all。统计比较专门针对held-out；不是另做dev显著性选择。
''')
# Replace approximate prose numbers with actual AUC, so summaries cannot drift.
aucrows=csvread(R/'robustness_auc.csv');al={(r['mode'],r['method'],r['metric']):float(r['auc_0_100']) for r in aucrows}
lines=[s.replace('0.531310/0.533949（精确值以表为准）',f"{al['memory','H_Raw_RawERK','MRR@10']:.6f}/{al['field','H_Raw_RawERK','MRR@10']:.6f}") for s in lines]
for name in ['by_category.csv','by_conversation.csv','by_gold_erk_coverage.csv']:
    section_csv('第一阶段 '+name,S/name,True)
add('<details>\n<summary>扩展15组：所有split、缺失条件、问题类型的完整六指标均值表</summary>')
mean_tables(cat,True);add('</details>')
for title,p in [('扩展实验全部逐种子总体数据',R/'overall.csv'),('扩展实验全部逐种子分类数据',R/'by_category.csv'),('各条件主指标区间与下降量',R/'robustness_summary.csv'),('各条件全部方法配对统计',R/'paired_all_comparisons.csv'),('AUC全部方法配对统计',R/'robustness_auc_paired.csv')]:section_csv(title,p,True)
add('### 实际结构覆盖：83个条件')
coverage=[js(p) for p in sorted((R/'conditions').glob('*.json'))]
table([{k:r[k] for k in ['mode','rate','seed','records','initial_nonempty','final_nonempty']} for r in coverage])
add('### 逐题数据与复现入口')
for p in [S/'per_query.csv',S/'documents.json',S/'memory_inputs.json',R/'queries.json',R/'memory_order.json',R/'methods.json',R/'contract.json',R/'auc_protocol.json',ROOT/'experiments/retrieval/run_raw_preserving_stage1.py',ROOT/'experiments/retrieval/run_representation_robustness.py',ROOT/'experiments/retrieval/supplement_robustness_auc.py']:
    assert p.exists();add('- '+link(p))
add('逐条件 NPZ：'+f'[conditions目录](<{(R/"conditions").as_posix()}>)'+'。metrics维度为method×query×6；六指标顺序见本文表头；top10为memory_order.json索引；field_bits位0/1/2对应E/R/K。不是只保存汇总分数。')
add('### 数据来源哈希清单\n\n下表用于核对本报告读取的文件，不替代各次实验原有contract。')
table([{'source':k,'sha256':v} for k,v in sorted(sources.items())])
add('''## 8. 最后：这些实验如何回答 P1-1？

### 8.1 逐项对应问题

| P1-1要求 | 本轮对应证据 | 可以得出的回答 |
|---|---|---|
| 补 Raw-only、ERK-only、RawERK | 三种表示各有 Dense/BM25 单路，并形成完整3×3混合组合 | 对照包含真正两路都不附加原文的ERK+ERK，不把Dense仍用Raw的设置称作完全去Raw |
| 补 RawE/RawER等 | 第一阶段保留八种Raw结构组件表示和ERK-only词法对照 | 可分析结构组件贡献，而非只比较两个最终系统 |
| 单独检验保留原文的作用 | 完全去Raw对比；固定BM25只换Dense；固定Dense只换BM25；共享缺失掩码 | 原文在Dense中有直接收益；词法原文的价值更突出地体现于结构缺失后的稳定匹配 |
| 报MRR、Recall、类型效果 | 全六指标，完整分类表；test/dev/all分开；逐种子与配对区间齐全 | 不依赖单点数字或选择性样本报告 |
| ERK-only很好时如实呈现 | Raw+ERK在clean/10%MRR更高，clean nDCG/Hit1也更高 | 不宣称RawERK在所有条件和指标下必胜 |
| “证据保真性”有何证据 | gold memory检索覆盖、完整原文保留对照及缺失诊断 | 直接证据是检索表示的覆盖与鲁棒性；未在本轮直接测试Reader生成答案的事实保真 |

### 8.2 可以直接用于回复共同作者或审稿人的中文版

我们补充了包含 Raw、ERK-only 和 RawERK 的完整表示对照：六个单路基线及九个 Dense–BM25 混合组合，并报告 held-out 问题的完整排名、证据覆盖和分类结果。在固定检索协议下，完全移除原文的 ERK-only 双路方案低于当前方案；固定词法表示后，Dense(Raw) 也优于 Dense(RawERK)。另一方面，Dense(Raw)+BM25(ERK-only) 在无额外缺失时取得更高的 MRR 点估计，这一结果被完整保留。进一步的共享结构缺失实验显示，BM25中保留原文能缓解结构信息减少带来的损失。当前 Dense(Raw)+BM25(RawERK) 组合在两种缺失模式下取得全部15组中最高的五项全范围鲁棒性AUC点估计。因此，这些实验支持原文语义通道与原文保留的结构增强词法通道这一选择，而不是证明每个通道都加入原文或结构必然提升所有指标。

### 8.3 最精简结论

**P1-1已得到直接实验回应：原文为检索提供稳定的语义与覆盖保障，结构字段提供额外词法线索；Dense(Raw)+BM25(RawERK)的依据是整体质量与缺失鲁棒性的平衡，而非clean MRR始终最高。**

若将P1-1严格理解为“Reader答案语义保真性”，本系列完成的是检索层证据，而不是生成层证明。论文应将两者分开表述。
''')
DEST.parent.mkdir(exist_ok=True)
DEST.write_text('\n'.join(lines),encoding='utf-8')
assert len([p for p in (R/'conditions').glob('*.npz')])==83
assert len(methods)==15 and len(overall)==83*3*15 and len(cat)==83*3*4*15
assert len(aucrows)==2*15*5
print(json.dumps({'path':str(DEST),'bytes':DEST.stat().st_size,'lines':len(lines),'tables':sum(l.startswith('| ---') for l in lines),'source_rows':counts,'sources':len(sources)},ensure_ascii=False,indent=2))
