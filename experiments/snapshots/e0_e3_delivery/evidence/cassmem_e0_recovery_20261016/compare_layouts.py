"""Read-only aggregation of measured raw events; separate analysis outputs."""
import json,collections,statistics,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
C=Path('D:/memorytable/cassmem_e0_formal_20261015/run_b9da27ebdd6e')
N=HERE/'resume_a3d619eff461'
OUT=HERE/'comparison';OUT.mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def dist(xs):return {'n':len(xs),'mean':float(np.mean(xs)),'p50':float(np.percentile(xs,50)),'p95':float(np.percentile(xs,95)),'p99':float(np.percentile(xs,99))}
groups=collections.defaultdict(lambda:collections.defaultdict(list));runs=[];manifest={}
for root in (C,N):
 for d in sorted(root.iterdir()):
  if not (d/'point.json').exists():continue
  pt=load(d/'point.json');key=(pt['layout'],pt['rate']);g=groups[key];vals=[]
  for name in ('events.jsonl','reads.jsonl'):
   path=d/'measured'/name;manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
   rs=lines(path)
   assert len({r['event_id'] for r in rs})==len(rs)
   for r in rs:
    assert r['final_status']=='ok'
    def ms(a,b):return (r[b]-r[a])/1e6
    if r['op']=='update':
     v={'arrival_ready':ms('actual_enqueue_ns','ready_check_done_ns'),'app_queue':ms('actual_enqueue_ns','worker_start_ns'),
        'id_lock_wait':ms('id_lock_wait_start_ns','id_lock_acquired_ns'),'raw_call':ms('raw_call_start_ns','raw_ack_ns'),
        'structured':ms('structured_start_ns','structured_done_ns'),'canonical':ms('canonical_check_start_ns','canonical_check_done_ns'),
        'sparse':ms('sparse_start_ns','sparse_done_ns'),'dense':ms('dense_start_ns','dense_done_ns'),'ready_check':ms('ready_check_start_ns','ready_check_done_ns'),
        'update_worker_total':ms('worker_start_ns','response_finish_ns'),
        'post_ready_audit':ms('post_ready_audit_start_ns','post_ready_audit_done_ns'),
        'post_update_query':ms('post_update_query_start_ns','post_update_query_done_ns'),
        'submission_lag':ms('planned_arrival_ns','actual_enqueue_ns')}
     components=['app_queue','id_lock_wait','raw_call','structured','canonical','sparse','dense','ready_check']
     v['other_ready']=v['arrival_ready']-sum(v[k] for k in components)
     assert v['other_ready']>=-1e-5
     for k,x in v.items():g[k].append(x)
     g['top10_hit'].append(int(r['post_update_target_present']));vals.append(v['arrival_ready'])
    else:
     v={'read_query_latency':ms('actual_enqueue_ns','query_finish_ns'),'read_response_with_audit':ms('actual_enqueue_ns','response_finish_ns'),
        'read_app_queue':ms('actual_enqueue_ns','worker_start_ns'),'read_candidate_access':ms('worker_start_ns','candidate_ids_done_ns'),
        'read_scoring':sum(r.get(k,0) for k in ('sparse_scoring_ns','dense_scoring_ns','fusion_ns'))/1e6}
     for k,x in v.items():g[k].append(x)
  runs.append({**pt,**dist(vals)})
result=[]
for (layout,rate),g in sorted(groups.items()):
 row={'layout':layout,'rate':rate,'metrics':{k:dist(v) for k,v in g.items() if k!='top10_hit'},'top10_hits':sum(g['top10_hit']),
      'per_run_p95':[r['p95'] for r in runs if r['layout']==layout and r['rate']==rate]}
 result.append(row)
(OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(OUT/'per_run.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
for r in result:
 m=r['metrics'];print(r['layout'],r['rate'],'n',m['arrival_ready']['n'],'ready',*[round(m['arrival_ready'][k],2) for k in ('p50','p95','p99')],'run_p95',[round(x,2) for x in r['per_run_p95']],'means',{k:round(m[k]['mean'],2) for k in ('app_queue','raw_call','structured','canonical','sparse','dense','ready_check')},'read_p95',round(m['read_query_latency']['p95'],2))

order=['cassandra-base','cassandra-materialized','neo4j-native','neo4j-materialized']
ordered=sorted(result,key=lambda r:(r['rate'],order.index(r['layout'])))
report=['# E0 四布局延迟与阶段耗时对比','',
'## 1. 统计口径','',
'100K 逻辑记忆，四布局各3个更新率、每点3个种子，36点；8 CPU/12 GiB单节点数据库容器，32个共享worker，每次更新伴随19次读取。因此0.25/0.5/1更新每秒分别对应5/10/20个总请求每秒。',
'正式阶段共2520次更新、47880次读取；warm-up和集成smoke排除。四布局均按修正后的精确向量哈希校验通过，全部请求在120秒drain截止前完成。原始FAIL和中断证据仍保留。',
'下表按同布局、同更新率合并3次重复的逐事件记录计算分位数，更新样本量分别为90/180/360；不是平均三次p95，也不把逐事件数当独立实验重复。每点p95另见per_run.json。',
'arrival-to-ready从实际入队到目标内容/版本检查结束，包含应用排队。raw call是客户端观察到的写调用，不是数据库内部commit。read query latency到排名返回为止；worker还会执行额外审计，单列其完整响应边界。','',
'## 2. 更新到检索就绪延迟（ms）','',
'| 更新/s | 布局 | p50 | p95 | p99 | 三次运行p95范围 |','|---:|---|---:|---:|---:|---:|']
for r in ordered:
 a=r['metrics']['arrival_ready'];ps=r['per_run_p95'];report.append(f"| {r['rate']:g} | {r['layout']} | {a['p50']:.2f} | {a['p95']:.2f} | {a['p99']:.2f} | {min(ps):.2f}–{max(ps):.2f} |")
report+=['','## 3. 更新阶段均值（ms）','',
'使用均值便于分解；各阶段p95不可相加。列出主要非重叠区间，未列出的锁获取/调度残余另存summary.json。scope锁计时与阶段嵌套，未重复相加。连接池等待和独立关系清理没有独立观测值，不填零。',
'','| 更新/s | 布局 | 应用排队 | raw写调用 | 结构/关系维护 | canonical核验 | sparse更新 | dense更新 | ready核验 | 总到达→ready |',
'|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in ordered:
 m=r['metrics'];vs=[m[k]['mean'] for k in ('app_queue','raw_call','structured','canonical','sparse','dense','ready_check','arrival_ready')]
 report.append(f"| {r['rate']:g} | {r['layout']} | "+' | '.join(f'{x:.2f}' for x in vs)+' |')
report+=['','## 4. 查询与额外审计边界（ms）','',
'| 更新/s | 布局 | 到查询结束p95（含排队） | 查询均值（含排队） | 完整读任务均值（含审计） | 读取排队均值 |',
'|---:|---|---:|---:|---:|---:|']
for r in ordered:
 m=r['metrics'];vs=[m['read_query_latency']['p95'],m['read_query_latency']['mean'],m['read_response_with_audit']['mean'],m['read_app_queue']['mean']]
 report.append(f"| {r['rate']:g} | {r['layout']} | "+' | '.join(f'{x:.2f}' for x in vs)+' |')
report+=['','## 5. 能支持的结论','',
'1. **低负载并非Cassandra全面领先。** 在0.25和0.5更新/s下，Neo4j native的ready p95为189.93/186.39ms，低于Cassandra base的272.76/248.75ms。四布局应用排队均值约0.1ms，此时主要是执行阶段差异。',
'2. **当前流程在1更新/s出现明显的排队转折。** Cassandra两布局ready p95约225ms；Neo4j native/materialized为53.64/48.29秒。三个种子均观察到该现象，Neo4j单次运行p95区间分别为52.22–53.97秒和47.13–50.00秒，非单个重复的偶发异常。',
'3. **差距主要来自应用排队，不是几十秒的写调用。** 在1更新/s，Neo4j native平均ready27.49秒，其中排队25.93秒（约94.3%）；materialized平均24.53秒，其中排队23.00秒（约93.8%）。raw写调用均值仍约7.5ms。应写queue-inclusive arrival-to-ready，不能称数据库commit延迟。',
'4. **物化收益取决于负载与阶段。** Cassandra物化降低canonical核验均值，且在0.25/0.5更新/s降低ready p95；但1更新/s的p95为225.90ms，与base224.55ms相近，不能宣称尾延迟始终更优。Neo4j物化在1更新/s降低积压水平，但低负载ready p95不优于native。',
'5. **全部最终完成不等于持续低延迟。** 36点都满足发出负载门槛，并在固定drain内完成；Neo4j在1更新/s仍有显著积压。应将offered-load valid、截止前ready比例和延迟同时展示。',
'6. **查询与更新的优劣可以不同。** 低负载Cassandra查询p95约22–28ms，Neo4j约37–39ms；但低负载更新ready反而Neo4j更快。不能用单一后端排名替代不同操作的成本分析。','',
'## 6. 解释边界与建议','',
'最重要的混杂因素是审计开销：每次读取在得到排名后，还通过后端读取graph/projection/relations等进行核验，且这些任务占用同一个32-worker池。1更新/s下Neo4j额外读审计均值约1.13–1.18秒，更新后的额外审计也约1.11–1.15秒；其资源竞争可以反馈到后续查询与更新。因此当前结果属于“带完整版本/内容检查的受控服务流程”，不是纯检索/写入路径的最大容量测试。',
'ready_check自身也是服务定义的一部分，应与仅用于实验的post-ready/read审计区分。当前代码在raw、结构维护之外，canonical与ready检查包含多次后端往返；不能从总延迟直接推断数据库存储引擎本身的瓶颈。',
'更新率按固定顺序逐步增加，缓存/运行顺序未交叉随机化；Cassandra延迟随负载提高而下降不能解释为“负载越高越快”。跨后端运行时间也不同，应避免不受控的因果归因。',
'建议保留本轮E0完整结果；若要证明生产serving容量优势，另行冻结“完整审计 vs 采样/移出计时审计”的对照，保持相同请求流、资源、ready服务语义和独立正确性抽查，检查Neo4j积压是否仍存在。此建议尚未启动。',
'p99在每组仅90/180/360次更新下应作为描述性结果；不要据此声称稳定的极端尾延迟或统计显著性。','',
'## 7. 可用于论文的结论草案','',
'> 在相同100K记忆规模和资源预算下，四种布局均完成全部预定请求并在固定drain内达到目标版本就绪。低负载下，Neo4j布局表现出较低的更新就绪延迟；随着请求率增至每秒1次更新和19次读取，Cassandra两种布局保持约225ms的arrival-to-ready p95，而Neo4j布局在当前完整检查流程中出现明显应用排队。阶段分析表明，这一差异主要来自排队与维护检查路径，raw写调用仍保持毫秒量级。该结果强调了同时报告负载有效性、排队、组件维护和检索就绪时间的重要性。','',
'## 8. 数据文件','',
'- summary.json：12组完整分布、各阶段均值与分位数。',
'- per_run.json：36次独立运行的ready分布。',
'- source_manifest.json：本次聚合读取的72份正式更新/读取日志及SHA256。',
'- 上游Cassandra使用原始事件并配合另存的corrected revalidation；Neo4j使用resume_a3d619eff461，未混入失败装载或smoke。']
(OUT/'E0_FOUR_LAYOUT_COMPARISON_ZH.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
