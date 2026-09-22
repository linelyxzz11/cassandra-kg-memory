# 3 System Design

## 3.1 Persistent Agent Memory Service

本文考虑持续运行的 long-term agent memory service。agent 跨 session 与用户及在线服务交互，memory 随交互不断增长和更新；query 通常在已知 conversation 或 user scope 下访问历史内容，并通过 online retrieval 为当前任务返回证据。该记忆层在单次请求结束后继续保存和维护服务状态，是 agentic Web infrastructure 中连接历史交互与后续任务的持久访问组件。

这一部署方式同时带来检索与服务挑战。scope 限定了访问范围，但相关证据仍可能分散于不同 session，并以语义改写或具体实体、关系线索出现；多用户、多 scope 的并发读取还会与持续更新共享后端和索引资源。与此同时，数据库写入只表示原始记录已经持久化，并不意味着结构化视图和检索索引已经可以使用。为读取效率而改变物理布局时，候选内容和排序行为也需要保持与同一逻辑记忆状态一致。

CassMem 因而以共同的逻辑记忆表示连接四项设计目标：利用原文与结构线索支持 scope 内的 structured retrieval；围绕稳定访问模式组织可持续维护的 online serving state；在不同物理访问路径之间保持 backend-independent retrieval semantics；并以 retrieval freshness 描述新 memory 从写入到具备检索条件及被查询实际命中的过程。后续各节分别说明这些目标如何落实到记忆投影、混合检索、Cassandra 物理组织和更新路径中。


## 3.2 Structured Memory Representation

在持续运行的长期记忆服务中，新的交互不断加入历史记录，同一实体或话题的相关信息也逐渐分散到不同 session。已知 conversation scope 可以缩小访问范围，但查询仍需从其中定位具体事实：语义相近的记忆可能描述不同对象，同一事实也可能采用不同表述。因此，检索既需要保留原文中的语境，也需要能够直接匹配实体、关系与关键词等具体线索。

这一需求决定了结构化表示需要与原始内容互补。仅使用压缩后的结构字段，可能遗漏原文中的限定条件与细节；在每次查询时重新解析历史内容，又会把结构提取的工作反复放到在线路径上。CassMem 因而将已抽取的实体、关系和关键词与来源 memory 关联，构造保留原文的 RawERK 检索视图。原始文本提供语义上下文，附加字段集中呈现可供词项匹配的线索，二者共同参与后续证据查找。

为使这一视图能够随记忆持续维护，CassMem 将来源内容与派生表示关联到同一 memory identity 和版本。记 $m_i^v$ 为第 $i$ 条记忆的版本 $v$，RawERK 文本 $r_i$ 随同来源字段构成检索投影 $p_i^v$：

$$
\begin{aligned}
m_i^v
&=\langle s_i,id_i,v,x_i,E_i,R_i,K_i,G_i,h_i\rangle,\\
r_i
&=\operatorname{Join}_{\mathrm{newline}}\!
  \left[x_i,\phi(\texttt{E:},E_i),\phi(\texttt{R:},R_i),\phi(\texttt{K:},K_i)\right],\\
p_i^v=\rho(m_i^v)
&=\langle m_i^v,r_i\rangle .
\end{aligned}
\tag{1}
$$

其中，$(s_i,id_i,v)$ 标识记忆的 scope、身份和版本；$x_i$ 为原文，$(E_i,R_i,K_i)$ 为实体、关系和关键词，$G_i$ 与 $h_i$ 分别保留局部关系事实及关联 embedding 的内容指纹。$\rho$ 表示从这些字段构造检索投影的固定过程。

RawERK 将一条记忆的原始叙述与抽取出的结构线索放入同一份可索引文本，使词项匹配能够直接利用这两类内容。例如，查询涉及某个人物及其活动时，实体和关系字段提供明确的匹配线索，原文则保留活动发生的条件与具体描述。为此，式（1）按原文、实体、关系、关键词的顺序组织内容，由 $\phi$ 为非空结构字段添加标签，再由 $\operatorname{Join}$ 以换行符连接。局部关系事实仍单独保存，供关系访问使用；RawERK 负责为同一条 memory 提供文本检索入口。

上述逻辑表示为检索与持久化提供共同的内容约定：原文用于语义表示与证据读取，RawERK 用于词项匹配，局部关系事实用于结构化访问，memory identity 则将这些访问结果关联到同一来源。后端可以分别保存字段并在读取时重建 RawERK，也可以预先维护该视图；两种方式均应提供相同的检索内容。由此，检索器依赖明确的逻辑字段与视图，存储层则能够按访问需求安排这些内容的物理组织。

## 3.3 Scope-local hybrid retrieval

跨会话查询与历史记忆之间既可能存在表述差异，也可能依赖具体的人物、活动或关系。Dense 检索通过原始文本的语义表示发现措辞不同但内容相关的记忆，适合补充词项匹配难以覆盖的改写；然而，语义相近的内容也可能涉及不同对象，仅凭相似度难以充分区分这些细节。BM25 则直接利用 RawERK 中的实体名称、关系描述和关键词，为具体事实提供词项匹配依据，但其召回依赖查询与记忆之间的词项重合。因此，CassMem 同时保留两路候选，并融合它们对同一 memory 的支持，使语义关联与具体线索共同决定证据排序。

对于 query $q$，Dense 通道以原文 embedding 的 cosine similarity 计算 $u_d(q,i)$，BM25 通道以 RawERK 文本计算 $u_l(q,i)$。两路在所属 scope 的记忆投影集合 $\mathcal V_{s(q)}$ 内分别取得 Top-50 候选，记为 $C_d(q)$ 和 $C_l(q)$，再按 memory identity 合并去重。候选并集保留任一路召回的记忆，共同身份使两类匹配信号汇集到同一来源，随后通过融合形成统一排序。

融合还需要处理两路分数缺乏共同尺度的问题。Cosine similarity 与 BM25 的取值范围不同，同一通道的分数分布也会随 query 改变，直接加权可能使数值幅度较大的通道主导结果。CassMem 因而在每个 query 的各通道 Top-50 候选内进行 Z-score 标准化，以当前候选分布为参照，衡量一条记忆在本路结果中的相对突出程度。融合权重作用于这些相对分数，在保留通道内排序与标准化分数间隔的同时，减弱原始分值尺度对组合结果的影响。

记 $\mu_{q,c}$ 为通道 $c$ 返回候选的平均分数，$\delta_{q,c}$ 为经过数值稳定处理的总体标准差，标准化与融合计算为：

$$
\begin{aligned}
z_c(q,i)
&=\frac{u_c(q,i)-\mu_{q,c}}{\delta_{q,c}},\qquad i\in C_c(q),\\
S(q,i)
&=\alpha\,\bar z_d(q,i)+(1-\alpha)\,\bar z_l(q,i),\\
\mathcal O_k(q)
&=\operatorname{StableTopK}_{i\in C_d(q)\cup C_l(q)} S(q,i),
\qquad k=10 .
\end{aligned}
\tag{2}
$$

其中，$\bar z_c$ 对已召回候选沿用 $z_c$，对该通道未召回的候选取其最低标准化分数；空通道取 0。这使仅被一路发现的记忆仍可参与排序，无需扩展召回范围来补算另一路分数。$\alpha$ 控制语义与词项证据的相对贡献，并在开发集上确定后固定用于评测。融合分数 $S$ 汇合两类证据的相对支持程度，并输出 Top-10 记忆供后续读取。


面向持续运行的记忆服务，上述设计通过保留原文的结构化视图集中呈现事实线索，以两路候选扩大证据覆盖，并通过查询内标准化协调语义匹配与词项匹配的相对支持。由此，跨会话记忆中的表述差异和具体事实线索能够在同一 scope 内共同参与证据选择，检索过程无需在查询时重新抽取历史内容的结构字段。要将这一访问方式持续提供给在线请求，还需要维护不断增长的记忆及其派生视图，并控制读取与更新的开销。下一节围绕这些需求设计 Cassandra 的持久化组织和物化路径。

## 3.4 Cassandra-native structured memory serving

前两节确定了检索所需的记忆内容与排序方式，后端则需要将这些内容组织为可持续维护的在线访问状态。在已知 conversation/user scope 的 long-term memory workload 中，查询首先需要取得范围内的 memory identity，随后按需读取记忆内容；结构化访问还需要恢复一条记忆关联的实体与关系，或根据指定关系筛选候选；词项检索则使用由原文和结构字段形成的 RawERK。CassMem 在逻辑层保留 graph-shaped memory 的实体、mention 与关系语义，在物理层根据这些相对稳定的访问模式，将其映射为面向查询的 wide-column serving state。该映射并非将图数据库中的记录直接迁移到 Cassandra，而是分别组织 scope-local memory、局部图记录、关系候选与检索视图，并以共同的 memory identity 和检索投影保持上层 retrieval semantics。

scope-local memory fetch 以 scope 为分区边界，以 memory identity 为分区内的聚簇键。记忆行保存原文、版本与 embedding 内容指纹，查询可从指定分区枚举 memory ID，也可结合 memory ID 定位单条记录。这一组织将访问范围直接落实到存储键上：候选枚举不需要扫描其他 scope，也不必同时取回全部文本与图记录。范围内的候选数量仍随该 scope 的记忆增长，因而 scope 分区限定的是访问边界，而不是固定的查询工作量。

实体与关系按其在记忆中的来源组织。实体字典以 scope 分区；mention 和局部边分别以 (scope, memory ID) 为联合分区键，在分区内按实体 ID 或边的序号区分记录。边记录保留来源实体、关系类型和目标实体，使读取一条记忆时能够恢复其局部关系事实，并追溯到对应的原始内容。基础布局还保存按 (scope, source entity) 分区的出边副本，以及按 scope 分区的边记录，分别表达来源实体邻接与范围内关系访问。这里的图结构由实体、mention 和带来源的边共同保存，不依赖在查询时从 RawERK 文本中重新推断关系。

对于显式给定关系条件的候选访问，系统提供基础与物化两种布局，将筛选工作分别安排在读取与更新阶段。基础路径读取 scope 内的边记录，在应用层匹配关系类型并对 memory ID 去重。它复用范围内的关系记录，按需完成筛选，无需为这一操作维护按类型分区的候选表；其读取工作量随 scope 内的边数增长。物化路径面向重复的关系筛选需求，维护以 (scope, relation) 为联合分区键的候选表，在分区内保留 memory ID、边序号及端点。查询可直接访问指定关系分区并归并候选，减少对其他关系记录的读取与过滤；分类组织及候选记录维护则在结构化写入阶段完成。两种布局明确了按需处理与预先组织之间的工作分配，具体选择由部署配置确定。

candidate projection 则解决候选内容如何转为可用的 RawERK retrieval view。基础路径将原文保存在记忆表，将实体、关系、关键词及 triples 保存在以 (scope, memory ID) 分区的特征表；请求 RawERK 时，分别读取原文与结构字段，按式（1）重建检索文本。物化路径在结构化写入阶段生成同一文本，并将 RawERK 与原文、结构字段共同保存在记忆行中，使该视图可通过一次点读取得。前者采用 read-time reconstruction，后者采用 update-time materialization。物化的是与 query 无关的检索文本，而不是某个 query 的分数或 Top-10；原始字段及局部图记录仍独立可用，完整图内容的读取仍需要访问 mention 和边记录。

这些派生状态由应用层显式维护。基础布局保存特征与多个读取方向的边副本；物化布局维护 RawERK 和 scope–relation 候选表，并保留按 memory 组织的 mention 与边。两种布局围绕候选定位和投影读取组织不同的记录集合，在保留逻辑内容与关系来源的前提下，调整文本组装、关系筛选及记录维护所发生的阶段。

上述物理状态通过共同的 memory identity 与检索层连接。Cassandra 提供 scope 或关系约束下的候选 ID，以及可读取的记忆内容和 RawERK 视图；应用层索引使用原始文本的语义表示与 RawERK 的词项表示，并按照第 3.3 节的规则完成召回和融合。候选范围、内容读取与证据排序由此具有明确接口，后端可以调整字段布局和物化路径，而不重新定义检索表示与打分方式。

相应地，backend-independent retrieval semantics 约束后端实现需要保持的行为：给定相同版本的逻辑记忆、scope 和关系条件，各读取路径应返回相同的候选身份与检索投影；进一步固定索引内容、打分器、候选深度和同分规则后，应得到相同的有序 Top-10。这一约定将候选内容和排序行为作为检索层与持久化层之间的稳定接口，使物理布局能够围绕读写开销调整，同时保持上层证据选择规则一致。至此，结构化记忆被组织为面向 scope、关系和检索视图的持久访问状态；持续更新还需要将这些状态的维护与索引更新衔接，下一节进一步描述这一过程及其检索可见性。


## 3.5 Retrieval and ranking freshness

持续运行的记忆服务需要将新交互及时转化为可供查询使用的证据。数据库提交仅完成原始记忆的持久化；结构化视图、关系候选以及稀疏与向量索引仍需更新，检索器才能利用本次新增的内容。因此，CassMem 将更新的生效过程从 database-level visibility 延伸到 retrieval-level freshness，沿以下路径组织记忆更新：

memory arrival → backend commit → structured view visible → retrieval index visible → final ranking visible。

新 memory 到达服务队列后，系统首先提交原始记录，再维护结构字段、局部关系和 RawERK 视图，使内容能够通过后端访问路径读取；随后更新稀疏与向量索引，使两路检索均可访问目标版本。共同的 memory identity 和版本将这些阶段关联到同一次更新。结构化内容与关系候选的回读检查，以及两路索引的版本检查，共同确认更新已具备参与检索的条件。查询在所属 scope 内取得候选并执行第 3.3 节的融合排序，新 memory 被选入 Top-$k$ 后，才成为该次查询可直接使用的证据。最后一步取决于 query 与记忆的相关性，因此索引可见与排名可见需要分别描述。

更新过程从状态就绪与查询命中两个层面记录。令 $a_i^v$ 为更新进入服务队列的时刻，$t_{\mathrm{index}}(m_i^v)$ 为两路索引更新及版本检查完成的时刻。记 $R_i^v$ 为目标版本的就绪检查结果：后端结构化投影、关系候选、稀疏索引和向量索引均通过检查时，$R_i^v=1$。在此条件下，time-to-retrievable 定义为：

$$
\begin{aligned}
T_{\mathrm{retrievable}}(m_i^v)
&=t_{\mathrm{index}}(m_i^v)-a_i^v,
\qquad R_i^v=1.
\end{aligned}
\tag{3}
$$

$T_{\mathrm{retrievable}}$ 表示新 memory 从到达到完成检索所需状态维护的观测时延，包含排队、后端处理和索引更新。未通过 $R_i^v$ 的更新作为状态维护失败单独记录，不进入成功就绪时延的统计。该定义把 database commit 与 retrieval readiness 区分开来：只有结构化内容、关系候选和两路索引均达到目标版本，新 memory 才具备参与后续检索的条件。

索引就绪描述 memory 已经能够参与检索，但 agent 最终使用的是排序后的证据，因此 retrieval freshness 还需要延伸到 ranking freshness。后者关注完成维护的新内容经过候选生成与融合后，是否及时反映在当前 query 的返回排序中。系统在索引就绪后执行一次真实的 scope-local hybrid retrieval，并记录目标 memory 是否出现在观测 Top-10；对于最终命中的样本，进一步报告其在给定时间预算内完成检索的比例。该观测将后端和索引就绪与最终证据返回联系起来，使 freshness 不仅停留在数据库或索引可见性。

ranking freshness 的解释仍需区分更新延迟与检索相关性。即使所有状态均已更新，新 memory 也可能因与 query 的匹配程度不足而不属于 Top-10，因此单次未命中不能直接视为 backend freshness failure。严格的排名陈旧或 Top-k convergence 需要在相同逻辑快照和检索配置下，与独立的 fully synchronized reference ranking 比较。当前协议尚未构造这一 reference，因而本文将 Hit@10 及其限时比例作为 observed ranking freshness，单独报告状态检查结果，并不将全部未命中解释为 stale result。

通过区分 retrieval readiness 与 observed ranking freshness，在线服务能够分别描述更新处理进度和新证据的实际使用情况，将记忆管理的观测范围从数据库提交延伸到检索结果。


CassMem 围绕跨会话持续增长的记忆，将结构化表示、证据排序和在线维护连接为统一的访问过程。RawERK 保留原文并集中呈现结构线索，两路检索与 query-wise ZScore fusion 共同组织候选证据，Cassandra 的 scope 分区、关系访问与物化视图将这些内容落实为可持续维护的服务状态。共同的记忆身份与检索语义衔接表示和物理访问，retrieval readiness 与 observed ranking freshness 则进一步刻画更新何时能够进入检索并反映在返回证据中，使长期记忆的管理从历史内容保存延伸到持续在线的证据供给。
