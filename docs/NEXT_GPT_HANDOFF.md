\# Next GPT Handoff: Cassandra-KG Memory Project



\## Project Location



```text

D:\\memorytable\\cassandra-kg-memory

```



This is the formal Git repository. Earlier exploratory files exist under `D:\\memorytable`, but the repository above should be treated as the main project.



\---



\## Research Topic



```text

Profiling-guided optimization for multi-hop KG traversal on Cassandra

```



Chinese description:



```text

面向 Cassandra 知识图谱记忆系统的多跳查询性能分析与索引缓存优化

```



\---



\## Core Cassandra Tables



Initial three-table design:



```text

kg\_edges\_by\_src

kg\_edges\_by\_dst

kg\_edges\_by\_relation\_bucket

```



Later added relation-aware index table:



```text

kg\_edges\_by\_src\_relation

```



The new index supports:



```text

graph\_id + src\_id + relation

```



\---



\## Key Scripts



Initial step:



```text

scripts/initial\_step/insert\_sample\_edges.py

scripts/initial\_step/kg\_query\_3hop.py

scripts/initial\_step/sync\_reverse\_edges.py

scripts/initial\_step/query\_reverse.py

scripts/initial\_step/sync\_relation\_edges.py

scripts/initial\_step/benchmark\_queries.py

```



Core scripts:



```text

scripts/core/generate\_synthetic\_kg.py

scripts/core/generate\_high\_degree\_kg.py

scripts/core/bulk\_insert\_kg.py

scripts/core/bulk\_insert\_kg\_v2.py

scripts/core/bulk\_insert\_kg\_v3.py

scripts/core/query\_synthetic\_paths.py

scripts/core/query\_by\_relation.py

scripts/core/query\_high\_degree\_forward.py

scripts/core/benchmark\_synthetic.py

scripts/core/benchmark\_synthetic\_parallel\_relation.py

scripts/core/benchmark\_high\_degree.py

scripts/core/benchmark\_depth\_profile.py

scripts/core/benchmark\_depth\_profile\_cache.py

scripts/core/benchmark\_depth\_profile\_parallel.py

scripts/core/benchmark\_parallel\_cache.py

scripts/core/sync\_src\_relation\_index.py

scripts/core/benchmark\_src\_relation\_index.py

scripts/core/generate\_multirelation\_index\_kg.py

scripts/core/benchmark\_path\_relation\_index.py

scripts/core/generate\_noisy\_path\_index\_kg.py

scripts/core/benchmark\_path\_relation\_index\_parallel.py

scripts/core/plot\_parallel\_frontier\_summary.py

scripts/core/plot\_parallel\_path\_relation\_index\_summary.py

```



\---



\## Current Main Results



\### Frontier-level parallel traversal



```text

synthetic\_high\_degree\_21k:

1131.530 ms -> 109.343 ms

10.35x speedup



synthetic\_10k:

1012.130 ms -> 69.085 ms

14.65x speedup

```



\### High-degree-aware cache



```text

parallel only:

120.282 ms

raw\_db = 5069



parallel + high-degree-aware cache:

57.295 ms

raw\_db = 69

```



\### Relation-aware source index



```text

1% selectivity:

111.040 ms -> 15.121 ms

7.34x speedup



10% selectivity:

112.608 ms -> 15.720 ms

7.16x speedup



50% selectivity:

91.281 ms -> 52.838 ms

1.73x speedup

```



\### Parallel + relation-aware index on noisy paths



```text

noisy path v1:

104.506 ms -> 67.518 ms

1.55x speedup



noisy path v1.5:

156.834 ms -> 71.503 ms

2.19x speedup



noisy path v2:

263.788 ms -> 67.626 ms

3.90x speedup

```



\---



\## Important Interpretation



The project has three optimization lines:



```text

1\. Frontier parallel:

&#x20;  reduces layer-wise one-hop query waiting time.



2\. High-degree-aware cache:

&#x20;  reduces repeated raw edge loading for expensive high-degree nodes.



3\. Relation-aware source index:

&#x20;  reduces invalid relation scans on multi-relation nodes.

```



The research direction should remain aligned with the professor's suggestion:



```text

Borrow ideas from Neo4j and optimize indexing and caching for Cassandra-KG traversal.

```



\---



\## Current File Organization



Large generated CSV files have been moved into:



```text

generated\_data/

```



Small sample data has been moved into:



```text

data\_sample/

```



Formal results are under:



```text

results/

```



Figures are under:



```text

figures/

```



\---



\## Recommended Next Steps



1\. Create an optimization roadmap figure;

2\. Build a combined benchmark:



&#x20;  \* serial baseline;

&#x20;  \* parallel only;

&#x20;  \* parallel + cache;

&#x20;  \* parallel + index;

&#x20;  \* parallel + cache + index;

3\. Analyze write amplification:



&#x20;  \* original design: 3 physical writes per logical edge;

&#x20;  \* with src-relation index: 4 physical writes per logical edge;

4\. Draft the conference paper outline.



```

```



