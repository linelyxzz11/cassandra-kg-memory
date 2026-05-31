\# Cassandra-KG Memory Project Process Record



\## 1. Project Overview



This project investigates how to support knowledge graph memory storage and multi-hop traversal on Apache Cassandra. Since Cassandra is not a native graph database, the project follows a query-driven data modelling strategy and implements graph traversal logic at the Python application layer.



The project started from basic knowledge graph modelling and gradually evolved into a profiling-guided optimization study for Cassandra-based KG traversal. The current research direction focuses on three optimization lines:



1\. Frontier-level parallel traversal;

2\. High-degree-aware one-hop cache;

3\. Relation-aware source index.



The current research topic can be summarized as:



```text

Profiling-guided optimization for multi-hop KG traversal on Cassandra

```



\---



\## 2. Knowledge Graph Modelling



The basic knowledge graph unit is a directed labelled edge:



```text

src\_id --relation--> dst\_id

```



Each edge includes:



\* `graph\_id`

\* `src\_id`

\* `src\_type`

\* `relation`

\* `dst\_id`

\* `dst\_type`

\* `confidence`

\* `source`

\* `edge\_id`

\* `created\_at`



The graph is represented as a directed labelled graph with edge properties.



\---



\## 3. Cassandra Table Design



The initial Cassandra-KG storage model contains three core access tables.



\### 3.1 `kg\_edges\_by\_src`



This table supports forward traversal from a source node.



Typical query:



```sql

WHERE graph\_id = ? AND src\_id = ?

```



It is the main table for one-hop query and multi-hop path expansion.



\### 3.2 `kg\_edges\_by\_dst`



This table supports reverse lookup from a destination node.



Typical query:



```sql

WHERE graph\_id = ? AND dst\_id = ?

```



It is used to answer questions such as which nodes point to a given target node.



\### 3.3 `kg\_edges\_by\_relation\_bucket`



This table supports relation-level scans.



Typical query:



```sql

WHERE graph\_id = ? AND relation = ? AND bucket = ?

```



The bucket field is used to avoid oversized partitions and to enable parallel bucket-level relation scans.



\### 3.4 `kg\_edges\_by\_src\_relation`



A fourth table was later added as a relation-aware source index.



Typical query:



```sql

WHERE graph\_id = ? AND src\_id = ? AND relation = ?

```



This table avoids scanning all outgoing edges under a source node when only a specific relation is needed.



\---



\## 4. Initial Prototype Stage



The initial prototype was implemented under `scripts/initial\_step/`.



Main scripts:



\* `insert\_sample\_edges.py`: inserts a small sample KG;

\* `kg\_query\_3hop.py`: performs early 3-hop traversal;

\* `sync\_reverse\_edges.py`: syncs data from source table to destination table;

\* `query\_reverse.py`: queries incoming edges by destination node;

\* `sync\_relation\_edges.py`: syncs data to relation bucket table;

\* `benchmark\_queries.py`: benchmarks forward, reverse, relation, and two-hop queries.



This stage verified that Cassandra can support basic forward lookup, reverse lookup, relation lookup, and simple multi-hop traversal through application-layer Python logic.



\---



\## 5. Synthetic KG Generation Stage



The project then moved from small handcrafted data to synthetic structured KG generation.



Main scripts:



\* `generate\_synthetic\_kg.py`

\* `bulk\_insert\_kg.py`

\* `bulk\_insert\_kg\_v2.py`

\* `query\_synthetic\_paths.py`

\* `query\_by\_relation.py`

\* `benchmark\_synthetic.py`



The synthetic KG contains structured chains such as:



```text

user --likes--> preference

preference --suitable\_for--> need

need --related\_to--> state

state --suggests--> strategy

```



and:



```text

user --studies--> topic

topic --includes--> technology

technology --used\_for--> task

task --requires--> resource

```



The benchmark results showed that multi-hop latency increases with depth, and relation-level bucket scans are expensive when performed serially.



\---



\## 6. Relation Bucket Parallel Optimization



The script `benchmark\_synthetic\_parallel\_relation.py` was implemented to parallelize relation bucket scans.



The original relation query scans buckets serially:



```text

bucket=0, bucket=1, ..., bucket=63

```



The optimized version queries multiple buckets concurrently.



Representative result on `synthetic\_10k`:



```text

Serial bucket relation query: approximately 998 ms

Parallel bucket relation query with 16 workers: approximately 33 ms

Speedup: approximately 29.61x

```



This verified that relation bucket tables need parallel querying to fully exploit the bucketed design.



\---



\## 7. High-degree KG Stage



A high-degree graph was generated using `generate\_high\_degree\_kg.py`.



Core structure:



```text

user\_big --mentions--> entity\_i

entity\_i --has\_feature--> feature\_i

feature\_i --supports--> need\_i

need\_i --related\_to--> state\_i

```



Representative configuration:



```text

high-degree node: user\_big

outgoing mentions edges: 5000

total logical edges: 21000

graph id: synthetic\_high\_degree\_21k

```



The script `benchmark\_high\_degree.py` showed that query latency increases significantly as fanout increases.



\---



\## 8. Depth-level Profiling Stage



The script `benchmark\_depth\_profile.py` was implemented to profile traversal cost level by level.



Recorded metrics include:



\* level;

\* frontier size;

\* Cassandra query count;

\* raw edge count;

\* expanded edge count;

\* new paths;

\* next frontier size;

\* total paths;

\* average latency;

\* p95 latency.



A key finding is:



```text

fanout controls expanded edges, but does not prevent Cassandra from reading all raw outgoing edges of a source node.

```



For example, in the high-degree graph:



```text

level 1 raw edges = 5000

level 1 expanded edges = 20

```



This finding motivated both high-degree-aware cache and relation-aware index optimization.



\---



\## 9. Cache Optimization Stage



The script `benchmark\_depth\_profile\_cache.py` was implemented to evaluate one-hop cache strategies.



Cache evolution:



1\. Unlimited one-hop cache;

2\. Bounded LRU cache;

3\. High-degree-priority cache.



The unlimited cache was used only as an upper-bound reference. The main practical direction shifted to finite-capacity cache.



A key observation is that ordinary LRU can suffer from cache thrashing in sequential multi-hop traversal when the cache capacity is smaller than the traversal working set.



The high-degree-aware cache prioritizes nodes with large raw edge counts. It is designed to protect expensive high-degree nodes such as `user\_big`.



Representative result under parallel + cache:



```text

Parallel only:

latency = 120.282 ms

raw edges from Cassandra = 5069



Parallel + high-degree-aware cache:

latency = 57.295 ms

raw edges from Cassandra = 69

```



This shows that high-degree-aware cache can effectively reduce repeated raw edge loading for expensive nodes.



\---



\## 10. Frontier-level Parallel Traversal



The script `benchmark\_depth\_profile\_parallel.py` was implemented to parallelize the expansion of nodes within the same frontier level.



This is different from relation bucket parallelism:



```text

relation bucket parallel:

parallelizes bucket scans in kg\_edges\_by\_relation\_bucket



frontier parallel:

parallelizes multiple src\_id one-hop queries in kg\_edges\_by\_src

```



Representative results:



```text

synthetic\_high\_degree\_21k:

serial latency approximately 1131.530 ms

parallel latency with 16 workers approximately 109.343 ms

speedup approximately 10.35x



synthetic\_10k:

serial latency approximately 1012.130 ms

parallel latency with 32 workers approximately 69.085 ms

speedup approximately 14.65x

```



This shows that a major bottleneck in Cassandra-KG traversal is the serial waiting time of many one-hop queries within each frontier layer.



\---



\## 11. Parallel + Cache Combination



The script `benchmark\_parallel\_cache.py` combines frontier-level parallel execution with bounded one-hop cache.



Representative result on `synthetic\_high\_degree\_21k`:



```text

parallel only:

latency = 120.282 ms

raw\_db = 5069



parallel + LRU:

latency = 120.222 ms

raw\_db = 5069



parallel + high-degree-aware cache:

latency = 57.295 ms

raw\_db = 69

```



This shows that high-degree-aware cache is more suitable than ordinary LRU for high-degree KG traversal workloads.



\---



\## 12. Relation-aware Source Index



A new table `kg\_edges\_by\_src\_relation` was added.



The script `sync\_src\_relation\_index.py` backfills existing edges from `kg\_edges\_by\_src` into the new index table.



The script `bulk\_insert\_kg\_v3.py` maintains the four access tables during insertion:



1\. `kg\_edges\_by\_src`;

2\. `kg\_edges\_by\_dst`;

3\. `kg\_edges\_by\_relation\_bucket`;

4\. `kg\_edges\_by\_src\_relation`.



Write amplification changes from:



```text

1 logical edge -> 3 physical writes

```



to:



```text

1 logical edge -> 4 physical writes

```



This provides the basis for discussing read performance improvement versus write amplification cost.



\---



\## 13. Single-node Relation Index Benchmark



The script `generate\_multirelation\_index\_kg.py` generates multi-relation high-degree nodes.



The script `benchmark\_src\_relation\_index.py` compares:



```text

baseline\_src\_scan:

read all outgoing edges and filter relation in Python



src\_relation\_index:

directly query by graph\_id + src\_id + relation

```



Representative selectivity results:



```text

1% selectivity:

10000 -> 100 raw edges

111.040 ms -> 15.121 ms

7.34x speedup



10% selectivity:

10000 -> 1000 raw edges

112.608 ms -> 15.720 ms

7.16x speedup



50% selectivity:

10000 -> 5000 raw edges

91.281 ms -> 52.838 ms

1.73x speedup

```



This verifies that relation-aware index is effective for low-selectivity relation queries on multi-relation nodes.



\---



\## 14. Noisy Multi-hop Path Index Benchmark



The script `generate\_noisy\_path\_index\_kg.py` generates noisy multi-hop path graphs.



Target path:



```text

user\_path\_mix --likes--> preference\_i

preference\_i --suitable\_for--> need\_i

need\_i --related\_to--> state\_i

state\_i --suggests--> strategy\_i

```



Noise relations are added to every level, forcing the baseline to scan many irrelevant outgoing edges.



The script `benchmark\_path\_relation\_index\_parallel.py` compares parallel baseline traversal and parallel relation-index traversal.



Representative results:



```text

noisy path v1:

baseline raw = 12580

index raw = 80

latency = 104.506 ms -> 67.518 ms

speedup = 1.55x



noisy path v1.5:

baseline raw = 62080

index raw = 80

latency = 156.834 ms -> 71.503 ms

speedup = 2.19x



noisy path v2:

baseline raw = 125080

index raw = 80

latency = 263.788 ms -> 67.626 ms

speedup = 3.90x

```



This demonstrates that as relation noise increases, baseline raw scans and latency increase, while relation-aware index keeps raw scans stable.



\---



\## 15. Current Research Contributions



The current project can support the following potential paper contributions:



1\. Depth-level profiling for Cassandra-based KG traversal;

2\. Frontier-level parallel traversal;

3\. High-degree-aware one-hop cache;

4\. Relation-aware source index;

5\. Read performance versus write amplification analysis.



\---



\## 16. Current Core Results



| Optimization              |          Scenario |    Baseline |  Optimized | Speedup |

| ------------------------- | ----------------: | ----------: | ---------: | ------: |

| Frontier parallel         | high-degree graph | 1131.530 ms | 109.343 ms |  10.35x |

| Frontier parallel         |  structured graph | 1012.130 ms |  69.085 ms |  14.65x |

| High-degree cache         |          user\_big |  120.282 ms |  57.295 ms |   2.10x |

| Relation index            |    1% selectivity |  111.040 ms |  15.121 ms |   7.34x |

| Relation index            |   10% selectivity |  112.608 ms |  15.720 ms |   7.16x |

| Relation index            |   50% selectivity |   91.281 ms |  52.838 ms |   1.73x |

| Parallel + relation index |     noisy path v1 |  104.506 ms |  67.518 ms |   1.55x |

| Parallel + relation index |   noisy path v1.5 |  156.834 ms |  71.503 ms |   2.19x |

| Parallel + relation index |     noisy path v2 |  263.788 ms |  67.626 ms |   3.90x |



\---



\## 17. Next Steps



Recommended next steps:



1\. Commit the cleaned project structure to Git;

2\. Produce an optimization roadmap figure;

3\. Build a unified combined pipeline benchmark:



&#x20;  \* baseline serial;

&#x20;  \* parallel only;

&#x20;  \* parallel + cache;

&#x20;  \* parallel + index;

&#x20;  \* parallel + cache + index;

4\. Analyze write amplification from 3x to 4x;

5\. Start drafting the conference paper outline.



```

```



