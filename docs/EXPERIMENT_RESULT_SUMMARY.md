\# Experiment Result Summary



\## Research Direction



This project studies profiling-guided optimization for multi-hop knowledge graph traversal on Cassandra. The current optimization work focuses on three directions:



1\. Frontier-level parallel traversal;

2\. High-degree-aware one-hop cache;

3\. Relation-aware source index.



\---



\## 1. Frontier-level Parallel Traversal



The baseline traversal queries each frontier node serially. The optimized traversal queries nodes within the same frontier level concurrently.



\### Results



| Graph                       | Baseline Latency | Optimized Latency | Speedup |

| --------------------------- | ---------------: | ----------------: | ------: |

| `synthetic\_high\_degree\_21k` |      1131.530 ms |        109.343 ms |  10.35x |

| `synthetic\_10k`             |      1012.130 ms |         69.085 ms |  14.65x |



\### Finding



Frontier-level parallel traversal significantly reduces wall-clock latency by avoiding serial waiting across multiple one-hop Cassandra queries.



\---



\## 2. High-degree-aware Cache



A high-degree-aware cache prioritizes nodes with large one-hop adjacency lists.



\### Result



| Method                             |    Latency | Raw Edges from Cassandra |

| ---------------------------------- | ---------: | -----------------------: |

| Parallel only                      | 120.282 ms |                     5069 |

| Parallel + high-degree-aware cache |  57.295 ms |                       69 |



\### Finding



High-degree-aware cache effectively avoids repeated raw edge loading for expensive high-degree nodes.



\---



\## 3. Relation-aware Source Index



A new table `kg\_edges\_by\_src\_relation` supports direct query by:



```text

graph\_id + src\_id + relation

```



This avoids reading all outgoing edges before relation filtering.



\### Single-node Selectivity Results



| Selectivity | Baseline Latency | Index Latency | Speedup | Raw Scan Reduction |

| ----------: | ---------------: | ------------: | ------: | -----------------: |

|          1% |       111.040 ms |     15.121 ms |   7.34x |       10000 -> 100 |

|         10% |       112.608 ms |     15.720 ms |   7.16x |      10000 -> 1000 |

|         50% |        91.281 ms |     52.838 ms |   1.73x |      10000 -> 5000 |



\### Finding



Relation-aware index is highly effective for low-selectivity relation queries on multi-relation nodes.



\---



\## 4. Parallel + Relation-aware Index on Noisy Multi-hop Paths



Noisy path graphs were generated to evaluate index performance under increasing relation noise.



\### Results



| Graph           | Baseline Raw | Index Raw | Baseline Latency | Index Latency | Speedup |

| --------------- | -----------: | --------: | ---------------: | ------------: | ------: |

| noisy path v1   |       12,580 |        80 |       104.506 ms |     67.518 ms |   1.55x |

| noisy path v1.5 |       62,080 |        80 |       156.834 ms |     71.503 ms |   2.19x |

| noisy path v2   |      125,080 |        80 |       263.788 ms |     67.626 ms |   3.90x |



\### Finding



As relation noise increases, baseline raw edge scans grow significantly, while relation-aware index keeps raw scans stable. This demonstrates the scalability of relation-aware indexing for noisy multi-hop traversal.



\---



\## 5. Current Research Contributions



The current stage supports the following potential contributions:



1\. A depth-level profiling method for Cassandra-based KG traversal;

2\. A frontier-level parallel traversal strategy;

3\. A high-degree-aware one-hop cache strategy;

4\. A relation-aware source index for reducing invalid relation scans;

5\. A discussion of read performance improvement versus write amplification.



\---



\## 6. Next Step



The next experiment should build a unified optimization pipeline benchmark:



```text

serial baseline

parallel only

parallel + high-degree-aware cache

parallel + relation-aware index

parallel + cache + index

```



This will help transform the current experimental results into a more complete conference-paper-level evaluation.



```

```



