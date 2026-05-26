&#x20;**Cassandra KG Memory Experiment**



This repository contains an experimental prototype for storing and querying knowledge graph triples on Apache Cassandra. The project is motivated by AI memory systems, where long-term memories may need to be represented not only as text records, but also as entity-relation triples with limited multi-hop access.



The goal is not to replace native graph databases. Instead, this project evaluates how far Cassandra's query-driven data modeling, wide-column tables, denormalized views, and application-layer traversal can support common knowledge graph access patterns.



**1. Motivation**



AI memory systems often require:



\- long-term memory storage

\- user/session/agent scoped retrieval

\- entity-relation representation

\- forward and reverse relation lookup

\- limited multi-hop access

\- traceable and benchmarkable query behavior



Cassandra does not provide native graph traversal. Therefore, this project follows a query-driven design: each common access pattern is mapped to a dedicated Cassandra table.



**2. Storage Model**



The current design uses three denormalized edge tables.



**2.1 Forward edge table**



`kg\_edges\_by\_src`



This table supports outgoing edge queries from a source node.



Example access pattern:





Given src\_id, find all outgoing edges.





Example:





user\_000001 --likes--> preference\_000001





&#x20;**2.2 Reverse edge table**



`kg\_edges\_by\_dst`



This table supports incoming edge queries to a destination node.



Example access pattern:





Given dst\_id, find all incoming edges.





Example:





Who points to need\_000001?





**2.3 Relation bucket table**



`kg\_edges\_by\_relation\_bucket`



This table supports global relation-level queries.



Example access pattern:





Given relation, find all edges with this relation.





Since a relation may contain many edges, the table uses bucket partitioning:





(graph\_id, relation, bucket)





Relation queries can be executed either serially across buckets or in parallel.



**3. Query Layer**



Cassandra is used for one-hop edge access. Multi-hop traversal is implemented in Python at the application layer.



The query layer supports:



\* relation whitelist

\* max fanout control

\* path-level visited set

\* logical edge deduplication

\* path score calculation

\* longest-path-first display



The core idea is:





Cassandra retrieves one-hop edges.

Python expands paths layer by layer.





&#x20;**4. Repository Structure**





cassandra-kg-memory/

├── schema/

│   └── cassandra\_tables.cql

├── scripts/

│   ├── core/

│   │   ├── generate\_synthetic\_kg.py

│   │   ├── generate\_high\_degree\_kg.py

│   │   ├── bulk\_insert\_kg.py

│   │   ├── bulk\_insert\_kg\_v2.py

│   │   ├── query\_synthetic\_paths.py

│   │   ├── query\_by\_relation.py

│   │   ├── query\_high\_degree\_forward.py

│   │   ├── benchmark\_synthetic.py

│   │   ├── benchmark\_synthetic\_parallel\_relation.py

│   │   └── benchmark\_high\_degree.py

│   └── initial\_step/

│       ├── insert\_sample\_edges.py

│       ├── kg\_query\_3hop.py

│       ├── sync\_reverse\_edges.py

│       ├── sync\_relation\_edges.py

│       ├── query\_reverse.py

│       └── benchmark\_queries.py

├── data\_sample/

├── results/

├── docs/

├── figures/

├── requirements.txt

└── README.md





**5. Core Scripts**



**Synthetic KG generation**



`scripts/core/generate\_synthetic\_kg.py`



Generates structured synthetic KG data with:



\* anchor users

\* multi-hop path templates

\* random noise triples



Example:





python scripts/core/generate\_synthetic\_kg.py --edges 1000 --users 100 --anchors 5 --chains-per-anchor 5 --noise-ratio 0.30 --output edges\_structured\_1k.csv





&#x20;**High-degree KG generation**



`scripts/core/generate\_high\_degree\_kg.py`



Generates a high-degree node graph where `user\_big` has many outgoing edges.



Example:



python scripts/core/generate\_high\_degree\_kg.py --high-degree 5000 --tail-depth 4 --noise-edges 1000 --output edges\_high\_degree\_21k.csv





&#x20;**Bulk insertion**



`scripts/core/bulk\_insert\_kg.py`



`scripts/core/bulk\_insert\_kg\_v2.py`



Writes KG triples into three Cassandra tables.



Example:





python scripts/core/bulk\_insert\_kg.py --file edges\_structured\_1k.csv --graph-id synthetic\_1k --bucket-count 32





For high-degree experiments:





python scripts/core/bulk\_insert\_kg\_v2.py --file edges\_high\_degree\_21k.csv --graph-id synthetic\_high\_degree\_21k --bucket-count 64 --bucket-mode dst





**Multi-hop path query**



`scripts/core/query\_synthetic\_paths.py`



Example:





python scripts/core/query\_synthetic\_paths.py --graph-id synthetic\_10k --start user\_000001 --depth 5 --fanout 20 --limit 30 --longest-first





**Relation query**



`scripts/core/query\_by\_relation.py`



Example:





python scripts/core/query\_by\_relation.py --graph-id synthetic\_10k --relation suitable\_for --bucket-count 64 --limit 20





**Synthetic benchmark**



`scripts/core/benchmark\_synthetic.py`



Example:





python scripts/core/benchmark\_synthetic.py --graph-id synthetic\_10k --start user\_000001 --reverse-dst need\_000001 --relation suitable\_for --bucket-count 64 --fanout 20 --repeat 20 --warmup 3





**Parallel relation benchmark**



`scripts/core/benchmark\_synthetic\_parallel\_relation.py`



Example:





python scripts/core/benchmark\_synthetic\_parallel\_relation.py --graph-id synthetic\_10k --relation suitable\_for --bucket-count 64 --workers 16 --repeat 20 --warmup 3





&#x20;**High-degree benchmark**



`scripts/core/benchmark\_high\_degree.py`



Example:





python scripts/core/benchmark\_high\_degree.py --graph-id synthetic\_high\_degree\_21k --start user\_big --fanout 100 --repeat 10 --warmup 2





**6. Initial Prototype Scripts**



The `scripts/initial\_step/` directory contains early feasibility validation scripts.



These scripts were used to validate:



\* small-scale KG edge insertion

\* forward edge queries

\* reverse edge materialization

\* relation index materialization

\* early 3-hop application-layer traversal

\* early query benchmark



They are kept for reproducibility and development history.



**7. Current Experimental Findings**



Current experiments show:



1\. Cassandra can support stable one-hop forward and reverse edge queries.

2\. Multi-hop access must be implemented at the application layer.

3\. Multi-hop latency increases with query depth and fanout.

4\. Relation bucket queries are expensive when scanned serially.

5\. Parallel bucket querying significantly reduces relation query latency.

6\. High-degree node one-hop reading is feasible at 5000 outgoing edges, but multi-hop expansion becomes expensive without fanout control.



**8. Limitations**



This project is an experimental prototype.



Current limitations include:



\* synthetic data only

\* no comparison with native graph databases yet

\* no distributed Cassandra cluster evaluation yet

\* limited write-throughput analysis

\* limited real AI memory workload evaluation



**9. Planned Work**



Future work includes:



\* scaling synthetic KG to 100k edges

\* adding write amplification and storage cost analysis

\* generating benchmark figures from CSV results

\* evaluating more high-degree settings

\* comparing different bucket strategies

\* preparing a technical report or conference-style paper draft



