from cassandra.cluster import Cluster

session = Cluster(["127.0.0.1"], port=9042).connect("ai_memory")

for eid in ["entity_0", "entity_3094", "entity_9999", "nonexistent"]:
    try:
        rows = session.execute(
            "SELECT src_id, relation, dst_id, source FROM kg_edges_by_src WHERE graph_id=%s AND src_id=%s",
            ("synth_1M", eid),
        )
        result = list(rows)
        print(f"src={eid}: {len(result)} edges returned")
        if result:
            print(f"  first: {result[0]}")
    except Exception as ex:
        print(f"src={eid}: ERROR - {ex}")

session.cluster.shutdown()