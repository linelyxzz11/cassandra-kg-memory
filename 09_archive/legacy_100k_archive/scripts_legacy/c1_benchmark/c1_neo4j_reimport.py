import csv, json, hashlib, os, sys, random
from pathlib import Path
from neo4j import GraphDatabase

GR = 'synth_100000_1781447372'
NEO_L = 'C1KGNode'
NEO_R = 'C1KG_EDGE'
FAN = 20
REPORT = Path('D:/memorytable/cassandra-kg-memory/reports/c1_preflight_100k')
REPORT.mkdir(parents=True, exist_ok=True)
MANIFEST = Path('D:/memorytable/cassandra-kg-memory/results/c1_manifest_100k_h2.jsonl')
CSV = Path('D:/memorytable/cassandra-kg-memory/results/c1_source_100k.csv')
PWD = os.environ['NEO4J_PASSWORD']

d = GraphDatabase.driver('bolt://127.0.0.1:7687', auth=('neo4j', PWD))
print('[1/5] Clear + constraint...', flush=True)
with d.session() as s:
    s.run(f'MATCH (n:{NEO_L}) DETACH DELETE n')
    s.run(f'CREATE CONSTRAINT c1_uq IF NOT EXISTS FOR (n:{NEO_L}) REQUIRE (n.graph_id, n.node_id) IS UNIQUE')
print('  Done', flush=True)

print('[2/5] Import from CSV...', flush=True)
edges = []
with CSV.open(encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        edges.append({'g': r['graph_id'], 's': r['src_id'], 'd': r['dst_id'],
                      'rel': r['relation'], 'src': r['source']})
n_edges = len(edges)
print(f'  Loaded {n_edges} edges', flush=True)

nodes_set = set()
for e in edges:
    nodes_set.add((e['g'], e['s']))
    nodes_set.add((e['g'], e['d']))
nl = [{'g': n[0], 'nid': n[1]} for n in nodes_set]
for i in range(0, len(nl), 5000):
    with d.session() as s:
        s.run(f'UNWIND $rows AS r MERGE (n:{NEO_L} {{graph_id: r.g, node_id: r.nid}})', rows=nl[i:i+5000]).consume()
print(f'  Nodes: {len(nl)}', flush=True)

for i in range(0, n_edges, 5000):
    batch = []
    for e in edges[i:i+5000]:
        batch.append({'g': e['g'], 's': e['s'], 'd': e['d'], 'rel': e['rel'], 'src': e['src']})
    with d.session() as s:
        s.run(
            f'UNWIND $rows AS r '
            f'MATCH (s:{NEO_L} {{graph_id: r.g, node_id: r.s}}) '
            f'MATCH (d:{NEO_L} {{graph_id: r.g, node_id: r.d}}) '
            f'CREATE (s)-[:{NEO_R} {{relation: r.rel, source: r.src, graph_id: r.g}}]->(d)',
            rows=batch,
        ).consume()
    print(f'  Edges: {min(i+5000, n_edges)}/{n_edges}', flush=True)

print('[3/5] Full-graph import preflight...', flush=True)
cass_set = set()
for e in edges:
    cass_set.add((GR, e['s'], e['rel'], e['d'], e['src']))
neo_set = set()
with d.session() as s:
    rows = s.run(
        f'MATCH (s:{NEO_L})-[r:{NEO_R}]->(d:{NEO_L}) '
        'RETURN s.node_id AS sid, r.relation AS rel, d.node_id AS did, r.source AS src'
    )
    for row in rows:
        neo_set.add((GR, row['sid'], row['rel'], row['did'], row['src'] or ''))
missing = sorted(cass_set - neo_set)
extra = sorted(neo_set - cass_set)
fp = {'cassandra': len(cass_set), 'neo4j': len(neo_set), 'mismatches': len(missing)+len(extra)}
with (REPORT/'full_graph_import_preflight_rerun.json').open('w') as f:
    json.dump(fp, f, indent=2)
if missing or extra:
    with (REPORT/'full_graph_import_mismatches_rerun.jsonl').open('w') as f:
        for m in missing[:50]: f.write(json.dumps({'type':'missing','key':list(m)})+'\n')
        for e in extra[:50]: f.write(json.dumps({'type':'extra','key':list(e)})+'\n')
print(json.dumps(fp, indent=2), flush=True)
if fp['mismatches'] > 0:
    print('ABORT', flush=True)
    d.close(); sys.exit(1)

print('[4/5] Neo4j spotcheck...', flush=True)
with MANIFEST.open() as f:
    queries = [json.loads(line) for line in f]
rng = random.Random(42)
checks = rng.sample(queries, 10)
results = []
for q in checks:
    frontier = {(q['seed_id'], (q['seed_id'],), ())}
    for rel in q['relation_path']:
        sources = sorted({s[0] for s in frontier})
        se = {}
        for src in sources:
            with d.session() as s:
                rows = list(s.run(
                    f'MATCH (n:{NEO_L} {{graph_id: $gid, node_id: $nid}})'
                    f'-[r:{NEO_R} {{relation: $rel}}]->'
                    f'(m:{NEO_L} {{graph_id: $gid}}) '
                    'RETURN n.node_id AS s, r.relation AS r, m.node_id AS d, coalesce(r.source, "") AS src '
                    'ORDER BY r, d, src',
                    gid=GR, nid=src, rel=rel))
            se[src] = [(row['s'], row['r'], row['d'], row['src']) for row in rows]
        nf = set()
        for src, np, ep in frontier:
            for e in se.get(src, [])[:FAN]:
                if e[2] not in np:
                    nf.add((e[2], np+(e[2],), ep+(e[3],)))
        frontier = nf
        if not frontier:
            break
    neo_paths = tuple(sorted({s[2] for s in frontier}))
    canon = sorted([list(p) for p in sorted(neo_paths)])
    h = hashlib.sha256(json.dumps(canon, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    ok = h == q['expected_path_hash']
    results.append({'query_id': q['query_id'], 'match': ok})
    mark = 'MATCH' if ok else 'MISMATCH'
    print(f'  {q["query_id"]}: {mark} (paths={len(neo_paths)})', flush=True)
    if not ok:
        print(f'    expected={q["expected_path_hash"][:16]} got={h[:16]}', flush=True)

with (REPORT/'neo4j_spotcheck_after_reimport.jsonl').open('w') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False)+'\n')
n_ok = sum(1 for r in results if r['match'])
print(f'  {n_ok}/10', flush=True)

d.close()
if n_ok < 10:
    print('Spotcheck FAILED', flush=True)
    sys.exit(1)
print('[5/5] All gates passed. Ready for C1 pilot.', flush=True)
