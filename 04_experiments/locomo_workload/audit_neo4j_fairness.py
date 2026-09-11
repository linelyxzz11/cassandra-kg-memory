"""Capture Neo4j indexes, PROFILE plans and runtime facts before scaling concurrency."""
from __future__ import annotations
import json,os,platform,time
from pathlib import Path
from run_canonical_live_gate import ROOT,env
OUT=ROOT/'05_reports'/'locomo_workload_100k'/'neo4j_fairness'
def plan_dict(p):
 if isinstance(p,dict):
  return {'operator':p.get('operatorType') or p.get('operator_type') or '', 'arguments':p.get('args') or p.get('arguments') or {},'identifiers':p.get('identifiers') or [],'children':[plan_dict(x) for x in p.get('children',[])]}
 return {'operator':p.operator_type,'arguments':dict(p.arguments),'identifiers':list(p.identifiers),'children':[plan_dict(x) for x in p.children]}
def operators(p):return [p['operator']]+[z for c in p['children'] for z in operators(c)]
def main():
 env();from neo4j import GraphDatabase,__version__ as driver_version
 d=GraphDatabase.driver(os.getenv('NEO4J_URI','bolt://localhost:7687'),auth=(os.getenv('NEO4J_USER','neo4j'),os.getenv('NEO4J_PASSWORD')),max_connection_pool_size=128);d.verify_connectivity();OUT.mkdir(parents=True,exist_ok=True)
 payload={'status':'PASS','driver_version':driver_version,'pool_max':128,'host':platform.node(),'plans':{},'indexes':[]}
 with d.session(database=os.getenv('NEO4J_DATABASE','neo4j')) as s:
  payload['server_version']=s.run('CALL dbms.components() YIELD versions RETURN versions[0] AS version').single()['version']
  payload['indexes']=[dict(r) for r in s.run("SHOW INDEXES YIELD name,state,type,entityType,labelsOrTypes,properties WHERE any(x IN labelsOrTypes WHERE x STARTS WITH 'LWV1') RETURN name,state,type,entityType,labelsOrTypes,properties")]
  queries={
   'native_candidates':("PROFILE MATCH (m:LWV1NativeMemory {scope_id:$s}) RETURN m.memory_id AS id",{'s':'lr000000::conv-26'}),
   'materialized_candidates':("PROFILE MATCH (m:LWV1MatMemory {scope_id:$s}) RETURN m.memory_id AS id",{'s':'lr000000::conv-26'}),
   'native_projection':("PROFILE MATCH (m:LWV1NativeMemory {scope_id:$s,memory_id:$id})-[:HAS_FEATURE]->(f:LWV1Feature) RETURN m.version,f.entities,f.relations,f.keywords,f.triples",{'s':'lr000000::conv-26','id':'lr000000::conv-26_session_1_D1:1'}),
   'materialized_projection':("PROFILE MATCH (m:LWV1MatMemory {scope_id:$s,memory_id:$id}) RETURN m.version,m.rawerk",{'s':'lr000000::conv-26','id':'lr000000::conv-26_session_1_D1:1'})}
  for name,(q,p) in queries.items():
   result=s.run(q,**p);list(result);summary=result.consume();plan=plan_dict(summary.profile);ops=operators(plan);payload['plans'][name]={'plan':plan,'operators':ops,'contains_all_nodes_scan':any('AllNodesScan' in x for x in ops),'contains_label_scan':any('ByLabelScan' in x for x in ops),'contains_index':any('Index' in x for x in ops)}
 if any(x['state']!='ONLINE' for x in payload['indexes']) or any(v['contains_all_nodes_scan'] for v in payload['plans'].values()) or any(payload['plans'][x]['contains_label_scan'] for x in ('native_candidates','materialized_candidates')):payload['status']='FAIL'
 (OUT/'neo4j_fairness_audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8');print(json.dumps({'status':payload['status'],'server_version':payload['server_version'],'indexes':len(payload['indexes']),'plans':{k:{x:y for x,y in v.items() if x!='plan'} for k,v in payload['plans'].items()}},indent=2));d.close()
 if payload['status']!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
