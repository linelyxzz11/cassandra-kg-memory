"""Experiment-only Neo4j candidate return optimization; same sorted memory IDs."""
from cassmem.backend.live_cells_graph import Neo4jGraphCells

class Neo4jCollectedGraphCells(Neo4jGraphCells):
    def ids(self, cell, scope):
        label='LWV2NativeMemory' if cell=='neo4j-native' else 'LWV2MatMemory'
        rows=self.run(f'MATCH (m:{label} {{scope_id:$s}}) RETURN collect(m.memory_id) AS ids',s=scope)
        return sorted(rows[0]['ids'])

    def validate_candidate_equivalence(self, cell, scopes):
        checks=[]
        for scope in sorted(scopes):
            original=super().ids(cell,scope)
            revised=self.ids(cell,scope)
            checks.append({'scope':scope,'count':len(original),'exact_match':original==revised})
        return checks
