"""Read-only provenance checks against frozen inputs and historical source."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/retrieval/protocol_audit'
REV = '369e6db'


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def historical(path):
    return subprocess.check_output(['git', 'show', f'{REV}:{path}'], cwd=ROOT).decode('utf-8-sig')


def main():
    memories = rows(ROOT / 'data/locomo_memory_records.csv')
    features = rows(ROOT / 'data/frozen_retrieval/p3_memory_features.csv')
    edges = rows(ROOT / 'data/frozen_retrieval/locomo_kg_edges_spacy.csv')
    gold = rows(ROOT / 'data/retrieval_gold/locomo_cat1_4_gold_memory.csv')
    source_path = '03_src/memory/locomo_retrieval_new_directions.py'
    source = historical(source_path)
    tree = ast.parse(source)
    stop_node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'STOP' for t in n.targets))
    stop = set(next(n.value for n in ast.walk(stop_node.value) if isinstance(n, ast.Constant) and isinstance(n.value, str)).split())

    def tokens(value):
        return [t for t in re.sub(r'[^a-z0-9\s]', ' ', value.lower()).split() if t not in stop and len(t) >= 2]

    mapped = defaultdict(list)
    ordered = sorted(memories, key=lambda row: row['memory_id'])
    for edge in edges:
        for memory in ordered:
            ev = edge['evidence']
            if memory['sample_id'] == edge['graph_id'] and (memory['dia_id'] == ev or ev in memory['memory_id']):
                mapped[memory['memory_id']].append(edge)
                break
    memory_by_id = {r['memory_id']: r for r in memories}
    mismatches = Counter()
    examples = []
    for feature in features:
        mid = feature['memory_id']
        linked = mapped[mid]
        ents = sorted({t for e in linked for t in tokens(e['src_id']) + tokens(e['dst_id'])})[:15]
        rels = sorted({t for e in linked for t in tokens(e['relation'])})[:10]
        keywords = sorted(set(tokens(memory_by_id[mid]['text'])) | set(tokens(', '.join(ents))))[:20] if linked else []
        predicted = {'entities': ', '.join(ents), 'relations': ', '.join(rels), 'keywords': ', '.join(keywords)}
        for field, value in predicted.items():
            if feature[field] != value:
                mismatches[field] += 1
                if len(examples) < 10:
                    examples.append({'memory_id': mid, 'field': field})
    arrays = {}
    for name in ('memory', 'qa'):
        path = ROOT / f'data/locomo_{name}_bge_large.npy'
        array = np.load(path, mmap_mode='r')
        arrays[name] = {'shape': list(array.shape), 'dtype': str(array.dtype), 'finite': bool(np.isfinite(array).all()), 'zero_rows': int((np.linalg.norm(array, axis=1) == 0).sum()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {
        'historical_revision': REV, 'reconstruction_source': source_path,
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'memory_rows': len(memories), 'feature_rows': len(features),
        'edge_rows': len(edges), 'edge_source_counts': dict(Counter(r['source'] for r in edges)),
        'reconstructed_field_mismatches': {f: mismatches[f] for f in ('entities', 'relations', 'keywords')},
        'mismatch_examples': examples,
        'scope_count': len({r['sample_id'] for r in memories}),
        'queries_by_split': dict(Counter(r['split'] for r in gold)),
        'evaluable_queries_by_split': dict(Counter(r['split'] for r in gold if int(r['gold_count']) > 0)),
        'embedding_arrays': arrays,
        'interpretation': 'Reconstruction equality supports derivation from the supplied KG edge artifact. That artifact is observation-derived. It does not establish raw-turn-only extraction or absence of upstream benchmark annotation leakage.'
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'provenance.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
