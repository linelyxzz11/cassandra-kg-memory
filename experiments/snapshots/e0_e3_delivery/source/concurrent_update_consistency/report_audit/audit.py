"""Read existing experiment logs; emit derived report evidence, never run workloads."""
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/formal_20261013_v2'

def stats(xs):
    return dict(n=len(xs), mean=float(np.mean(xs)), p50=float(np.percentile(xs, 50)),
                p95=float(np.percentile(xs, 95))) if xs else dict(n=0)

groups = defaultdict(lambda: dict(counts=Counter(), durations=defaultdict(list)))
points = []
for folder in sorted(RUN.iterdir()):
    if not (folder / 'summary.json').exists():
        continue
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8-sig'))
    layout, mode, rate, rep = folder.name.rsplit('_', 3)
    group = groups[f'{layout}_{rate}']
    counts, durations = group['counts'], group['durations']
    point_counts = Counter()
    deadline = json.loads((folder / 'measured_deadline.json').read_text(encoding='utf-8-sig'))['deadline_ns']
    for name in ('updates', 'business_reads', 'observation_reads', 'post_ready_valid_reads',
                 'post_ready_target_failures', 'mixed_reads_all_windows', 'errors',
                 'ready_updates', 'deadline_censored_updates'):
        counts[name] += summary[name]
    counts['points'] += 1
    counts['load_valid_points'] += int(summary['offered_load_valid'])
    with (folder / 'measured_observations.jsonl').open(encoding='utf-8-sig') as f:
        for line in f:
            r = json.loads(line)
            if r['kind'] == 'update':
                if 'ready_ns' not in r or r['ready_ns'] is None:
                    continue
                assert r['ready_check']['raw_version'] == 2, (folder, r['event_id'])
                for name, start, end in (
                    ('queue_ms', 'submit_ns', 'start_ns'),
                    ('active_ms', 'start_ns', 'ready_ns'),
                    ('arrival_ready_ms', 'submit_ns', 'ready_ns'),
                    ('raw_call_ms', 'raw_call_start_ns', 'raw_ack_ns')):
                    durations[name].append((r[end] - r[start]) / 1e6)
                if r['ready_ns'] > deadline:
                    point_counts['overdue_recomputed'] += 1
                    counts['overdue_not_started_at_deadline'] += int(r['start_ns'] > deadline)
                continue
            point_counts['read_rows'] += 1
            phase = r['phase_class']
            counts['phase_' + phase] += 1
            follow = r.get('followup', False)
            post = phase == 'post_ready'
            counts['followups'] += int(follow)
            if post:
                counts['postready_followup' if follow else 'postready_business'] += 1
                counts['postready_start_by_deadline' if r['start_ns'] <= deadline else 'postready_start_after_deadline'] += 1
                point_counts['postready_recomputed'] += 1
            # Every formal target is version 2; source records start at version 1.
            # Null/non-candidate is distinct from a returned old candidate.
            for branch in ('sparse', 'dense'):
                version = r.get(f'target_{branch}_candidate_version')
                if version is not None:
                    counts[branch + '_target_candidate_observations'] += 1
                    if int(version) < 2:
                        counts[branch + '_old_target_candidate'] += 1
                        counts[branch + '_old_target_candidate_' + phase] += 1
            if r.get('stale_relation'):
                counts['stale_relation_' + phase] += 1
    assert point_counts['read_rows'] == summary['observation_reads'], folder
    assert point_counts['postready_recomputed'] == summary['post_ready_valid_reads'], folder
    assert point_counts['overdue_recomputed'] == summary['deadline_censored_updates'], folder
    intervals = json.loads((folder / 'measured_intervals.json').read_text(encoding='utf-8-sig'))
    for r in intervals:
        if r['bad_observations']:
            counts['events_with_bad_observation'] += 1
            durations['observed_bad_span_ms'].append((r['last_bad_end_ns'] - r['first_bad_start_ns']) / 1e6)
            if r['first_later_good_end_ns'] is not None:
                durations['first_bad_to_later_good_ms'].append((r['first_later_good_end_ns'] - r['first_bad_start_ns']) / 1e6)
            counts['unrecovered_events'] += int(r['unrecovered_in_observation_window'])
    points.append(dict(point=folder.name, **summary))

output = dict(points=points, groups={k: dict(counts=dict(v['counts']),
              durations={n: stats(a) for n, a in v['durations'].items()}) for k, v in groups.items()},
              note='Durations pooled across modes/repeats; observation spans are not continuous stale durations.')
dest = Path(__file__).with_name('audit_results.json')
dest.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
for key, v in output['groups'].items():
    print(key, json.dumps(v, ensure_ascii=False))
print('Validated', len(points), 'points; generated', dest)
