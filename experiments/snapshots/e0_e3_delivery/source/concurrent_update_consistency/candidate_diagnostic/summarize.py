"""Pool raw per-query observations; keep mean/p95 units explicit."""
import json
from pathlib import Path
import run_diagnostic as diag

HERE=Path(__file__).resolve().parent
out={}
groups={
    'results_v2':['micro_1_stream','micro_1_collect','micro_32_stream','micro_32_collect',
                  'reads_stream_audit','reads_stream_light','reads_collect_audit','reads_collect_light'],
    'matched_index_v2':['stream','collect']}
for folder,families in groups.items():
    for family in families:
        rows=[]
        for f in (HERE/folder).glob(family+'_r*.jsonl'):
            if '_emitter' not in f.name:rows.extend(json.loads(x) for x in f.open(encoding='utf-8'))
        if not rows:continue
        result={'queries':len(rows)}
        for k in ('candidate_ms','acquire_ms','run_ms','fetch_ms','extract_sort_ms','client_thread_cpu_ms',
                  'server_available_ms','server_consumed_ms','queue_ms','active_ms','total_ms','audit_ms'):
            vals=[r[k] for r in rows if r.get(k) is not None]
            if vals:result[k]=diag.stats(vals)
        for k in ('candidate_ok','top10_ok','audit_ok','exact_ids'):
            if any(k in r for r in rows):result[k+'_failures']=sum(r.get(k) is False for r in rows)
        out[folder+'/'+family]=result
diag.write_json(HERE/'pooled_summary.json',out)
print(json.dumps(out,indent=2))
