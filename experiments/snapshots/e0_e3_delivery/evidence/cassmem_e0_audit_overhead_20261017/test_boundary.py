import ast,json,platform,sys
from pathlib import Path
import run as R

s=R.phase_source()
replacement="                row['read_audit_enabled']=self.audit_arm=='full'\n                if self.audit_arm=='full':\n                    row['read_audit_start_ns']=time.perf_counter_ns();row['audit']=self.capture(i);row['read_audit_done_ns']=time.perf_counter_ns()\n"
assert s.count(replacement)==1
assert s.replace(replacement,"                row['audit']=self.capture(i)\n")==R.F.phase_source()
assert "self.service.apply(r,self.vectors[i][2],hook=hook)" in s
matrix=R.F.load(R.OUT/'formal_matrix.json')
groups={}
for p in matrix:groups.setdefault((p['layout'],p['rate'],p['seed']),[]).append(p)
assert len(groups)==36
assert all({p['audit_arm'] for p in pair}=={'full','read_audit_off'} and len({p['trace'] for p in pair})==1 for pair in groups.values())
compile(R.campaign_source(),'<test>','exec')
R.F.verify(R.F.load(R.OUT/'source_manifest.json'))
R.F.save(R.OUT/'boundary_tests.json',dict(passed=True,exact_original_phase_recovered_after_reversing_single_read_audit_edit=True,paired_points=36,total_points=72,update_service_unchanged=True,test_sha256=R.F.sha(__file__)))
R.F.save(R.OUT/'current_environment.json',dict(python=sys.version,platform=platform.platform(),cpu_count=R.F.psutil.cpu_count(),ram=R.F.psutil.virtual_memory().total,docker=R.F.docker('version','--format','{{json .Server}}')))
print('PASS: exact single-read-audit boundary, 36 paired points, frozen hashes.')
