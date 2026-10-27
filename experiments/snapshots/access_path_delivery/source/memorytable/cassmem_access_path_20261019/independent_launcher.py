"""Detached launcher with preserved tests, stdout/stderr and exit evidence."""
import subprocess, sys, time, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
out=ROOT/'run_v1'
stamp=time.strftime('%Y%m%d_%H%M%S')
with (out/f'launcher_{stamp}.log').open('x',encoding='utf-8') as log:
    def emit(value):log.write(json.dumps(value)+'\n');log.flush()
    tests=subprocess.run([sys.executable,'-B',str(ROOT/'test_campaign.py')],cwd=ROOT,capture_output=True,text=True)
    evidence=dict(returncode=tests.returncode,stdout=tests.stdout,stderr=tests.stderr)
    (out/f'tests_{stamp}.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    emit(dict(stage='offline_tests',returncode=tests.returncode))
    if tests.returncode:raise SystemExit(tests.returncode)
    frozen=subprocess.run([sys.executable,'-B',str(ROOT/'campaign.py'),'freeze'],cwd=ROOT,stdout=log,stderr=log)
    if frozen.returncode:raise SystemExit(frozen.returncode)
    child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'campaign.py'),'run'],cwd=ROOT,stdout=log,stderr=log)
    emit(dict(stage='campaign_started',pid=child.pid))
    code=child.wait();emit(dict(stage='campaign_exited',returncode=code,unix_time=time.time()))
    raise SystemExit(code)
