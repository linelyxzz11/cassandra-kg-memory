import subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
with (root/('backend_'+time.strftime('%Y%m%d_%H%M%S')+'.log')).open('w',encoding='utf-8') as f:
    r=subprocess.run([sys.executable,'-u',str(root/'run_backend.py')],cwd=root,stdout=f,stderr=subprocess.STDOUT)
    f.write('\nEXIT '+str(r.returncode)+'\n')
