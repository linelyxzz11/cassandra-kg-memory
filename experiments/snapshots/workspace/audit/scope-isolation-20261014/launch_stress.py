import subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
with (root/('stress_'+time.strftime('%Y%m%d_%H%M%S')+'.log')).open('w',encoding='utf-8') as f:
    p=subprocess.run([sys.executable,'-u',str(root/'run_stress.py')],cwd=root,stdout=f,stderr=subprocess.STDOUT)
    f.write('\nEXIT '+str(p.returncode)+'\n')
