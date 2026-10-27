import os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ['SCOPE_B_OUTPUT']='backend_retry1'
os.environ['SCOPE_B_REUSE_CASSANDRA']='cassmem-scope-b-20261014161714-cassandra'
with (root/('backend_retry1_'+time.strftime('%Y%m%d_%H%M%S')+'.log')).open('w',encoding='utf-8') as f:
    r=subprocess.run([sys.executable,'-u',str(root/'run_backend.py')],cwd=root,stdout=f,stderr=subprocess.STDOUT)
    f.write('\nEXIT '+str(r.returncode)+'\n')
