from pathlib import Path
import os,sys,subprocess
r=Path(__file__).resolve().parent
os.environ['OMNIAWARE_ASSETS']=str(r/'assets')
subprocess.run([sys.executable,str(r/'omniaware/scripts/data_prep/prepare_okvqa_knowledge.py'),'--action','download'],check=True)
subprocess.run([sys.executable,str(r/'resume_dpr_build.py')],check=True)
