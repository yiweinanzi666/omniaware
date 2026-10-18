import importlib.util,pathlib,os,json,sys
root=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('prep',root/'prepare_v3.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
os.environ.update(OMNIAWARE_ASSETS=str(p.ASSETS),OMNIAWARE_CACHE=str(p.ASSETS.parent/'cache'),OMNIAWARE_WEBQA_OPEN_DIR=str(p.ASSETS/'webqa_open'))
p.run([str(p.ENV/'bin/python'),'-m','pytest','-p','no:warnings'],'pytest')
print('PYTEST_PASSED',flush=True)
