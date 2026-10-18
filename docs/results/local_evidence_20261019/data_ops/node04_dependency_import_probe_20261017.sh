set +e
/home/user/anaconda3/bin/python - <<'PY'
import numpy
print('numpy',numpy.__version__,numpy.__file__)
for name in ('scipy','scipy.linalg','sklearn','torchvision','transformers'):
  try:
    mod=__import__(name,fromlist=['*'])
    print('OK',name,getattr(mod,'__version__',''))
  except Exception as exc:
    print('FAIL',name,type(exc).__name__,repr(exc))
PY
exit 0
