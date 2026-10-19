/home/user/anaconda3/bin/python -u - <<'PY'
import os, pathlib, xml.etree.ElementTree as ET, urllib.parse
audit=pathlib.Path('/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/preflight_0242')
for test in ET.parse(audit/'pytest.xml').iter('testcase'):
    failure=test.find('failure')
    if failure is not None:
        print('FAILED_TEST',test.attrib)
        print((failure.text or '')[:9000])
for name, value in os.environ.items():
    if 'proxy' in name.lower():
        print('PROXY_SETTING', name, urllib.parse.urlparse(value).hostname if '://' in value else value)
PY
cd /home/user/Futaoran/omniaware_v2_exec || exit 1
git -C external/M2RAG rev-parse HEAD
git -C external/M2RAG status --short
ls -ld external/M2RAG assets/cache/official_m2rag /home/user/Futaoran/omniaware/assets/cache/official_m2rag
