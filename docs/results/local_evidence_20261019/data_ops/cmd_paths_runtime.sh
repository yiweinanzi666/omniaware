export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware/cache
test -f /home/user/Futaoran/omniaware/assets/data/WebQA_train_val.json; echo test_file=$?
test -d /home/user/Futaoran/omniaware/assets/cache/webqa_full4966_images/images; echo test_dir=$?
/home/user/Futaoran/omniaware_v2_python311/bin/python - <<'PY'
import os
from omniaware import paths
print(os.environ.get('OMNIAWARE_ASSETS'))
print(paths.WEBQA_JSON, paths.WEBQA_JSON.is_file())
print(paths.WEBQA_IMAGES, paths.WEBQA_IMAGES.is_dir())
print(paths.LOCAL_CACHE, (paths.LOCAL_CACHE/'relevance_webqa_full.json').is_file())
PY
