set -e
mkdir -p /home/user/Futaoran/omniaware_pydeps /home/user/Futaoran/transformers_download
nohup /home/user/anaconda3/bin/pip download --no-deps --dest /home/user/Futaoran/transformers_download 'transformers==5.5.3' > /home/user/Futaoran/transformers_download.log 2>&1 &
echo PID=$!
