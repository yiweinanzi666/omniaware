find /home/user/anaconda3/pkgs/numpy-1.26.4-py312hc5e2394_0 -maxdepth 4 -type f -name '__init__.py' -o -name '_multiarray_umath*.so' | head -10
find /home/user/anaconda3/lib/python3.12/site-packages/numpy -maxdepth 1 -type f -printf '%f\n' | head
