cd /home/user/Futaoran/omniaware_v2_exec
grep -n "add_argument\|def main" scripts/analysis/image_evidence_check.py | tail -40
sed -n '300,430p' scripts/analysis/image_evidence_check.py
