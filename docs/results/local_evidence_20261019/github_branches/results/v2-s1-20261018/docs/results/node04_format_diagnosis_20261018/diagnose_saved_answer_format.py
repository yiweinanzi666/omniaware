"""Read-only reconstruction of saved S1 token outputs. No model forward or rescoring."""
import sys, pathlib, json, hashlib
root=pathlib.Path('/home/user/Futaoran/omniaware_v2_selfcheck_20261018')
sys.path.insert(0,str(root/'src'))
from transformers import AutoTokenizer
from omniaware.answers import parse_answer
tok=AutoTokenizer.from_pretrained('/home/user/Futaoran/omniaware/assets/cache/qwen3-vl-8b-instruct',local_files_only=True)
base=root/'runs/fp32_20261018'
report={'purpose':'Reconstruct saved outputs, not new inference or official scoring','datasets':{}}
for ds in ['webqa_full','okvqa_gs']:
    p=base/f'v2_s1_{ds}'/f'{ds}.jsonl'
    raw=p.read_bytes();rows=[json.loads(x) for x in raw.decode('utf8').splitlines()]
    checks=[]
    for i,r in enumerate(rows):
        for name,a in r['actions'].items():
            if not a.get('available'):continue
            decoded=tok.decode(a['decode']['tokens'],skip_special_tokens=True)
            parsed=parse_answer(decoded,answer_first=r['answer_format']=='answer_first')
            fields=['answer','short','answer_first_ok','answer_empty','citation_only','refusal']
            differences={k:{'saved':a[k],'reconstructed':parsed[k]} for k in fields if a[k]!=parsed[k]}
            checks.append({'qid':r['qid'],'action':name,'first20':i<20,'question':r['question'],
                           'raw':decoded,'parsed':parsed,'differences':differences,
                           'hit_token_cap':a['decode']['hit_token_cap'],
                           'short_words':len(parsed['short'].split())})
    report['datasets'][ds]={'input_sha256':hashlib.sha256(raw).hexdigest(),'questions':len(rows),
          'available_actions':len(checks),'parse_mismatches':sum(bool(x['differences']) for x in checks),
          'token_cap_hits':sum(x['hit_token_cap'] for x in checks),'checks':checks}
dest=pathlib.Path('/home/user/Futaoran/omniaware_selfcheck_logs_20261018/saved_format_diagnosis.json')
assert not dest.exists(),str(dest)
dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:{f:v[f] for f in ['questions','available_actions','parse_mismatches','token_cap_hits']} for k,v in report['datasets'].items()}))
