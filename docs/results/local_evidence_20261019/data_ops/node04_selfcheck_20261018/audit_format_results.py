"""Audit completed format-revision S1 without changing any answers or scoring them."""
import json,pathlib,hashlib,collections,sys
root=pathlib.Path('/home/user/Futaoran/omniaware_v2_format_20261018')
sys.path.insert(0,str(root/'src'))
from omniaware import prompts
base=root/'runs/fp32_format_20261018'
oldbase=pathlib.Path('/home/user/Futaoran/omniaware_v2_selfcheck_20261018/runs/fp32_20261018')
report={}
for dataset in ['webqa_full','okvqa_gs']:
    d=base/('v2_s1_'+dataset);oldroot=oldbase/('v2_s1_'+dataset)
    rows=[json.loads(x) for x in (d/(dataset+'.jsonl')).read_text().splitlines()]
    old={r['qid']:r for r in [json.loads(x) for x in (oldroot/(dataset+'.jsonl')).read_text().splitlines()]}
    counts={};changed_prompts=0;input_differences=[]
    for r in rows:
        for name,a in r['actions'].items():
            prev=old[r['qid']]['actions'][name]
            if a.get('available')!=prev.get('available'):input_differences.append([r['qid'],name,'availability'])
            if not a.get('available'):continue
            c=counts.setdefault(name,collections.Counter());inp=a['input']
            c['available']+=1;c['images']+=inp['n_images'];c['visual_tokens']+=sum(inp['image_tokens'])
            c['empty_non_refusal']+=bool(a.get('answer_empty')) and not a.get('refusal')
            c['refusal']+=bool(a.get('refusal'));c['hit_token_cap']+=a['decode']['hit_token_cap']
            c['missing_topk_text']+=any('text' not in v for v in a['first_token']['topk'])
            if name=='direct':c['bad_direct_image_count']+=inp['n_images']!=(0 if dataset=='webqa_full' else 1)
            if dataset=='webqa_full' and 'image' in r.get('gold_modalities',[]) and name=='static_image':
                c['image_questions']+=1;c['missing_images_or_tokens']+=inp['n_images']<=0 or not inp['image_tokens'] or any(x<=0 for x in inp['image_tokens'])
            for field in ['n_images','image_tokens']:
                if inp[field]!=prev['input'][field]:input_differences.append([r['qid'],name,field])
            if a['chunks']!=prev['chunks']:input_differences.append([r['qid'],name,'chunks'])
            changed_prompts+=inp['prompt_sha256']!=prev['input']['prompt_sha256']
    env=[json.loads(x) for x in (d/(dataset+'.env.ndjson')).read_text().splitlines()]
    config=json.loads((d/(dataset+'_config.json')).read_text());oldconfig=json.loads((oldroot/(dataset+'_config.json')).read_text())
    log=(d/(dataset+'.log')).read_text();completion=json.loads((d/(dataset+'.merged.json')).read_text())
    hashes={k:hashlib.sha256((d/completion[k]).read_bytes()).hexdigest()==completion[k+'_sha256'] for k in ['output','config']}
    instruction=prompts.instruction_text(dataset)
    report[dataset]={'questions':len(rows),'same_qids_as_previous':set(old)=={r['qid'] for r in rows},
       'changed_prompt_hashes':changed_prompts,'evidence_image_differences':input_differences,'input_trace_counts':counts,
       'instruction_matches_approved':config['instruction']==instruction and config['instruction_sha256']==prompts.instruction_sha256(instruction),
       'frozen_config_comparison':{k:{'old':oldconfig.get(k),'new':config.get(k),'equal':k in config and k in oldconfig and oldconfig[k]==config[k]} for k in ['max_new_tokens','max_pixels','split_seed','split_qids_sha256','evaluation_qids_sha256','dtype']},
       'env':env,'prompt_rows':len((d/(dataset+'.prompts.ndjson')).read_text().splitlines()),
       'log_has_env':'[env]' in log,'log_has_done':'[done]' in log,'completion_hashes_match':hashes,
       'completion':completion,'format':json.loads((d/(dataset+'.answer_format.json')).read_text()),
       'first20':[{'qid':r['qid'],'question':r['question'],'actions':{k:{f:a.get(f) for f in ['short','answer','answer_first_ok','refusal']} for k,a in r['actions'].items() if a.get('available')}} for r in rows[:20]]}
dest=base/'s1_review.json';assert not dest.exists()
dest.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:{f:v[f] for f in ['questions','same_qids_as_previous','changed_prompt_hashes','evidence_image_differences','input_trace_counts','instruction_matches_approved','completion_hashes_match','format']} for k,v in report.items()},ensure_ascii=False))
