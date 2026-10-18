"""Read completed FP32 S1 outputs and preserve trace/format/provenance evidence."""
import json,pathlib,hashlib,collections
base=pathlib.Path('/home/user/Futaoran/omniaware_v2_selfcheck_20261018/runs/fp32_20261018')
oldbase=pathlib.Path('/home/user/Futaoran/omniaware_v2_exec/runs')
report={}
for dataset in ['webqa_full','okvqa_gs']:
    root=base/('v2_s1_'+dataset);oldroot=oldbase/('v2_s1_'+dataset)
    rows=[json.loads(x) for x in (root/(dataset+'.jsonl')).read_text().splitlines()]
    old={r['qid']:r for r in [json.loads(x) for x in (oldroot/(dataset+'.jsonl')).read_text().splitlines()]}
    counts={};different_inputs=[]
    for r in rows:
        for name,a in r['actions'].items():
            if not a.get('available'):continue
            c=counts.setdefault(name,collections.Counter());inp=a['input']
            c['available']+=1;c['images']+=inp['n_images'];c['visual_tokens']+=sum(inp['image_tokens'])
            c['empty_non_refusal']+=bool(a.get('answer_empty')) and not a.get('refusal')
            c['refusal']+=bool(a.get('refusal'))
            c['missing_topk_text']+=any('text' not in v for v in a['first_token']['topk'])
            if name=='direct':c['bad_direct_image_count']+=inp['n_images']!=(0 if dataset=='webqa_full' else 1)
            if dataset=='webqa_full' and 'image' in r.get('gold_modalities',[]) and name=='static_image':
                c['image_questions']+=1;c['missing_images_or_tokens']+=inp['n_images']<=0 or not inp['image_tokens'] or any(x<=0 for x in inp['image_tokens'])
            prev=old.get(r['qid'],{}).get('actions',{}).get(name,{})
            if inp['prompt_sha256']!=prev.get('input',{}).get('prompt_sha256'):different_inputs.append([r['qid'],name])
    env=[json.loads(x) for x in (root/(dataset+'.env.ndjson')).read_text().splitlines()]
    config=json.loads((root/(dataset+'_config.json')).read_text())
    oldconfig=json.loads((oldroot/(dataset+'_config.json')).read_text())
    log=(root/(dataset+'.log')).read_text()
    report[dataset]={'questions':len(rows),'same_qids_as_bf16':set(old)=={r['qid'] for r in rows},
       'input_hash_differences_vs_bf16':different_inputs,'input_trace_counts':counts,
       'config_comparison':{k:{'old':oldconfig.get(k),'new':config.get(k),'equal':oldconfig.get(k)==config.get(k)} for k in ['instruction','instruction_sha256','max_new_tokens','max_pixels','split_seed','split_qids_sha256','evaluation_qids_sha256']},
       'env':env,'prompt_rows':len((root/(dataset+'.prompts.ndjson')).read_text().splitlines()),
       'log_has_env':'[env]' in log,'log_has_done':'[done]' in log,
       'completion':json.loads((root/(dataset+'.merged.json')).read_text()),
       'format':json.loads((root/(dataset+'.answer_format.json')).read_text()),
       'first20':[{'qid':r['qid'],'question':r['question'],'actions':{k:{f:a.get(f) for f in ['short','answer','answer_first_ok','refusal']} for k,a in r['actions'].items() if a.get('available')}} for r in rows[:20]]}
dest=base/'s1_review.json';dest.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:{f:v[f] for f in ['questions','same_qids_as_bf16','input_hash_differences_vs_bf16','input_trace_counts','prompt_rows','format']} for k,v in report.items()},ensure_ascii=False))
