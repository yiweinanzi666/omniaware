"""Export deterministic first-20 format review and aggregate real S1 input traces (not semantic labeling)."""
import json,pathlib,collections
base=pathlib.Path(__file__).parent
result={}
for dataset in ['webqa','okvqa']:
    records=[json.loads(line) for line in (base/(dataset+'_s1.jsonl')).read_text(encoding='utf8').splitlines()]
    audit=[];counts={}
    for r in records:
        for name,a in r['actions'].items():
            if not a.get('available'):continue
            c=counts.setdefault(name,collections.Counter())
            c['available']+=1
            inp=a.get('input',{})
            c['images']+=inp.get('n_images',0)
            c['visual_tokens']+=sum(inp.get('image_tokens',[]))
            c['refusal']+=bool(a.get('refusal'))
            c['empty_non_refusal']+=bool(a.get('answer_empty')) and not bool(a.get('refusal'))
            c['missing_first_token_text']+=any('text' not in t for t in a.get('first_token',{}).get('topk',[]))
            if name=='direct':c['incorrect_direct_image_count']+=inp.get('n_images')!=(0 if dataset=='webqa' else 1)
            if dataset=='webqa' and 'image' in r.get('gold_modalities',[]) and name=='static_image':
                c['image_question_count']+=1
                c['missing_image_input']+=inp.get('n_images',0)<=0
                c['invalid_image_token_count']+=not inp.get('image_tokens') or any(x<=0 for x in inp.get('image_tokens',[]))
    for r in records[:20]:
        audit.append({'qid':r['qid'],'question':r['question'],'actions':{
            k:{'short':a.get('short'),'answer':a.get('answer'),'refusal':a.get('refusal'),'answer_first_ok':a.get('answer_first_ok')}
            for k,a in r['actions'].items() if a.get('available')}})
    result[dataset]={'questions':len(records),'input_trace_counts':counts,'first20_review':audit}
(base/'s1_trace_and_format_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
for dataset,r in result.items():
    print(dataset,json.dumps(r['input_trace_counts']))
    for item in r['first20_review']:
        print(json.dumps({'qid':item['qid'],'question':item['question'],
                          'outputs':{k:(a['short'],a['answer']) if dataset=='webqa' else a['answer'] for k,a in item['actions'].items()}},ensure_ascii=True))
