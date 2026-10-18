import pathlib,json,hashlib,shutil
root=pathlib.Path('D:/WWW2027/data_ops/node04_fp32_code_20261018')
ops=pathlib.Path('D:/WWW2027/data_ops/node04_selfcheck_20261018')
dest=root/'docs/results/node04_format_diagnosis_20261018'
dest.mkdir(exist_ok=True)
for name in ['saved_format_diagnosis.json','diagnose_saved_answer_format.py']:
    shutil.copyfile(ops/name,dest/name)
report=json.loads((dest/'saved_format_diagnosis.json').read_text(encoding='utf8'))
# Explicitly identified sentence outputs from the previously fixed first20 sample; no gold-dependent rewrite.
indices={0:['direct','static_text'],2:['static_image'],5:['direct'],8:['direct','static_image','rerank_static'],12:['static_image','static_mixed'],15:['direct','static_image','static_mixed'],18:['direct'],19:['direct']}
rows=[json.loads(x) for x in (root/'runs/fp32_20261018/v2_s1_webqa_full/webqa_full.jsonl').read_text(encoding='utf8').splitlines()]
pairs={(rows[i]['qid'],a) for i,actions in indices.items() for a in actions}
examples=[c for c in report['datasets']['webqa_full']['checks'] if (c['qid'],c['action']) in pairs]
assert len(examples)==14
evidence={'scope':'14 explicitly identified full-sentence first-line examples in fixed first20 WebQA; not an exhaustive semantic classifier or new pass threshold',
 'examples':examples,'automatic_true':sum(c['parsed']['answer_first_ok'] is True for c in examples)}
(dest/'manual_sentence_examples.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in dest.iterdir() if p.is_file() and p.name not in ['MANIFEST.json','.gitattributes']}
(dest/'MANIFEST.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf8')
(dest/'.gitattributes').write_text('* -text\n',encoding='ascii')
print(json.dumps({'datasets':{k:{f:v[f] for f in ['available_actions','parse_mismatches','token_cap_hits']} for k,v in report['datasets'].items()},'manual_sentence_examples':len(examples),'automatic_true':evidence['automatic_true']}))
