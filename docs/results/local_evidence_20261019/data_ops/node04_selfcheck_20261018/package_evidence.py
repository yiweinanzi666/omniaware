import hashlib,json,pathlib,shutil
base=pathlib.Path(__file__).parent
reports=pathlib.Path('D:/WWW2027/data_ops/node04_v2_20261017/review_7ccf676/docs/results')
dest=reports/'node04_selfcheck_20261018'
dest.mkdir(parents=True,exist_ok=True)
names=['cpu_checks.sh','cpu_source_commit.txt','pytest.log','pytest.xml','pip_check.txt',
       'diagnose_precision.py','precision_checks.sh','precision_diagnostic.json',
       'diagnose_layers.py','layer_diagnostic.json','diagnose_mlp_input.py','mlp_input_diagnostic.json',
       'full_fp32_checks.sh','verify_fp32_text_image.json','verify_fp32_text_image.log','verify_fp32_text_image.exit',
       'verify_real_model_8b_bfloat16_failed.json','inspect_s1.py','s1_trace_and_format_review.json',
       'audit_s1_provenance.py','s1_provenance_review.json','image_evidence_check.json']
for name in names:shutil.copyfile(base/name,dest/name)
for name in ['20261018_node04_code_review.md','20261018_node04_precision_proposal.md']:
    shutil.copyfile(base/name,reports/name)
manifest={'source_commit':'28e9514dc432d6891a492390bc4bdef176c519ea',
          'overall_acceptance':'BLOCKED_BF16_AND_MANUAL_FORMAT',
          'files':{name:{'bytes':(dest/name).stat().st_size,'sha256':hashlib.file_digest((dest/name).open('rb'),'sha256').hexdigest()} for name in names},
          'review_inputs_retained_locally':{name:hashlib.file_digest((base/name).open('rb'),'sha256').hexdigest() for name in ['webqa_s1.jsonl','okvqa_s1.jsonl','source_28e9514.tar']}}
(dest/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(json.dumps({'files':len(names),'bytes':sum(x['bytes'] for x in manifest['files'].values())}))
