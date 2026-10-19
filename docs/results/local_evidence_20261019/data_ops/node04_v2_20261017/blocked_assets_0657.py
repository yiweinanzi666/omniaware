import datetime, hashlib, json, pathlib
A = pathlib.Path('/home/user/Futaoran/omniaware/assets')
L = A / 'manifests/v2_run_20261017'
record = {'start': datetime.datetime.now().astimezone().isoformat()}
p = A.parent / 'cache/okvqa_gs112k/corpus.jsonl'
rows = 0
blank = 0
errors = []
ids = []
for number, line in enumerate(p.open('rb'), 1):
    if not line.strip():
        blank += 1
        continue
    try:
        value = json.loads(line)
        rows += 1
        ids.append(value.get('id'))
    except Exception as e:
        errors.append({'line': number, 'error': str(e)})
record['corpus'] = {'path': str(p), 'physical_lines': number, 'json_rows': rows, 'blank_lines': blank, 'parse_errors': errors, 'unique_ids': len(set(ids)), 'first_id': ids[0], 'last_id': ids[-1], 'required_rows': 112724, 'pass': rows == 112724 and not errors}
raw = json.loads((A / 'data/WebQA_train_val.json').read_text())
# Only four failed validation qids; no additional directory scan.
qids = ['d5d27f720dba11ecb1e81171463288e9','d5d570a60dba11ecb1e81171463288e9','d5d5ec0c0dba11ecb1e81171463288e9','d5d8eeca0dba11ecb1e81171463288e9']
record['missing_webqa_val_images'] = []
for qid in qids:
    for kind in ['img_posFacts', 'img_negFacts']:
        for fact in raw[qid][kind]:
            path = A / 'cache/webqa_full4966_images/images' / (str(fact['image_id']) + '.jpg')
            if not path.exists():
                record['missing_webqa_val_images'].append({'qid': qid, 'fact_kind': kind, 'image_id': fact['image_id'], 'path': str(path)})
record['csv_files'] = []
for name in ['okvqa_train_corpus.csv', 'okvqa_full_corpus.csv']:
    p = A / 'downloads/okvqa_gs' / name
    row = {'path': str(p), 'exists': p.exists()}
    if p.exists():
        row.update(bytes=p.stat().st_size, sha256=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest())
    record['csv_files'].append(row)
record['end'] = datetime.datetime.now().astimezone().isoformat()
with (L / 'blocked_assets_0657.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record), flush=True)
