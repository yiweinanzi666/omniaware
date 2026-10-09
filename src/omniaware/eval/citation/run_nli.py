"""Future CPU-only E5 inference: complete citation union, no truncation or windows."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'cache/e5_citation_support/deberta_nli/b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7'


def expected_checker():
    directory = Path(__file__).resolve().parent
    manifest_path = directory / 'checker_asset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    runtime = json.loads((directory / 'verification.json').read_text())['dependency_versions']
    return dict(model=manifest['model'], revision=manifest['revision'], id2label=manifest['id2label'],
        asset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        torch=runtime['torch'], transformers=runtime['transformers'], device='cpu', dtype='float32',
        batch_size=1, truncation=False, windows=False, decision='argmax(logits) == 0')


def run(source, output):
    if output.exists():
        raise FileExistsError(f'Preserve existing inference artifact: {output}')
    manifest = json.loads(Path(__file__).with_name('checker_asset_manifest.json').read_text())
    for record in manifest['files']:
        with (MODEL / record['file']).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256']
    # Imports and model allocation occur only when this future-stage command is invoked.
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    identity = expected_checker()
    assert (torch.__version__, transformers.__version__) == (identity['torch'], identity['transformers'])
    assert (manifest['model'], manifest['revision'], manifest['id2label']) == (identity['model'], identity['revision'], identity['id2label'])
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL), local_files_only=True,
        dtype=torch.float32).to('cpu').eval()
    assert model.config.id2label == {0:'entailment', 1:'neutral', 2:'contradiction'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as destination, source.open() as requests:
        for line in requests:
            request = json.loads(line)
            record = dict(dataset=request['dataset'], qid=request['qid'], variant=request['variant'],
                statement_index=request['statement_index'], checker=identity,
                input_sha256=hashlib.sha256(line.encode()).hexdigest())
            if request['status'] == 'ALL_CITATIONS_INVALID':
                record.update(status='STRUCTURAL_ZERO', entailment=0, logits=None, input_tokens=0)
            else:
                assert request['status'] == 'NLI_INPUT_ONLY' and request['premise'] and request['hypothesis']
                inputs = tokenizer(request['premise'], request['hypothesis'], return_tensors='pt',
                                   truncation=False, padding=False)
                started = time.perf_counter()
                with torch.inference_mode():
                    logits = model(**inputs).logits[0]
                record.update(status='NLI_COMPLETED', logits=logits.tolist(),
                    predicted_label_id=int(logits.argmax()), entailment=int(logits.argmax() == 0),
                    input_tokens=int(inputs['input_ids'].shape[1]), elapsed_seconds=time.perf_counter()-started)
            destination.write(json.dumps(record) + '\n')
            destination.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
