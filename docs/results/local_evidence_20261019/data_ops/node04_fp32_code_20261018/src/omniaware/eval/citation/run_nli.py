"""Future CPU-only E5 inference: each statement's own citations as premise, never truncated.

A premise longer than one 512-token DeBERTa window is split into overlapping windows exactly as in the Faith/AP
checker (attribution.premise_windows); the window with the highest entailment probability decides, and its
logits are the recorded logits (attribution.window_decision). One rule serves both citation metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from ... import paths

MODEL = paths.NLI_CHECKER


def expected_checker():
    directory = Path(__file__).resolve().parent
    manifest_path = directory / 'checker_asset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    import torch
    import transformers
    return dict(model=manifest['model'], revision=manifest['revision'], id2label=manifest['id2label'],
        asset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        torch=torch.__version__, transformers=transformers.__version__, device='cpu', dtype='float32',
        batch_size=1, truncation=False, windows='512 tokens, stride half a window, max entailment probability',
        decision='argmax(logits of the deciding window) == 0')


def run(source, output):
    if output.exists():
        raise FileExistsError(f'Preserve existing inference artifact: {output}')
    manifest = json.loads(Path(__file__).with_name('checker_asset_manifest.json').read_text())
    for record in manifest['files']:
        with (MODEL / record['file']).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256']
    # Imports and model allocation occur only when this future-stage command is invoked.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    identity = expected_checker()
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
                from .attribution import premise_windows
                p_ids = tokenizer.encode(request['premise'], add_special_tokens=False)
                h_ids = tokenizer.encode(request['hypothesis'], add_special_tokens=False)
                started = time.perf_counter()
                scored = []
                with torch.inference_mode():
                    for win in premise_windows(p_ids, h_ids):
                        ids = tokenizer.build_inputs_with_special_tokens(win, h_ids)
                        types = tokenizer.create_token_type_ids_from_sequences(win, h_ids)
                        logits = model(input_ids=torch.tensor([ids]), token_type_ids=torch.tensor([types]),
                                       attention_mask=torch.ones(1, len(ids), dtype=torch.long)).logits[0]
                        scored.append((float(logits.softmax(-1)[0]), logits, len(ids)))
                _, logits, _ = max(scored, key=lambda w: w[0])
                record.update(status='NLI_COMPLETED', logits=logits.tolist(),
                    predicted_label_id=int(logits.argmax()), entailment=int(logits.argmax() == 0),
                    windows=len(scored), input_tokens=sum(n for _, _, n in scored),
                    elapsed_seconds=time.perf_counter()-started)
            destination.write(json.dumps(record) + '\n')
            destination.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
