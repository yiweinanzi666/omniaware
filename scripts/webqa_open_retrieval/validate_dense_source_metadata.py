"""Independently check source joins and real reader inputs; no QA model is loaded."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
from unittest.mock import patch

import torch
from transformers import AutoProcessor

from . import join_dense_source_metadata as J
from experiments.eab_first_token.runs.claude_direct_repair_20261005.round_017.candidates import phase1_executor as E


def main():
    torch.set_num_threads(2)
    base = J.C.HERE
    output = base/'dense_sources_engineering/train256'
    selection = base/'dense_relevance_engineering/train256'
    result = J.join(selection, output, True)
    rows = [json.loads(line) for line in (output/'sources.jsonl').read_text().splitlines()]
    selected = [json.loads(line) for line in (selection/'selected.jsonl').read_text().splitlines()]
    cohort = [json.loads(line) for line in J.COHORT.read_text().splitlines()]
    provenance = {r['source_id']: r for r in
                  (json.loads(line) for line in (output/'provenance.jsonl').read_text().splitlines())}
    wanted = {s['source_id'] for row in rows for s in row['sources']}
    references = {}
    for name, key in (('text_documents.jsonl', 'document_id'), ('image_corpus.jsonl', 'fact_id')):
        with (base/name).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row[key] in wanted:
                    references[row[key]] = row
    assert references.keys() == wanted == provenance.keys()
    extra_captions = 0
    for row, query, retrieval in zip(rows, cohort, selected, strict=True):
        assert row['qid'] == query['qid'] == retrieval['qid'] and row['question'] == query['question']
        assert not (E.P.FORBIDDEN_KEYS | {'u_initial', 'gate_enabled'}) & set(E.P._keys(row))
        assert len(row['sources']) == 5
        for source, candidate in zip(row['sources'], retrieval['selected_sources'], strict=True):
            assert all(source[k] == candidate[k] for k in ('source_id', 'modality', 'rank', 'r_j'))
            original = references[source['source_id']]
            if source['modality'] == 'text':
                assert source['text'] == original['text']
                assert provenance[source['source_id']]['source_snippet_ids'] == original['source_snippet_ids']
            else:
                variants = sorted(set(re.sub(r'\s+', ' ', v).strip() for v in original['captions'])-{''})
                assert source['text'].split('\n') == variants
                assert provenance[source['source_id']]['caption_variants'] == variants
                assert J.captions(list(reversed(original['captions']))) == variants
                assert J.C.sha(source['image_path']) == source['image_sha256']
                extra_captions += len(variants)-1
    config = json.loads(E.T17_CONFIG.read_text())
    assert all(J.C.sha(E.MODEL/name) == config['model_assets_sha256'][name] for name in E.SMALL_ASSETS)
    processor = AutoProcessor.from_pretrained(E.MODEL, local_files_only=True)
    counter = E.P.TokenCounter(processor)
    lengths, prompt_inputs = [], []
    for row in rows:
        messages, prompt, spans, images = E.P.conversation(processor, E.P.WEBQA, row, [(i, 1) for i in range(5)])
        lengths.append(counter(prompt, E.P.image_paths(messages)))
        prompt_inputs.append((messages, spans, images))
    rendered = []
    for index in sorted(range(len(rows)), key=lambda i: (-lengths[i], rows[i]['qid']))[:4]:
        messages, spans, images = prompt_inputs[index]
        tensors, bias, physical = E.P.process_conversation(processor, messages, spans, images, device='cpu')
        actual = int(tensors['input_ids'].shape[1])
        assert actual == lengths[index] and bias.shape[-1] == actual
        rendered.append(dict(qid=rows[index]['qid'], input_tokens=actual,
                             input_tensors=E.physical.input_identity(tensors)))
    frozen = {str(p): J.C.sha(p) for p in output.iterdir() if p.name != 'worker.lock'}
    with patch.object(J, 'captions', side_effect=AssertionError('Completed bundle recomputed metadata')):
        assert J.join(selection, output, True)['new_join'] is False
    assert frozen == {path: J.C.sha(path) for path in frozen}
    rejected = {}
    for key, call in (('empty_caption', lambda: J.captions(['  ', '\n'])),
                      ('nonstring_caption', lambda: J.captions(['caption', None]))):
        try:
            call()
        except ValueError as error:
            rejected[key] = str(error)
        else:
            raise AssertionError('Invalid caption metadata accepted')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'sources.jsonl'
        path.write_text('unfinished prior content\n')
        try:
            J.write_rows(path, rows)
        except ValueError as error:
            rejected['uncompleted_changed_bundle_preserved'] = str(error)
        else:
            raise AssertionError('Changed incomplete bundle overwritten')
        assert path.read_text() == 'unfinished prior content\n'
    proof = dict(status='PASS_ENGINEERING_PREFIX_SOURCE_JOIN', checked_utc=datetime.now(timezone.utc).isoformat(),
        questions=len(rows), selected_sources=1280, unique_sources=len(wanted),
        all_source_id_rank_relevance_and_text_matches=True, extra_caption_occurrences_retained=extra_captions,
        global_caption_audit=result['global_caption_audit'],
        rank_order_static_input_tokens=dict(min=min(lengths), max=max(lengths), total=sum(lengths),
                                           over_historical1536=sum(n > 1536 for n in lengths)),
        real_processor_checks=rendered, reentry_new_joins=0, reentry_byte_changes=0, invalid_input_rejections=rejected,
        bundle_completion_sha256=J.C.sha(output/'completion.json'),
        source_sha256={str(p): J.C.sha(p) for p in (Path(__file__), Path(J.__file__), J.PROTOCOL)},
        labels_read=False, QA_model_loads=0, answer_generations=0, MPS_calls=0,
        scope='1024-vector engineering indices only; actual U, full retrieval and replication/permutation budget remain pending')
    J.C.save(base/'DENSE_SOURCE_METADATA_INDEPENDENT_VALIDATION.json', proof)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
