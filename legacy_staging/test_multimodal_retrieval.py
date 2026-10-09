"""Ranking selection/order and label-free input contract, without model inference."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.eab_first_token.multimodal_retrieval import (
    load_cached_records, load_inputs, prepare_config, rank_sources, row_hash, run, sha256_file,
)


def candidates():
    return [dict(modality='image' if i%2 else 'text', fact_id=str(i), text=f'Original {i}',
                 **({'caption':f'Original {i}'} if i%2 else {'fact':f'Original {i}'})) for i in range(6)]


class RetrievalTests(unittest.TestCase):
    def test_ce_selection_and_bge_presentation_are_distinct(self):
        sources = candidates()
        original = copy.deepcopy(sources)
        scored, selected = rank_sources(sources, [.1,.5,.9,.2,.3,1.], [6,5,4,3,2,-10])
        self.assertEqual([c['fact_id'] for c in selected], ['2','1','4','3','0'])
        self.assertEqual([c['ce_rank'] for c in selected], [3,2,5,4,1])
        self.assertEqual([c['bge_rank'] for c in selected], [2,3,4,5,6])
        self.assertEqual(scored[5]['bge_cosine'],1.)
        self.assertEqual(sources,original)
        self.assertTrue(all(c['text'].startswith('Original') for c in selected))

    def test_shared_fact_id_across_modalities_and_ties_remain_distinct(self):
        sources = candidates()
        sources[1]['fact_id']='0'
        _, selected = rank_sources(sources,[.5]*6,[1.]*6)
        self.assertEqual([c['source_id'] for c in selected], ['image:0','text:0','text:2','image:3','text:4'])
        _, reversed_selection = rank_sources(list(reversed(sources)),[.5]*6,[1.]*6)
        self.assertEqual(selected,reversed_selection)

    def test_nonfinite_or_misaligned_model_outputs_raise(self):
        with self.assertRaises(ValueError):
            rank_sources(candidates(),[.5]*5,[1.]*6)
        with self.assertRaises(ValueError):
            rank_sources(candidates(),[.5]*6,[1.,2.,3.,4.,5.,float('nan')])

    def test_input_preserves_caption_and_rejects_label_or_empty_fallback(self):
        row=dict(qid='q',question='Which item?',source_split='train',subset='development',candidates=candidates())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'inputs.jsonl'
            path.write_text(json.dumps(row)+'\n')
            self.assertEqual(load_inputs(path),[row])
            poisoned=copy.deepcopy(row)
            poisoned['candidates'][0]['role']='pos'
            path.write_text(json.dumps(poisoned)+'\n')
            with self.assertRaises(ValueError):load_inputs(path)
            missing=copy.deepcopy(row)
            missing['candidates'][1]['text']=''
            path.write_text(json.dumps(missing)+'\n')
            with self.assertRaises(ValueError):load_inputs(path)

    def test_saved_configuration_does_not_require_current_code_or_runtime_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            inputs = output/'inputs.jsonl'
            inputs.write_text('frozen inputs\n')
            saved = dict(identity=dict(inputs_sha256=sha256_file(inputs), source_code_sha256='archived code',
                                       torch='archived runtime', batch_size=2, threads=4))
            config_path = output/'config.json'
            config_path.write_text(json.dumps(saved))
            before = config_path.read_bytes()
            args = SimpleNamespace(inputs=inputs, output=output, batch_size=9, threads=9)
            self.assertEqual(prepare_config([], args), saved)
            self.assertEqual(config_path.read_bytes(), before)
            inputs.write_text('different scientific input\n')
            with self.assertRaisesRegex(ValueError, 'input differs'):
                prepare_config([], args)

    def test_completed_cache_resume_validates_content_without_loading_models(self):
        row = dict(qid='q', question='Which item?', source_split='train', subset='development', candidates=candidates())
        bge = dict(qid='q', source_row_sha256=row_hash(row), config_sha256='recorded provenance',
                   bge_cosines=[.1,.5,.9,.2,.3,1.], candidate_input_tokens=[4]*6, bge_seconds=1.)
        scored, selected = rank_sources(row['candidates'], bge['bge_cosines'], [6,5,4,3,2,-10])
        for candidate in scored:
            candidate.update(bge_input_tokens=4, ce_input_tokens=8)
        record = dict(row, candidates=scored, source_row_sha256=row_hash(row), config_sha256='recorded provenance',
                      ordered_sources=selected, selected_source_ids=[c['source_id'] for c in selected],
                      selected_image_count=2, bge_seconds=1., ce_and_order_seconds=2.)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            inputs = output/'inputs.jsonl'
            inputs.write_text(json.dumps(row)+'\n')
            (output/'config.json').write_text(json.dumps(dict(identity=dict(inputs_sha256=sha256_file(inputs),
                batch_size=2, threads=4, candidate_occurrences=6, source_code_sha256='archived code'))))
            (output/'bge_scores.jsonl').write_text(json.dumps(bge)+'\n')
            result_path = output/'retrieval.jsonl'
            result_path.write_text(json.dumps(record)+'\n')
            args = SimpleNamespace(inputs=inputs, output=output, batch_size=9, threads=9)
            with patch('transformers.AutoTokenizer.from_pretrained', side_effect=AssertionError('model load')), \
                    patch('torch.set_num_threads') as set_threads:
                self.assertEqual(run(args)['selected_sources'], 5)
                set_threads.assert_called_once_with(4)
                for field, value in [('source_row_sha256', 'wrong source'), ('ordered_sources', []),
                                     ('selected_source_ids', list(reversed(record['selected_source_ids'])))]:
                    damaged = dict(record, **{field:value})
                    result_path.write_text(json.dumps(damaged)+'\n')
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        run(args)
                result_path.write_text(json.dumps(record)+'\n')
                for values in ([.1]*5, [.1]*5+[float('nan')]):
                    (output/'bge_scores.jsonl').write_text(json.dumps(dict(bge, bge_cosines=values))+'\n')
                    with self.subTest(values=values), self.assertRaises(ValueError):
                        run(args)

    def test_cached_qids_cannot_hide_duplicate_or_unexpected_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cache.jsonl'
            for records in ([dict(qid='q'), dict(qid='q')], [dict(qid='other')]):
                path.write_text(''.join(json.dumps(row)+'\n' for row in records))
                with self.assertRaises(ValueError):
                    load_cached_records(path, [dict(qid='q')])


if __name__=='__main__':
    unittest.main()
