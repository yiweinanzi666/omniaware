"""Official-core synthetic comparison and real Qwen token offsets on CPU."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.special import softmax
import spacy
import torch
from transformers import AutoTokenizer

from experiments.eab_first_token.dragin_rind import (
    align_qwen_tokens, calibrate_threshold, generation_entropy_nats,
    received_attention, rind_word_scores, score_draft,
)


ROOT = Path(__file__).parent


class RindNumericalTests(unittest.TestCase):
    def test_actual_official_function_matches_entropy_attention_and_word_core(self):
        source = ast.parse((ROOT/'cache/official_dragin/src/generate.py').read_text())
        cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'BasicGenerator')
        function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'generate_attn')
        namespace = dict(np=np, torch=torch, softmax=softmax)
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'official_generate_attn_only', 'exec'), namespace)
        attention = torch.tensor([[[[1.,0,0],[.9,.1,0],[.2,.3,.5]],
                                   [[1.,0,0],[.1,.9,0],[.4,.3,.3]]]])
        scores = [torch.tensor([[.5,1.,-.2]]), torch.tensor([[0.,0.,0.]]), torch.tensor([[2.,0.,-1.]])]
        class Model:
            device = 'cpu'
            def generate(self, **kwargs):
                return SimpleNamespace(sequences=torch.tensor([[1,10,11,12]]), scores=scores)
            def __call__(self, *args, **kwargs):
                return SimpleNamespace(attentions=(attention,))
        tokenizer = SimpleNamespace(encode=lambda *a,**k:torch.tensor([[1]]),
            convert_ids_to_tokens=lambda ids:['▁Al','pha','▁beta'], decode=lambda ids:'Alpha beta')
        original = namespace['generate_attn'](SimpleNamespace(tokenizer=tokenizer, model=Model(), space_token='▁'),
                                               'prompt', 3, use_entropy=True)
        entropy = generation_entropy_nats(scores)
        importance = received_attention(attention, 3)
        np.testing.assert_allclose(importance, [1.,.6,.4], rtol=1e-7)
        weights = np.array([[1,1,0],[0,0,1]], dtype=float)
        word_attention, word_entropy, values, masked = rind_word_scores(entropy, importance, weights, [0,0], [True,True])
        np.testing.assert_allclose(word_attention, original[2], rtol=1e-7)
        np.testing.assert_allclose(word_entropy, original[4], rtol=1e-7)
        expected = np.asarray(original[2])/sum(original[2])*2*np.asarray(original[4])
        np.testing.assert_allclose(values, expected, rtol=1e-7)
        np.testing.assert_equal(values, masked)
        self.assertAlmostEqual(entropy[1], np.log(3), places=6)

    def test_unlabeled_threshold_ties_are_deterministic_and_never_split_scores(self):
        result = calibrate_threshold(np.arange(1,501))
        self.assertEqual((result['threshold'], result['retrieve_count']), (108.,392))
        self.assertEqual(result['achieved_rate'], .784)
        self.assertEqual(result['reference_threshold'], 1.2)
        tied = calibrate_threshold([0.,1.,1.,1.,2.], target_rate=.5)
        self.assertEqual((tied['threshold'], tied['retrieve_count'], tied['threshold_tie_count']), (1.,1,3))
        self.assertEqual(calibrate_threshold([0.,0.,0.])['retrieve_count'], 0)
        for bad in ([], [float('nan')], [-1.]):
            with self.assertRaises(ValueError):calibrate_threshold(bad)

    def test_processed_masked_logits_and_misaligned_attention(self):
        value = generation_entropy_nats([torch.tensor([[0., -torch.inf, 0.]])])[0]
        self.assertAlmostEqual(value, np.log(2), places=6)
        with self.assertRaises(ValueError):generation_entropy_nats([torch.tensor([[-torch.inf,-torch.inf]])])
        with self.assertRaises(ValueError):received_attention(torch.ones(1,1,2,3), 2)


class QwenAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT/'cache/qwen3-vl-8b-instruct', local_files_only=True)
        cls.nlp = spacy.load('en_core_web_sm')

    def score_text(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)+[self.tokenizer.eos_token_id]
        scores = [torch.zeros(1,4) for _ in ids]
        attention = torch.eye(len(ids)).reshape(1,1,len(ids),len(ids))
        return score_draft(ids, scores, attention, self.tokenizer, self.nlp)

    def test_real_multitoken_punctuation_decimal_and_unicode_alignment(self):
        text = 'The blue cats cost 12.50 dollars.\nNew York has 1,000 towers! Café 東京。'
        result = self.score_text(text)
        self.assertEqual(result['draft_text'], text)
        self.assertEqual(result['offset_method'], 'tokenizer_offsets')
        decimal = next(w for w in result['words'] if w['text']=='12.50')
        thousand = next(w for w in result['words'] if w['text']=='1,000')
        self.assertTrue(decimal['content_word'] and thousand['content_word'])
        self.assertIn(13, [result['draft_token_ids'][i] for i in decimal['token_indices']])
        self.assertEqual(result['special_token_positions'], [len(result['draft_token_ids'])-1])
        self.assertTrue(all(result['special_token_positions'][0] not in w['token_indices'] for w in result['words']))
        for word in result['words']:
            self.assertEqual(text[word['start']:word['end']], word['text'])
            self.assertTrue(word['token_indices'])
        weights = np.zeros(len(result['draft_token_ids']))
        for word in result['words']:
            weights[word['token_indices']] += word['token_weights']
        self.assertTrue(np.all(weights <= 1.))
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(13), '.')

    def test_valid_noncanonical_generated_segmentation_uses_original_ids(self):
        ids = sum((self.tokenizer.encode(letter, add_special_tokens=False) for letter in 'cats'), [])
        self.assertNotEqual(ids, self.tokenizer.encode('cats', add_special_tokens=False))
        result = align_qwen_tokens(self.tokenizer, ids)
        self.assertEqual(result['text'], 'cats')
        self.assertEqual(result['offsets'], [(0,1),(1,2),(2,3),(3,4)])
        self.assertEqual(result['offset_method'], 'generated_byte_offsets_checked_against_decode')

    def test_post_pos_mask_trigger_and_empty_special_only_draft(self):
        determiner = self.score_text('the')
        self.assertTrue(determiner['pre_mask_trigger'])
        self.assertTrue(determiner['mask_removed_all_hits'])
        self.assertFalse(determiner['retrieve'])
        self.assertEqual(determiner['masked_max_score'], 0.)
        empty = self.score_text('')
        self.assertEqual(empty['words'], [])
        self.assertEqual(empty['draft_text'], '')
        self.assertFalse(empty['retrieve'])


if __name__ == '__main__':
    unittest.main()
