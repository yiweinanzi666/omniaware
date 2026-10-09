"""Check actual native prompts, source preservation and document-only extraction."""

import json
import unittest

from transformers import AutoProcessor

from experiments.eab_first_token.m2rag_qwen import LocalQwenVLM
from experiments.eab_first_token.m2rag_webqa import (
    direct_answer_document, extractor_messages, official_query,
    official_stage_one_inputs, prepare_sources,
)
from experiments.eab_first_token.multimodal_packing import IMAGE_SIZE, pack_multimodal
from experiments.eab_first_token.run_webqa import ROOT


class M2RAGWebQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(ROOT/'cache/qwen3-vl-8b-instruct', local_files_only=True)
        cls.backend = LocalQwenVLM(None, cls.processor, max_new_tokens=384, image_size=IMAGE_SIZE)
        cls.sources = [
            dict(source_id='text:1', fact_id='1', modality='text', text='Shared snippet.', r_j=.25),
            dict(source_id='image:30218009', fact_id='30218009', modality='image', text='Shared snippet.', r_j=.7),
            dict(source_id='text:2', fact_id='2', modality='text', text='A different fact.', r_j=.4),
            dict(source_id='image:30009006', fact_id='30009006', modality='image', text='Shared snippet.', r_j=.9),
            dict(source_id='text:3', fact_id='3', modality='text', text='The last fact.', r_j=.2),
        ]
        for source in cls.sources:
            if source['modality']=='image':
                record=json.loads((ROOT/'cache/webqa_parquet_images'/(source['fact_id']+'.json')).read_text())
                source['image_path']=record['local_path']

    def test_common_truncation_preserves_source_identity_and_selected_pixels(self):
        original=[dict(self.sources[1],text='An original caption. '*300)]
        prepared=prepare_sources(self.processor.tokenizer,original,
            {'30218009':self.sources[1]['image_path']},{'image:30218009':.7})
        self.assertEqual(prepared[0]['source_id'],original[0]['source_id'])
        self.assertEqual(len(prepared[0]['kept_text_token_ids']),256)
        self.assertGreater(prepared[0]['full_text_tokens'],256)
        self.assertEqual(prepared[0]['text'],self.processor.tokenizer.decode(prepared[0]['kept_text_token_ids']))
        self.assertEqual(original[0]['text'],'An original caption. '*300)
        query=official_query('synthetic','Synthetic question?',prepared)
        image=query['webpages'][0]['images'][0]
        self.assertEqual(image['image_id'],30218009)
        self.assertEqual(image['detailed_image_caption'],prepared[0]['text'])
        self.assertEqual(image['cached_image_url'],prepared[0]['image_path'])

    def test_native_single_and_multi_prepare_without_model_calls_and_map_only_sources(self):
        query=official_query('synthetic','Does the Shared snippet. repeat?',self.sources)
        _,_,direct=pack_multimodal(self.processor,query['content'],self.sources)
        expected=[source['source_id'] for source in self.sources]
        for multi in (False,True):
            with self.subTest(multi_stage=multi):
                inputs,relevance,trace=official_stage_one_inputs(self.backend,self.processor,
                    query,self.sources,multi_stage=multi,image_size=IMAGE_SIZE)
                self.assertEqual(trace['official_source_order'],expected)
                self.assertEqual(trace['pixel_values_sha256'],direct['pixel_values_sha256'])
                self.assertEqual(trace['image_grid_thw'],direct['image_grid_thw'])
                self.assertEqual(inputs['input_ids'].shape,relevance.shape)
                self.assertEqual([row['source_id'] for row in trace['sources']],expected)
                for row,source in zip(trace['sources'],self.sources):
                    for start,end in row['text_char_spans']:
                        self.assertEqual(trace['prompt'][start:end],source['text'])
                    self.assertGreater(len(row['text_token_indices']),0)
                    self.assertEqual(bool(row['vision_token_indices']),source['modality']=='image')
                first_source_start=min(row['expanded_text_char_spans'][0][0] for row in trace['sources'])
                prefix=[i for i,(a,z) in enumerate(trace['offset_mapping']) if z<=first_source_start]
                self.assertTrue((relevance[0,prefix]==0).all().item())
                self.assertEqual(self.backend.calls,[])

    def test_extractor_reads_generated_caption_and_retained_image_only(self):
        retained=self.sources[1]['image_path']
        result=dict(placeholder_response='First generated line.\n<IMAGE_PLACEHOLDER>\nFinal generated line.',
            output_images=[dict(image_caption='A generated caption.',
                detailed_image_caption='Original source fact must not enter extractor.',cached_image_url=retained)])
        messages=extractor_messages('Synthetic question?',result)
        text=''.join(m['text'] for m in messages if m['type']=='text')
        self.assertIn('First generated line.',text)
        self.assertIn('Final generated line.',text)
        self.assertIn('A generated caption.',text)
        self.assertNotIn('Original source fact',text)
        self.assertEqual([m['image_url']['url'] for m in messages if m['type']=='image_url'],[retained])
        direct=extractor_messages('Synthetic question?',direct_answer_document('Exact direct answer.'))
        self.assertEqual(len(direct),1)
        self.assertTrue(direct[0]['text'].endswith('Exact direct answer.'))


if __name__=='__main__':
    unittest.main()
