"""Source preservation, truncation and document-only extraction for the M2RAG WebQA interface.

The legacy native-prompt span test depended on the old packing pipeline and was removed with
official_stage_one_inputs; tokenizer and images are synthetic here.
"""

import types
import unittest

from omniaware.baselines.m2rag_webqa import (
    direct_answer_document, extractor_messages, official_query, prepare_sources,
)


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, tokens):
        return " ".join(f"w{t}" for t in tokens)


class M2RAGWebQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = types.SimpleNamespace(tokenizer=WordTokenizer())
        cls.sources = [
            dict(source_id='text:1', fact_id='1', modality='text', text='Shared snippet.', r_j=.25),
            dict(source_id='image:30218009', fact_id='30218009', modality='image', text='Shared snippet.', r_j=.7),
            dict(source_id='text:2', fact_id='2', modality='text', text='A different fact.', r_j=.4),
            dict(source_id='image:30009006', fact_id='30009006', modality='image', text='Shared snippet.', r_j=.9),
            dict(source_id='text:3', fact_id='3', modality='text', text='The last fact.', r_j=.2),
        ]
        for source in cls.sources:
            if source['modality']=='image':
                source['image_path']=f"/synthetic/{source['fact_id']}.jpg"

    def test_prepare_sources_keeps_text_identity_and_selected_pixels(self):
        original=[dict(self.sources[1],text='An original caption. '*300)]
        prepared=prepare_sources(self.processor.tokenizer,original,
            {'30218009':self.sources[1]['image_path']},{'image:30218009':.7})
        self.assertEqual(prepared[0]['source_id'],original[0]['source_id'])
        # The caller cuts the text to the shared word budget; prepare_sources passes it through unchanged.
        self.assertNotIn('kept_text_token_ids',prepared[0])
        self.assertGreater(prepared[0]['full_text_tokens'],256)
        self.assertEqual(prepared[0]['text'],original[0]['text'])
        self.assertEqual(original[0]['text'],'An original caption. '*300)
        query=official_query('synthetic','Synthetic question?',prepared)
        image=query['webpages'][0]['images'][0]
        self.assertEqual(image['image_id'],30218009)
        self.assertEqual(image['detailed_image_caption'],prepared[0]['text'])
        self.assertEqual(image['cached_image_url'],prepared[0]['image_path'])

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

    def test_question_images_reach_the_official_query_and_the_extractor(self):
        question_image='/synthetic/question.jpg'
        query=official_query('synthetic','Synthetic question?',[],query_images=[question_image])
        self.assertEqual([(a['valid'],a['cached_image_url'],a['image_url']) for a in query['aux_images']],
            [(True,question_image,question_image)])
        self.assertEqual(official_query('synthetic','Synthetic question?',[])['aux_images'],[])
        result=dict(placeholder_response='Generated answer.',output_images=[])
        messages=extractor_messages('Synthetic question?',result,query_images=[question_image])
        self.assertEqual([m['image_url']['url'] for m in messages if m['type']=='image_url'],[question_image])
        self.assertIn('Generated answer.',''.join(m['text'] for m in messages if m['type']=='text'))
        self.assertEqual(len(extractor_messages('Synthetic question?',result)),1)


if __name__=='__main__':
    unittest.main()
