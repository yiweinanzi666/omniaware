"""Cross-encoder selection and BGE presentation order (verbatim legacy tests)."""
import copy
import unittest

from omniaware.rerank import rank_sources


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
