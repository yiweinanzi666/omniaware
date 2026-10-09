import unittest
from omniaware.answer_format import parse_mmqa


class AnswerFormatTests(unittest.TestCase):
    def test_multiple_answers_keep_item_boundaries(self):
        self.assertEqual(parse_mmqa('["Amores Possíveis", "O Xangô de Baker Street"]'),
                         (["Amores Possíveis", "O Xangô de Baker Street"], True))

    def test_commas_slashes_and_duplicates_are_preserved(self):
        values = ['1,573,587', 'Arzu / Kız İsmail', 'Ford', 'Ford']
        import json
        self.assertEqual(parse_mmqa(json.dumps(values)), (values, True))

    def test_invalid_json_is_scored_as_raw_answer(self):
        for text in ['A, B', '["A",', '{"answer": "A"}', '[1, "A"]', '```json\n["A"]\n```']:
            self.assertEqual(parse_mmqa(text), ([text], False))


if __name__ == '__main__':
    unittest.main()
