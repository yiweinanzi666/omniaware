"""Expose the pinned MultiModalQA metrics without changing their functions."""

import runpy

from .. import paths


SOURCE = paths.MMQA_OFFICIAL_EVAL
namespace = runpy.run_path(str(SOURCE / 'common_utils.py'))
namespace['__name__'] = 'pinned_mmqa_evaluate'
# The original wildcard import is supplied by the common_utils namespace above.
source = (SOURCE / 'evaluate.py').read_text().replace('from common_utils import *', '')
exec(compile(source, str(SOURCE / 'evaluate.py'), 'exec'), namespace)
list_em = namespace['list_em']
list_f1 = namespace['list_f1']
evaluate_predictions = namespace['evaluate_predictions']


def score_predictions(predictions, references):
    """References contain outer alternatives, each holding a complete gold list."""
    gold = {row['qid']: row['gold_answers'] for row in references}
    return evaluate_predictions(predictions, gold)
