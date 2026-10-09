"""How often the retrieved top-5 passages contain an annotator answer, on consumed OK-VQA train dev128, split by
closed-book correctness. Diagnostic only (uses consumed train references; no method uses it). CPU only."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

from experiments.eab_first_token.score_okvqa_development import TRAIN_ANNOTATIONS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / 'runs/okvqa_draft_query_codex_20261005'
AUDIT = RUN / 'RESULTS_AUDIT_B_all128.jsonl'
SUMMARY = RUN / 'qa_output/scored/summary.json'


def norm(text):
    return ' ' + re.sub(r'[^0-9a-z]+', ' ', text.lower()).strip() + ' '


def main():
    rows = [json.loads(line) for line in AUDIT.open()]
    scores = json.loads(SUMMARY.read_text())['scores']
    with zipfile.ZipFile(TRAIN_ANNOTATIONS) as archive:
        document = json.loads(archive.read('mscoco_train2014_annotations.json'))
    wanted = {r['question_id'] for r in rows}
    answers = {a['question_id']: {x['answer'] for x in a['answers']} for a in document['annotations'] if a['question_id'] in wanted}
    if len(rows) != 128 or set(answers) != wanted:
        raise ValueError('expected 128 annotated development questions')
    table, cases = {}, []
    for r in rows:
        q = r['qid']
        golds = [norm(a) for a in answers[r['question_id']] if norm(a).strip()]
        closed_ok = scores['closed-book']['per_question_points'][q] > 0
        for name, key in (('question', 'original_top5'), ('draft_expanded', 'draft_expanded_top5')):
            text = norm(' '.join(c['text'] for c in r['retrieval'][key]))
            hit = any(g in text for g in golds)
            cell = table.setdefault(name, {}).setdefault('closed_correct' if closed_ok else 'closed_wrong', dict(questions=0, passages_contain_answer=0))
            cell['questions'] += 1
            cell['passages_contain_answer'] += hit
            if name == 'draft_expanded' and hit and not closed_ok:
                cases.append(dict(qid=q, question=r['question'], closed=r['predictions']['closed-book'],
                                  draft_expanded_answer=r['predictions']['draft-expanded-bm25-static'],
                                  draft_expanded_points=scores['draft-expanded-bm25-static']['per_question_points'][q]))
    result = dict(questions=128, audit_sha256=hashlib.sha256(AUDIT.read_bytes()).hexdigest(), match='any annotator answer as a whole-word substring of the concatenated top-5 text',
                  table=table, closed_wrong_but_draft_expanded_passages_contain_answer=cases)
    (HERE / 'okvqa_passage_answer_recall.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
