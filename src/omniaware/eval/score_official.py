"""Score a run's saved predictions with the unmodified official benchmark evaluators.

    python -m omniaware.eval.score_official runs/<tag>/<dataset>.jsonl [--reference static] [--bart]
                                            [--device cuda] [--subset]

Every arm in the prediction file (direct, static, EAB variants, gated arms) is scored
with the same official evaluator, and each arm is compared with the reference arms by a
paired bootstrap over questions (clustered by source video for VideoRAG).  Gold answers are read
from the original annotation files (WebQA and the UniversalRAG suite: the reference stored with the
question).

Integrity: the scored questions must be exactly the evaluation set of the run's configuration
(``<dataset>_config.json``: count and sha256 of the sorted ids), unless --subset is given. The gold answers in
the predictions must equal those the dataset loader gives for the same question. WebQA questions with more
than one reference answer are refused.

Metrics: OK-VQA VQA accuracy (VQAEval, 10 annotators); WebQA acc_approx (published validation
approximation), optionally x normalized ParaBank BARTScore (--bart); HybridQA EM / F1 and
VideoRAG-Synth ROUGE-L, optionally BERTScore (--bart), as in UniversalRAG (eval/universalrag_metrics.py).
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np

from .. import answers, data, paths, runio


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def paired(a, b, cluster_ids=None):
    """Paired comparison of b against a over the same questions.

    ``paired_ci95_pp`` is the percentile bootstrap interval of the mean difference (clusters resampled when
    given). ``p_two_sided`` is the bootstrap test of H0: mean difference = 0, using the bootstrap distribution
    re-centred on zero: P(|delta* - delta| >= |delta|), with the +1 correction so it is never exactly 0."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays differ in length: {a.shape} vs {b.shape}")
    delta = b - a
    rng = np.random.default_rng(20261008)
    boot = []
    if cluster_ids is not None:
        groups = {}
        for i, group in enumerate(cluster_ids):
            groups.setdefault(group, []).append(i)
        grouped = list(groups.values())
        sums = np.array([delta[g].sum() for g in grouped])
        counts = np.array([len(g) for g in grouped])
    for _ in range(20):
        if cluster_ids is None:
            ids = rng.integers(len(a), size=(250, len(a)))
            boot.extend(delta[ids].mean(axis=1).tolist())
        else:
            ids = rng.integers(len(sums), size=(250, len(sums)))
            boot.extend((sums[ids].sum(axis=1) / counts[ids].sum(axis=1)).tolist())
    boot = np.asarray(boot)
    point = float(delta.mean())
    p = (np.sum(np.abs(boot - point) >= abs(point)) + 1) / (len(boot) + 1) if point != 0 else 1.0
    return {'baseline': float(a.mean()), 'ours': float(b.mean()),
            'delta_pp': point, 'relative_percent': float(100 * delta.mean() / a.mean()) if a.mean() else None,
            'paired_ci95_pp': np.quantile(boot, [.025, .975]).tolist(), 'p_two_sided': float(min(p, 1.0)),
            'wins': int((delta > 0).sum()), 'losses': int((delta < 0).sum()), 'ties': int((delta == 0).sum())}


def score_okvqa(preds, qids):
    with zipfile.ZipFile(paths.OKVQA_ANNOTATIONS) as archive:
        name = next(n for n in archive.namelist() if n.endswith('.json'))
        original = {str(r['question_id']): r for r in json.loads(archive.read(name))['annotations']}
    official = module(paths.THIRD_PARTY / 'vqa_eval' / 'vqaEval_py3.py', 'official_vqa')
    scores = {}
    for arm, answers in preds.items():
        truth = SimpleNamespace(qa=deepcopy({q: original[q] for q in qids}), getQuesIds=lambda: qids)
        prediction = SimpleNamespace(qa={q: {'answer': answers[q]} for q in qids})
        evaluator = official.VQAEval(truth, prediction, n=10)
        with redirect_stdout(StringIO()):
            evaluator.evaluate(qids)
        scores[arm] = {q: {'vqa_accuracy': evaluator.evalQA[q]} for q in qids}
    return scores, ['vqa_accuracy'], None


def default_device(device=None):
    if device:
        return device
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def score_webqa(preds, qids, records, bart=False, device=None):
    from .webqa_acc import load_official_metric, score_references
    metric, provenance = load_official_metric()
    for q in qids:
        # The official train/val release has exactly one reference per question (eval_webqa.md, webqa_acc).
        # Several references are refused rather than truncated to the first one.
        if len(records[q]['golds']) != 1:
            raise ValueError(f"{q}: {len(records[q]['golds'])} reference answers; the WebQA train/val release has "
                             "exactly one per question, so multiple references are refused, not truncated")
    rows = []
    for arm, answers in preds.items():
        for q in qids:
            r = records[q]
            value, _ = score_references(answers[q], r['golds'], r['qcate'], metric)
            rows.append({'qid': q, 'variant': arm, 'prediction': answers[q], 'references': r['golds'],
                         'category': r['qcate'], 'acc_approx': value})
    metrics = ['acc_approx']
    extra = {'acc_approx_provenance': provenance, 'primary_metric': 'acc_approx'}
    if bart:
        # A BARTScore failure is recorded and the acc scores already computed are still written.
        try:
            from .webqa_bart import add_fluency, load_parabank
            scorer, _ = load_parabank(device=default_device(device))
            fluent_rows = add_fluency(rows, scorer, variants=list(preds))
        except Exception as exc:
            extra['bart_error'] = f'{type(exc).__name__}: {exc}'
        else:
            rows = fluent_rows
            for row in rows:
                row['normalized_bart'] = row['normalized_bart_score']
                row['QA_approx'] = row['acc_approx_x_normalized_bart']
            metrics.extend(['normalized_bart', 'QA_approx'])
    scores = {arm: {} for arm in preds}
    for row in rows:
        cell = {m: 100 * row[m] for m in metrics if m in row}
        for m in metrics:
            cell.setdefault(m, None)
        scores[row['variant']][row['qid']] = cell
    extra['subsets'] = _webqa_subsets(scores, qids, records, metrics)
    extra['normalized_bart'] = _arm_means(scores, qids, 'normalized_bart') if 'normalized_bart' in metrics else None
    extra['QA_approx'] = _arm_means(scores, qids, 'QA_approx') if 'QA_approx' in metrics else None
    return scores, metrics, extra


def webqa_subset_name(record: dict) -> str:
    """Image question vs text question. Gold modalities win; qcate is only a fallback."""
    mods = record.get('gold_modalities')
    if isinstance(mods, (list, tuple, set)):
        return 'image_question' if 'image' in mods else 'text_question'
    if str(record.get('qcate') or '').lower() in {'image', 'img'}:
        return 'image_question'
    return 'text_question'


def _mean_defined(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _arm_means(scores, qids, metric):
    if not qids:
        return {arm: None for arm in scores}
    out = {}
    for arm in scores:
        vals = [scores[arm][q].get(metric) for q in qids]
        vals = [v for v in vals if v is not None]
        out[arm] = float(np.mean(vals)) if vals else None
    return out


def _webqa_subsets(scores, qids, records, metrics):
    groups = {'image_question': [], 'text_question': []}
    for q in qids:
        groups[webqa_subset_name(records[q])].append(q)
    return {name: {'n': len(qs), **{m: _arm_means(scores, qs, m) for m in metrics}}
            for name, qs in groups.items()}


def diagnostic_substring_accuracy(prediction: str, reference: str) -> float:
    """Diagnostic only: 1 if the reference is a substring of the prediction. Not an official score.

    Paper-table scripts must not call this. OK-VQA uses ``score_okvqa``; WebQA uses acc_approx.
    """
    pred, ref = (prediction or '').lower().strip(), (reference or '').lower().strip()
    if not ref:
        return 0.0
    return 1.0 if ref in pred else 0.0


def webqa_references():
    """qid -> gold answers, re-read by the WebQA loader from the annotation file (train and val splits)."""
    references = {}
    for split in ('train', 'val'):
        for e in data.load_webqa('all', split=split):
            references[e['qid']] = e['golds']
    return references


def universalrag_references(family):
    """qid -> gold answers, re-read from the UniversalRAG query file of this family (data.universalrag_golds)."""
    query_file = next(qf for qf, fam, _ in data.UNIVERSALRAG_TARGETS.values() if fam == family)
    queries = json.loads((paths.UNIVERSALRAG_ROOT / 'dataset' / 'query' / f'{query_file}.json').read_text(encoding='utf8'))
    return {str(q['index']): data.universalrag_golds(q) for q in queries}


def check_golds(records, qids, references):
    """The gold answers stored with each prediction must equal the ones the dataset loader gives for its qid."""
    for q in qids:
        if str(q) not in references:
            raise ValueError(f'{q}: not in the dataset annotation; its gold answers cannot be verified')
        if list(records[q]['golds']) != list(references[str(q)]):
            raise ValueError(f"{q}: gold answers {records[q]['golds']} in the predictions differ from the dataset's "
                             f"{references[str(q)]}; re-run the predictions after changing the data")


def check_question_set(path, qids, subset=False):
    """The scored questions must be exactly the evaluation questions the run was configured for: their count and
    the sha256 of their sorted ids (<dataset>_config.json next to the predictions). A partial or foreign file is
    refused unless subset=True, and the summary then says so."""
    path = Path(path)
    config_path = path.with_name(f'{path.stem}_config.json')
    if not config_path.exists():
        if subset:
            return {'mode': 'subset', 'questions': len(qids), 'expected_questions': None}
        raise ValueError(f'{path.name}: no {config_path.name} next to the predictions, so the question set cannot be '
                         'checked; pass --subset to score it anyway')
    config = json.loads(config_path.read_text(encoding='utf8'))
    expected_n, expected_sha = config.get('evaluation_questions'), config.get('evaluation_qids_sha256')
    if subset:
        return {'mode': 'subset', 'questions': len(qids), 'expected_questions': expected_n}
    if expected_sha is None:
        raise ValueError(f'{config_path.name} has no evaluation_qids_sha256; pass --subset to score it anyway')
    if len(qids) != expected_n or runio.qid_set_sha256(qids) != expected_sha:
        raise ValueError(f'{path.name}: {len(qids)} questions, but the configuration expects {expected_n} and this '
                         'question set differs; finish the run, or pass --subset to score a part of it')
    return {'mode': 'exact', 'questions': len(qids), 'question_set_sha256': expected_sha}


SCORERS = {'okvqa': score_okvqa}


def score_run(path, references=('static', 'direct'), bart=False, device=None, subset=False):
    path = Path(path)
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    qids = [r['qid'] for r in records]
    assert len(qids) == len(set(qids)), 'duplicate question ids'
    by_qid = {r['qid']: r for r in records}
    families = {r['dataset'] for r in records}
    assert len(families) == 1, f'mixed datasets: {families}'
    family = families.pop()
    arms = list(records[0]['preds'])
    for r in records:
        assert set(r['preds']) == set(arms), f"{r['qid']}: arms differ from the first record"
    question_check = check_question_set(path, qids, subset=subset)
    empty_flags = []
    preds = {}
    for arm in arms:
        preds[arm] = {}
        for q in qids:
            parsed = answers.parse_answer(by_qid[q]['preds'][arm] or '')
            preds[arm][q] = parsed['answer']
            empty_flags.append(parsed['answer_empty'])
    if family == 'webqa':
        check_golds(by_qid, qids, webqa_references())
        scores, metrics, extra = score_webqa(preds, qids, by_qid, bart=bart, device=device)
    elif family in ('hybridqa', 'videorag'):
        from .universalrag_metrics import score as universalrag_score
        check_golds(by_qid, qids, universalrag_references(family))
        scores, metrics, extra = universalrag_score(family, preds, qids, by_qid, bertscore=False)
        if bart and family == 'videorag':
            try:
                scores, metrics, _ = universalrag_score(family, preds, qids, by_qid, bertscore=True)
            except Exception as exc:
                extra['bart_error'] = f'{type(exc).__name__}: {exc}'
    else:
        scores, metrics, extra = SCORERS[family](preds, qids)
    extra = extra or {}
    clusters = extra.get('clusters')
    summary = {'scored_utc': datetime.now(timezone.utc).isoformat(), 'predictions': str(path),
               'predictions_sha256': hashlib.sha256(raw).hexdigest(), 'dataset': family, 'n': len(qids),
               'question_check': question_check, 'subset': question_check['mode'] == 'subset',
               'metrics': metrics,
               'means': {arm: {m: _mean_defined([scores[arm][q][m] for q in qids]) for m in metrics}
                         for arm in arms},
               'comparisons': {}, **answers.summarize_empty(empty_flags)}
    if 'bart_error' in extra:
        summary['bart_error'] = extra['bart_error']
    for ref in references:
        if ref in arms:
            summary['comparisons'][ref] = {
                arm: {m: paired([scores[ref][q][m] for q in qids], [scores[arm][q][m] for q in qids], clusters)
                      for m in metrics} for arm in arms if arm != ref}
    if 'acc_approx_provenance' in extra:
        summary['webqa_metric_provenance'] = extra['acc_approx_provenance']
        summary['primary_metric'] = 'acc_approx'
        summary['normalized_bart'] = extra.get('normalized_bart')
        summary['QA_approx'] = extra.get('QA_approx')
        summary['subsets'] = extra.get('subsets')
    if 'bertscore' in extra:          # fixed model and reference rule of the BERTScore column
        summary['bertscore'] = extra['bertscore']
    details = [{'qid': q, 'scores': {arm: scores[arm][q] for arm in arms}} for q in qids]
    return summary, details


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('predictions', type=Path, help='<run>/<dataset>.jsonl written by omniaware.run')
    parser.add_argument('--reference', action='append', default=None,
                        help='reference arm(s) for paired comparisons (default: static and direct)')
    parser.add_argument('--bart', action='store_true',
                        help='WebQA: add the ParaBank BARTScore composite; VideoRAG: add BERTScore F1')
    parser.add_argument('--device', default=None,
                        help='device of the WebQA BARTScore model (default: cuda if available, else cpu)')
    parser.add_argument('--subset', action='store_true',
                        help='score part of the evaluation set: skip the check against the run configuration '
                             '(the summary is marked subset)')
    args = parser.parse_args()
    summary, details = score_run(args.predictions, tuple(args.reference or ('static', 'direct')), bart=args.bart,
                                 device=args.device, subset=args.subset)
    stem = args.predictions.with_suffix('')
    Path(f'{stem}.official_scores.jsonl').write_text(
        ''.join(runio.dumps_strict(d, ensure_ascii=False) + '\n' for d in details), encoding='utf8')
    Path(f'{stem}.official_summary.json').write_text(
        runio.dumps_strict(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(summary['means'], indent=2))
    if summary.get('empty_answer_warning'):
        print(summary['empty_answer_warning'], flush=True)
    if 'bart_error' in summary:
        print(f"[warning] BARTScore failed, acc scores written without it: {summary['bart_error']}", flush=True)


if __name__ == '__main__':
    main()
