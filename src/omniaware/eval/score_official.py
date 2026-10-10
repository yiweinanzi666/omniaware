"""Score a run's saved predictions with the unmodified official benchmark evaluators.

    python -m omniaware.eval.score_official runs/<tag>/<dataset>.jsonl [--reference static] [--bart]

Every arm in the prediction file (direct, static, EAB variants, gated arms) is scored
with the same official evaluator, and each arm is compared with the reference arms by a
paired bootstrap over questions (clustered by source video for VideoRAG).  Gold answers are read
from the original annotation files (WebQA and the UniversalRAG suite: the reference stored with the
question).

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

from .. import paths


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def paired(a, b, cluster_ids=None):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
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
    return {'baseline': float(a.mean()), 'ours': float(b.mean()),
            'delta_pp': float(delta.mean()), 'relative_percent': float(100 * delta.mean() / a.mean()) if a.mean() else None,
            'paired_ci95_pp': np.quantile(boot, [.025, .975]).tolist(),
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


def score_webqa(preds, qids, records, bart=False):
    from .webqa_acc import load_official_metric, score_references
    metric, provenance = load_official_metric()
    rows = []
    for arm, answers in preds.items():
        for q in qids:
            r = records[q]
            value, _ = score_references(answers[q], r['golds'][:1], r['qcate'], metric)
            rows.append({'qid': q, 'variant': arm, 'prediction': answers[q], 'references': r['golds'][:1],
                         'category': r['qcate'], 'acc_approx': value})
    metrics = ['acc_approx']
    if bart:
        from .webqa_bart import add_fluency, load_parabank
        scorer, _ = load_parabank(device='cpu')
        rows = add_fluency(rows, scorer, variants=list(preds))
        metrics.append('acc_approx_x_normalized_bart')
    scores = {arm: {} for arm in preds}
    for row in rows:
        scores[row['variant']][row['qid']] = {m: 100 * row[m] for m in metrics}
    return scores, metrics, {'acc_approx_provenance': provenance}


SCORERS = {'okvqa': score_okvqa}


def score_run(path, references=('static', 'direct'), bart=False):
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
    preds = {arm: {q: by_qid[q]['preds'][arm] or '' for q in qids} for arm in arms}
    if family == 'webqa':
        scores, metrics, extra = score_webqa(preds, qids, by_qid, bart=bart)
    elif family in ('hybridqa', 'videorag'):
        from .universalrag_metrics import score as universalrag_score
        scores, metrics, extra = universalrag_score(family, preds, qids, by_qid, bertscore=bart)
    else:
        scores, metrics, extra = SCORERS[family](preds, qids)
    extra = extra or {}
    clusters = extra.get('clusters')
    summary = {'scored_utc': datetime.now(timezone.utc).isoformat(), 'predictions': str(path),
               'predictions_sha256': hashlib.sha256(raw).hexdigest(), 'dataset': family, 'n': len(qids),
               'metrics': metrics,
               'means': {arm: {m: float(np.mean([scores[arm][q][m] for q in qids])) for m in metrics}
                         for arm in arms},
               'comparisons': {}}
    for ref in references:
        if ref in arms:
            summary['comparisons'][ref] = {
                arm: {m: paired([scores[ref][q][m] for q in qids], [scores[arm][q][m] for q in qids], clusters)
                      for m in metrics} for arm in arms if arm != ref}
    if 'acc_approx_provenance' in extra:
        summary['webqa_metric_provenance'] = extra['acc_approx_provenance']
    details = [{'qid': q, 'scores': {arm: scores[arm][q] for arm in arms}} for q in qids]
    return summary, details


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('predictions', type=Path, help='<run>/<dataset>.jsonl written by omniaware.run')
    parser.add_argument('--reference', action='append', default=None,
                        help='reference arm(s) for paired comparisons (default: static and direct)')
    parser.add_argument('--bart', action='store_true',
                        help='WebQA: add the ParaBank BARTScore composite; VideoRAG: add BERTScore F1')
    args = parser.parse_args()
    summary, details = score_run(args.predictions, tuple(args.reference or ('static', 'direct')), bart=args.bart)
    stem = args.predictions.with_suffix('')
    Path(f'{stem}.official_scores.jsonl').write_text(
        ''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in details))
    Path(f'{stem}.official_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary['means'], indent=2))


if __name__ == '__main__':
    main()
