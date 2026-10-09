"""Score saved predictions with the unmodified official benchmark evaluators."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT / 'experiments/eab_first_token'
sys.path.insert(0, str(ROOT))
from experiments.eab_first_token.mmqa_official_eval import score_predictions


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


def mmqa_scores(answers, refs):
    means, fractions = score_predictions(answers, refs)
    scores = {q: {m: 100 * v for m, v in row.items()} for q, row in fractions.items()}
    for metric, mean in means.items():
        assert np.isclose(mean, np.mean([row[metric] for row in scores.values()]))
    return scores


def video_official_comparison(a, b, valid_a, valid_b, cluster_ids):
    """Use the official script's parsed-answer denominator in both arms."""
    result = paired(a, b, cluster_ids)
    values = np.column_stack([a, b, valid_a, valid_b]).astype(float)
    groups = {}
    for group, row in zip(cluster_ids, values):
        groups.setdefault(group, np.zeros(4))[:] += row
    sums = np.asarray(list(groups.values()))
    total = sums.sum(axis=0)
    baseline, ours = total[:2] / total[2:]
    rng = np.random.default_rng(20261008)
    boot = []
    for _ in range(20):
        sampled = sums[rng.integers(len(sums), size=(250, len(sums)))].sum(axis=1)
        rates = np.divide(sampled[:, :2], sampled[:, 2:], out=np.zeros_like(sampled[:, :2]), where=sampled[:, 2:] != 0)
        boot.extend((rates[:, 1] - rates[:, 0]).tolist())
    result.update(baseline=float(baseline), ours=float(ours), delta_pp=float(ours - baseline),
                  relative_percent=float(100 * (ours - baseline) / baseline) if baseline else None,
                  paired_ci95_pp=np.quantile(boot, [.025, .975]).tolist(),
                  baseline_answered=int(total[2]), ours_answered=int(total[3]))
    return result


def main():
    results = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'official_only': True, 'datasets': {}}
    for ds, total in [('mmqa_table', 299), ('okvqa', 5046), ('videomme', 900)]:
        source_paths = sorted((HERE / 'outputs').glob('worker_*/' + ds + '.jsonl'))
        records = [json.loads(line) for path in source_paths for line in path.read_text().splitlines() if line]
        qids = [r['qid'] for r in records]
        assert len(qids) == len(set(qids)), 'Duplicate predictions'
        if len(records) != total:
            results['datasets'][ds] = {'status': 'GENERATION_INCOMPLETE', 'completed': len(records), 'total': total}
            continue
        payload = json.loads((HERE / ('input_' + ds + '.json')).read_text())
        expected = {e['qid'] for e in payload['rows']}
        assert set(qids) == expected
        preds = {'ours': {r['qid']: r['prediction'] for r in records}}
        scores = {}
        if ds == 'mmqa_table':
            with gzip.open(ASSETS / 'data/mmqa_official/MMQA_dev.jsonl.gz', 'rt') as stream:
                original = {r['qid']: r for r in map(json.loads, stream)}
            refs = [{'qid': q, 'gold_answers': [[str(a['answer']) for a in original[q]['answers']]]} for q in qids]
            cache = json.loads((HERE / 'baseline_mmqa_table.json').read_text())
            preds['static'] = {q: cache[q]['static']['original_prediction'] for q in qids}
            for name, answers in preds.items():
                scores[name] = mmqa_scores(answers, refs)
            metric_names = ['list_f1', 'list_em']
        elif ds == 'okvqa':
            with zipfile.ZipFile(ASSETS / 'data/okvqa_local/original/mscoco_val2014_annotations.json.zip') as archive:
                original = {str(r['question_id']): r for r in json.loads(archive.read('mscoco_val2014_annotations.json'))['annotations']}
            cache = json.loads((HERE / 'baseline_okvqa.json').read_text())
            for name in ('always-bm25-static', 'trigger-static', 'no-retrieval'):
                preds[name] = {q: cache[q][name]['original_prediction'] for q in qids}
            official = module(ASSETS / 'cache/okvqa_eval_20261004/vqaEval_py3.py', 'official_vqa')
            for name, answers in preds.items():
                truth = SimpleNamespace(qa=deepcopy({q: original[q] for q in qids}), getQuesIds=lambda: qids)
                prediction = SimpleNamespace(qa={q: {'answer': answers[q]} for q in qids})
                evaluator = official.VQAEval(truth, prediction, n=10)
                with redirect_stdout(StringIO()):
                    evaluator.evaluate(qids)
                scores[name] = {q: {'vqa_accuracy': evaluator.evalQA[q]} for q in qids}
            metric_names = ['vqa_accuracy']
        else:
            import pyarrow.parquet as pq
            original = {r['question_id']: r for r in pq.read_table(ASSETS / 'data/videomme_short/official_annotations.parquet').to_pylist()}
            official = module(HERE / 'official_video_eval.py', 'official_video')
            preds['static'] = {r['qid']: r['baseline_prediction'] for r in records}
            valid = {}
            parsed = {}
            for name, answers in preds.items():
                extracted = {q: official.extract_characters_regex(a) for q, a in answers.items()}
                parsed[name] = extracted
                valid[name] = sum(bool(v) for v in extracted.values())
                scores[name] = {q: {'choice_accuracy': 100 * int(extracted[q] == original[q]['answer'])} for q in qids}
                grouped = {}
                for q in qids:
                    ref = original[q]
                    item = grouped.setdefault(ref['videoID'], {k: ref[k] for k in ('duration', 'domain', 'sub_category')})
                    item.setdefault('questions', []).append({'task_type': ref['task_type'], 'answer': ref['answer'], 'response': answers[q]})
                artifact = HERE / ('video_official_input_' + name + '.json')
                artifact.write_text(json.dumps(list(grouped.values()), ensure_ascii=False) + '\n')
                log = StringIO()
                with redirect_stdout(log):
                    official.eval_your_results(str(artifact), video_types=['short'], skip_missing=False)
                (HERE / ('video_official_' + name + '.log')).write_text(log.getvalue())
            metric_names = ['choice_accuracy']
        comparisons = {}
        clusters = [original[q]['videoID'] for q in qids] if ds == 'videomme' else None
        for name in preds:
            if name != 'ours':
                comparisons[name] = {m: paired([scores[name][q][m] for q in qids],
                                              [scores['ours'][q][m] for q in qids], clusters) for m in metric_names}
        details = [{'qid': q, 'predictions': {m: p[q] for m, p in preds.items()},
                    'official_scores': {m: p[q] for m, p in scores.items()}} for q in qids]
        target = HERE / (ds + '_official_scores.jsonl')
        target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in details))
        calibration = set(payload['calibration_qids'])
        held = [q for q in qids if q not in calibration]
        held_clusters = [original[q]['videoID'] for q in held] if ds == 'videomme' else None
        excluded = {name: {m: paired([scores[name][q][m] for q in held], [scores['ours'][q][m] for q in held], held_clusters)
                           for m in metric_names} for name in preds if name != 'ours'}
        result = {'status': 'COMPLETE', 'n': total, 'comparisons': comparisons,
                  'excluding_calibration': {'n': len(held), 'comparisons': excluded},
                  'hit_token_cap': sum(r['hit_token_cap'] for r in records),
                  'scores_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                  'prediction_source_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
        if ds == 'videomme':
            result['valid_options'] = valid
            result['fixed_denominator_diagnostic'] = {'n': total, 'comparisons': comparisons,
                                                      'excluding_calibration': result['excluding_calibration']}
            def official_comparisons(ids):
                return {'static': {'choice_accuracy': video_official_comparison(
                    [scores['static'][q]['choice_accuracy'] for q in ids],
                    [scores['ours'][q]['choice_accuracy'] for q in ids],
                    [bool(parsed['static'][q]) for q in ids],
                    [bool(parsed['ours'][q]) for q in ids], [original[q]['videoID'] for q in ids])}}
            result['comparisons'] = official_comparisons(qids)
            result['excluding_calibration'] = {'n': len(held), 'comparisons': official_comparisons(held)}
            result['official_script_denominator'] = 'The unmodified official script divides correct answers by parsed answers separately for each arm.'
            result['official_script_accuracy'] = {name: sum(v['choice_accuracy'] for v in scores[name].values()) / valid[name] if valid[name] else 0 for name in preds}
        results['datasets'][ds] = result
    (HERE / 'OFFICIAL_RESULTS.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({ds: {k: v for k, v in result.items() if k in ('status', 'n', 'completed', 'total', 'comparisons', 'valid_options')}
                      for ds, result in results['datasets'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
