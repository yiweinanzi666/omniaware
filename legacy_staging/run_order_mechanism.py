"""T17: fixed-content permutations with static, replication and V3.1-timed EAB."""

from datetime import datetime, timezone
import gc
import itertools
import json
import math
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, GenerationConfig, Qwen3VLForConditionalGeneration

from .eab_profiles import evidence_token_map
from .qwen_paper_eab import qwen_paper_eab
from .run_complete_answer_development import make_record
from .run_reranker_development import save_json
from .run_sentence_selection import row_digest
from .run_token_budget import text_digest
from .run_webqa import ROOT, SYSTEM, digest, observe, read_jsonl, save_record, synchronize
from .sentence_selection import render_input


SOURCE = ROOT / 'runs/reranker_order_development'
RANKING = ROOT / 'runs/reranker_development/reranking.jsonl'
CALIBRATION = ROOT / 'runs/webqa_text_development/calibration_summary.json'
MODEL = ROOT / 'cache/qwen3-vl-8b-instruct'
OUTPUT = ROOT / 'runs/order_mechanism_development'
PROTOCOL = ROOT / 'T17_ORDER_MECHANISM_PROTOCOL.md'
METHODS = ('static', 'replication', 'paper-eab')
PERMUTATION_SEEDS = (17, 23, 29)
GAMMA, TAU = 1.2, .4


def permutation_schedule():
    anchors = [tuple(range(5)), tuple(reversed(range(5)))]
    available = [p for p in itertools.permutations(range(5)) if p not in anchors]
    schedule = [dict(order_id=name, kind='anchor', seed=None, permutation=list(p))
        for name, p in zip(('sorted', 'reverse'), anchors, strict=True)]
    for seed in PERMUTATION_SEEDS:
        chosen = random.Random(seed).sample(available, 4)
        for index, p in enumerate(chosen):
            schedule.append(dict(order_id=f's{seed}-p{index+1}', kind='random', seed=seed, permutation=list(p)))
        available = [p for p in available if p not in chosen]
    return schedule


def replication_counts(relevances, uncertainty, gate):
    return [math.ceil(math.exp(GAMMA * uncertainty * value)) if gate else 1 for value in relevances]


def prepare_input(tokenizer, original, order, method, relevances, observation):
    counts = replication_counts(relevances, observation['u_initial'], observation['gate_enabled']) if method == 'replication' else [1]*5
    blocks, weights, positions = [], [], []
    for rank in order['permutation']:
        for copy_index in range(counts[rank]):
            block = original['evidence_blocks'][rank]
            positions.append(dict(fact_id=block['fact_id'], original_rank=rank+1, copy_index=copy_index+1,
                input_block_position=len(blocks)+1))
            blocks.append(block)
            weights.append(relevances[rank])
    content = '\n\n'.join([b['block'] for b in blocks] + [f"Question: {original['question']}"])
    inputs, trace = render_input(tokenizer, content, SYSTEM)
    mapping = evidence_token_map(tokenizer, trace['rendered_prompt'], [b['block'] for b in blocks], weights)
    trace.update(evidence_blocks=blocks, evidence_ids=[b['fact_id'] for b in blocks],
        evidence_sha256=text_digest('\n\n'.join(b['block'] for b in blocks)),
        source_evidence_sha256=original['evidence_sha256'], source_row_sha256=row_digest(original),
        evidence_token_counts=mapping['evidence_token_counts'], block_positions=positions,
        per_original_block_relevance=relevances, per_original_block_copies=counts,
        order_id=order['order_id'], order_kind=order['kind'], permutation_seed=order['seed'],
        permutation=order['permutation'], method=method, max_chunk_tokens=256, max_new_tokens=128,
        u_initial=observation['u_initial'], gate_enabled=observation['gate_enabled'])
    return inputs, mapping['evidence_relevance'], trace


def prepare_config(originals):
    path = OUTPUT / 'config.json'
    if path.exists():
        return json.loads(path.read_text())
    reference = json.loads((SOURCE / 'config.json').read_text())
    sources = {'t15_predictions.jsonl': SOURCE / 'predictions.jsonl',
        't15_config.json': SOURCE / 'config.json', 't14_reranking.jsonl': RANKING,
        'calibration_summary.json': CALIBRATION, PROTOCOL.name: PROTOCOL}
    for name in ('run_order_mechanism.py', 'score_order_mechanism.py', 'qwen_paper_eab.py',
                 'test_qwen_paper_eab.py', 'test_order_mechanism.py', 'eab_profiles.py',
                 'run_complete_answer_development.py', 'run_sentence_selection.py',
                 'run_webqa.py', 'sentence_selection.py', 'run_token_budget.py',
                 'run_reranker_development.py', 'prepare_webqa.py', 'qwen_adapter.py', 'eab.py',
                 'score_evidence_order.py', 'score_webqa.py', 'score_bart.py'):
        sources[name] = ROOT / name
    config = dict(experiment='T17 fixed-content order mechanism', stage='development',
        created_utc=datetime.now(timezone.utc).isoformat(), samples=len(originals),
        methods=list(METHODS), orders=permutation_schedule(),
        planned_generations=len(originals)*len(permutation_schedule())*len(METHODS), planned_observations=len(originals),
        model_path=str(MODEL), model_revision=reference['model_revision'],
        model_assets_sha256=reference['model_assets_sha256'], seed=17, device='mps', dtype='bfloat16', attention='sdpa',
        torch=torch.__version__, transformers=transformers.__version__, system_prompt=SYSTEM,
        gamma=GAMMA, tau_bias=TAU, beta=.3, max_chunk_tokens=256, max_new_tokens=128,
        eab_timing='first generated query row; first affects second answer token; all decoder layers',
        eab_precision='float32 relevance, multiplication and additive-mask combination, then model dtype',
        replication='gated ceil(exp(gamma * U * r)) adjacent whole-block copies; no additional truncation',
        permutation_definition='12 unique random permutations from 3 seeds, 4 each, plus sorted/reverse anchors. Greedy generation uses seed17. Not 3 independent generation repetitions.',
        input_definition='T15 CE-selected/BGE-ordered saved blocks; labels remain bound to facts; static and EAB identical tokens',
        bootstrap=dict(unit='question, retaining all permutations and methods',resamples=10000,seed=20261002),
        cost_boundary='Static: saved CE+sorting and new generation. Replication/EAB: additionally new question-only observation. Common BGE, loads, CPU preparation, I/O and scoring excluded. Segmented timings, not end-to-end.',
        sources={name:dict(path=str(p),sha256=digest(p)) for name,p in sources.items()})
    snapshot = OUTPUT / 'source'
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, p in sources.items():
        (snapshot/name).write_bytes(p.read_bytes())
    save_json(path, config)
    return config


def generate(originals, config):
    path = OUTPUT / 'predictions.jsonl'
    observations_path = OUTPUT / 'observations.jsonl'
    saved = {(r['qid'],r['variant']):r for r in read_jsonl(path)}
    observations = {r['qid']:r for r in read_jsonl(observations_path)}
    ranking = {r['qid']:r['selected_item'] for r in read_jsonl(RANKING)}
    calibration = json.loads(CALIBRATION.read_text())
    config_hash = digest(OUTPUT / 'config.json')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    torch.manual_seed(17)
    print(json.dumps(dict(event='loading_model', planned_generations=config['planned_generations'])),flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
        device_map='mps',dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    eos = model.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos,int) else eos
    decode = GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=128,
        eos_token_id=eos,pad_token_id=tokenizer.pad_token_id,use_cache=True)
    for index, original in enumerate(originals):
        qid = original['qid']
        if qid not in observations:
            observation = observe(model,tokenizer,original,SimpleNamespace(max_chunk_tokens=256,device='mps',beta=.3))
            _, observation_trace = render_input(tokenizer, f"Question: {original['question']}", SYSTEM)
            observation.update(input_tokens=observation_trace['input_tokens'],
                input_token_ids=observation_trace['input_token_ids'])
            observation.update(u_initial=float(np.clip((observation['score']-calibration['median'])/(1.5*calibration['iqr'])+.5,0,1)))
            observation.update(gate_enabled=observation['u_initial']>TAU,config_sha256=config_hash)
            save_record(observations_path,observation)
            observations[qid] = observation
        observation = observations[qid]
        by_fact = {b['fact_id']:b['relevance'] for b in ranking[qid]['evidence']}
        relevances = [by_fact[b['fact_id']] for b in original['evidence_blocks']]
        schedule = [(order,method) for order in config['orders'] for method in METHODS]
        schedule.sort(key=lambda pair:text_digest(qid+pair[0]['order_id']+pair[1]))
        for order,method in schedule:
            variant = method+'__'+order['order_id']
            if (qid,variant) in saved:
                continue
            cpu_inputs,relevance,trace = prepare_input(tokenizer,original,order,method,relevances,observation)
            inputs = {k:v.to('mps') for k,v in cpu_inputs.items()}
            synchronize('mps')
            tick = time.perf_counter()
            with torch.inference_mode(), qwen_paper_eab(model,evidence_relevance=relevance,
                    uncertainty=observation['u_initial'],gamma=GAMMA,
                    enabled=method=='paper-eab' and observation['gate_enabled']):
                output = model.generate(**inputs,generation_config=decode)
            synchronize('mps')
            elapsed = time.perf_counter()-tick
            tokens = output[0,trace['input_tokens']:].cpu().tolist()
            row = make_record(original,variant,trace,config_hash,tokens,elapsed,tokenizer,eos_ids)
            row.update(observation_seconds=observation['observation_seconds'],
                reranker_seconds=original['reranker_seconds'],reorder_seconds=original['reorder_seconds'],
                measured_method_seconds=elapsed+original['reranker_seconds']+original['reorder_seconds']+
                    (observation['observation_seconds'] if method!='static' else 0))
            save_record(path,row)
            saved[qid,variant] = row
            print(json.dumps(dict(event='prediction',sample=index+1,total=len(originals),variant=variant,
                completed_generations=len(saved),planned_generations=config['planned_generations'],
                seconds=elapsed,input_tokens=trace['input_tokens'],output_tokens=len(tokens))),flush=True)
            del output,inputs
        gc.collect()
    save_json(OUTPUT/'completion.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        samples=len(originals),predictions=len(saved),observations=len(observations),
        config_sha256=config_hash,predictions_sha256=digest(path),observations_sha256=digest(observations_path)))
    del model
    gc.collect()
    torch.mps.empty_cache()


def main():
    from .score_order_mechanism import score
    originals = [r for r in read_jsonl(SOURCE/'predictions.jsonl') if r['variant']=='cross-selected-bge-order']
    config = prepare_config(originals)
    if not (OUTPUT/'completion.json').exists():
        generate(originals,config)
    if not (OUTPUT/'scored/summary.json').exists():
        score(OUTPUT,config)


if __name__=='__main__':
    main()
