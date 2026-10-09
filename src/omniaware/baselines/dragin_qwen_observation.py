"""Single-question DRAGIN observation on an already loaded, eval-mode Qwen.

Sequential inference only. The draft-only probe returns native SDPA outputs;
CPU float64 probabilities are reconstructed separately for RIND scoring.
"""

from contextlib import contextmanager
from copy import deepcopy
import time
from unittest.mock import patch

import torch
import torch.nn.functional as F

from .dragin_rind import REFERENCE_THRESHOLD, score_draft


@contextmanager
def capture_draft_sdpa(model):
    """Capture all post-RoPE final-layer Q/K/mask without replacing its output."""
    attention = model.model.language_model.layers[-1].self_attn
    original_forward, original_sdpa = attention.forward, F.scaled_dot_product_attention
    capture = {"calls": 0}

    def tap(*args, **kwargs):
        output = original_sdpa(*args, **kwargs)
        query, key = args[:2]
        mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
        scale = kwargs.get("scale")
        capture.update(
            # Keep device transfer and widening separate: MPS bf16 needs this.
            query=query.detach().cpu().double(), key=key.detach().cpu().double(),
            mask=None if mask is None else mask.detach().cpu(),
            query_shape=list(query.shape), key_shape=list(key.shape),
            query_dtype=str(query.dtype), key_dtype=str(key.dtype),
            mask_shape=None if mask is None else list(mask.shape),
            mask_dtype=None if mask is None else str(mask.dtype),
            scale=query.shape[-1] ** -.5 if scale is None else scale,
            is_causal=kwargs.get("is_causal", args[5] if len(args) > 5 else False),
            dropout_p=kwargs.get("dropout_p", args[4] if len(args) > 4 else 0.),
            enable_gqa=kwargs.get("enable_gqa", False), calls=capture["calls"] + 1,
        )
        return output

    def forward(*args, **kwargs):
        with patch.object(F, "scaled_dot_product_attention", tap):
            return original_forward(*args, **kwargs)

    with patch.object(attention, "forward", forward):
        yield capture


def reconstruct_draft_attention(capture):
    """Reconstruct every [B,H,query,key] probability, including causal diagonal."""
    if capture["calls"] != 1 or capture["dropout_p"] != 0:
        raise ValueError("Need exactly one final-layer SDPA call with dropout disabled")
    query, key = capture["query"], capture["key"]
    if not torch.isfinite(query).all() or not torch.isfinite(key).all():
        raise ValueError("SDPA query and key must be finite")
    if capture["enable_gqa"]:
        key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    scores = (query @ key.transpose(-2, -1)) * capture["scale"]
    if capture["is_causal"]:
        blocked = torch.ones(scores.shape[-2:], dtype=torch.bool).triu(1)
        scores.masked_fill_(blocked, -torch.inf)
    mask = capture["mask"]
    if mask is not None:
        if mask.dtype == torch.bool:
            scores.masked_fill_(~mask, -torch.inf)
        else:
            scores += mask.double()
    # Native SDPA returns zero for a wholly masked row.
    weights = scores.softmax(dim=-1).masked_fill(torch.isneginf(scores).all(-1, keepdim=True), 0.)
    if not torch.isfinite(weights).all():
        raise ValueError("Reconstructed attention is not finite")
    return weights


def _synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def observe_dragin_draft(model, prepared_inputs, tokenizer, nlp, *, generation_config,
                         threshold=REFERENCE_THRESHOLD, call_record=None):
    """Generate one question-only draft and score one fresh draft-only forward.

    Pass a GenerationConfig with an explicit max_new_tokens. Greedy/single
    output and score capture are fixed here; EOS, padding, suppressions and
    other processors come from the copied config and are recorded in full.
    Returns JSON-compatible diagnostics. Large scores and Q/K are temporary;
    token entropy and reduced attention retain the numerical inputs to RIND.
    An optional mutable call_record retains completed stages if a later stage fails.
    """
    input_ids = prepared_inputs["input_ids"]
    config = deepcopy(generation_config)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1]:
        raise ValueError("Need one nonempty question-only token sequence")
    if model.training or not isinstance(config.max_new_tokens, int) or config.max_new_tokens < 1:
        raise ValueError("Need an eval-mode model and a positive max_new_tokens")
    config.do_sample, config.num_beams, config.num_return_sequences = False, 1, 1
    config.return_dict_in_generate, config.output_scores, config.use_cache = True, True, True
    config.output_logits, config.output_attentions, config.output_hidden_states = False, False, False
    progress = {} if call_record is None else call_record
    progress.update(status="started", active_stage="generation", generation_config=config.to_dict(),
                    generation_calls=0, extra_forward_calls=0)
    device = input_ids.device
    _synchronize(device)
    started = time.perf_counter()
    progress["generation_calls"] = 1
    generated = model.generate(**prepared_inputs, generation_config=config)
    _synchronize(device)
    generation_seconds = time.perf_counter() - started
    progress["draft_generation_seconds"] = generation_seconds
    draft = generated.sequences[:, input_ids.shape[1]:].clone()
    processed_scores = generated.scores
    del generated  # Release the generation KV cache before the fresh probe.
    ids = draft[0].detach().cpu().tolist()
    if not ids or len(processed_scores) != len(ids):
        raise ValueError("Each emitted draft ID needs one processed generation score")
    progress.update(draft_token_ids=ids, generated_tokens=len(ids),
        raw_draft_text=tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        processed_score_count=len(processed_scores), active_stage="draft_probe")

    probe_inputs = dict(input_ids=draft, attention_mask=torch.ones_like(draft))
    probe_parameters = dict(use_cache=False, logits_to_keep=1)
    _synchronize(device)
    probe_started = time.perf_counter()
    progress["extra_forward_calls"] = 1
    with capture_draft_sdpa(model) as capture:
        model(**probe_inputs, **probe_parameters)
    _synchronize(device)
    probe_seconds = time.perf_counter() - probe_started
    progress.update(draft_probe_seconds=probe_seconds, active_stage="rind_scoring")
    scoring_started = time.perf_counter()
    attention = reconstruct_draft_attention(capture)
    diagnostic = score_draft(ids, processed_scores, attention, tokenizer, nlp, threshold=threshold)
    scoring_seconds = time.perf_counter() - scoring_started

    eos_ids = config.eos_token_id
    eos_ids = [] if eos_ids is None else [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
    ended_with_eos = ids[-1] in eos_ids
    reached_cap = len(ids) >= config.max_new_tokens
    record = dict(
        **diagnostic,
        raw_draft_text=tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        input_token_ids=input_ids[0].detach().cpu().tolist(),
        generation_inputs={key: value.detach().cpu().tolist() for key, value in prepared_inputs.items()},
        probe_input_token_ids=ids, probe_parameters=probe_parameters,
        generation_config=config.to_dict(), generated_tokens=len(ids),
        eos_emitted=any(token in eos_ids for token in ids), ended_with_eos=ended_with_eos,
        reached_max_new_tokens=reached_cap, length_limit_hit=reached_cap and not ended_with_eos,
        stop_reason="eos" if ended_with_eos else "length_limit" if reached_cap else "other",
        processed_score_count=len(processed_scores), processed_score_shape=list(processed_scores[0].shape),
        sdpa={key: value for key, value in capture.items() if key not in ("query", "key", "mask")},
        attention_reconstruction="CPU float64 softmax from all final-layer post-RoPE Q/K and native mask/scale",
        draft_generation_seconds=generation_seconds, draft_probe_seconds=probe_seconds,
        rind_scoring_seconds=scoring_seconds, total_seconds=time.perf_counter() - started,
        generation_calls=1, extra_forward_calls=1,
        device=str(device), model_dtype=str(next(model.parameters()).dtype),
    )
    progress.update(record, status="completed", active_stage=None)
    return record
