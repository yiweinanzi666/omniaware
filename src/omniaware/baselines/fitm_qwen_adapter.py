"""Found-in-the-Middle Qwen adaptation (matched-length null reference).

Sequential inference; no model loading. Null content, independent layer/head
weights, frozen alpha and query timing are explicit project adaptation choices.
"""

from contextlib import contextmanager, ExitStack
import hashlib
import time
from unittest.mock import patch

import torch
import torch.nn.functional as F

from .fitm_attention import (TEMPERATURE, calibrated_alpha, document_means,
                             redistribute_document_attention, validate_documents)


METHOD_NAME = "Found-in-the-Middle Qwen adaptation (matched-length null reference)"
DUMMY_TEXT = "This is a reference document."


def document_indices(sources):
    return [sorted(set(source["text_token_indices"] + source["vision_token_indices"])) for source in sources]


def selected_layers(model):
    count = len(model.model.language_model.layers)
    if count < 16:
        raise ValueError("This adapter requires the specified last 16 decoder layers")
    return list(range(count - 16, count))


def _last_log_attention(query, key, mask, *, scale, is_causal, enable_gqa):
    query_index = query.shape[-2] - 1
    query = query[..., -1:, :].float()
    key = key.float()
    if enable_gqa:
        key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    scores = query @ key.transpose(-2, -1) * (query.shape[-1] ** -.5 if scale is None else scale)
    if is_causal:
        scores.masked_fill_(torch.arange(key.shape[-2], device=key.device) > query_index, -torch.inf)
    if mask is not None:
        mask = mask[..., -1:, :] if mask.ndim >= 2 else mask
        if mask.dtype == torch.bool:
            scores.masked_fill_(~mask, -torch.inf)
        else:
            scores += mask.float()
    return scores.log_softmax(-1)


def corrected_sdpa_output(original, query, key, value, mask, alpha, documents,
                          *, scale=None, is_causal=False, enable_gqa=False):
    log_attention = _last_log_attention(query, key, mask, scale=scale,
                                      is_causal=is_causal, enable_gqa=enable_gqa)
    adjusted, statistics = redistribute_document_attention(log_attention, alpha, documents)
    values = value.float()
    if enable_gqa:
        values = values.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
    correction = (adjusted - log_attention.exp()) @ values
    result = original.clone()
    result[..., -1:, :] = (original[..., -1:, :].float() + correction).to(original.dtype)
    return result, statistics


@contextmanager
def qwen_fitm(model, documents, *, alpha=None):
    """Observe last prompt row if alpha=None; otherwise calibrate the last row.

    In generate this selects the final prefill row, then every single-row decode.
    Alpha maps actual layer indices to [batch,query_heads,1,documents] tensors.
    Disabled/observation mode returns the original SDPA output exactly.
    """
    layers = selected_layers(model)
    state = dict(layer_indices=layers, means={}, calls={layer: 0 for layer in layers}, first={}, last={})
    original_sdpa = F.scaled_dot_product_attention

    def wrap(attention, layer):
        original_forward = attention.forward
        def tap(*args, **kwargs):
            original = original_sdpa(*args, **kwargs)
            query, key, value = args[:3]
            mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
            dropout = kwargs.get("dropout_p", args[4] if len(args) > 4 else 0.)
            if dropout:
                raise ValueError("FITM inference requires dropout disabled")
            parameters = dict(scale=kwargs.get("scale"),
                is_causal=kwargs.get("is_causal", args[5] if len(args) > 5 else False),
                enable_gqa=kwargs.get("enable_gqa", False))
            validate_documents(documents, key.shape[-2])
            state["calls"][layer] += 1
            if alpha is None:
                log_attention = _last_log_attention(query, key, mask, **parameters)
                state["means"][layer] = document_means(log_attention, documents)
                return original
            result, statistics = corrected_sdpa_output(original, query, key, value, mask,
                alpha[layer], documents, **parameters)
            statistics.update(query_tokens=query.shape[-2], key_tokens=key.shape[-2])
            state["first"].setdefault(layer, statistics)
            state["last"][layer] = statistics
            return result
        def forward(*args, **kwargs):
            with patch.object(F, "scaled_dot_product_attention", tap):
                return original_forward(*args, **kwargs)
        return forward

    with ExitStack() as stack:
        for layer in layers:
            attention = model.model.language_model.layers[layer].self_attn
            stack.enter_context(patch.object(attention, "forward", wrap(attention, layer)))
        yield state


def _tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def make_null_reference(inputs, sources, target, tokenizer):
    """Replace one source at its original physical text/image positions."""
    reference = {key: value.clone() for key, value in inputs.items()}
    ordinary = [token for token in tokenizer.encode(DUMMY_TEXT, add_special_tokens=False)
                if token not in set(tokenizer.all_special_ids)]
    text_indices = sources[target]["text_token_indices"]
    if not ordinary:
        raise ValueError("Dummy text must have ordinary tokenizer IDs")
    replacement = [ordinary[index % len(ordinary)] for index in range(len(text_indices))]
    if text_indices:
        reference["input_ids"][0, text_indices] = torch.tensor(replacement, device=inputs["input_ids"].device)
    metadata = dict(target_document=target, source_id=sources[target].get("source_id"),
        dummy_text=DUMMY_TEXT, dummy_base_token_ids=ordinary, text_token_indices=text_indices,
        replacement_token_ids=replacement, visual_null=None,
        reference_input_token_ids=reference["input_ids"][0].detach().cpu().tolist(),
        non_target_documents="original contents retained", dummy_constant_relevance_proven=False)
    visual = sorted((index for index, source in enumerate(sources) if source["vision_token_indices"]),
                    key=lambda index: min(sources[index]["vision_token_indices"]))
    if target in visual:
        grids = inputs["image_grid_thw"].detach().cpu().tolist()
        if len(visual) != len(grids):
            raise ValueError("Current packing requires one image grid per visual source")
        image_index = visual.index(target)
        sizes = [t * h * w for t, h, w in grids]
        start = sum(sizes[:image_index])
        stop = start + sizes[image_index]
        original_pixels = reference["pixel_values"][start:stop]
        metadata.update(visual_null="processor_pixel_values_zero", image_index=image_index,
            image_grid_thw=grids[image_index], pixel_patch_range=[start, stop],
            original_pixel_patch_sha256=_tensor_hash(original_pixels))
        original_pixels.zero_()
        metadata["reference_pixel_patch_sha256"] = _tensor_hash(original_pixels)
    metadata["input_tensor_sha256"] = {key: _tensor_hash(value) for key, value in reference.items()}
    return reference, metadata


def _synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def prepare_fitm_calibration(model, inputs, sources, tokenizer, *, call_record=None):
    """K null forwards + one unbiased actual-prompt forward; return alpha, record."""
    _synchronize(model.device)
    started = time.perf_counter()
    documents = document_indices(sources)
    validate_documents(documents, inputs["input_ids"].shape[1])
    if inputs["input_ids"].shape[0] != 1 or model.training:
        raise ValueError("Need one prepared question and an eval-mode model")
    layers = selected_layers(model)
    record = {} if call_record is None else call_record
    record.update(method=METHOD_NAME, status="started", temperature=TEMPERATURE,
        layer_indices=layers, document_token_indices=documents,
        source_ids=[source.get("source_id") for source in sources],
        document_token_counts=list(map(len, documents)), dummy_text=DUMMY_TEXT,
        dummy_constant_relevance_proven=False, references=[], extra_forward_calls_started=0,
        forward_parameters=dict(use_cache=False, logits_to_keep=1), forward_attempts=[],
        reference_forward_seconds=0., reference_preparation_seconds=0.,
        actual_observation_seconds=0., input_token_ids=inputs["input_ids"][0].detach().cpu().tolist(),
        input_tensor_sha256={key: _tensor_hash(value) for key, value in inputs.items()},
        alpha_policy="independent layer/query-head; frozen before first answer token",
        application="last prefill query and every decode query; per-query mean denominator and C")
    baseline = {}

    def observe(prepared):
        _synchronize(model.device)
        before = time.perf_counter()
        record["extra_forward_calls_started"] += 1
        attempt = dict(stage=record["active_stage"], status="started")
        record["forward_attempts"].append(attempt)
        try:
            with qwen_fitm(model, documents) as capture:
                model(**prepared, **record["forward_parameters"])
            _synchronize(model.device)
            elapsed = time.perf_counter() - before
            if any(capture["calls"][layer] != 1 for layer in layers):
                raise ValueError("Each selected layer needs one actual SDPA observation")
            attempt.update(status="completed", synchronized_seconds=elapsed)
            return capture["means"], elapsed
        except BaseException as error:
            attempt.update(status="failed", elapsed_seconds_until_failure=time.perf_counter() - before,
                           error_type=type(error).__name__, error=str(error))
            raise

    try:
        for target in range(len(documents)):
            record["active_stage"] = f"reference_{target}"
            before = time.perf_counter()
            reference, metadata = make_null_reference(inputs, sources, target, tokenizer)
            _synchronize(model.device)
            metadata["preparation_seconds"] = time.perf_counter() - before
            record["reference_preparation_seconds"] += metadata["preparation_seconds"]
            record["references"].append(metadata)
            means, seconds = observe(reference)
            metadata["forward_seconds"] = seconds
            record["reference_forward_seconds"] += seconds
            metadata["dummy_mean_attention"] = {str(layer): means[layer][..., target].cpu().tolist() for layer in layers}
            for layer in layers:
                if layer not in baseline:
                    baseline[layer] = torch.empty_like(means[layer])
                baseline[layer][..., target] = means[layer][..., target]
        record["active_stage"] = "unbiased_actual_prompt"
        original, seconds = observe(inputs)
        record["actual_observation_seconds"] = seconds
        alpha = {layer: calibrated_alpha(original[layer], baseline[layer]) for layer in layers}
        record.update(original_document_means={str(layer): original[layer].cpu().tolist() for layer in layers},
            reference_document_means={str(layer): baseline[layer].cpu().tolist() for layer in layers},
            calibrated_document_scores={str(layer): (original[layer] - baseline[layer]).cpu().tolist() for layer in layers},
            alpha={str(layer): alpha[layer].cpu().tolist() for layer in layers},
            status="completed", active_stage=None,
            extra_forward_seconds=record["reference_forward_seconds"] + seconds,
            total_calibration_seconds=time.perf_counter() - started)
        return alpha, record
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error),
                      total_calibration_seconds=time.perf_counter() - started)
        raise
