"""Qwen3-VL runner with Evidence Attention Bias (EAB).

EAB (paper Eq. 7-8) adds ``M_bias[a, b] = value_j`` to the attention logits of every text
decoder layer when key position b lies in evidence chunk j and query position a >= q_start.
The bias is injected by wrapping the global ``sdpa`` attention function; vision-tower
attention and positions outside the selected rows are untouched.

All methods for one example share the same token sequence.  The heavy prefix (query
image + evidence) is prefilled once; each method then runs the short question suffix and
greedy decoding with its own bias, and the KV cache is cropped back afterwards.  Rows
before ``q_start`` are not biased, so sharing the prefix is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from PIL import Image
from transformers import AttentionInterface, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

EPS = 1e-9


# ------------------------------------------------------------ attention hook --
class _State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bias: torch.Tensor | None = None   # [P] float, per key column
        self.q_start: int = 0                    # first biased query row (absolute)
        self.layers: set[int] | None = None      # None = all text layers
        self.capture_layer: int | None = None
        self.captured: torch.Tensor | None = None


STATE = _State()
_ORIG_SDPA = sdpa_attention_forward


def _eab_sdpa(module, query, key, value, attention_mask, **kwargs):
    layer = getattr(module, "_erc_layer", None)
    if layer is None:
        return _ORIG_SDPA(module, query, key, value, attention_mask, **kwargs)
    st = STATE
    q_len, kv_len = query.shape[2], key.shape[2]
    q_pos0 = kv_len - q_len

    if st.capture_layer is not None and layer == st.capture_layer:
        k = key if key.shape[1] == query.shape[1] else repeat_kv(key, query.shape[1] // key.shape[1])
        scale = kwargs.get("scaling") or module.scaling
        logits = (query[:, :, -1:, :].float() @ k.float().transpose(-1, -2)) * scale
        st.captured = torch.softmax(logits, dim=-1).mean(dim=1)[0, 0].detach().float().cpu()

    if st.bias is not None and (st.layers is None or layer in st.layers):
        first_row = max(st.q_start - q_pos0, 0)
        if first_row < q_len:
            dtype = query.dtype
            neg = torch.finfo(dtype).min
            if attention_mask is None:
                rows = torch.arange(q_pos0, kv_len, device=query.device)[:, None]
                cols = torch.arange(kv_len, device=query.device)[None, :]
                mask = torch.zeros(q_len, kv_len, device=query.device, dtype=dtype)
                mask = mask.masked_fill(cols > rows, neg)[None, None]
            elif attention_mask.dtype == torch.bool:
                mask = torch.zeros(attention_mask.shape, device=query.device, dtype=dtype)
                mask = mask.masked_fill(~attention_mask, neg)
            else:
                mask = attention_mask.to(dtype).clone()
            col = torch.zeros(kv_len, device=query.device, dtype=dtype)
            n = min(kv_len, st.bias.numel())
            col[:n] = st.bias[:n].to(device=query.device, dtype=dtype)
            mask = mask.expand(mask.shape[0], mask.shape[1], q_len, kv_len).clone()
            mask[..., first_row:, :] += col
            attention_mask = mask
    return _ORIG_SDPA(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register("sdpa", _eab_sdpa)


def _crop(cache, length: int) -> None:
    """Truncate the KV cache to ``length`` tokens (negative form works on old and new transformers)."""
    extra = cache.get_seq_length() - length
    if extra > 0:
        cache.crop(-extra)


# ----------------------------------------------------------------- prompts --
@dataclass
class Segment:
    text: str | None = None
    image: str | None = None
    chunk: str | None = None      # evidence chunk id (span bookkeeping)
    max_pixels: int | None = None # per-image pixel cap (None: model default)


@dataclass
class Encoded:
    input_ids: torch.Tensor                 # [1, P]
    mm_token_type_ids: torch.Tensor         # [1, P]
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    spans: dict[str, tuple[int, int]]       # chunk id -> [start, end)
    split: int                              # first token of the question suffix
    length: int


@dataclass
class BiasSpec:
    values: dict[str, float]                # chunk id -> additive logit bias
    q_start: int                            # absolute first biased query row
    layers: set[int] | None = None


@dataclass
class Result:
    text: str
    first_logits: torch.Tensor = field(repr=False)
    tokens: list[int] = field(default_factory=list)


class QwenEAB:
    def __init__(self, model_path, device: str | None = None, dtype=torch.bfloat16,
                 max_pixels: int = 256 * 32 * 32, min_pixels: int = 64 * 32 * 32):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tok = self.processor.tokenizer
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.layers = []
        for m in self.model.modules():
            if m.__class__.__name__ == "Qwen3VLTextAttention":
                m._erc_layer = m.layer_idx
                self.layers.append(m.layer_idx)
        self.n_layers = len(self.layers)
        cfg = self.model.config
        self.image_token_id = cfg.image_token_id
        self.vision_start_id = cfg.vision_start_token_id
        self.vision_end_id = cfg.vision_end_token_id
        self.merge = self.processor.image_processor.merge_size
        self.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        self.eos = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id,
                    self.tok.convert_tokens_to_ids("<|endoftext|>")}
        self.vocab_size = self.model.config.text_config.vocab_size

    # ------------------------------------------------------------ encoding --
    def encode(self, segments: list[Segment], split_marker: int) -> Encoded:
        """segments[split_marker:] form the question suffix (must be text only)."""
        image_segs = [s for s in segments if s.image is not None]
        images = [Image.open(s.image).convert("RGB") for s in image_segs]
        pixel_values = grid = None
        if images and any(s.max_pixels for s in image_segs):
            parts = []
            for s, im in zip(image_segs, images):
                cap = s.max_pixels or self.size["longest_edge"]
                size = {"shortest_edge": min(self.size["shortest_edge"], cap), "longest_edge": cap}
                parts.append(self.processor.image_processor(images=[im], size=size, return_tensors="pt"))
            pixel_values = torch.cat([p["pixel_values"] for p in parts], dim=0)
            grid = torch.cat([p["image_grid_thw"] for p in parts], dim=0)
        elif images:
            img = self.processor.image_processor(images=images, size=self.size, return_tensors="pt")
            pixel_values, grid = img["pixel_values"], img["image_grid_thw"]
        ids: list[int] = []
        types: list[int] = []
        spans: dict[str, list[int]] = {}
        split = None
        k = 0
        for i, s in enumerate(segments):
            if i == split_marker:
                split = len(ids)
            start = len(ids)
            if s.image is not None:
                n = int(grid[k].prod().item()) // (self.merge ** 2)
                k += 1
                ids += [self.vision_start_id] + [self.image_token_id] * n + [self.vision_end_id]
                types += [0] + [1] * n + [0]
            else:
                t = self.tok.encode(s.text, add_special_tokens=False)
                ids += t
                types += [0] * len(t)
            if s.chunk is not None:
                if s.chunk in spans:
                    spans[s.chunk][1] = len(ids)
                else:
                    spans[s.chunk] = [start, len(ids)]
        if split is None:
            split = len(ids)
        return Encoded(
            input_ids=torch.tensor([ids]), mm_token_type_ids=torch.tensor([types]),
            pixel_values=pixel_values, image_grid_thw=grid,
            spans={c: (a, b) for c, (a, b) in spans.items()}, split=split, length=len(ids))

    # ------------------------------------------------------------ forward ---
    def _bias_tensor(self, enc: Encoded, spec: BiasSpec) -> torch.Tensor:
        col = torch.zeros(enc.length, dtype=torch.float32)
        for cid, v in spec.values.items():
            if cid in enc.spans and v != 0.0:
                a, b = enc.spans[cid]
                col[a:b] = float(v)
        return col.to(self.device)

    def _activate(self, enc: Encoded, spec: BiasSpec | None):
        STATE.bias = None if spec is None else self._bias_tensor(enc, spec)
        STATE.q_start = 0 if spec is None else spec.q_start
        STATE.layers = None if spec is None else spec.layers

    @torch.no_grad()
    def _prefix(self, enc: Encoded):
        self.model.model.rope_deltas = None
        STATE.reset()
        kw = dict(input_ids=enc.input_ids[:, :enc.split].to(self.device),
                  mm_token_type_ids=enc.mm_token_type_ids[:, :enc.split].to(self.device),
                  use_cache=True, logits_to_keep=1)
        if enc.pixel_values is not None:
            kw.update(pixel_values=enc.pixel_values.to(self.device, self.model.dtype),
                      image_grid_thw=enc.image_grid_thw.to(self.device))
        out = self.model(**kw)
        if self.model.model.rope_deltas is None:
            self.model.model.rope_deltas = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        return out.past_key_values

    @torch.no_grad()
    def _step(self, ids: torch.Tensor, cache, spec_enc=None, capture=False) -> torch.Tensor:
        STATE.capture_layer = self.layers[-1] if capture else None
        STATE.captured = None
        out = self.model(input_ids=ids.to(self.device), past_key_values=cache, use_cache=True, logits_to_keep=1)
        STATE.capture_layer = None
        return out.logits[0, -1].float()

    @torch.no_grad()
    def _decode(self, cache, first_logits: torch.Tensor, max_new_tokens: int) -> list[int]:
        tokens = []
        logits = first_logits
        for _ in range(max_new_tokens):
            nxt = int(torch.argmax(logits).item())
            if nxt in self.eos:
                break
            tokens.append(nxt)
            if len(tokens) == max_new_tokens:
                break
            logits = self._step(torch.tensor([[nxt]]), cache)
        return tokens

    @torch.no_grad()
    def run(self, enc: Encoded, make_variants, max_new_tokens: int, decode_static=True):
        """Run the unbiased prefill (needed for U), then every variant returned by
        ``make_variants(grounding_attention, unbiased_first_logits)``.

        ``decode_static`` (bool or callable(grounding, logits) -> bool) controls whether the
        unbiased answer itself is decoded and returned as "static".
        Returns (results, grounding attention, unbiased first-token logits).
        """
        cache = self._prefix(enc)
        suffix = enc.input_ids[:, enc.split:]
        self._activate(enc, None)
        base_logits = self._step(suffix, cache, capture=True)
        grounding = STATE.captured
        results = {}
        if decode_static(grounding, base_logits) if callable(decode_static) else decode_static:
            toks = self._decode(cache, base_logits, max_new_tokens)
            results["static"] = Result(self.tok.decode(toks, skip_special_tokens=True).strip(), base_logits.cpu(), toks)
            _crop(cache, enc.length)
        variants = make_variants(grounding, base_logits) if make_variants else {}
        # variants that keep the prompt unbiased share the static suffix cache
        late = {k: v for k, v in variants.items() if v is None or v.q_start >= enc.length}
        early = {k: v for k, v in variants.items() if k not in late}
        unbiased = results.get("static")
        for name, spec in late.items():
            if spec is None and unbiased is not None:
                results[name] = unbiased
                continue
            self._activate(enc, spec)
            toks = self._decode(cache, base_logits, max_new_tokens)
            self._activate(enc, None)
            results[name] = Result(self.tok.decode(toks, skip_special_tokens=True).strip(), base_logits.cpu(), toks)
            if spec is None:
                unbiased = results[name]
            _crop(cache, enc.length)
        for name, spec in early.items():
            _crop(cache, enc.split)
            self._activate(enc, spec)
            logits = self._step(suffix, cache)
            toks = self._decode(cache, logits, max_new_tokens)
            self._activate(enc, None)
            results[name] = Result(self.tok.decode(toks, skip_special_tokens=True).strip(), logits.cpu(), toks)
        STATE.reset()
        return results, grounding, base_logits.cpu()


# ------------------------------------------------------------- uncertainty --
def predictive_entropy(logits: torch.Tensor) -> float:
    logp = torch.log_softmax(logits.float(), dim=-1)
    h = -(logp.exp() * logp).sum().item()
    return h / math.log(logits.numel())


def grounding_entropy(attn: torch.Tensor | None, token_idx: list[int]) -> float:
    if attn is None or len(token_idx) <= 1:
        return 0.0
    a = attn[token_idx].double()
    a = a / (a.sum() + EPS)
    h = -(a * torch.log(a + EPS)).sum().item()
    return h / math.log(max(len(token_idx), 2))


def chunk_mass(attn: torch.Tensor | None, spans: dict[str, tuple[int, int]]) -> dict[str, float]:
    if attn is None:
        return {}
    return {c: float(attn[a:b].sum()) for c, (a, b) in spans.items()}
