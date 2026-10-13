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
import time
from dataclasses import dataclass, field

import torch
from PIL import Image
from transformers import (AttentionInterface, AutoConfig, AutoProcessor, InternVLForConditionalGeneration,
                          Qwen3VLForConditionalGeneration)
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
        self.blocked: torch.Tensor | None = None  # [P] bool per key column; True keys are hidden from every row
        self.capture_layer: int | None = None
        self.captured: torch.Tensor | None = None


STATE = _State()
_ORIG_SDPA = sdpa_attention_forward


def _additive_mask(attention_mask, query, q_pos0: int, q_len: int, kv_len: int) -> torch.Tensor:
    """Additive float mask for the rows of this forward pass: the causal mask when the model passes none,
    the boolean mask converted to 0 / finfo.min, or the additive float mask as given."""
    dtype = query.dtype
    neg = torch.finfo(dtype).min
    if attention_mask is None:
        rows = torch.arange(q_pos0, kv_len, device=query.device)[:, None]
        cols = torch.arange(kv_len, device=query.device)[None, :]
        mask = torch.zeros(q_len, kv_len, device=query.device, dtype=dtype)
        return mask.masked_fill(cols > rows, neg)[None, None]
    if attention_mask.dtype == torch.bool:
        mask = torch.zeros(attention_mask.shape, device=query.device, dtype=dtype)
        return mask.masked_fill(~attention_mask, neg)
    return attention_mask.to(dtype)


def _eab_sdpa(module, query, key, value, attention_mask, **kwargs):
    layer = getattr(module, "_erc_layer", None)
    if layer is None:
        return _ORIG_SDPA(module, query, key, value, attention_mask, **kwargs)
    st = STATE
    q_len, kv_len = query.shape[2], key.shape[2]
    q_pos0 = kv_len - q_len
    biased = st.bias is not None and (st.layers is None or layer in st.layers)
    first_row = max(st.q_start - q_pos0, 0)
    biased = biased and first_row < q_len
    capturing = st.capture_layer is not None and layer == st.capture_layer
    if not (biased or capturing or st.blocked is not None):
        return _ORIG_SDPA(module, query, key, value, attention_mask, **kwargs)

    # One additive mask serves both the captured row and the real attention call, so the captured
    # attention is the softmax the model actually uses (attention mask, EAB column bias and blocked keys included).
    mask = _additive_mask(attention_mask, query, q_pos0, q_len, kv_len)
    if biased or st.blocked is not None:
        mask = mask.expand(mask.shape[0], mask.shape[1], q_len, kv_len).clone()
    if biased:
        col = torch.zeros(kv_len, device=query.device, dtype=query.dtype)
        n = min(kv_len, st.bias.numel())
        col[:n] = st.bias[:n].to(device=query.device, dtype=query.dtype)
        mask[..., first_row:, :] += col
    if st.blocked is not None:
        # Blocked keys are -inf on every row (prefill, suffix and decoding). Position 0 is never blocked
        # (_blocked_tensor), so every row keeps an unblocked key and no softmax row is all -inf.
        hidden = torch.zeros(kv_len, dtype=torch.bool, device=query.device)
        n = min(kv_len, st.blocked.numel())
        hidden[:n] = st.blocked[:n].to(query.device)
        mask[..., hidden] = float("-inf")
    if biased or st.blocked is not None:
        attention_mask = mask
    if capturing:
        # Head-averaged attention of the last query row, i.e. the t_end row of a prefill (the row that
        # predicts the first answer token); the suffix forward always ends on that row.
        k = key if key.shape[1] == query.shape[1] else repeat_kv(key, query.shape[1] // key.shape[1])
        scale = kwargs.get("scaling") or module.scaling
        logits = (query[:, :, -1:, :].float() @ k.float().transpose(-1, -2)) * scale + mask[..., -1:, :].float()
        st.captured = torch.softmax(logits, dim=-1).mean(dim=1)[0, 0].detach().float().cpu()
    return _ORIG_SDPA(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register("sdpa", _eab_sdpa)


def require_sdpa(model) -> None:
    """EAB works only through the globally registered sdpa function. Any other attention backend would
    silently turn every EAB variant into the static arm, so loading fails instead."""
    impl = getattr(model.config.text_config, "_attn_implementation", None)
    if impl != "sdpa":
        raise RuntimeError(f"EAB requires sdpa attention in the text decoder, got {impl!r}")


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
    blocked: frozenset = frozenset()        # chunk ids no row may attend to (interventions.mask_span); no EAB on them


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
    seconds: float = 0.0                                          # decode time of this arm
    # Head-averaged last-layer attention of the t_end row (the prefill row that predicts the first answer
    # token), computed under this arm's own bias; equal to the static arm's for decode-only variants.
    attention: torch.Tensor | None = field(default=None, repr=False)
    # Logprob of each greedy token under the distribution that produced it. Empty when the decode
    # path did not record them (a one-token result can still be scored from first_logits).
    token_logprobs: list[float] | None = None


def tag_text_layers(model) -> list[int]:
    """Mark the decoder self-attention modules of the language model (vision attention stays untouched)."""
    layers = []
    for i, layer in enumerate(model.model.language_model.layers):
        layer.self_attn._erc_layer = i
        layers.append(i)
    return layers


class QwenEAB:
    """EAB runner for decoder-token VLMs: Qwen3-VL (kind 'qwen3_vl') and native InternVL3.5 (kind 'internvl').

    Both use a ChatML prompt and put image tokens into the decoder sequence, so evidence spans and the
    SDPA bias hook are shared; only image tokenisation and the prefill inputs differ. InternVL images
    use ``max_patches`` 448x448 tiles of 256 tokens (1 tile = Qwen3-VL's default 256-token budget).
    """

    def __init__(self, model_path, device: str | None = None, dtype=torch.bfloat16,
                 max_pixels: int = 256 * 32 * 32, min_pixels: int = 64 * 32 * 32, max_patches: int = 1):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.kind = "internvl" if AutoConfig.from_pretrained(model_path).model_type == "internvl" else "qwen3_vl"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tok = self.processor.tokenizer
        cls = InternVLForConditionalGeneration if self.kind == "internvl" else Qwen3VLForConditionalGeneration
        self.model = cls.from_pretrained(model_path, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        require_sdpa(self.model)
        self.layers = tag_text_layers(self.model)
        self.n_layers = len(self.layers)
        cfg = self.model.config
        self.image_token_id = cfg.image_token_id
        if self.kind == "internvl":
            self.vision_start_id = self.processor.start_image_token_id
            self.vision_end_id = self.processor.end_image_token_id
            self.image_seq_length = self.processor.image_seq_length
            self.max_patches = max_patches
        else:
            self.vision_start_id = cfg.vision_start_token_id
            self.vision_end_id = cfg.vision_end_token_id
            self.merge = self.processor.image_processor.merge_size
            self.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        self.eos = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id,
                    self.tok.convert_tokens_to_ids("<|endoftext|>")}
        self.vocab_size = len(self.tok)     # |V| of the next-token entropy; padded output rows are excluded

    def _internvl_images(self, images):
        out = self.processor.image_processor(images=images, crop_to_patches=self.max_patches > 1,
                                             max_patches=self.max_patches, return_tensors="pt")
        counts = [int(n) * self.image_seq_length for n in out["num_patches"]]
        return out["pixel_values"], counts

    # ------------------------------------------------------------ encoding --
    def encode(self, segments: list[Segment], split_marker: int) -> Encoded:
        """segments[split_marker:] form the question suffix (must be text only)."""
        image_segs = [s for s in segments if s.image is not None]
        images = [Image.open(s.image).convert("RGB") for s in image_segs]
        pixel_values = grid = None
        token_counts = None
        if images and getattr(self, "kind", "qwen3_vl") == "internvl":
            pixel_values, token_counts = self._internvl_images(images)
        elif images and any(s.max_pixels for s in image_segs):
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
                n = token_counts[k] if token_counts else int(grid[k].prod().item()) // (self.merge ** 2)
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
            if cid in enc.spans and v != 0.0 and cid not in enc.blocked:
                a, b = enc.spans[cid]
                col[a:b] = float(v)
        return col.to(self.device)

    def _blocked_tensor(self, enc: Encoded) -> torch.Tensor | None:
        if not enc.blocked:
            return None
        hidden = torch.zeros(enc.length, dtype=torch.bool)
        for cid in enc.blocked:
            a, b = enc.spans[cid]
            if a == 0:   # a row whose only visible key would be blocked would be all -inf (NaN)
                raise ValueError(f"blocked chunk {cid} starts at token 0; keep a prefix before the evidence")
            hidden[a:b] = True
        return hidden.to(self.device)

    def _activate(self, enc: Encoded, spec: BiasSpec | None):
        STATE.bias = None if spec is None else self._bias_tensor(enc, spec)
        STATE.q_start = 0 if spec is None else spec.q_start
        STATE.layers = None if spec is None else spec.layers

    @torch.no_grad()
    def _prefix(self, enc: Encoded):
        STATE.reset()
        STATE.blocked = self._blocked_tensor(enc)   # applies to the prefill, the suffix and the decoding of run()
        kw = dict(input_ids=enc.input_ids[:, :enc.split].to(self.device), use_cache=True, logits_to_keep=1)
        if getattr(self, "kind", "qwen3_vl") == "internvl":   # 1-D RoPE, no grid or token types
            if enc.pixel_values is not None:
                kw.update(pixel_values=enc.pixel_values.to(self.device, self.model.dtype))
            return self.model(**kw).past_key_values
        self.model.model.rope_deltas = None
        kw.update(mm_token_type_ids=enc.mm_token_type_ids[:, :enc.split].to(self.device))
        if enc.pixel_values is not None:
            kw.update(pixel_values=enc.pixel_values.to(self.device, self.model.dtype),
                      image_grid_thw=enc.image_grid_thw.to(self.device))
        out = self.model(**kw)
        if self.model.model.rope_deltas is None:
            self.model.model.rope_deltas = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        return out.past_key_values

    @torch.no_grad()
    def _step(self, ids: torch.Tensor, cache, spec_enc=None, capture=False) -> torch.Tensor:
        STATE.capture_layer = self.layers[getattr(self, "capture_at", -1)] if capture else None
        STATE.captured = None
        out = self.model(input_ids=ids.to(self.device), past_key_values=cache, use_cache=True, logits_to_keep=1)
        STATE.capture_layer = None
        return out.logits[0, -1].float()

    def _sync(self):
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        elif str(self.device) == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def _decode(self, cache, first_logits: torch.Tensor, max_new_tokens: int) -> tuple[list[int], list[float]]:
        """Greedy decoding from the first answer token's logits, with each chosen token's logprob."""
        tokens, logprobs = [], []
        logits = first_logits
        for _ in range(max_new_tokens):
            logp = torch.log_softmax(logits.float(), dim=-1)
            nxt = int(torch.argmax(logits).item())
            if nxt in self.eos:
                break
            tokens.append(nxt)
            logprobs.append(float(logp[nxt]))
            if len(tokens) == max_new_tokens:
                break
            logits = self._step(torch.tensor([[nxt]]), cache)
        return tokens, logprobs

    def _timed_decode(self, cache, first_logits, max_new_tokens):
        self._sync()
        t0 = time.perf_counter()
        toks, logprobs = self._decode(cache, first_logits, max_new_tokens)
        self._sync()
        return toks, time.perf_counter() - t0, logprobs

    @torch.no_grad()
    def run(self, enc: Encoded, make_variants, max_new_tokens: int, decode_static=True):
        """Run the unbiased prefill (needed for U), then every variant returned by
        ``make_variants(grounding_attention, unbiased_first_logits)``.

        ``decode_static`` (bool or callable(grounding, logits) -> bool) controls whether the
        unbiased answer itself is decoded and returned as "static".
        Every Result.attention is captured on the t_end row (the last row of the suffix forward), under
        that arm's own bias. Returns (results, grounding attention, unbiased first-token logits).
        """
        self._sync()
        t0 = time.perf_counter()
        cache = self._prefix(enc)
        suffix = enc.input_ids[:, enc.split:]
        self._activate(enc, None)
        base_logits = self._step(suffix, cache, capture=True)
        grounding = STATE.captured
        self._sync()
        self.last_prefill_seconds = time.perf_counter() - t0
        results = {}
        if decode_static(grounding, base_logits) if callable(decode_static) else decode_static:
            toks, sec, logprobs = self._timed_decode(cache, base_logits, max_new_tokens)
            results["static"] = Result(self.tok.decode(toks, skip_special_tokens=True).strip(), base_logits.cpu(), toks,
                                       sec, grounding, logprobs)
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
            toks, sec, logprobs = self._timed_decode(cache, base_logits, max_new_tokens)
            self._activate(enc, None)
            # Decode-only bias starts after the t_end row, so that row (and its attention) is unbiased.
            results[name] = Result(self.tok.decode(toks, skip_special_tokens=True).strip(), base_logits.cpu(), toks,
                                   sec, grounding, logprobs)
            if spec is None:
                unbiased = results[name]
            _crop(cache, enc.length)
        for name, spec in early.items():
            self._sync()
            t0 = time.perf_counter()
            _crop(cache, enc.split)
            self._activate(enc, spec)
            logits = self._step(suffix, cache, capture=True)
            attention = STATE.captured
            toks, sec, logprobs = self._timed_decode(cache, logits, max_new_tokens)
            self._activate(enc, None)
            results[name] = Result(self.tok.decode(toks, skip_special_tokens=True).strip(), logits.cpu(), toks,
                                   time.perf_counter() - t0, attention, logprobs)
        STATE.reset()
        return results, grounding, base_logits.cpu()


# ------------------------------------------------------------- uncertainty --
def predictive_entropy(logits: torch.Tensor, vocab_size: int) -> float:
    """Next-token entropy over the first ``vocab_size`` logits (the tokenizer's vocabulary), normalised by
    log|V|. Padding rows of the output layer are excluded from the distribution."""
    if not 1 < vocab_size <= logits.numel():
        raise ValueError(f"vocab_size {vocab_size} must lie in (1, {logits.numel()}]")
    logp = torch.log_softmax(logits[:vocab_size].float(), dim=-1)
    h = -(logp.exp() * logp).sum().item()
    return h / math.log(vocab_size)


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
