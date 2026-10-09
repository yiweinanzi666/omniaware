"""Tiny random Qwen3-VL wrapped like QwenEAB (CPU, no weights) and a text-only encoded prompt."""
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from omniaware.qwen_eab import QwenEAB, Segment


class CharTokenizer:
    """Deterministic text-only stand-in for the Qwen tokenizer."""

    eos_token_id = 157

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 140 + 1 for ch in text]

    def decode(self, tokens, skip_special_tokens=True):
        tokens = [int(t) for t in tokens]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in (155, 156, 157)]
        return " ".join(map(str, tokens))

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 155, "<|endoftext|>": 156}[token]


def make_qwen():
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=160, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2, deepstack_visual_indexes=[0]),
        image_token_id=150, video_token_id=151, vision_start_token_id=152, vision_end_token_id=153)
    config.text_config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(config).eval()
    # Same attribute setup as QwenEAB.__init__, without loading a checkpoint.
    q = QwenEAB.__new__(QwenEAB)
    q.device, q.model, q.tok, q.processor = "cpu", model, CharTokenizer(), None
    q.layers = []
    for m in model.modules():
        if m.__class__.__name__ == "Qwen3VLTextAttention":
            m._erc_layer = m.layer_idx
            q.layers.append(m.layer_idx)
    q.n_layers = len(q.layers)
    q.eos = {155, 156, 157}
    q.vocab_size = 160
    q.image_token_id = 150
    return q


def encode(q):
    segments = [Segment(text="<user>"), Segment(text="Document 1: alpha beta gamma", chunk="c1"),
                Segment(text="Document 2: delta epsilon", chunk="c2"), Segment(text="Question: which? Answer:")]
    return q.encode(segments, split_marker=3)


def make_internvl():
    """Tiny random native InternVL (Qwen3 language model) wrapped like QwenEAB(kind='internvl')."""
    from transformers import InternVLConfig, InternVLForConditionalGeneration
    from omniaware.qwen_eab import tag_text_layers
    config = InternVLConfig(
        text_config=dict(model_type="qwen3", vocab_size=160, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         max_position_embeddings=512),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                           image_size=[56, 56], patch_size=[14, 14]),
        image_token_id=150)
    config.text_config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = InternVLForConditionalGeneration(config).eval()
    q = QwenEAB.__new__(QwenEAB)
    q.kind, q.device, q.model, q.tok, q.processor = "internvl", "cpu", model, CharTokenizer(), None
    q.layers = tag_text_layers(model)
    q.n_layers = len(q.layers)
    q.eos = {155, 156, 157}
    q.vocab_size = 160
    q.image_token_id = 150
    return q
