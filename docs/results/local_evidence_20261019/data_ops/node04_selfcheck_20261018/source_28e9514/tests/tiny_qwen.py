"""Tiny random Qwen3-VL used by baseline tests (verbatim from legacy test_qwen_adapter.py)."""

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration


def tiny_model(backend="eager"):
    torch.manual_seed(17)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=64, pad_token_id=0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000., "mrope_section": [2, 1, 1]},
        ),
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
            patch_size=2, spatial_merge_size=2, temporal_patch_size=2,
            out_hidden_size=32, num_position_embeddings=16,
            deepstack_visual_indexes=[0],
        ),
        image_token_id=61, video_token_id=59, vision_start_token_id=60,
        vision_end_token_id=62, attn_implementation="eager",
    )
    model = Qwen3VLForConditionalGeneration(config).eval()
    model.model.language_model.set_attn_implementation(backend)
    return model


def image_prompt():
    return dict(
        input_ids=torch.tensor([[2, 60, 61, 62, 6, 7]]),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        mm_token_type_ids=torch.tensor([[0, 0, 1, 0, 0, 0]]),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        # Four random flattened 2-frame, 2x2 RGB patches form one image token.
        pixel_values=torch.randn(4, 24, generator=torch.Generator().manual_seed(18)),
    )
