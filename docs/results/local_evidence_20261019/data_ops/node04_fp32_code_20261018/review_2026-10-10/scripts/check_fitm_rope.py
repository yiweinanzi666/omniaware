import sys
sys.argv = ["x"]
exec(open("check_fitm_span.py").read().split("model = model18()")[0])
import torch
model = model18()
with torch.inference_mode():
    pos_orig = model.model.get_rope_index(inputs["input_ids"], image_grid_thw=inputs["image_grid_thw"],
                                          attention_mask=inputs["attention_mask"], mm_token_type_ids=inputs["mm_token_type_ids"])[0]
    pos_ref = model.model.get_rope_index(ref["input_ids"], image_grid_thw=ref["image_grid_thw"],
                                         attention_mask=ref["attention_mask"], mm_token_type_ids=ref["mm_token_type_ids"])[0]
print("orig rope positions (3 x T):\n", pos_orig[:, 0].tolist())
print("null-ref rope positions:\n", pos_ref[:, 0].tolist())
print("identical:", torch.equal(pos_orig, pos_ref))
