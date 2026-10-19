"""Local Qwen backend for the pinned official M2RAG summarizer.

The caller owns model/processor loading and generation/image budgets. This
backend preserves the official flat-content-to-one-user-message convention.
"""

import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch
from transformers import GenerationConfig


class LocalQwenVLM:
    name = "vlm-local-qwen3-vl"

    def __init__(self, model, processor, *, max_new_tokens, image_paths=None, image_size=None):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_new_tokens = max_new_tokens
        self.image_paths = image_paths or {}
        self.image_size = image_size
        self.calls = []

    def num_tokens_from_text(self, text):
        return len(self.tokenizer(text).input_ids)

    def trim_text(self, text, max_tokens, side="right"):
        tokens = self.tokenizer(text).input_ids
        if len(tokens) > max_tokens:
            if side == "left":
                tokens = tokens[-max_tokens:] if max_tokens else []
            elif side == "right":
                tokens = tokens[:max_tokens]
            else:
                raise ValueError(f"Invalid trimming side: {side}")
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _conversation(self, messages):
        content = []
        for message in messages:
            kind = message["type"]
            if kind == "text":
                content.append({"type": "text", "text": message["text"]})
            elif kind == "image_caption":
                content.append({"type": "text", "text": message["image_caption"]["caption"]})
            elif kind == "image_url":
                url = message["image_url"]["url"]
                path = str(self.image_paths.get(url, url))
                if path.startswith("file://"):
                    path = unquote(urlparse(path).path)
                elif "://" in path or path.startswith("data:"):
                    raise ValueError(f"Image URL needs an explicit local image_paths mapping: {url}")
                content.append({"type": "image", "path": str(Path(path).resolve(strict=True))})
            else:
                raise ValueError(f"Unsupported official content type: {kind}")
        return [{"role": "user", "content": content}]

    def _synchronize(self):
        if self.model.device.type == "mps":
            torch.mps.synchronize()
        elif self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)

    @torch.inference_mode()
    def chat(self, messages):
        """Run the official call shape with the constructor's fixed greedy budget."""
        conversation = self._conversation(messages)
        processor_kwargs = {"add_special_tokens": False}
        if self.image_size is not None:
            processor_kwargs["images_kwargs"] = {"size": self.image_size}
        started = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, processor_kwargs=processor_kwargs,
        ).to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        decode = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens,
            eos_token_id=eos, pad_token_id=self.tokenizer.pad_token_id, use_cache=True,
        )
        grid = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else []
        call = {
            "status": "started", "messages": conversation,
            "input_token_ids": inputs["input_ids"][0].tolist(),
            "generation": {"max_new_tokens": decode.max_new_tokens, "do_sample": decode.do_sample,
                           "num_beams": decode.num_beams, "eos_token_id": eos,
                           "pad_token_id": decode.pad_token_id, "use_cache": decode.use_cache},
            "image_size": dict(self.image_size or self.processor.image_processor.size),
            "image_grid_thw": grid,
            "vision_tokens": sum(t * h * w for t, h, w in grid) // self.processor.image_processor.merge_size ** 2,
        }
        self.calls.append(call)
        self._synchronize()
        generation_started = time.perf_counter()
        try:
            generated = self.model.generate(**inputs, generation_config=decode)
            self._synchronize()
        except Exception as error:
            call.update(status="failed", error_type=type(error).__name__, error=str(error),
                        generation_attempt_seconds=time.perf_counter() - generation_started,
                        elapsed_seconds=time.perf_counter() - started)
            raise
        generation_seconds = time.perf_counter() - generation_started
        tokens = generated[0, prompt_tokens:].tolist()
        response = self.tokenizer.decode(tokens, skip_special_tokens=True)
        eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
        ended_with_eos = bool(tokens and tokens[-1] in eos_ids)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": len(tokens),
                 "total_tokens": prompt_tokens + len(tokens)}
        call.update({
            "status": "completed", "response": response, "usage": usage,
            "ended_with_eos": ended_with_eos,
            "hit_token_cap": len(tokens) >= self.max_new_tokens and not ended_with_eos,
            "output_token_ids": tokens, "generation_seconds": generation_seconds,
            "elapsed_seconds": time.perf_counter() - started,
        })
        return {"response": response, "usage": usage}
