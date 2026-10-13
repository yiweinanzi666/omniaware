"""Frozen answer instruction for each dataset family.

Every arm of one run uses the same string. The wording is the frozen default; a caller may pass
``overrides`` (family -> string) but the test suite does not choose it. ``EXCEPTIONS`` lists arms
that cannot share the string, with a reason. It is empty unless a baseline's own protocol forbids
the shared instruction.
"""
from __future__ import annotations

import hashlib

INSTRUCTIONS = {
    "webqa": "Answer in one complete sentence that restates the key object in the question.",
    "okvqa": "Answer in no more than 3 words. Give the answer only.",
    "hybridqa": "Please respond with only the exact answer.",
    "videorag": "Please respond in a complete sentence.",
}

# arm name -> why that arm cannot use INSTRUCTIONS. Empty by default.
EXCEPTIONS: dict[str, str] = {}


def family_of(dataset: str) -> str:
    if dataset.startswith("okvqa"):
        return "okvqa"
    if dataset.startswith("webqa"):
        return "webqa"
    if dataset.startswith("hybrid"):
        return "hybridqa"
    if dataset.startswith("video"):
        return "videorag"
    if dataset in INSTRUCTIONS:
        return dataset
    raise KeyError(dataset)


def instruction_text(dataset: str, overrides: dict | None = None) -> str:
    family = family_of(dataset)
    if overrides and family in overrides:
        return overrides[family]
    return INSTRUCTIONS[family]


def instruction_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def instruction_for_style(dataset: str, answer_style: str, *, cite_instruction: str,
                          short_instruction: str, overrides: dict | None = None) -> dict:
    """The instruction this run actually decodes with, plus its sha256. Same for every arm."""
    if answer_style == "cite":
        text = cite_instruction
    elif answer_style == "short":
        text = short_instruction
    else:
        text = instruction_text(dataset, overrides)
    return {"instruction": text, "instruction_sha256": instruction_sha256(text),
            "instruction_exceptions": dict(EXCEPTIONS)}
