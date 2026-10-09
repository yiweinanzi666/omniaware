"""Output serialization only. No aliases, semantic matching, or gold answers."""
import json

MMQA_INSTRUCTION = (
    'Return only a JSON array of answer strings. Include every item requested by the question. '
    'For a single answer, return a one-element array. Do not include explanations.'
)


def parse_mmqa(text):
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [text], False
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value, True
    return [text], False
