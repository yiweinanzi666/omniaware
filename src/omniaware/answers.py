"""Parse a decoded string before it is scored.

A blank answer, a citation-only dump (``Sources:`` / ``[E<j>]`` lines and nothing else), or a refusal is
``answer_empty``. A real answer followed by a citation line keeps the answer text. The only rewrite is stripping
those citation lines. Nothing else is truncated or rewritten.

A refusal is recognised in two forms.

- A bare refusal phrase ("I don't know.", "Insufficient information.") is matched as the whole answer.
- A refusal sentence is one sentence that (1) opens with an apology or an inability clause, (2) names what is
  missing (question, information, evidence, context, image, ...) before its first sentence break, and (3) has no
  comma, semicolon or colon after the refusal clause. A sentence that goes on to give an answer after a comma is
  therefore an answer.

Examples: "I'm sorry, I cannot answer this question." and "I cannot determine the answer from the provided image."
are refusals. "I cannot see any person." (no naming word, the verb is not a refusal verb), "No", "None" and
"I cannot determine the exact number, it is about five." are answers.
"""
from __future__ import annotations

import re

# A line that is only a citation dump. Inline ``[E1]`` inside a sentence is not this.
_SOURCES_LINE = re.compile(r"(?is)^\s*sources\s*:.*$")
_CITE_TOKEN_LINE = re.compile(r"(?is)^\s*(?:\[E\d+\][\s,]*)+$")
# Whole-answer refusal phrases. Bare ``no`` / ``none`` are left alone: they are real OK-VQA answers.
_REFUSAL_PHRASE = re.compile(
    r"(?is)^\s*(?:"
    r"i\s+(?:do\s+not|don't|dont|cannot|can't|cant)\s+(?:know|answer)(?:\s+this(?:\s+question)?)?"
    r"|i(?:'m|\s+am)\s+(?:sorry|unable)(?:\s*,)?(?:\s+i\s+(?:do\s+not|don't|cannot|can't)\s+(?:know|answer))?"
    r"|unable\s+to\s+answer"
    r"|insufficient\s+information"
    r"|no\s+answer"
    r"|i\s+refuse\s+to\s+answer"
    r")\s*[.!]?\s*$"
)
# One refusal sentence (see the module docstring). The lookahead requires a naming word before the first sentence break.
_REFUSAL_SENTENCE = re.compile(
    r"(?is)^(?=[^.!?]*\b(?:question|information|evidence|context|details|provided|supplied|given|available|"
    r"image|images|text|passage|photo|picture)\b)"
    r"\s*(?:(?:i'?m|i\s+am)\s+sorry|sorry|i\s+apologi[sz]e|unfortunately)?\s*,?\s*"
    r"(?:i\s+(?:cannot|can'?t|can\s+not|could\s+not|couldn'?t|do\s+not|don'?t|am\s+unable\s+to|'?m\s+unable\s+to"
    r"|am\s+not\s+able\s+to|'?m\s+not\s+able\s+to|am\s+not\s+sure)\s+"
    r"(?:answer|determine|know|tell|say|provide|identify|confirm|find|help|assist|be\s+sure|figure\s+out)"
    r"|(?:the\s+)?(?:provided|supplied|given|available|input)\s+(?:\w+\s+){0,2}?"
    r"(?:does\s+not|do\s+not|doesn'?t|don'?t)\s+(?:contain|provide|include|mention|state|say|show|answer|tell|have)"
    r"|(?:there\s+(?:is|are)\s+)?(?:not\s+enough|insufficient)\s+(?:information|evidence|context|details)"
    r"|(?:i\s+)?(?:cannot|can'?t|unable\s+to)\s+(?:answer|determine))"
    r"[^,;:!?]*[.!]?\s*$"
)


def is_refusal(answer: str) -> bool:
    """True when the whole answer (citation lines already removed) is a refusal."""
    return bool(_REFUSAL_PHRASE.match(answer) or _REFUSAL_SENTENCE.match(answer))


def parse_answer(text: str | None) -> dict:
    """Return ``answer``, ``answer_empty``, ``citation_only`` for one decoded string."""
    raw = "" if text is None else str(text)
    if not raw.strip():
        return {"answer": "", "answer_empty": True, "citation_only": False}
    answer_lines, cite_lines = [], []
    for line in raw.splitlines():
        if not line.strip():
            continue
        if _SOURCES_LINE.match(line) or _CITE_TOKEN_LINE.match(line):
            cite_lines.append(line.strip())
        else:
            answer_lines.append(line.strip())
    if not answer_lines:
        # Only citation lines are left (a non-blank raw string always has a line in one of the two lists).
        return {"answer": "", "answer_empty": True, "citation_only": True}
    answer = "\n".join(answer_lines).strip()
    if is_refusal(answer):
        return {"answer": "", "answer_empty": True, "citation_only": False}
    return {"answer": answer, "answer_empty": False, "citation_only": False}


def question_empty_flags(record: dict) -> list[bool]:
    """``answer_empty`` of every decoded arm of one question (parsed from ``preds``, which keeps the raw model
    output) and of every available v2 action. Unavailable actions decoded nothing and are not counted."""
    flags = [parse_answer(text)["answer_empty"] for text in (record.get("preds") or {}).values()]
    flags += [bool(a["answer_empty"]) for a in (record.get("actions") or {}).values() if a.get("available", True)]
    return flags


def summarize_empty(flags: list[bool]) -> dict:
    """Empty-answer rate over decoded strings. Above 0.5% the summary carries a warning line."""
    n = len(flags)
    rate = (sum(bool(f) for f in flags) / n) if n else 0.0
    out = {"empty_answer_rate": rate, "n_answers": n, "n_empty": int(sum(bool(f) for f in flags))}
    if rate > 0.005:
        out["empty_answer_warning"] = f"EMPTY ANSWER RATE {rate:.6f}"
    return out
