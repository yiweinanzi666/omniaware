"""Parse a decoded string before it is scored.

A blank answer, a citation-only dump (``Sources:`` / ``[E<j>]`` lines and nothing else), or a
refusal is ``answer_empty``. A real answer followed by a citation line keeps the answer text.
The only rewrite is stripping those citation lines. Nothing else is truncated or rewritten.
"""
from __future__ import annotations

import re

# A line that is only a citation dump. Inline ``[E1]`` inside a sentence is not this.
_SOURCES_LINE = re.compile(r"(?is)^\s*sources\s*:.*$")
_CITE_TOKEN_LINE = re.compile(r"(?is)^\s*(?:\[E\d+\][\s,]*)+$")
# Whole-answer refusals. Bare ``no`` / ``none`` are left alone: they are real OK-VQA answers.
_REFUSAL = re.compile(
    r"(?is)^\s*(?:"
    r"i\s+(?:do\s+not|don't|dont|cannot|can't|cant)\s+(?:know|answer)(?:\s+this(?:\s+question)?)?"
    r"|i(?:'m|\s+am)\s+(?:sorry|unable)(?:\s*,)?(?:\s+i\s+(?:do\s+not|don't|cannot|can't)\s+(?:know|answer))?"
    r"|unable\s+to\s+answer"
    r"|insufficient\s+information"
    r"|no\s+answer"
    r"|i\s+refuse\s+to\s+answer"
    r")\s*[.!]?\s*$"
)


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
    if cite_lines and not answer_lines:
        return {"answer": "", "answer_empty": True, "citation_only": True}
    answer = "\n".join(answer_lines).strip()
    if not answer:
        return {"answer": "", "answer_empty": True, "citation_only": bool(cite_lines)}
    if _REFUSAL.match(answer):
        return {"answer": "", "answer_empty": True, "citation_only": False}
    return {"answer": answer, "answer_empty": False, "citation_only": False}


def summarize_empty(flags: list[bool]) -> dict:
    """Empty-answer rate over decoded strings. Above 0.5% the summary carries a warning line."""
    n = len(flags)
    rate = (sum(bool(f) for f in flags) / n) if n else 0.0
    out = {"empty_answer_rate": rate, "n_answers": n, "n_empty": int(sum(bool(f) for f in flags))}
    if rate > 0.005:
        out["empty_answer_warning"] = f"EMPTY ANSWER RATE {rate:.6f}"
    return out
