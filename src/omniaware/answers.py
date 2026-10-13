"""Parse a decoded string before it is scored.

A blank answer, a citation-only dump (``Sources:`` / ``[E<j>]`` lines and nothing else), or a refusal is
``answer_empty``. A real answer followed by a citation line keeps the answer text. The only rewrite is stripping
those citation lines. Nothing else is truncated or rewritten. ``refusal`` marks the third kind, so that the run
summaries can keep refusals apart from blank and citation-only output (``summarize_empty``).

A refusal is recognised in two forms.

- A bare refusal phrase ("I don't know.", "Insufficient information.") is matched as the whole answer.
- A refusal sentence is one sentence that (1) opens with an apology or an inability clause, (2) names what is
  missing (question, information, evidence, context, image, ...) before its first sentence break, and (3) has no
  comma, semicolon or colon after the refusal clause. A sentence that goes on to give an answer after a comma is
  therefore an answer.

Examples: "I'm sorry, I cannot answer this question." and "I cannot determine the answer from the provided image."
are refusals. "I cannot see any person." (no naming word, the verb is not a refusal verb), "No", "None" and
"I cannot determine the exact number, it is about five." are answers.

An answer-first output (``answer_first=True``, the WebQA format of prompts.ANSWER_FIRST) has the short answer on its
first line and one complete sentence after it. ``short`` is the first line (an "Answer:" label is dropped) and
``answer``, the scored text, is the sentence: the WebQA references are sentences. When the sentence line is missing,
``answer`` falls back to the short line and ``answer_first_ok`` is False. For a plain output ``short`` equals
``answer`` and ``answer_first_ok`` is None.
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


_SHORT_LABEL = re.compile(r"(?i)^\s*(?:short\s+)?answer\s*:\s*")
_SENTENCE_LABEL = re.compile(r"(?i)^\s*(?:(?:full|complete)\s+)?sentence\s*:\s*")


def _empty(citation_only: bool = False, refusal: bool = False) -> dict:
    return {"answer": "", "short": "", "answer_empty": True, "citation_only": citation_only, "refusal": refusal,
            "answer_first_ok": None}


def parse_answer(text: str | None, answer_first: bool = False) -> dict:
    """Return ``answer`` (the scored text), ``short``, ``answer_empty``, ``citation_only``, ``refusal`` and
    ``answer_first_ok`` for one decoded string."""
    raw = "" if text is None else str(text)
    if not raw.strip():
        return _empty()
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
        return _empty(citation_only=True)
    if answer_first:
        short = _SHORT_LABEL.sub("", answer_lines[0]).strip()
        sentence = " ".join(_SENTENCE_LABEL.sub("", line).strip() for line in answer_lines[1:]).strip()
        answer, form_ok = (sentence or short), bool(sentence)
        if not short:                                  # "Answer:" alone on the first line
            short = answer
        if is_refusal(answer) or is_refusal(short):
            return _empty(refusal=True)
        return {"answer": answer, "short": short, "answer_empty": False, "citation_only": False, "refusal": False,
                "answer_first_ok": form_ok}
    answer = "\n".join(answer_lines).strip()
    if is_refusal(answer):
        return _empty(refusal=True)
    return {"answer": answer, "short": answer, "answer_empty": False, "citation_only": False, "refusal": False,
            "answer_first_ok": None}


def empty_kind(parsed: dict) -> str | None:
    """Why a parsed answer or a v2 action record is empty: "blank", "citation_only" or "refusal"; None for an answer.

    A v2 action record written before the ``refusal`` field existed has no way to tell a refusal from a blank
    decode, so it counts as blank.
    """
    if not parsed.get("answer_empty"):
        return None
    if parsed.get("citation_only"):
        return "citation_only"
    if parsed.get("refusal"):
        return "refusal"
    return "blank"


def question_empty_kinds(record: dict) -> list[tuple[str, str | None]]:
    """``(arm, empty_kind)`` for every decoded string of one question: each arm of ``preds`` (parsed from the raw
    output) and each available v2 action, named ``action:<name>``. Unavailable actions decoded nothing and are not
    counted."""
    answer_first = record.get("answer_format") == "answer_first"
    kinds = [(arm, empty_kind(parse_answer(text, answer_first))) for arm, text in (record.get("preds") or {}).items()]
    kinds += [(f"action:{name}", empty_kind(action)) for name, action in (record.get("actions") or {}).items()
              if action.get("available", True)]
    return kinds


def summarize_empty(kinds: list[tuple[str, str | None]]) -> dict:
    """Empty-answer statistics over decoded strings, from the ``(arm, kind)`` pairs of ``question_empty_kinds``.

    ``empty_answer_rate`` counts blank and citation-only strings. Those mean the prompt or the decode is broken, and
    this is the rate ``--max-empty-rate`` checks; above 0.5% the summary carries a warning line. A refusal is the
    model declining to answer. It is scored like any empty answer, but it is reported apart (``refusal_rate``, and
    per arm in ``empty_by_arm``) so that it neither stops a run nor hides: a high refusal rate on one arm, for
    example image evidence the model says it cannot see, is a finding to inspect in S1.
    """
    n = len(kinds)
    by_arm: dict[str, dict[str, int]] = {}
    totals = {"blank": 0, "citation_only": 0, "refusal": 0}
    for arm, kind in kinds:
        entry = by_arm.setdefault(arm, {"n": 0, "blank": 0, "citation_only": 0, "refusal": 0})
        entry["n"] += 1
        if kind is not None:
            entry[kind] += 1
            totals[kind] += 1
    n_empty = totals["blank"] + totals["citation_only"]
    rate = (n_empty / n) if n else 0.0
    out = {"empty_answer_rate": rate, "refusal_rate": (totals["refusal"] / n) if n else 0.0, "n_answers": n,
           "n_empty": n_empty, "n_blank": totals["blank"], "n_citation_only": totals["citation_only"],
           "n_refusal": totals["refusal"], "empty_by_arm": by_arm}
    if rate > 0.005:
        out["empty_answer_warning"] = f"EMPTY ANSWER RATE {rate:.6f}"
    return out
