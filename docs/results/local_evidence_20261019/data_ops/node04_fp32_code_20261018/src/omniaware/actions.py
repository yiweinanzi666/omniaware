"""The v2 action set. Defined once; every other module imports it from here."""
from __future__ import annotations

DIRECT = "direct"
EVIDENCE_ACTIONS = ("static_text", "static_image", "static_mixed", "rerank_static")
V2_ACTIONS = (DIRECT,) + EVIDENCE_ACTIONS
