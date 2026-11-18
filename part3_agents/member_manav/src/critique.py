"""Critique: per-attempt decision logic. Pure -- the CLIP score and the VLM judgement
are passed in as already-computed values, so this is fully testable without loading
anything. Unsafe attempts never reach decide() at all (the orchestrator discards them
directly); this only decides between a SAFE attempt passing or needing another try. The
revision cap is the orchestrator's responsibility, not this function's -- it just says
accept/revise for one attempt, with no notion of how many attempts remain.
"""
from __future__ import annotations

from typing import Optional

from .state import CritiqueResult

MAX_REVISIONS = 2

# at most 2 revisions: each step gets stronger (more edit strength, a new seed, and a
# rephrased subject in the prompt). Used for a "revise" (didn't pass) and for a
# "discarded" (unsafe) attempt alike -- the orchestrator applies the same strategy
# either way to get a materially different next try.
REVISION_STRATEGIES = [
    {"strength_delta": 0.15, "seed_offset": 1, "rephrase": False,
     "note": "increase the edit strength and use a new seed"},
    {"strength_delta": 0.25, "seed_offset": 2, "rephrase": True,
     "note": "increase the edit strength further, rephrase the subject, and use a new seed"},
]


def decide(clip_score: Optional[float], clip_threshold: float, vlm_verdict: Optional[str],
           safe: bool = True) -> CritiqueResult:
    """Decides accept vs. revise for one SAFE attempt. Never call this for an unsafe
    attempt -- the orchestrator builds a "discarded" CritiqueResult directly for those."""
    passes_clip = clip_score is not None and clip_score >= clip_threshold
    passes_vlm = vlm_verdict == "yes"
    decision = "accept" if (passes_clip and passes_vlm) else "revise"
    return CritiqueResult(clip_score=clip_score, clip_threshold=clip_threshold,
                           vlm_verdict=vlm_verdict, safe=safe, decision=decision)


def next_revision_strategy(revision_count: int) -> dict:
    return REVISION_STRATEGIES[min(revision_count, len(REVISION_STRATEGIES) - 1)]
