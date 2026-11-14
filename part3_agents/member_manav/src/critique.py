"""Critique: decides accept / revise / reject from a CLIP score, a yes/no VLM
judgement and the safety flag. Pure decision logic -- the CLIP score and the VLM
judgement are passed in as already-computed values, so this is fully testable without
loading anything.
"""
from __future__ import annotations

from typing import Optional

from .state import CritiqueResult

MAX_REVISIONS = 2

# at most 2 revisions: each step gets stronger (more edit strength, a new seed, and a
# rephrased subject in the prompt)
REVISION_STRATEGIES = [
    {"strength_delta": 0.15, "seed_offset": 1, "rephrase": False,
     "note": "increase the edit strength and use a new seed"},
    {"strength_delta": 0.25, "seed_offset": 2, "rephrase": True,
     "note": "increase the edit strength further, rephrase the subject, and use a new seed"},
]


def decide(clip_score: Optional[float], clip_threshold: float, vlm_verdict: Optional[str],
           safe: bool, revision_count: int) -> CritiqueResult:
    if not safe:
        return CritiqueResult(clip_score=clip_score, clip_threshold=clip_threshold,
                               vlm_verdict=vlm_verdict, safe=False, decision="reject")

    passes_clip = clip_score is not None and clip_score >= clip_threshold
    passes_vlm = vlm_verdict == "yes"
    if passes_clip and passes_vlm:
        return CritiqueResult(clip_score=clip_score, clip_threshold=clip_threshold,
                               vlm_verdict=vlm_verdict, safe=True, decision="accept")

    if revision_count >= MAX_REVISIONS:
        # out of revisions: accept the best attempt rather than looping forever
        return CritiqueResult(clip_score=clip_score, clip_threshold=clip_threshold,
                               vlm_verdict=vlm_verdict, safe=True, decision="accept")

    strategy = REVISION_STRATEGIES[min(revision_count, len(REVISION_STRATEGIES) - 1)]
    return CritiqueResult(clip_score=clip_score, clip_threshold=clip_threshold,
                           vlm_verdict=vlm_verdict, safe=True, decision="revise",
                           revision_instruction=strategy["note"])


def next_revision_strategy(revision_count: int) -> dict:
    return REVISION_STRATEGIES[min(revision_count, len(REVISION_STRATEGIES) - 1)]
