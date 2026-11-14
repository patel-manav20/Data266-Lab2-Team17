"""Orchestration: plain Python functions passing one AgentState from agent to agent.
No framework. Every agent call is timed and appended to the trace; every failure (bad
image, empty request, a generation error, an out-of-memory error) turns into a clear
message on the state, never a crash -- and a JSON trace is saved in every case,
including failures.

All the real model calls are injected as callables (`vlm_call`, `generate_fn`,
`clip_score_fn`, `vlm_judge_fn`, `classifier_fn`) so this module runs exactly the same
way in the Mac stub tests and in the real Colab notebook -- only what is passed in for
those callables differs.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from . import critique as critique_mod
from . import prompt_engineering as pe
from . import vision as vision_mod
from .planner import plan
from .state import AgentState, GenerationRecord
from .trace import save_trace

SEED = 266
START_STRENGTH = 0.6
MAX_STRENGTH = 0.95


def run(
    request: str,
    image_path: Optional[str],
    *,
    vlm_call: Callable[[str, str], str],
    generate_fn: Callable[..., GenerationRecord],
    classifier_fn: Optional[Callable[[str], str]] = None,
    clip_score_fn: Optional[Callable[[str, str], float]] = None,
    vlm_judge_fn: Optional[Callable[[str, str], str]] = None,
    clip_threshold: float = 0.25,
    critique_enabled: bool = True,
    trace_dir: str = "outputs/traces",
    request_id: str = "req",
) -> AgentState:
    state = AgentState(request=request)
    state.image.path = image_path

    try:
        _run_pipeline(state, vlm_call=vlm_call, generate_fn=generate_fn, classifier_fn=classifier_fn,
                       clip_score_fn=clip_score_fn, vlm_judge_fn=vlm_judge_fn,
                       clip_threshold=clip_threshold, critique_enabled=critique_enabled)
    except Exception as e:  # never crash -- always leave a clear message and a trace
        state.add_error(f"Something went wrong while handling this request ({e.__class__.__name__}: {e}).")
    finally:
        save_trace(state, trace_dir, request_id)
    return state


def _run_pipeline(state: AgentState, *, vlm_call, generate_fn, classifier_fn,
                   clip_score_fn, vlm_judge_fn, clip_threshold, critique_enabled) -> None:
    t0 = time.perf_counter()
    plan(state, classifier_fn=classifier_fn)
    state.log("Planner", time.perf_counter() - t0, f"route={state.route}")

    if state.route == "unsupported":
        return

    if state.image.path:
        t0 = time.perf_counter()
        user_question = state.request if state.route == "caption_question" else None
        vision_mod.run_vision(state, vlm_call, user_question=user_question)
        cap = state.vision.caption[:60] if state.vision else ""
        state.log("Vision Understanding", time.perf_counter() - t0, f"caption={cap!r}")

    if state.route == "caption_question":
        return

    t0 = time.perf_counter()
    prompt, negative = pe.build_generation_prompt(state.request, state.vision, state.route,
                                                    instruction_flagged=state.safety.instruction_in_image)
    state.prompt, state.negative_prompt = prompt, negative
    state.log("Prompt Engineering", time.perf_counter() - t0, prompt[:80])

    seed = SEED
    strength = START_STRENGTH
    gen: Optional[GenerationRecord] = None
    while True:
        t0 = time.perf_counter()
        try:
            gen = generate_fn(state=state, prompt=prompt, negative_prompt=negative,
                               seed=seed, strength=strength)
        except Exception as e:
            state.add_error(f"Image generation failed ({e.__class__.__name__}: {e}).")
            state.log("Image Generation", time.perf_counter() - t0, "failed")
            return
        state.generation = gen
        state.generation_history.append(gen)
        state.log("Image Generation", time.perf_counter() - t0, f"{gen.model} seed={seed}")

        if not critique_enabled:
            state.critique = critique_mod.CritiqueResult(safe=not gen.nsfw_flagged,
                                                           decision="accept" if not gen.nsfw_flagged else "reject")
            break

        t0 = time.perf_counter()
        clip_score = clip_score_fn(gen.output_path, prompt) if clip_score_fn else None
        verdict_text = vlm_judge_fn(gen.output_path, prompt) if vlm_judge_fn else None
        result = critique_mod.decide(clip_score, clip_threshold, verdict_text,
                                      safe=not gen.nsfw_flagged, revision_count=state.revision_count)
        state.critique = result
        state.log("Critique", time.perf_counter() - t0,
                   f"decision={result.decision} clip={clip_score}")

        if result.decision != "revise":
            break

        strategy = critique_mod.next_revision_strategy(state.revision_count)
        state.revision_count += 1
        strength = min(MAX_STRENGTH, strength + strategy["strength_delta"])
        seed += strategy["seed_offset"]
        if strategy["rephrase"]:
            prompt = pe.apply_revision(prompt, result.revision_instruction)

    state.final_image_path = gen.output_path if gen else None
    state.final_message = _final_message(state)


def _final_message(state: AgentState) -> str:
    if state.critique is None:
        return "Done."
    if state.critique.decision == "reject":
        return "The result did not pass the safety check, so no image is shown."
    if state.revision_count >= critique_mod.MAX_REVISIONS and state.critique.decision == "accept":
        return f"Here is the result after {state.revision_count} revision(s) (reached the revision limit)."
    if state.revision_count > 0:
        return f"Here is the result after {state.revision_count} revision(s)."
    return "Here is the result."
