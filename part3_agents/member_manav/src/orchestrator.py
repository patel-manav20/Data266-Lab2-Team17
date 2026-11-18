"""Orchestration: plain Python functions passing one AgentState from agent to agent.
No framework. Every agent call is timed and appended to the trace; every failure (bad
image, empty request, a generation error, an out-of-memory error) turns into a clear
message on the state, never a crash -- and a JSON trace is saved in every case,
including failures.

Split into two phases so a fair "with vs. without critique" comparison is possible:
`prepare()` runs the Planner, Vision Understanding and Prompt Engineering exactly once;
`generate_and_critique()` continues from an already-prepared state through the
generation/critique loop. `run()` composes both for the normal single-call case;
`run_both()` prepares ONCE and runs generate_and_critique() twice (critique on and off)
on separate deep copies of that same prepared state, so both modes see the identical
route, vision output and prompt, and attempt 1 of both starts from the same seed (266).

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
GENERATION_ROUTES = {"style_change", "edit", "domain_style"}


def prepare(state: AgentState, *, vlm_call: Callable[[str, str], str],
            classifier_fn: Optional[Callable[[str], str]] = None) -> AgentState:
    """Planner, then Vision Understanding (if there's an image), then Prompt
    Engineering (if the route needs a generation). Leaves state.prompt/negative_prompt
    set for generation routes; for caption_question/unsupported this is the whole
    pipeline and state.final_message is already set."""
    t0 = time.perf_counter()
    plan(state, classifier_fn=classifier_fn)
    state.log("Planner", time.perf_counter() - t0, f"route={state.route}")

    if state.route == "unsupported":
        return state

    if state.image.path:
        t0 = time.perf_counter()
        user_question = state.request if state.route == "caption_question" else None
        vision_mod.run_vision(state, vlm_call, user_question=user_question)
        cap = state.vision.caption[:60] if state.vision else ""
        state.log("Vision Understanding", time.perf_counter() - t0, f"caption={cap!r}")

    if state.route == "caption_question":
        return state

    t0 = time.perf_counter()
    prompt, negative = pe.build_generation_prompt(state.request, state.vision, state.route,
                                                    instruction_flagged=state.safety.instruction_in_image)
    state.prompt, state.negative_prompt = prompt, negative
    state.log("Prompt Engineering", time.perf_counter() - t0, prompt[:80])
    return state


def generate_and_critique(
    state: AgentState,
    *,
    generate_fn: Callable[..., GenerationRecord],
    clip_score_fn: Optional[Callable[[str, str], float]] = None,
    vlm_judge_fn: Optional[Callable[[str, str], str]] = None,
    clip_threshold: float = 0.25,
    critique_enabled: bool = True,
) -> AgentState:
    """Continues from an already-prepared state. A new torch.Generator is created fresh
    inside generate_fn on every call, seeded with that attempt's seed (attempt 1 always
    uses SEED=266) -- never reused across attempts or across states. An attempt flagged
    unsafe by the safety checker is DISCARDED: never returned, never counted as the best
    attempt, and the loop tries again if the cap allows. When critique is off there is
    only ever one attempt, by definition -- no loop, no retry even on an unsafe result."""
    if state.route not in GENERATION_ROUTES or state.prompt is None:
        return state

    prompt = state.prompt
    seed = SEED
    strength = START_STRENGTH
    while True:
        t0 = time.perf_counter()
        try:
            gen = generate_fn(state=state, prompt=prompt, negative_prompt=state.negative_prompt,
                               seed=seed, strength=strength)
        except Exception as e:
            state.add_error(f"Image generation failed ({e.__class__.__name__}: {e}).")
            state.log("Image Generation", time.perf_counter() - t0, "failed")
            return state
        state.generation_history.append(gen)
        state.log("Image Generation", time.perf_counter() - t0, f"{gen.model} seed={seed}")

        if not critique_enabled:
            clip_score = clip_score_fn(gen.output_path, prompt) if clip_score_fn else None
            result = critique_mod.CritiqueResult(
                clip_score=clip_score, safe=not gen.nsfw_flagged,
                decision="discarded" if gen.nsfw_flagged else "accept")
            state.critique_history.append(result)
            break  # critique off: exactly one attempt, no retry, no exception

        t0 = time.perf_counter()
        if gen.nsfw_flagged:
            result = critique_mod.CritiqueResult(safe=False, decision="discarded")
            state.log("Critique", time.perf_counter() - t0, "decision=discarded (unsafe)")
        else:
            clip_score = clip_score_fn(gen.output_path, prompt) if clip_score_fn else None
            verdict_text = vlm_judge_fn(gen.output_path, prompt) if vlm_judge_fn else None
            result = critique_mod.decide(clip_score, clip_threshold, verdict_text, safe=True)
            state.log("Critique", time.perf_counter() - t0,
                       f"decision={result.decision} clip={clip_score}")
        state.critique_history.append(result)

        if result.decision == "accept":
            break
        if state.revision_count >= critique_mod.MAX_REVISIONS:
            break  # out of attempts, regardless of whether the last one was discarded or just didn't pass

        strategy = critique_mod.next_revision_strategy(state.revision_count)
        state.revision_count += 1
        strength = min(MAX_STRENGTH, strength + strategy["strength_delta"])
        seed += strategy["seed_offset"]
        if strategy["rephrase"]:
            prompt = pe.apply_revision(prompt, strategy["note"])

    _finalize(state)
    return state


def _finalize(state: AgentState) -> None:
    """Picks the best SAFE attempt across the whole run (highest CLIP among safe
    attempts, not necessarily the last one) and decides the overall outcome: "accept" if
    the last attempt tried fully passed; "stopped_at_cap" if the cap was reached but at
    least one attempt was safe; "reject" only if NO attempt was ever safe."""
    if not state.critique_history:
        return
    last = state.critique_history[-1]

    best_idx, best_clip = None, None
    for idx, c in enumerate(state.critique_history):
        if not c.safe:
            continue
        if best_idx is None:
            best_idx = idx  # first safe attempt, used if none ever got a CLIP score
        if c.clip_score is not None and (best_clip is None or c.clip_score > best_clip):
            best_idx, best_clip = idx, c.clip_score

    if last.decision == "accept":
        overall, chosen_idx = "accept", len(state.generation_history) - 1
    elif best_idx is not None:
        overall, chosen_idx = "stopped_at_cap", best_idx
    else:
        overall, chosen_idx = "reject", None

    if chosen_idx is not None:
        chosen = state.critique_history[chosen_idx]
        state.generation = state.generation_history[chosen_idx]
        state.critique = critique_mod.CritiqueResult(
            clip_score=chosen.clip_score, clip_threshold=chosen.clip_threshold,
            vlm_verdict=chosen.vlm_verdict, safe=True, decision=overall)
        state.final_image_path = state.generation.output_path
    else:
        state.generation = None
        state.critique = critique_mod.CritiqueResult(safe=False, decision="reject")
        state.final_image_path = None

    state.final_message = _final_message(state)


def _final_message(state: AgentState) -> str:
    if state.critique is None:
        return "Done."
    if state.critique.decision == "reject":
        return "None of the attempts passed the safety check, so no image is shown."
    if state.critique.decision == "stopped_at_cap":
        return (f"Here is the closest safe attempt after {state.revision_count} revisions; "
                "it may not fully match what you asked for.")
    if state.revision_count > 0:
        return f"Here is the result after {state.revision_count} revision(s)."
    return "Here is the result."


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
        prepare(state, vlm_call=vlm_call, classifier_fn=classifier_fn)
        generate_and_critique(state, generate_fn=generate_fn, clip_score_fn=clip_score_fn,
                               vlm_judge_fn=vlm_judge_fn, clip_threshold=clip_threshold,
                               critique_enabled=critique_enabled)
    except Exception as e:  # never crash -- always leave a clear message and a trace
        state.add_error(f"Something went wrong while handling this request ({e.__class__.__name__}: {e}).")
    finally:
        save_trace(state, trace_dir, request_id)
    return state


def run_both(
    request: str,
    image_path: Optional[str],
    *,
    vlm_call: Callable[[str, str], str],
    generate_fn: Callable[..., GenerationRecord],
    classifier_fn: Optional[Callable[[str], str]] = None,
    clip_score_fn: Optional[Callable[[str, str], float]] = None,
    vlm_judge_fn: Optional[Callable[[str, str], str]] = None,
    clip_threshold: float = 0.25,
    trace_dir: str = "outputs/traces",
    request_id_prefix: str = "req",
) -> tuple[AgentState, AgentState]:
    """The fair with/without-critique comparison: prepare ONCE, then branch into two
    independent generate_and_critique() calls (critique on, then off) from identical
    deep copies of that one prepared state. Returns (state_on, state_off); both traces
    are saved, suffixed "_on" / "_off"."""
    shared = AgentState(request=request)
    shared.image.path = image_path
    try:
        prepare(shared, vlm_call=vlm_call, classifier_fn=classifier_fn)
    except Exception as e:
        shared.add_error(f"Something went wrong while handling this request ({e.__class__.__name__}: {e}).")

    state_on = shared.model_copy(deep=True)
    state_off = shared.model_copy(deep=True)

    for state, critique_enabled, suffix in ((state_on, True, "on"), (state_off, False, "off")):
        try:
            generate_and_critique(state, generate_fn=generate_fn, clip_score_fn=clip_score_fn,
                                   vlm_judge_fn=vlm_judge_fn, clip_threshold=clip_threshold,
                                   critique_enabled=critique_enabled)
        except Exception as e:
            state.add_error(f"Something went wrong while handling this request ({e.__class__.__name__}: {e}).")
        finally:
            save_trace(state, trace_dir, f"{request_id_prefix}_{suffix}")

    return state_on, state_off
