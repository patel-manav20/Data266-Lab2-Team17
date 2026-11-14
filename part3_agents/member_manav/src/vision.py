"""Vision Understanding: a quality pre-check that needs no model, a prompt builder, a
strict-JSON parser with a plain-text fallback, and the agent function that wires them
to an injected VLM call. `vlm_call` is the only place a real model is touched, so this
whole module is testable on a laptop with a stub `vlm_call` and no downloads.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

import numpy as np
from PIL import Image

from .planner import looks_like_instruction
from .state import AgentState, VisionOutput

TOO_SMALL_PX = 64
TOO_DARK_MEAN = 15.0  # mean pixel value, 0-255 scale

DEFAULT_QUESTIONS = [
    "What is the overall scene or setting?",
    "Is there any text visible in the image? If so, what does it say?",
]

_COMMON_NOUNS = ["dog", "cat", "person", "people", "car", "bus", "bicycle", "table",
                 "cup", "sofa", "couch", "sign", "beach", "park", "kitchen", "street"]


def quality_precheck(image_path: str) -> Optional[VisionOutput]:
    """Returns a VisionOutput if the image is too small or too dark to describe with
    confidence, computed with plain PIL/numpy -- no model call. Returns None otherwise,
    meaning the real VLM should run."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        arr = np.asarray(img)
    if min(w, h) < TOO_SMALL_PX:
        msg = "The image is too small to describe with confidence. Please provide a clearer image."
        return VisionOutput(caption=msg, too_dark_or_small=True, qa={"default": msg})
    if float(arr.mean()) < TOO_DARK_MEAN:
        msg = "The image is too dark to describe with confidence. Please provide a clearer image."
        return VisionOutput(caption=msg, too_dark_or_small=True, qa={"default": msg})
    return None


def build_vlm_prompt(user_question: Optional[str]) -> str:
    questions = list(DEFAULT_QUESTIONS)
    if user_question:
        questions.insert(0, user_question)
    q_list = "\n".join(f"- {q}" for q in questions)
    return (
        "Look at the image and answer in STRICT JSON with exactly these keys: "
        '"caption" (one sentence), "objects" (a list of the main objects visible), '
        '"visible_text" (any text visible in the image, or an empty string), '
        '"qa" (an object mapping each question below to a short answer).\n'
        f"Questions:\n{q_list}\n"
        "Return only the JSON object, nothing else."
    )


def _fallback_object_guess(raw: str) -> list[str]:
    low = raw.lower()
    return [n for n in _COMMON_NOUNS if n in low]


def parse_vlm_output(raw: str, user_question: Optional[str] = None) -> VisionOutput:
    """Parses the VLM's JSON response. Falls back to a plain-text reading (first line as
    the caption, a small keyword scan for objects) when the model doesn't return clean
    JSON, which real small VLMs occasionally do."""
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            objects = data.get("objects", [])
            qa = data.get("qa", {})
            return VisionOutput(
                caption=str(data.get("caption", "")).strip(),
                objects=[str(o) for o in objects] if isinstance(objects, list) else [],
                visible_text=str(data.get("visible_text", "")).strip(),
                qa={str(k): str(v) for k, v in qa.items()} if isinstance(qa, dict) else {},
            )
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    caption = raw.strip().splitlines()[0] if raw.strip() else ""
    qa = {user_question: raw.strip()} if user_question else {}
    return VisionOutput(caption=caption, objects=_fallback_object_guess(raw), visible_text="", qa=qa)


def format_caption_answer(vision: VisionOutput, user_question: Optional[str]) -> str:
    parts = [vision.caption]
    if vision.visible_text:
        parts.append(f'Visible text in the image: "{vision.visible_text}"')
    if user_question and user_question in vision.qa:
        parts.append(vision.qa[user_question])
    return " ".join(p for p in parts if p)


def run_vision(state: AgentState, vlm_call: Callable[[str, str], str],
               user_question: Optional[str] = None) -> AgentState:
    pre = quality_precheck(state.image.path)
    if pre is not None:
        state.vision = pre
        state.final_message = pre.caption
        return state

    prompt = build_vlm_prompt(user_question)
    raw = vlm_call(state.image.path, prompt)
    vision = parse_vlm_output(raw, user_question=user_question)

    instr_found, _pat = looks_like_instruction(vision.visible_text)
    state.safety.instruction_in_image = instr_found
    if instr_found:
        state.safety.instruction_text_found = vision.visible_text

    state.vision = vision
    if state.route == "caption_question":
        state.final_message = format_caption_answer(vision, user_question)
    return state
