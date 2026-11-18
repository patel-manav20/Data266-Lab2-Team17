"""Prompt Engineering: turns the request and the Vision output into a generation
prompt and a negative prompt. The user's request is kept verbatim as the core
instruction. Ordinary visible text found in the input image is quoted as image content
("the image shows the text \"...\""), never appended as something to obey. Text the
Planner/Vision already flagged as instruction-like (see planner.looks_like_instruction)
is handled more strictly still: it is never copied into the generation prompt at all --
not even in quotes -- so an injection attempt cannot influence the image, and a
generic note is used instead. The raw flagged text stays only in the safety trace
(state.safety.instruction_text_found) for audit purposes.
"""
from __future__ import annotations

from typing import Optional

from .state import VisionOutput

DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, extra limbs, watermark, text artifacts"
DOMAIN_STYLE_NEGATIVE_EXTRA = "photorealistic, realistic skin texture"


def build_generation_prompt(request: str, vision: Optional[VisionOutput], route: str,
                             instruction_flagged: bool = False) -> tuple[str, str]:
    parts = [request.strip().rstrip(".")]
    if vision is not None:
        if vision.caption:
            parts.append(f"(starting image: {vision.caption})")
        if vision.visible_text and instruction_flagged:
            # flagged as instruction-like: never copied into the prompt, not even quoted
            parts.append("(the original image also contains some text, not used here)")
        elif vision.visible_text:
            # ordinary text: quoted as CONTENT ONLY, never treated as an instruction
            parts.append(f'(the original image shows the text "{vision.visible_text}" written in it)')
    prompt = ", ".join(p for p in parts if p)

    negative = DEFAULT_NEGATIVE_PROMPT
    if route == "domain_style":
        negative = f"{negative}, {DOMAIN_STYLE_NEGATIVE_EXTRA}"
    return prompt, negative


def apply_revision(prompt: str, instruction: str) -> str:
    """Appends a concrete revision instruction from the Critique agent (e.g. "rephrase
    the subject") to the existing prompt, rather than rebuilding it from scratch."""
    return f"{prompt}, {instruction}"
