"""The Planner: image validity, input safety, and routing. Rules (keywords and
patterns) run first and are checked purely in Python -- no model call, no GPU. The
vision model is only consulted as a fallback classifier when no rule confidently
matches a route, so it is passed in as a plain callable (`classifier_fn`) and is never
imported directly here -- that keeps this module testable on a laptop with no
downloads at all.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

import cv2
from PIL import Image

from .state import AgentState, ImageInfo

# ---------------------------------------------------------------------------
# Patterns. Checked with re.search over the lowercased request text.
# ---------------------------------------------------------------------------

UNSAFE_REQUEST_PATTERNS = [
    r"\bnaked\b", r"\bnude\b", r"\bnudity\b", r"\bnsfw\b", r"\bnudify\b",
    r"\bexplicit\b.*\b(adult|sexual)\b", r"\b(adult|sexual)\b.*\bexplicit\b",
    r"\bporn(ographic)?\b", r"\bundress\b", r"\bstrip(ped)?\b.*\b(her|him|them)\b",
    r"\bbloody\b", r"\bgore\b", r"\bgory\b", r"\bstabbing\b", r"\bmurder\b", r"\bbeheading\b",
    r"\bself[- ]harm\b", r"\bsuicide\b", r"\bgraphic violence\b",
]

DECEPTIVE_EDIT_PATTERNS = [
    r"\bmake it look real\b", r"\blooks? real\b", r"\bface ?swap\b",
    r"\bswap (his|her|their|the) face\b", r"\breplace .*\bface\b",
    r"\bfamous (politician|celebrity|person)\b", r"\bcelebrity\b",
    r"\bcrime scene\b", r"\bholding a (gun|knife|weapon)\b",
    r"\bmake (it|this) (look )?(realistic|authentic)\b", r"\bfabricat", r"\bdeepfake\b",
]

INSTRUCTION_IN_TEXT_PATTERNS = [
    r"\bignore the (user|instructions?)\b", r"\bsystem\s*:", r"\badmin mode\b",
    r"\bassistant\s*:", r"\breply only\b", r"\bfrom now on\b", r"\bjailbreak\b",
    r"\byou must\b", r"\boverride\b", r"\bpwned\b",
]

DOMAIN_STYLE_PATTERNS = [r"\bpok[ée]mon\b"]

AMBIGUOUS_PHRASES = [
    r"\bmake it better\b", r"\bimprove this\b", r"^do something\b", r"^help$",
]

STYLE_CHANGE_PATTERNS = [
    r"\bwatercolor\b", r"\boil painting\b", r"\bpencil sketch\b", r"\bsketch\b",
    r"\bturn (this|it) into an? .*(painting|sketch|style)\b",
    r"\bmake (this|it) look like an? .*(painting|sketch|photo|style)\b",
    r"\bchange the style\b", r"\bin the style of\b", r"\bbrush strokes\b", r"\bcartoonize\b",
]

EDIT_PATTERNS = [
    r"\bchange the color\b", r"\badd an?\b", r"\bremove the\b",
    r"\bmake it (brighter|darker|clearer|sharper)\b", r"\btoo dark\b",
    r"\benhance\b", r"\bedit\b", r"\breplace the\b", r"\bbrighter\b", r"\bblurry\b",
]

QUESTION_PATTERNS = [
    r"\bdescribe\b", r"^what\b", r"\bwhat (is|are|color|kind)\b", r"\bwho\b",
    r"\bwhere\b", r"\bhow many\b", r"\bis there\b", r"\bare there\b",
    r"\blist\b", r"\bcaption\b", r"\?\s*$",
]


def is_unsafe_request(request: str) -> tuple[bool, Optional[str]]:
    text = request.lower()
    for pat in UNSAFE_REQUEST_PATTERNS:
        if re.search(pat, text):
            return True, pat
    return False, None


_FACE_CASCADE: Optional["cv2.CascadeClassifier"] = None


def has_face(image_path: str) -> bool:
    """A face presence check. Pure CPU (OpenCV's bundled Haar cascade), no GPU, no
    network -- cheap enough to run on every request that carries an image.

    Histogram-equalizes the grayscale image before detecting: a plain read with the
    cascade's usual settings missed a real, clearly visible face in one of the
    red-team photos (R07) simply because of its lighting/contrast; equalizing first
    caught it, with the cascade's normal (already fairly conservative) settings
    otherwise unchanged, and without adding false positives on faceless photos
    (checked against every scenario image: dogs, cats, street scenes, text images)."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0


def is_real_person_deceptive_edit(request: str, image_path: Optional[str]) -> tuple[bool, Optional[str]]:
    if not image_path or not has_face(image_path):
        return False, None
    text = request.lower()
    for pat in DECEPTIVE_EDIT_PATTERNS:
        if re.search(pat, text):
            return True, pat
    return False, None


def looks_like_instruction(text: str) -> tuple[bool, Optional[str]]:
    """Flags visible-image-text that reads like an instruction. This flag is for the
    trace only -- it is never what keeps the text from being obeyed. What keeps it from
    being obeyed is that Prompt Engineering (prompt_engineering.py) always quotes
    visible_text as image CONTENT, never as something to follow, regardless of this flag.
    """
    if not text:
        return False, None
    low = text.lower()
    for pat in INSTRUCTION_IN_TEXT_PATTERNS:
        if re.search(pat, low):
            return True, pat
    return False, None


def check_image(image_path: Optional[str]) -> ImageInfo:
    if not image_path:
        return ImageInfo(path=None, valid=True)
    p = Path(image_path)
    if not p.exists():
        return ImageInfo(path=image_path, valid=False, error=f"file not found: {image_path}")
    if p.stat().st_size == 0:
        return ImageInfo(path=image_path, valid=False, error="the image file is empty (0 bytes)")
    try:
        with Image.open(p) as img:
            img.verify()
        # verify() only checks the file structure, not that every pixel is actually
        # decodable -- a truncated JPEG can pass it cleanly. convert() forces a full
        # decode, which is what actually catches a cut-off file like S14/R10's.
        with Image.open(p) as img:
            img = img.convert("RGB")
            w, h = img.size
        return ImageInfo(path=image_path, valid=True, width=w, height=h)
    except Exception as e:
        return ImageInfo(path=image_path, valid=False,
                          error=f"could not open the image file ({e.__class__.__name__})")


def route_request(request: str, classifier_fn: Optional[Callable[[str], str]] = None) -> tuple[str, str]:
    text = request.lower().strip()
    if not text:
        return "unsupported", "empty request"
    for pat in DOMAIN_STYLE_PATTERNS:
        if re.search(pat, text):
            return "domain_style", pat
    for pat in AMBIGUOUS_PHRASES:
        if re.search(pat, text):
            return "unsupported", pat
    for pat in STYLE_CHANGE_PATTERNS:
        if re.search(pat, text):
            return "style_change", pat
    for pat in EDIT_PATTERNS:
        if re.search(pat, text):
            return "edit", pat
    for pat in QUESTION_PATTERNS:
        if re.search(pat, text):
            return "caption_question", pat
    if classifier_fn is not None:
        return classifier_fn(request), "vision-model classifier (rules unsure)"
    return "unsupported", "ambiguous (no rule matched)"


def plan(state: AgentState, classifier_fn: Optional[Callable[[str], str]] = None) -> AgentState:
    image_info = check_image(state.image.path)
    state.image = image_info
    if not image_info.valid:
        state.route = "unsupported"
        state.final_message = (
            f"I couldn't read the image you provided: {image_info.error}. "
            "Please try a valid JPG or PNG file."
        )
        return state

    if not state.request.strip():
        state.route = "unsupported"
        state.final_message = "Please tell me what you'd like me to do."
        return state

    unsafe, pat = is_unsafe_request(state.request)
    if unsafe:
        state.safety.unsafe_request = True
        state.safety.reason = f"matched unsafe-request pattern: {pat}"
        state.route = "unsupported"
        state.final_message = "I can't help with that request."
        return state

    deceptive, pat = is_real_person_deceptive_edit(state.request, image_info.path)
    if deceptive:
        state.safety.real_person_deceptive_edit = True
        state.safety.reason = f"matched a deceptive-edit pattern on a photo with a real person: {pat}"
        state.route = "unsupported"
        state.final_message = "I can't make a realistic or deceptive edit of a real person."
        return state

    route, _reason = route_request(state.request, classifier_fn=classifier_fn)
    state.route = route
    if route == "unsupported" and not state.final_message:
        state.final_message = (
            "I'm not sure what change you want -- could you say, for example, whether you "
            "want a style change, an edit, or a brighter photo?"
        )
    return state
