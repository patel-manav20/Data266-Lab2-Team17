"""The shared state every agent reads from and writes to. One pydantic model, passed
by plain Python functions from one agent to the next -- no framework, no hidden state.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

ROUTES = ("caption_question", "style_change", "edit", "domain_style", "unsupported")


class ImageInfo(BaseModel):
    path: Optional[str] = None
    valid: bool = False
    error: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class SafetyVerdict(BaseModel):
    unsafe_request: bool = False
    real_person_deceptive_edit: bool = False
    instruction_in_image: bool = False
    instruction_text_found: Optional[str] = None
    reason: Optional[str] = None

    @property
    def refused(self) -> bool:
        return self.unsafe_request or self.real_person_deceptive_edit


class VisionOutput(BaseModel):
    caption: str = ""
    objects: list[str] = Field(default_factory=list)
    visible_text: str = ""
    qa: dict[str, str] = Field(default_factory=dict)
    too_dark_or_small: bool = False


class GenerationRecord(BaseModel):
    model: str
    mode: str  # "text2img" | "img2img" | "instruct_edit"
    prompt: str
    negative_prompt: str = ""
    seed: int = 266
    steps: Optional[int] = None
    strength: Optional[float] = None
    image_guidance_scale: Optional[float] = None
    guidance_scale: Optional[float] = None
    output_path: Optional[str] = None
    time_s: float = 0.0
    nsfw_flagged: bool = False


class CritiqueResult(BaseModel):
    clip_score: Optional[float] = None
    clip_threshold: Optional[float] = None
    vlm_verdict: Optional[str] = None  # "yes" | "no" | None
    safe: bool = True
    # per-attempt: "accept" | "revise" | "discarded" (unsafe, never counted as best)
    # overall (state.critique after a run finishes): "accept" | "stopped_at_cap" | "reject"
    decision: str = "accept"
    revision_instruction: Optional[str] = None


class TraceEntry(BaseModel):
    agent: str
    time_s: float
    summary: str


class AgentState(BaseModel):
    request: str
    image: ImageInfo = Field(default_factory=ImageInfo)

    route: Optional[str] = None
    safety: SafetyVerdict = Field(default_factory=SafetyVerdict)

    vision: Optional[VisionOutput] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None

    generation: Optional[GenerationRecord] = None
    generation_history: list[GenerationRecord] = Field(default_factory=list)

    critique: Optional[CritiqueResult] = None
    critique_history: list[CritiqueResult] = Field(default_factory=list)
    revision_count: int = 0

    errors: list[str] = Field(default_factory=list)
    final_message: Optional[str] = None
    final_image_path: Optional[str] = None

    trace: list[TraceEntry] = Field(default_factory=list)

    def log(self, agent: str, time_s: float, summary: str) -> None:
        self.trace.append(TraceEntry(agent=agent, time_s=time_s, summary=summary))

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.final_message = message
