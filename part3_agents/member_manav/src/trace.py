"""One JSON trace file per request, written next to the agents' own outputs."""
from __future__ import annotations

from pathlib import Path

from .state import AgentState


def save_trace(state: AgentState, out_dir: str, request_id: str) -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    trace_path = out_path / f"trace_{request_id}.json"
    trace_path.write_text(state.model_dump_json(indent=2))
    return str(trace_path)
