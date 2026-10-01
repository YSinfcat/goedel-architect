"""
Lightweight event tracer for the Goedel-Architect proving loop.

Events are emitted at each tool call / result inside _run_loop so that
graph_viz.py can build an interactive visualization of the proving steps.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    kind: str          # theorem_start | tool_call | tool_result | model_text | final_verify
    thm_name: str
    turn: int = 0
    call_id: str | None = None
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    result: str | None = None
    ok: bool | None = None
    # Refinement round the event belongs to (1-based; None for events
    # outside the phase loop). Previously every iteration's events were
    # indistinguishable in a trace, so cross-round token aggregation and
    # visualization mixed rounds together (review IV.6).
    iteration: int | None = None
    ts: float = field(default_factory=time.time)


class NullTracer:
    def emit(self, event: TraceEvent) -> None:
        pass


class JsonlTracer:
    """Thread-safe JSONL writer — one line per event."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._f = self.path.open("a", encoding="utf-8", buffering=1)

    def emit(self, event: TraceEvent) -> None:
        line = json.dumps(asdict(event)) + "\n"
        with self._lock:
            self._f.write(line)

    def close(self) -> None:
        with self._lock:
            self._f.close()

    def __del__(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
