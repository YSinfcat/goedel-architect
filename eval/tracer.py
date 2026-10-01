"""DEPRECATED shim - the canonical implementation lives in src/tracer.py.

This file was a drifted duplicate of the src copy (missing the `iteration`
field); it remains only as a re-export so external imports keep working.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
_CANONICAL = _SRC / "tracer.py"

_existing = sys.modules.get("tracer")
if _existing is not None and getattr(_existing, "__file__", "") == str(_CANONICAL):
    _mod = _existing
else:
    _spec = importlib.util.spec_from_file_location("tracer_canonical", _CANONICAL)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["tracer_canonical"] = _mod
    _spec.loader.exec_module(_mod)

TraceEvent = _mod.TraceEvent
NullTracer = _mod.NullTracer
JsonlTracer = _mod.JsonlTracer
