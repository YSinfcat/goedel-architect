"""Backend-agnostic tool-call dispatch.

`repo_search`/`mathlib_search` dispatch (parse args -> call retrieval -> join
hits or fall back to "No results") used to be implemented independently in
prover.py and blueprint.py, and had already drifted (one coerced `k` to
`int`, the other didn't). ToolDispatcher and the shared handler factories
below give every phase one canonical implementation of each tool.
"""
from __future__ import annotations

from typing import Callable


class ToolDispatcher:
    """Maps a tool name to a handler, independent of which ModelBackend
    produced the call. Construct one per node (Phase 2) or per generation/
    refinement attempt (Phase 1/3); handlers may close over per-attempt
    mutable state (see prover.py's _ProverDispatcher)."""

    def __init__(self, handlers: dict[str, Callable[[dict], str]]):
        self._handlers = handlers

    def dispatch(self, name: str, args: dict) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return f"Tool unavailable: {name}. Valid tools: {', '.join(self._handlers)}."
        return handler(args)


def repo_search_handler(repo_retrieval) -> Callable[[dict], str]:
    def handler(args: dict) -> str:
        hits = repo_retrieval.search(args.get("query", ""), int(args.get("k", 10)))
        return "\n\n".join(h.format() for h in hits) or "No results in repo."
    return handler


def mathlib_search_handler(retrieval) -> Callable[[dict], str]:
    def handler(args: dict) -> str:
        hits = retrieval.search(args.get("query", ""), int(args.get("k", 10)))
        return "\n\n".join(h.format() for h in hits) or "No results found."
    return handler
