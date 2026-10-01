"""Mathlib semantic search via LeanSearch and Loogle.

Mirrors the "semantic Mathlib search service" from the paper (Section 4.1),
which is backed by the same APIs used by the LeanSearchClient Lean package.

Two backends (tried in order):
  1. LeanSearch (https://leansearch.net) — natural language queries
       POST /search  body: {"query": ["text"], "num_results": k}
  2. Loogle (https://loogle.lean-lang.org/json) — name/type pattern queries
       GET /json?q=<pattern>

LeanSearch is the primary backend; Loogle is the fallback when LeanSearch is
unavailable or when the query looks like a Lean identifier/type pattern.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

LEANSEARCH_URL = os.environ.get(
    "LEANSEARCHCLIENT_LEANSEARCH_API_URL",
    "https://leansearch.net/search",
)
LOOGLE_URL = os.environ.get(
    "LEANSEARCHCLIENT_LOOGLE_API_URL",
    "https://loogle.lean-lang.org/json",
)
DEFAULT_TIMEOUT = 15.0


@dataclass
class LemmaResult:
    name: str
    type_sig: str
    docstring: str
    score: float = 0.0

    def format(self) -> str:
        lines = [f"**{self.name}**"]
        if self.type_sig:
            lines.append(f"Type: `{self.type_sig}`")
        if self.docstring:
            lines.append(f"Doc: {self.docstring}")
        return "\n".join(lines)


class MathlibRetrieval:
    """Wrapper around LeanSearch + Loogle APIs."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self._client = httpx.Client(timeout=timeout)
        # Set by search() when the most recent call hit a backend failure;
        # empty string means the last search completed (possibly with zero
        # hits). See search().
        self.last_error = ""

    def search(self, query: str, k: int = 10) -> list[LemmaResult]:
        """Search Mathlib. Tries LeanSearch first, falls back to Loogle.

        An outage no longer masquerades as "no relevant lemmas": when BOTH
        backends fail, the returned list is empty AND `self.last_error`
        carries the failure so callers (and the prover's tool result text)
        can tell "service down" from a genuine zero-hit query.
        """
        self.last_error = ""
        results = self._leansearch(query, k)
        if not results:
            results = self._loogle(query, k)
        return results

    # ------------------------------------------------------------------

    def _leansearch(self, query: str, k: int) -> list[LemmaResult]:
        """Natural language search via LeanSearch."""
        try:
            resp = self._client.post(
                LEANSEARCH_URL,
                json={"query": [query], "num_results": k},
                headers={"accept": "application/json", "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return self._parse_leansearch(resp.json())
        except Exception as exc:
            self.last_error = f"LeanSearch unavailable: {exc}"
            return []

    def _loogle(self, query: str, k: int) -> list[LemmaResult]:
        """Name/type pattern search via Loogle."""
        try:
            resp = self._client.get(LOOGLE_URL, params={"q": query})
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", []) or []
            results = []
            for hit in hits[:k]:
                name = hit.get("name", "")
                if not name:
                    continue
                results.append(LemmaResult(
                    name=name,
                    type_sig=hit.get("type", ""),
                    docstring=hit.get("doc") or "",
                ))
            return results
        except Exception as exc:
            self.last_error = (self.last_error + "; " if self.last_error else "") + f"Loogle unavailable: {exc}"
            return []

    def _parse_leansearch(self, data: object) -> list[LemmaResult]:
        # Shape: [[{"result": {"name": [...], "type": str, "docstring": str}}]]
        results: list[LemmaResult] = []
        outer = data if isinstance(data, list) else []
        inner = outer[0] if outer and isinstance(outer[0], list) else outer
        for item in inner:
            r = item.get("result", item) if isinstance(item, dict) else {}
            name_parts = r.get("name", [])
            name = ".".join(name_parts) if isinstance(name_parts, list) else str(name_parts)
            if not name:
                continue
            results.append(LemmaResult(
                name=name,
                type_sig=r.get("type", ""),
                docstring=r.get("docstring", r.get("doc", "")),
            ))
        return results

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MathlibRetrieval:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
