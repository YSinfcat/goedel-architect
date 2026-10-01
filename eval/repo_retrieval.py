"""DEPRECATED shim - the canonical implementation lives in src/repo_retrieval.py.

This file was a drifted duplicate of the src copy (it had already lost the
cache staleness check; the two silently diverged). It remains only as a
re-export so any external code importing ``eval.repo_retrieval`` keeps
working; all fixes (content-hashed cache keys, embedding-model in the key)
land in the src module.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
_CANONICAL = _SRC / "repo_retrieval.py"

# When this shim is imported under the name "repo_retrieval" (eval/ ahead
# of src/ on sys.path), a plain `from repo_retrieval import ...` would
# re-import the shim itself - import the canonical module by path instead,
# reusing it when it is already loaded under its own name.
_existing = sys.modules.get("repo_retrieval")
if _existing is not None and getattr(_existing, "__file__", "") == str(_CANONICAL):
    _mod = _existing
else:
    _spec = importlib.util.spec_from_file_location("repo_retrieval_canonical", _CANONICAL)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["repo_retrieval_canonical"] = _mod
    _spec.loader.exec_module(_mod)

EMBED_MODEL = _mod.EMBED_MODEL
RepoDecl = _mod.RepoDecl
RepoRetrieval = _mod.RepoRetrieval
