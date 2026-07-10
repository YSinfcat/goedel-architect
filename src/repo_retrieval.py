"""
Embedding-based semantic search over a Lean repo's declarations.

Index is built once per repo via OpenAI text-embedding-3-small and cached
to disk as a .npy file. Subsequent runs load the cache instantly.

Usage:
    r = RepoRetrieval(Path("...lean_repos/VCV-io"), cache_dir=Path("results/"))
    hits = r.search("induction principle for OracleComp", k=8)
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 512  # max texts per embeddings API call


# ---------------------------------------------------------------------------
# Declaration extraction
# ---------------------------------------------------------------------------

_DECL_START_RE = re.compile(
    r"^[ \t]*(?:@\[[^\]]*\]\s*)*"
    r"(?:protected\s+|private\s+|noncomputable\s+)*"
    r"(theorem|lemma|def|abbrev|instance|inductive|structure|class)\s+"
    r"(\S+)",
    re.MULTILINE,
)

_BOUNDARY_RE = re.compile(
    r"^[ \t]*(?:namespace|end|section|open|variable|variables|universe)\b",
    re.MULTILINE,
)

_CTOR_LINE_RE = re.compile(r"^[ \t]*\|[ \t]*", re.MULTILINE)

_OPEN_BRACKETS = "([{"
_CLOSE_BRACKETS = ")]}"

_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|_")


def _format_sig(params: str, ty: str) -> str:
    """`RepoDecl.format()` always prepends its own " : " before signature -
    if signature already starts with ":" (from a naive `params : ty` join
    when params is empty), the result is a confusing "name : : Type"."""
    if params.strip() and ty.strip():
        sig = f"{params} : {ty}"
    else:
        sig = ty if ty.strip() else params
    return re.sub(r"\s+", " ", sig).strip()


def _scan_header(text: str, pos: int) -> tuple[int, int | None]:
    """From `pos` (just after a declaration's name), scan forward tracking
    bracket depth to find where the header ends: `:=` or `where` at depth 0.

    Also returns the position of the last depth-0 (top-level) `:`, which
    splits binder params from the return type - e.g. `(h : P) : Q := ...`
    has a depth-1 `:` inside the parens (part of the binder) and a depth-0
    `:` before `Q` (the actual type annotation) that we want.

    Many `def`/`abbrev` declarations have NO such top-level colon at all
    (`abbrev foo := bar`) - the previous version of this scanner treated
    "the next `:` anywhere in the file" as the type separator, which for
    these declarations meant scanning straight through `:=` into the START
    OF THE NEXT DECLARATION looking for a colon, and finding one inside
    that next declaration's own binder types (e.g. `def stuck (t: Tm) : ...`).
    That merged two unrelated declarations into one garbled, misleading
    signature - confirmed on `Frap/Types.lean`'s `step_normal_form`/`stuck`
    pair, which is why `repo_search` returned junk for exact-name lookups.
    """
    depth = 0
    last_colon = None
    n = len(text)
    i = pos
    while i < n:
        c = text[i]
        if c in _OPEN_BRACKETS:
            depth += 1
        elif c in _CLOSE_BRACKETS:
            depth -= 1
        elif depth <= 0:
            if text[i:i + 2] == ":=":
                return i, last_colon
            if c == ":":
                last_colon = i
            elif text[i:i + 5] == "where" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")) \
                    and (i + 5 >= n or not (text[i + 5].isalnum() or text[i + 5] == "_")):
                return i, last_colon
        i += 1
    return n, last_colon


def _find_block_end(text: str, pos: int) -> int:
    """End of an inductive/structure/class body: start of the next
    top-level declaration or namespace-ish keyword, or EOF."""
    matches = [m.start() for m in (_DECL_START_RE.search(text, pos), _BOUNDARY_RE.search(text, pos)) if m]
    return min(matches) if matches else len(text)


def _split_constructors(text: str, body_start: int, body_end: int) -> list[tuple[str, str, str, int]]:
    """Split an inductive/structure `:=`/`where` body into per-constructor
    (name, binders, type, line) tuples, one per `| ctorName ... : Type` line
    (continuation lines belonging to the same constructor are included up
    to the next `|`-line). Previously these constructors were never
    extracted at all (the old regex only matched theorem/lemma/def/abbrev/
    instance), so every inductive's constructors - e.g. `Step.st_pred0`,
    `BValue.bv_true` - were invisible to repo_search regardless of query."""
    body = text[body_start:body_end]
    markers = [(m.start(), m.end()) for m in _CTOR_LINE_RE.finditer(body)]
    if not markers:
        return []
    ctors: list[tuple[str, str, str, int]] = []
    for i, (mark_start, content_start) in enumerate(markers):
        content_end = markers[i + 1][0] if i + 1 < len(markers) else len(body)
        chunk = body[content_start:content_end].strip()
        if not chunk:
            continue
        name_m = re.match(r"(\S+)", chunk)
        if not name_m:
            continue
        name = name_m.group(1)
        rest = chunk[name_m.end():]
        depth = 0
        last_colon = None
        for j, c in enumerate(rest):
            if c in _OPEN_BRACKETS:
                depth += 1
            elif c in _CLOSE_BRACKETS:
                depth -= 1
            elif c == ":" and depth <= 0:
                last_colon = j
        if last_colon is not None:
            binders, ty = rest[:last_colon].strip(), rest[last_colon + 1:].strip()
        else:
            binders, ty = "", rest.strip()
        line = text[:body_start + mark_start].count("\n") + 1
        ctors.append((name, binders, ty, line))
    return ctors


@dataclass
class RepoDecl:
    kind: str
    name: str
    signature: str
    file: str
    line: int

    def format(self) -> str:
        return f"{self.kind} {self.name} : {self.signature.strip()}\n-- in {self.file}:{self.line}"

    def embed_text(self) -> str:
        """Text sent to the embedding model."""
        return f"{self.kind} {self.name} {self.signature}"


def _extract_decls(path: Path, repo_root: Path) -> list[RepoDecl]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    rel = str(path.relative_to(repo_root))
    decls: list[RepoDecl] = []
    for m in _DECL_START_RE.finditer(text):
        kind = m.group(1)
        name = m.group(2).strip()
        line = text[: m.start()].count("\n") + 1
        header_end, colon_pos = _scan_header(text, m.end())
        if colon_pos is not None:
            params, ty = text[m.end():colon_pos], text[colon_pos + 1:header_end]
        else:
            params, ty = text[m.end():header_end], ""
        sig = _format_sig(params, ty)
        decls.append(RepoDecl(kind=kind, name=name, signature=sig, file=rel, line=line))

        is_ctor_block = kind in ("inductive", "structure", "class") and text[header_end:header_end + 2] == ":="
        if not is_ctor_block and kind in ("inductive", "structure", "class"):
            where_len = 5 if text[header_end:header_end + 5] == "where" else 0
            is_ctor_block = where_len > 0
        if is_ctor_block:
            body_start = header_end + (2 if text[header_end:header_end + 2] == ":=" else 5)
            block_end = _find_block_end(text, body_start)
            for ctor_name, binders, ctor_ty, ctor_line in _split_constructors(text, body_start, block_end):
                decls.append(RepoDecl(
                    kind="constructor", name=f"{name}.{ctor_name}",
                    signature=_format_sig(binders, ctor_ty), file=rel, line=ctor_line,
                ))
    return decls


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class RepoRetrieval:
    """
    Semantic search over a Lean repo using OpenAI embeddings.

    The first call to search() builds an embedding index and writes it to
    cache_dir/<repo_hash>.npy + <repo_hash>.json. Subsequent calls load
    from cache.
    """

    _instances: dict[str, "RepoRetrieval"] = {}
    _instances_lock = threading.Lock()

    def __init__(self, repo_root: Path, cache_dir: Path | None = None) -> None:
        self.repo_root = Path(repo_root)
        self.cache_dir = Path(cache_dir) if cache_dir else self.repo_root.parent.parent / "repo_index_cache"
        self._client = OpenAI()
        self._decls: list[RepoDecl] = []
        self._embeddings: np.ndarray | None = None  # shape (N, D)
        self._loaded = False
        self._lock = threading.Lock()

    @classmethod
    def for_repo(cls, lean_src_dir: Path, repo_name: str, cache_dir: Path | None = None) -> "RepoRetrieval":
        key = str(lean_src_dir / repo_name)
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(lean_src_dir / repo_name, cache_dir=cache_dir)
        return cls._instances[key]

    # ---- public API -------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[RepoDecl]:
        self._ensure_loaded()
        if not self._decls or self._embeddings is None:
            return []

        q_emb = self._embed([query])[0]  # (D,)
        scores = self._embeddings @ q_emb  # cosine sim (already normalized)
        top_k = int(min(k, len(self._decls)))
        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [self._decls[i] for i in idx]

    # ---- loading & caching -----------------------------------------------

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._load()
            self._loaded = True

    def _load(self) -> None:
        if not self.repo_root.exists():
            return

        # Collect declarations
        for lean_file in self.repo_root.rglob("*.lean"):
            if ".lake" in lean_file.parts:
                continue
            for decl in _extract_decls(lean_file, self.repo_root):
                self._decls.append(decl)

        if not self._decls:
            return

        # Check cache
        cache_key = self._cache_key()
        emb_path  = self.cache_dir / f"{cache_key}.npy"
        meta_path = self.cache_dir / f"{cache_key}.json"

        if emb_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("n") == len(self._decls):
                self._embeddings = np.load(str(emb_path))
                return
            print(f"[repo_retrieval] Cache for {self.repo_root.name} is stale "
                  f"(cached n={meta.get('n')}, live n={len(self._decls)}) - rebuilding.")

        # Build index
        print(f"[repo_retrieval] Building embedding index for {self.repo_root.name} "
              f"({len(self._decls)} declarations) ...")
        texts = [d.embed_text() for d in self._decls]
        self._embeddings = self._embed(texts)

        # Save cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(emb_path), self._embeddings)
        meta_path.write_text(json.dumps({"repo": str(self.repo_root), "n": len(self._decls)}))
        print(f"[repo_retrieval] Index saved to {emb_path}")

    def _cache_key(self) -> str:
        h = hashlib.md5(str(self.repo_root.resolve()).encode()).hexdigest()[:12]
        return f"{self.repo_root.name}_{h}"

    # ---- embedding -------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts in batches, return L2-normalised float32 matrix."""
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            resp = self._client.embeddings.create(model=EMBED_MODEL, input=batch)
            vecs = np.array([item.embedding for item in resp.data], dtype=np.float32)
            # L2-normalise so dot product == cosine similarity
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs /= np.where(norms == 0, 1.0, norms)
            all_vecs.append(vecs)
        return np.concatenate(all_vecs, axis=0)
