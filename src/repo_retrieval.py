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

_DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*"
    r"(?:protected\s+|private\s+)?"
    r"(theorem|lemma|def|abbrev|noncomputable def|instance)\s+"
    r"(\S+)"
    r"([^:=]*?)"
    r"\s*:\s*"
    r"(.*?)(?=\s*:=|\s*where\s|\Z)",
    re.DOTALL | re.MULTILINE,
)

_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|_")


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
    for m in _DECL_RE.finditer(text):
        kind = m.group(1)
        name = m.group(2).strip()
        sig  = re.sub(r"\s+", " ", (m.group(3) + " : " + m.group(4))).strip()
        line = text[: m.start()].count("\n") + 1
        decls.append(RepoDecl(kind=kind, name=name, signature=sig, file=rel, line=line))
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

        # Check cache. The key embeds the embedding model and a content
        # fingerprint over every extracted declaration - previously the key
        # was just the repo path and the staleness check compared
        # declaration COUNT only, so edits that kept the count (a renamed
        # lemma, a changed signature) silently reused stale embeddings.
        cache_key = self._cache_key(self._decls)
        emb_path  = self.cache_dir / f"{cache_key}.npy"
        meta_path = self.cache_dir / f"{cache_key}.json"

        if emb_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            content_hash = self._content_hash(self._decls)
            if (meta.get("n") == len(self._decls)
                    and meta.get("content_hash") == content_hash
                    and meta.get("embed_model") == EMBED_MODEL):
                self._embeddings = np.load(str(emb_path))
                return
            print(f"[repo_retrieval] Cache for {self.repo_root.name} is stale "
                  f"(cached n={meta.get('n')}, live n={len(self._decls)}; "
                  f"content/embedding-model changed) - rebuilding.")

        # Build index
        print(f"[repo_retrieval] Building embedding index for {self.repo_root.name} "
              f"({len(self._decls)} declarations) ...")
        texts = [d.embed_text() for d in self._decls]
        self._embeddings = self._embed(texts)

        # Save cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(emb_path), self._embeddings)
        meta_path.write_text(json.dumps({
            "repo": str(self.repo_root),
            "n": len(self._decls),
            "embed_model": EMBED_MODEL,
            "content_hash": self._content_hash(self._decls),
        }))
        print(f"[repo_retrieval] Index saved to {emb_path}")

    @staticmethod
    def _content_hash(decls: list[RepoDecl]) -> str:
        """sha256 over every declaration's identity and signature text, so
        any content change (rename, signature edit, reorder) invalidates the
        cache even when the declaration count is unchanged."""
        h = hashlib.sha256()
        for d in decls:
            h.update(d.embed_text().encode("utf-8"))
        return h.hexdigest()[:16]

    def _cache_key(self, decls: list[RepoDecl]) -> str:
        h = hashlib.md5(str(self.repo_root.resolve()).encode()).hexdigest()[:12]
        return f"{self.repo_root.name}_{h}_{EMBED_MODEL.replace('/', '-')}_{self._content_hash(decls)}"

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
