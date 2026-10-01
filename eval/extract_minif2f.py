"""Build the {test,valid}.jsonl files run_minif2f.py expects from the
yangky11/miniF2F-lean4 mirror (one theorem per file, `:= by sorry` tail).

The official openai/miniF2F repo ships only Lean 3 on its default branch;
the Lean 4 statements live in community mirrors with per-problem .lean
files instead of jsonl. This extractor keeps the full preamble (imports,
set_option, opens) before each theorem so statements elaborate in the
same environment the mirror was built against.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
MIRROR = ROOT / "data" / "minif2f"

_THEOREM_RE = re.compile(r"^theorem\s+(\w+)", re.MULTILINE)


def extract(split: str) -> list[dict]:
    problems = []
    for f in sorted((MIRROR / "MiniF2F" / split.capitalize()).glob("*.lean")):
        text = f.read_text()
        m = _THEOREM_RE.search(text)
        if not m:
            continue
        name = m.group(1)
        # statement = preamble + theorem up to and including the sorry tail
        idx = m.start()
        preamble = text[:idx].strip()
        stmt = text[idx:].strip()
        problems.append({
            "name": name,
            "formal_statement": f"{preamble}\n\n{stmt}" if preamble else stmt,
            "split": split,
        })
    return problems


def main() -> None:
    for split in ("test", "valid"):
        problems = extract(split)
        out = MIRROR / f"{split}.jsonl"
        with open(out, "w") as f:
            for p in problems:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"{split}: {len(problems)} problems -> {out}")


if __name__ == "__main__":
    sys.exit(main())
