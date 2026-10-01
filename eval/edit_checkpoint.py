"""Human intervention CLI on checkpoints (review IV.3).

    python eval/edit_checkpoint.py status   CHECKPOINT.json
    python eval/edit_checkpoint.py set-proof CHECKPOINT.json NODE \
        --file proof.txt [--reason "..."]        # or --stdin
    python eval/edit_checkpoint.py lock     CHECKPOINT.json NODE [--reason "..."]
    python eval/edit_checkpoint.py unlock   CHECKPOINT.json NODE
    python eval/edit_checkpoint.py retry    CHECKPOINT.json NODE [--reason "..."]

Every mutation is journaled in the checkpoint (operator, before/after,
reason) and surfaces in the result's autonomy label; human proofs still
have to pass the same independent final verification as model proofs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import interventions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status", help="list nodes, statuses, sources, locks")
    p.add_argument("checkpoint", type=Path)

    p = sub.add_parser("set-proof", help="inject a human-written proof")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("node")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="file containing the proof body (`by ...`; a leading ':=' is normalized away)")
    src.add_argument("--stdin", action="store_true")
    p.add_argument("--reason", default="")

    for name, help_ in (("lock", "protect a node from stale-proof invalidation"),
                        ("unlock", "remove the lock")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("checkpoint", type=Path)
        p.add_argument("node")
        p.add_argument("--reason", default="")

    p = sub.add_parser("retry", help="clear a node's cached result so the next run re-attempts it")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("node")
    p.add_argument("--reason", default="")

    args = ap.parse_args()
    state = interventions.load_for_edit(args.checkpoint)

    if args.cmd == "status":
        for row in interventions.status(state):
            print(f"{row['node']:28} {row['status']:14} "
                  f"source={row['source'] or '-':6} "
                  f"{'LOCKED' if row['locked'] else ''}")
        print(f"interventions: {len(state.interventions)}  "
              f"autonomy: {interventions.autonomy_label(state)}")
        return

    if args.cmd == "set-proof":
        body = sys.stdin.read() if args.stdin else Path(args.file).read_text()
        interventions.set_proof(state, args.node, body, reason=args.reason)
    elif args.cmd == "lock":
        interventions.lock(state, args.node, reason=args.reason)
    elif args.cmd == "unlock":
        interventions.unlock(state, args.node, reason=args.reason)
    elif args.cmd == "retry":
        interventions.retry(state, args.node, reason=args.reason)

    interventions.save_after_edit(state, args.checkpoint)
    print(f"[{args.cmd}] recorded on {args.node}; "
          f"autonomy now: {interventions.autonomy_label(state)}")


if __name__ == "__main__":
    main()
