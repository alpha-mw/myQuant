#!/usr/bin/env python3
"""Build or verify one explicit local factor-governance canonical replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.factors.governance_canonical_replay import (  # noqa: E402
    CanonicalReplayError,
    produce_canonical_replay,
    verify_canonical_replay,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit canonical replay inputs without search or fallback."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument(
        "--draft-path",
        help="Exact draft path to validate and publish; omit for readback only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run local-only canonical production or verification."""

    args = parse_args(argv)
    try:
        if args.draft_path is None:
            result = verify_canonical_replay(
                private_root=args.private_root,
                registry_path=args.registry_path,
            )
        else:
            result = produce_canonical_replay(
                private_root=args.private_root,
                registry_path=args.registry_path,
                draft_path=args.draft_path,
            )
    except (CanonicalReplayError, OSError, ValueError):
        print("factor_governance_canonical_replay_status=blocked", file=sys.stderr)
        print("blocker=canonical_local_byte_readback_failed", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
