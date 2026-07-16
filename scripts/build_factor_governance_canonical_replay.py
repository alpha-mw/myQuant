#!/usr/bin/env python3
"""Retired v1 canonical-replay CLI; Factor v3 rejects legacy replay graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    del args
    print("factor_governance_canonical_replay_status=blocked", file=sys.stderr)
    print("blocker=legacy_canonical_replay_v1_retired", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
