#!/usr/bin/env python3
"""Retired v2 replay-evidence CLI; v3 evidence never auto-upgrades v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.factors.governance_protocol_v3 import (  # noqa: E402
    FORWARD_PRODUCTION_APPLY_BLOCKER,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-chain-replay-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    del args
    print(
        "factor_governance_evidence_blocked="
        "factor-governance-replay-evidence.v2_is_retired",
        file=sys.stderr,
    )
    print(f"production_apply_blocker={FORWARD_PRODUCTION_APPLY_BLOCKER}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
