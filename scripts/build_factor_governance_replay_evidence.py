#!/usr/bin/env python3
"""Build report-only normalized FactorGovernanceProtocol v2 replay evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.factors.governance_evidence import (  # noqa: E402
    produce_governance_replay_evidence,
    write_governance_replay_evidence,
)
from quant_investor.factors.governance_protocol_v2 import (  # noqa: E402
    canonical_replay_producer_control,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-chain-replay-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = Path(args.full_chain_replay_json).expanduser()
    output = Path(args.output_json).expanduser()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
        evidence = produce_governance_replay_evidence(raw)
        write_governance_replay_evidence(output, evidence)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"factor_governance_evidence_blocked={exc}", file=sys.stderr)
        return 2
    print(f"evidence_json={output}")
    print(f"evidence_hash={evidence['evidence_hash']}")
    control = canonical_replay_producer_control()
    print("production_apply_eligible=false")
    print(f"production_apply_blocker={control['blocker']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
