"""Additive dispatcher for the V17 v4 provisional forward command.

The legacy CLI implementation is content-addressed and remains byte-for-byte
unchanged.  Every command except ``run-provisional-forward`` is delegated to
that implementation with the original arguments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from . import cli as legacy_cli
from .provisional_forward import (
    NO_AUTHORITY,
    ProvisionalForwardError,
    run_provisional_forward,
)


def _daily_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-investor-v17-v4 research-daily-run")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--request-path", required=True)
    parser.add_argument("--request-sha256", required=True)
    return parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-investor-v17-v4 run-provisional-forward")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--request-path", required=True)
    parser.add_argument("--request-sha256", required=True)
    return parser


def _emit(value: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "research-daily-run":
        from quant_investor.research.research_scheduler import (
            ResearchLoopError,
            run_daily_research_loop,
        )

        args = _daily_parser().parse_args(values[1:])
        try:
            _emit(
                run_daily_research_loop(
                    str(Path(args.workspace_root).resolve()),
                    request_path=args.request_path,
                    request_sha256=args.request_sha256,
                )
            )
            return 0
        except ResearchLoopError as exc:
            _emit(
                {
                    "authority": dict(NO_AUTHORITY),
                    "blocker_code": exc.code,
                    "broker": False,
                    "default_protocol_state": "V15_DEFAULT",
                    "execution": False,
                    "factor_governance_write": False,
                    "global_activation_state": "INACTIVE",
                    "order": False,
                    "preserved_artifact_refs": list(exc.preserved_artifact_refs),
                    "production_governance_eligible": False,
                    "provider_calls": False,
                    "research_only": True,
                    "research_runtime_default": False,
                    "run_state": "RUN_BLOCKED",
                    "status": "BLOCKED",
                    "trade": False,
                }
            )
            return exc.exit_code
    if not values or values[0] != "run-provisional-forward":
        return legacy_cli.main(values)
    args = _parser().parse_args(values[1:])
    try:
        _emit(
            run_provisional_forward(
                str(Path(args.workspace_root).resolve()),
                request_path=args.request_path,
                request_sha256=args.request_sha256,
            )
        )
        return 0
    except ProvisionalForwardError as exc:
        _emit(
            {
                "authority": dict(NO_AUTHORITY),
                "blocker_code": exc.code,
                "default_protocol_state": "V15_DEFAULT",
                "factor_governance_write": False,
                "formal_activation_eligible": False,
                "global_activation_state": "INACTIVE",
                "preserved_artifact_refs": list(exc.preserved_artifact_refs),
                "production_governance_eligible": False,
                "promotion_eligible": False,
                "provider_calls": False,
                "research_runtime_available": True,
                "research_runtime_default": False,
                "run_state": "BLOCKED",
                "selector": False,
                "status": "BLOCKED",
            }
        )
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
