"""Public wrapper adding ``research-evaluate`` without changing V4 sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


class _ArgumentContractError(ValueError):
    pass


class _ResearchEvaluateParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentContractError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _ResearchEvaluateParser(prog="quant-investor-v17-v4 research-evaluate")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--request-path", required=True)
    parser.add_argument("--request-sha256", required=True)
    return parser


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_once(value: dict[str, Any], *, maximum: int) -> bool:
    raw = _canonical_bytes(value)
    within_limit = len(raw) <= maximum
    if len(raw) > maximum:
        from .receipts import blocked_envelope

        raw = _canonical_bytes(blocked_envelope(status="BLOCKED", blocker_code="limit_exceeded"))
    sys.stdout.buffer.write(raw)
    return within_limit


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch R2.2 exactly; delegate every legacy argument byte-for-byte."""

    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] != "research-evaluate":
        from quant_investor.v17_v4_runtime.cli_provisional import main as legacy_main

        return legacy_main(values)

    from .._core import IntelligenceContractError
    from .forward_evaluator import (
        ForwardEvaluationError,
        ImplementationIntegrityError,
        run_forward_research_evaluation,
    )
    from .receipts import MAX_ENVELOPE_BYTES, blocked_envelope

    try:
        args = _parser().parse_args(values[1:])
        workspace_root = Path(args.workspace_root)
        if not workspace_root.is_absolute():
            raise _ArgumentContractError("workspace-root must be absolute")
        result = run_forward_research_evaluation(
            str(workspace_root),
            request_path=args.request_path,
            request_sha256=args.request_sha256,
        )
        return 0 if _write_once(result, maximum=MAX_ENVELOPE_BYTES) else 2
    except _ArgumentContractError:
        _write_once(
            blocked_envelope(status="BLOCKED", blocker_code="argument_invalid"),
            maximum=MAX_ENVELOPE_BYTES,
        )
        return 2
    except ImplementationIntegrityError:
        _write_once(
            blocked_envelope(
                status="INTERNAL_ERROR",
                blocker_code="implementation_integrity_error",
            ),
            maximum=MAX_ENVELOPE_BYTES,
        )
        sys.stderr.write("research-evaluate internal error\n")
        return 3
    except ForwardEvaluationError as exc:
        _write_once(
            blocked_envelope(
                status="BLOCKED",
                blocker_code=exc.code,
                preserved_artifact_refs=exc.preserved_artifact_refs,
            ),
            maximum=MAX_ENVELOPE_BYTES,
        )
        return 2
    except IntelligenceContractError:
        _write_once(
            blocked_envelope(status="BLOCKED", blocker_code="evaluation_blocked"),
            maximum=MAX_ENVELOPE_BYTES,
        )
        return 2
    except Exception:
        _write_once(
            blocked_envelope(status="INTERNAL_ERROR", blocker_code="internal_error"),
            maximum=MAX_ENVELOPE_BYTES,
        )
        sys.stderr.write("research-evaluate internal error\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
