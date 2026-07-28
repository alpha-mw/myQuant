"""Dedicated, offline CLI for the V17 protocol-v3 research runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from quant_investor.v17_v3_contract.identities import require_sha256

from .activation import (
    ACTIVE,
    ActivationPublisher,
)
from .authority import PROTOCOL_VERSION, authority_envelope
from .redaction import assert_public_envelope_safe
from .service import (
    admitted_sources,
    analyze,
    build_initial_pool,
    calibrate,
    status,
    verify_runtime,
)
from .storage import SecureStore


def _emit(payload: dict[str, Any]) -> None:
    assert_public_envelope_safe(payload)
    sys.stdout.write(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _error_payload(exc: BaseException) -> dict[str, Any]:
    """Return an allowlisted error DTO without raw exception text."""

    payload = {
        "version": f"{PROTOCOL_VERSION}.cli-error.v1",
        "status": "BLOCKED",
        "error": type(exc).__name__,
        **authority_envelope(),
    }
    assert_public_envelope_safe(payload)
    return payload


def _workspace_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-root", default=str(Path.cwd()))


def _locator_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-locator", required=True)
    parser.add_argument("--expected-source-locator-sha256", required=True)
    _workspace_argument(parser)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-investor-v17-v3",
        description=(
            "isolated V17 v3 formal-research/shadow runtime; "
            "no production default, provider, execution, broker, order, or trade authority"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="verify exact V3 package and Phase-A boundary")

    admit = commands.add_parser(
        "admit-sources",
        help="admit one exact canonical source locator and role closure",
    )
    _locator_arguments(admit)
    calibration = commands.add_parser(
        "calibrate-fusion",
        help="run offline fusion calibration from admitted PIT sources",
    )
    _locator_arguments(calibration)
    preselect = commands.add_parser(
        "build-initial-pool",
        help="replay one exact PRESELECT locator and persist its immutable pool",
    )
    _locator_arguments(preselect)

    activation = commands.add_parser(
        "activate-formal-research",
        help=(
            "activate one exact typed PROMOTED receipt and formal candidate "
            "after validating their complete transitive closure"
        ),
    )
    activation.add_argument("--promotion-receipt", required=True)
    activation.add_argument("--expected-promotion-receipt-sha256", required=True)
    activation.add_argument("--formal-output")
    activation.add_argument("--expected-formal-output-sha256")
    _workspace_argument(activation)

    analysis = commands.add_parser(
        "analyze",
        help="run admitted-source-only shadow or formal-research analysis",
    )
    analysis.add_argument(
        "--mode",
        choices=("shadow", "formal-research"),
        required=True,
    )
    _locator_arguments(analysis)

    revoke = commands.add_parser(
        "revoke-formal-research",
        help="terminally revoke one exact ACTIVE formal-research cutoff",
    )
    revoke.add_argument("--strategy-id", required=True)
    revoke.add_argument("--cutoff", required=True)
    revoke.add_argument("--expected-active-receipt-sha256", required=True)
    revoke.add_argument("--reason", required=True)
    _workspace_argument(revoke)

    read_status = commands.add_parser(
        "status",
        help="revalidate current ACTIVE formal-research result",
    )
    read_status.add_argument("--strategy-id", required=True)
    _workspace_argument(read_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            _emit(verify_runtime().to_public_wire())
            return 0
        if args.command == "admit-sources":
            _, admission = admitted_sources(
                workspace_root=Path(args.workspace_root),
                locator_path=args.source_locator,
                expected_locator_sha256=args.expected_source_locator_sha256,
            )
            _emit(admission.to_public_wire())
            return 0
        if args.command == "calibrate-fusion":
            calibration_outcome = calibrate(
                workspace_root=Path(args.workspace_root),
                locator_path=args.source_locator,
                expected_locator_sha256=args.expected_source_locator_sha256,
            )
            _emit(calibration_outcome.to_public_wire())
            return 0 if calibration_outcome.promoted else 2
        if args.command == "build-initial-pool":
            _emit(
                build_initial_pool(
                    workspace_root=Path(args.workspace_root),
                    locator_path=args.source_locator,
                    expected_locator_sha256=(args.expected_source_locator_sha256),
                ).to_public_wire()
            )
            return 0
        if args.command == "activate-formal-research":
            store = SecureStore(Path(args.workspace_root))
            expected = require_sha256(
                args.expected_promotion_receipt_sha256,
                label="expected promotion receipt SHA-256",
            )
            promotion = store.read_path(args.promotion_receipt, expected)
            from quant_investor.v17_v3_contract.canonical import load_canonical_resource

            document = load_canonical_resource(
                promotion,
                label="promotion receipt",
            )
            if type(document) is not dict:
                raise ValueError("promotion receipt root must be an object")
            formal_output: bytes | None = None
            if args.formal_output is not None:
                if args.expected_formal_output_sha256 is None:
                    raise ValueError("formal output SHA-256 is required with formal output")
                formal_output = store.read_path(
                    args.formal_output,
                    require_sha256(
                        args.expected_formal_output_sha256,
                        label="expected formal output SHA-256",
                    ),
                )
            activation_outcome = ActivationPublisher(store).activate(
                strategy_id=document.get("strategy_id"),
                cutoff=document.get("cutoff"),
                promotion_receipt_bytes=promotion,
                promotion_receipt_path=args.promotion_receipt,
                expected_promotion_receipt_sha256=expected,
                formal_output_bytes=formal_output,
                formal_output_path=args.formal_output,
                expected_formal_output_sha256=(args.expected_formal_output_sha256),
            )
            _emit(activation_outcome.to_public_wire())
            return 0 if activation_outcome.status == ACTIVE else 2
        if args.command == "analyze":
            analysis_outcome = analyze(
                workspace_root=Path(args.workspace_root),
                mode=args.mode,
                locator_path=args.source_locator,
                expected_locator_sha256=args.expected_source_locator_sha256,
            )
            _emit(analysis_outcome.to_public_wire())
            return 2 if analysis_outcome.result.terminal.state.startswith("HARD_STOP_") else 0
        if args.command == "revoke-formal-research":
            revocation_outcome = ActivationPublisher(SecureStore(Path(args.workspace_root))).revoke(
                strategy_id=args.strategy_id,
                cutoff=args.cutoff,
                expected_active_receipt_sha256=args.expected_active_receipt_sha256,
                reason=args.reason,
            )
            _emit(revocation_outcome.to_public_wire())
            return 0 if revocation_outcome.status == "REVOKED" else 2
        if args.command == "status":
            _emit(
                status(
                    workspace_root=Path(args.workspace_root),
                    strategy_id=args.strategy_id,
                ).to_public_wire()
            )
            return 0
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        _emit(_error_payload(exc))
        return 2
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
