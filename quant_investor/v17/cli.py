"""Thin, offline CLI adapters for the seven v17 market subcommands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .latest import repair_terminal_latest
from .risk_policy import seal_risk_policy_from_owner_mandate
from .runtime import (
    finalize_shadow_run_from_file,
    prepare_shadow_run_from_file,
    receive_shadow_response_from_file,
    shadow_status,
)
from .source_maintain import maintain_sources_from_plan_file


def run_source_maintain(
    *,
    repo_root: str | Path,
    plan_path: str | Path,
    expected_plan_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    manifest, path, manifest_sha = maintain_sources_from_plan_file(
        repo_root,
        plan_path,
        expected_plan_sha256=expected_plan_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    return {
        "manifest_id": manifest["manifest_id"],
        "manifest_path": path.relative_to(Path(repo_root).absolute()).as_posix(),
        "manifest_sha256": manifest_sha,
        "authority": False,
    }


def run_risk_policy_seal(
    *,
    repo_root: str | Path,
    owner_mandate_path: str | Path,
    output_path: str | Path,
    expected_owner_mandate_sha256: str,
    validation_cutoff: str,
) -> dict[str, Any]:
    root = Path(repo_root).absolute()
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    private_root = root / "data" / "private" / "v17_sources"
    policy, output_sha = seal_risk_policy_from_owner_mandate(
        owner_mandate_path,
        output,
        expected_owner_mandate_sha256=expected_owner_mandate_sha256,
        output_root=private_root,
        validation_cutoff=validation_cutoff,
    )
    return {
        "policy_id": policy["policy_id"],
        "availability": policy["availability"],
        "output_path": output.relative_to(root).as_posix(),
        "output_sha256": output_sha,
        "semantic_sha256": policy["semantic_sha256"],
        "authority": False,
    }


def run_shadow_prepare(
    *,
    repo_root: str | Path,
    request_path: str | Path,
    expected_request_sha256: str,
    expected_ledger_sha256: str,
) -> dict[str, Any]:
    return prepare_shadow_run_from_file(
        repo_root,
        request_path=request_path,
        expected_request_sha256=expected_request_sha256,
        expected_ledger_sha256=expected_ledger_sha256,
    )


def run_shadow_receive(
    *,
    repo_root: str | Path,
    run_id: str,
    response_path: str | Path,
    expected_response_sha256: str,
    expected_ledger_sha256: str,
    expected_latest_sha256: str,
    failed_at: str,
) -> dict[str, Any]:
    return receive_shadow_response_from_file(
        repo_root,
        run_id=run_id,
        response_path=response_path,
        expected_response_sha256=expected_response_sha256,
        expected_ledger_sha256=expected_ledger_sha256,
        expected_latest_sha256=expected_latest_sha256,
        failed_at=failed_at,
    )


def run_shadow_finalize(
    *,
    repo_root: str | Path,
    run_id: str,
    finalization_path: str | Path,
    expected_finalization_sha256: str,
    expected_ledger_sha256: str,
    expected_latest_sha256: str,
    failed_at: str,
) -> dict[str, Any]:
    return finalize_shadow_run_from_file(
        repo_root,
        run_id=run_id,
        finalization_path=finalization_path,
        expected_finalization_sha256=expected_finalization_sha256,
        expected_ledger_sha256=expected_ledger_sha256,
        expected_latest_sha256=expected_latest_sha256,
        failed_at=failed_at,
    )


def run_shadow_status(*, repo_root: str | Path, run_id: str) -> dict[str, Any]:
    return shadow_status(repo_root, run_id)


def run_shadow_latest_repair(
    *,
    repo_root: str | Path,
    run_id: str,
    expected_ledger_sha256: str,
    expected_latest_sha256: str,
    repaired_at: str,
) -> dict[str, Any]:
    pointer, pointer_sha = repair_terminal_latest(
        repo_root,
        run_id=run_id,
        expected_ledger_sha256=expected_ledger_sha256,
        expected_latest_sha256=expected_latest_sha256,
        repaired_at=repaired_at,
    )
    return {
        "run_id": run_id,
        "terminal_state": pointer["terminal_state"],
        "publication_mode": pointer["publication_mode"],
        "latest_sha256": pointer_sha,
        "authority": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-investor market")
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("v17-source-maintain")
    source.add_argument("--plan", required=True)
    source.add_argument("--expected-plan-sha256", required=True)
    source.add_argument("--expected-manifest-sha256", required=True)

    risk = commands.add_parser("v17-risk-policy-seal")
    risk.add_argument("--owner-mandate", required=True)
    risk.add_argument("--output", required=True)
    risk.add_argument("--expected-owner-mandate-sha256", required=True)
    risk.add_argument("--validation-cutoff", required=True)

    prepare = commands.add_parser("v17-shadow-prepare")
    prepare.add_argument("--request", required=True)
    prepare.add_argument("--expected-request-sha256", required=True)
    prepare.add_argument("--expected-ledger-sha256", required=True)

    receive = commands.add_parser("v17-shadow-receive")
    receive.add_argument("--run-id", required=True)
    receive.add_argument("--response", required=True)
    receive.add_argument("--expected-response-sha256", required=True)
    receive.add_argument("--expected-ledger-sha256", required=True)
    receive.add_argument("--expected-latest-sha256", required=True)
    receive.add_argument("--failed-at", required=True)

    finalize = commands.add_parser("v17-shadow-finalize")
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--finalization", required=True)
    finalize.add_argument("--expected-finalization-sha256", required=True)
    finalize.add_argument("--expected-ledger-sha256", required=True)
    finalize.add_argument("--expected-latest-sha256", required=True)
    finalize.add_argument("--failed-at", required=True)

    status = commands.add_parser("v17-shadow-status")
    status.add_argument("--run-id", required=True)

    repair = commands.add_parser("v17-shadow-latest-repair")
    repair.add_argument("--run-id", required=True)
    repair.add_argument("--expected-ledger-sha256", required=True)
    repair.add_argument("--expected-latest-sha256", required=True)
    repair.add_argument("--repaired-at", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: str | Path | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(repo_root).absolute() if repo_root is not None else Path.cwd().absolute()
    try:
        if args.command == "v17-source-maintain":
            result = run_source_maintain(
                repo_root=root,
                plan_path=args.plan,
                expected_plan_sha256=args.expected_plan_sha256,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
        elif args.command == "v17-risk-policy-seal":
            result = run_risk_policy_seal(
                repo_root=root,
                owner_mandate_path=args.owner_mandate,
                output_path=args.output,
                expected_owner_mandate_sha256=args.expected_owner_mandate_sha256,
                validation_cutoff=args.validation_cutoff,
            )
        elif args.command == "v17-shadow-prepare":
            result = run_shadow_prepare(
                repo_root=root,
                request_path=args.request,
                expected_request_sha256=args.expected_request_sha256,
                expected_ledger_sha256=args.expected_ledger_sha256,
            )
        elif args.command == "v17-shadow-receive":
            result = run_shadow_receive(
                repo_root=root,
                run_id=args.run_id,
                response_path=args.response,
                expected_response_sha256=args.expected_response_sha256,
                expected_ledger_sha256=args.expected_ledger_sha256,
                expected_latest_sha256=args.expected_latest_sha256,
                failed_at=args.failed_at,
            )
        elif args.command == "v17-shadow-finalize":
            result = run_shadow_finalize(
                repo_root=root,
                run_id=args.run_id,
                finalization_path=args.finalization,
                expected_finalization_sha256=args.expected_finalization_sha256,
                expected_ledger_sha256=args.expected_ledger_sha256,
                expected_latest_sha256=args.expected_latest_sha256,
                failed_at=args.failed_at,
            )
        elif args.command == "v17-shadow-status":
            result = run_shadow_status(repo_root=root, run_id=args.run_id)
        else:
            result = run_shadow_latest_repair(
                repo_root=root,
                run_id=args.run_id,
                expected_ledger_sha256=args.expected_ledger_sha256,
                expected_latest_sha256=args.expected_latest_sha256,
                repaired_at=args.repaired_at,
            )
    except (OSError, ValueError) as exc:
        print(f"v17 command failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "main",
    "run_risk_policy_seal",
    "run_shadow_finalize",
    "run_shadow_latest_repair",
    "run_shadow_prepare",
    "run_shadow_receive",
    "run_shadow_status",
    "run_source_maintain",
]
