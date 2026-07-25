#!/usr/bin/env python3
"""Export the exact validated v17 shadow latest state to a private dashboard loader."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
from typing import Any

from quant_investor.v17.contracts import (
    V17ContractError,
    require_authority_false,
    require_exact_keys,
)
from quant_investor.v17.latest import read_latest_pointer
from quant_investor.v17.semantic import validate_semantic_seal
from quant_investor.v17.state_machine import (
    TERMINAL_OUTPUT_KEYS,
    TERMINAL_OUTPUT_VERSION,
)
from quant_investor.v17.storage import atomic_write_bytes, file_sha256, read_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_SCHEMA_VERSION = "dashboard_contract.v17-shadow.v1"
LATEST_RELATIVE_PATH = PurePosixPath("results/v17_shadow/_latest/shadow.json")
DEFAULT_SCHEMA_RELATIVE_PATH = PurePosixPath(
    "portfolio_dashboard/schema/dashboard_contract.v17-shadow.schema.json"
)
DEFAULT_OUTPUT_RELATIVE_PATH = PurePosixPath("portfolio_dashboard/generated/v17_shadow_latest.js")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_metadata(
    *,
    latest_pointer_sha256: str | None,
    ledger_sha256: str | None,
    output_sha256: str | None,
    readback_verified: bool,
) -> dict[str, Any]:
    return {
        "path": LATEST_RELATIVE_PATH.as_posix(),
        "latest_pointer_sha256": latest_pointer_sha256,
        "ledger_sha256": ledger_sha256,
        "output_sha256": output_sha256,
        "readback_verified": readback_verified,
        "fallback_used": False,
    }


def _unavailable_contract(
    *,
    schema_sha256: str,
    generated_at: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "schema_sha256": schema_sha256,
        "availability": "UNAVAILABLE",
        "generated_at": generated_at,
        "reason": reason,
        "source": _source_metadata(
            latest_pointer_sha256=None,
            ledger_sha256=None,
            output_sha256=None,
            readback_verified=False,
        ),
        "latest_pointer": None,
        "terminal_output": None,
        "authority": False,
    }


def build_v17_shadow_dashboard_contract(
    *,
    repo_root: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a read-only contract from one exact, hash-verified latest pointer.

    Missing or invalid latest evidence is represented as ``UNAVAILABLE``.  No
    legacy, sample, synthetic, or inferred source is consulted.
    """

    root = Path(repo_root).absolute()
    schema_path = root / Path(*DEFAULT_SCHEMA_RELATIVE_PATH.parts)
    schema_sha = file_sha256(schema_path)
    timestamp = generated_at or _utc_now()

    try:
        latest = read_latest_pointer(root, verify_targets=True)
    except (OSError, ValueError, V17ContractError):
        return _unavailable_contract(
            schema_sha256=schema_sha,
            generated_at=timestamp,
            reason="v17_latest_pointer_invalid",
        )
    if latest is None:
        return _unavailable_contract(
            schema_sha256=schema_sha,
            generated_at=timestamp,
            reason="v17_latest_pointer_missing",
        )

    pointer, pointer_sha = latest
    output_path = root / Path(*PurePosixPath(pointer["output_path"]).parts)
    try:
        before = file_sha256(output_path)
        output = validate_semantic_seal(read_json(output_path))
        after = file_sha256(output_path)
        if before != after or before != pointer["output_sha256"]:
            raise V17ContractError("v17 terminal output changed during dashboard export")
        require_exact_keys(
            output,
            TERMINAL_OUTPUT_KEYS,
            label="dashboard terminal output",
        )
        if output.get("version") != TERMINAL_OUTPUT_VERSION:
            raise V17ContractError("dashboard terminal output version mismatch")
        if (
            output.get("run_id") != pointer["run_id"]
            or output.get("terminal_state") != pointer["terminal_state"]
        ):
            raise V17ContractError("dashboard latest/output identity mismatch")
        require_authority_false(output.get("authority"))
    except (OSError, ValueError, V17ContractError):
        return _unavailable_contract(
            schema_sha256=schema_sha,
            generated_at=timestamp,
            reason="v17_terminal_output_invalid",
        )

    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "schema_sha256": schema_sha,
        "availability": "AVAILABLE",
        "generated_at": timestamp,
        "reason": None,
        "source": _source_metadata(
            latest_pointer_sha256=pointer_sha,
            ledger_sha256=pointer["ledger_sha256"],
            output_sha256=pointer["output_sha256"],
            readback_verified=True,
        ),
        "latest_pointer": pointer,
        "terminal_output": output,
        "authority": False,
    }


def dashboard_loader_bytes(contract: dict[str, Any]) -> bytes:
    encoded = (
        json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return ('"use strict";\n' f"window.V17ShadowLatest = {encoded};\n").encode("utf-8")


def export_v17_shadow_dashboard(
    *,
    repo_root: str | Path,
    output_path: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).absolute()
    target = (
        Path(output_path).absolute()
        if output_path is not None
        else root / Path(*DEFAULT_OUTPUT_RELATIVE_PATH.parts)
    )
    output_root = target.parent
    contract = build_v17_shadow_dashboard_contract(
        repo_root=root,
        generated_at=generated_at,
    )
    loader_sha = atomic_write_bytes(
        target,
        dashboard_loader_bytes(contract),
        root=output_root,
    )
    return {
        "availability": contract["availability"],
        "reason": contract["reason"],
        "output_path": str(target),
        "output_sha256": loader_sha,
        "source": contract["source"],
        "authority": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export exact v17 shadow latest evidence for the read-only dashboard",
    )
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output", default="")
    parser.add_argument("--generated-at", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = export_v17_shadow_dashboard(
        repo_root=args.repo_root,
        output_path=args.output or None,
        generated_at=args.generated_at or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["availability"] == "AVAILABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
