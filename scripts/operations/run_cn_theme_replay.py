#!/usr/bin/env python3
"""Replay one Top100 Theme day through DC primary and registered TDX fallback."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

import quant_investor
from quant_investor.intelligence._common import validate_stable_artifact
from quant_investor.intelligence.daily import validate_daily_research_policy
from quant_investor.market.credential_preflight import read_project_env_token
from quant_investor.market.tushare import (
    TushareContractError,
    build_theme_provider_execution_plan,
    derive_tdx_fallback_company_keyset,
)
from quant_investor.market.tushare._core import canonical_bytes
from quant_investor.market.tushare.theme_runtime import (
    ThemeCaptureSafetyError,
    capture_theme_plan,
    load_capture_root,
    load_exact_plan,
    write_exact,
)
from quant_investor.market.tushare_transport import OfficialTushareHttpsClient


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load_exact(
    *, workspace: Path, relative_path: str, expected_sha256: str, label: str
) -> tuple[dict[str, Any], bytes]:
    path = (workspace / relative_path).resolve(strict=True)
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise ThemeCaptureSafetyError(f"{label}_PATH_INVALID") from exc
    observed = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
    ):
        raise ThemeCaptureSafetyError(f"{label}_FILE_INVALID")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ThemeCaptureSafetyError(f"{label}_SHA_MISMATCH")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ThemeCaptureSafetyError(f"{label}_JSON_INVALID") from exc
    if type(value) is not dict or canonical_bytes(value) != raw:
        raise ThemeCaptureSafetyError(f"{label}_NOT_CANONICAL")
    return value, raw


def _owner_dir(path: Path) -> None:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    observed = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) & 0o022
    ):
        raise ThemeCaptureSafetyError("THEME_REPLAY_DIRECTORY_INVALID")


def _plan(
    *,
    path: Path,
    provider: str,
    trade_date: str,
    companies: list[str],
    document_observed_at: str,
    created_at: str,
) -> tuple[dict[str, Any], str]:
    if path.exists():
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        plan = load_exact_plan(path, digest)
    else:
        plan = build_theme_provider_execution_plan(
            provider=provider,
            trade_date=trade_date,
            company_keyset=companies,
            document_observed_at=document_observed_at,
            created_at=created_at,
        )
        digest = write_exact(path, plan)
    if (
        plan["provider"] != provider
        or plan["trade_date"] != trade_date
        or plan["company_keyset"] != companies
    ):
        raise ThemeCaptureSafetyError("THEME_REPLAY_PLAN_BINDING_MISMATCH")
    return plan, digest


def run(
    args: argparse.Namespace,
    *,
    client: Any | None = None,
    now: Callable[[], str] = _now,
    token_reader: Callable[[Path], str] = read_project_env_token,
) -> dict[str, Any]:
    workspace = Path(args.workspace_root).resolve(strict=True)
    selected_value, _selected_raw = _load_exact(
        workspace=workspace,
        relative_path=args.selected_symbols,
        expected_sha256=args.expected_selected_symbols_sha256,
        label="THEME_SELECTED_SYMBOLS",
    )
    selected = validate_stable_artifact(
        selected_value, expected_kind="daily_research_selected_symbols"
    )
    selected_payload = selected["payload"]
    trade_date = selected_payload["signal_date"]
    companies = sorted(selected_payload["ordered_symbols"], key=lambda value: value.encode("ascii"))
    if (
        selected_payload["symbol_count"] != len(companies)
        or len(companies) != len(set(companies))
        or trade_date < "20260827"
    ):
        raise ThemeCaptureSafetyError("THEME_SELECTED_SYMBOLS_INVALID")
    policy_value, _policy_raw = _load_exact(
        workspace=workspace,
        relative_path=args.policy,
        expected_sha256=args.expected_policy_sha256,
        label="THEME_POLICY",
    )
    policy = validate_daily_research_policy(policy_value)
    policy_payload = policy["payload"]
    if (
        policy_payload["strategy_id"] != selected_payload["strategy_id"]
        or policy_payload["technology_policy_state"] != "ACTIVE"
        or policy_payload["effective_signal_date"] > trade_date
        or policy_payload["theme_provider_precedence"] != ["TUSHARE_DC", "TUSHARE_TDX"]
    ):
        raise ThemeCaptureSafetyError("THEME_POLICY_BINDING_INVALID")

    expected_root = Path(args.expected_import_root).resolve(strict=True)
    import_origin = Path(quant_investor.__file__).resolve(strict=True)
    if expected_root != import_origin and expected_root not in import_origin.parents:
        raise ThemeCaptureSafetyError("THEME_IMPORT_ORIGIN_MISMATCH")

    suffix = selected_payload["symbol_set_sha256"][:12]
    theme_root = workspace / "data/private/intelligence_sources/theme"
    plans_root = theme_root / "plans"
    dc_parent = theme_root / "dc"
    tdx_parent = theme_root / "tdx"
    replay_root = theme_root / "replays"
    for path in (theme_root, plans_root, dc_parent, tdx_parent, replay_root):
        _owner_dir(path)

    created_at = now()
    document_observed_at = min(created_at, str(policy["created_at"]))
    dc_plan_path = plans_root / f"dc-{trade_date}-{suffix}.json"
    dc_plan, dc_plan_sha = _plan(
        path=dc_plan_path,
        provider="TUSHARE_DC",
        trade_date=trade_date,
        companies=companies,
        document_observed_at=document_observed_at,
        created_at=created_at,
    )
    dc_output = dc_parent / f"dc-theme-{trade_date}-{suffix}"

    token: str | None = None
    transport = client
    if args.allow_live and transport is None:
        token = token_reader(workspace / ".env")
        os.environ["TUSHARE_TOKEN"] = token
        os.environ["TUSHARE_URL"] = "https://api.tushare.pro/dataapi"
        transport = OfficialTushareHttpsClient(strict_decimal_decode=True)
    try:
        dc_summary = capture_theme_plan(
            plan_path=dc_plan_path,
            plan_sha256=dc_plan_sha,
            output_root=dc_output,
            allow_live=bool(args.allow_live),
            resume=dc_output.exists(),
            client=transport,
            now=now,
        )
        if not args.allow_live:
            return {
                "credential_source": "PROJECT_ENV",
                "dc": dc_summary,
                "network_attempts": 0,
                "status": "DRY_RUN_VALIDATED",
                "token_hash_recorded": False,
                "token_material_recorded": False,
            }
        dc_loaded_plan, dc_capture, dc_partitions = load_capture_root(dc_output)
        if (
            dc_loaded_plan != dc_plan
            or dc_capture["status"] not in {"COMPLETE", "PARTIAL"}
            or dc_partitions[0]["status"] == "INCOMPLETE"
        ):
            raise ThemeCaptureSafetyError("THEME_DC_CAPTURE_INCOMPLETE")
        fallback = derive_tdx_fallback_company_keyset(
            dc_plan=dc_plan,
            dc_capture=dc_capture,
            dc_partition_documents=dc_partitions,
        )
        tdx_summary = None
        tdx_plan_path = None
        tdx_plan_sha = None
        tdx_output = None
        if fallback:
            tdx_plan_path = plans_root / f"tdx-{trade_date}-{suffix}.json"
            _tdx_plan, tdx_plan_sha = _plan(
                path=tdx_plan_path,
                provider="TUSHARE_TDX",
                trade_date=trade_date,
                companies=fallback,
                document_observed_at=document_observed_at,
                created_at=created_at,
            )
            tdx_output = tdx_parent / f"tdx-theme-{trade_date}-{suffix}"
            tdx_summary = capture_theme_plan(
                plan_path=tdx_plan_path,
                plan_sha256=tdx_plan_sha,
                output_root=tdx_output,
                allow_live=True,
                resume=tdx_output.exists(),
                client=transport,
                now=now,
            )
            _loaded_plan, tdx_capture, _tdx_partitions = load_capture_root(tdx_output)
            if tdx_capture["status"] != "COMPLETE":
                raise ThemeCaptureSafetyError("THEME_TDX_CAPTURE_INCOMPLETE")

        receipt = {
            "schema_id": "cn-theme-daily-replay.v1",
            "trade_date": trade_date,
            "strategy_id": selected_payload["strategy_id"],
            "company_count": len(companies),
            "company_set_sha256": selected_payload["symbol_set_sha256"],
            "credential_source": "PROJECT_ENV",
            "env_file": ".env",
            "env_key": "TUSHARE_TOKEN",
            "token_material_recorded": False,
            "token_hash_recorded": False,
            "dc_plan_ref": {
                "path": dc_plan_path.relative_to(workspace).as_posix(),
                "sha256": dc_plan_sha,
            },
            "dc_capture_ref": {
                "path": (dc_output / "capture.json").relative_to(workspace).as_posix(),
                "sha256": hashlib.sha256((dc_output / "capture.json").read_bytes()).hexdigest(),
            },
            "fallback_company_keyset": fallback,
            "tdx_plan_ref": (
                None
                if tdx_plan_path is None
                else {
                    "path": tdx_plan_path.relative_to(workspace).as_posix(),
                    "sha256": tdx_plan_sha,
                }
            ),
            "tdx_capture_ref": (
                None
                if tdx_output is None
                else {
                    "path": (tdx_output / "capture.json").relative_to(workspace).as_posix(),
                    "sha256": hashlib.sha256(
                        (tdx_output / "capture.json").read_bytes()
                    ).hexdigest(),
                }
            ),
            "status": "COMPLETE",
            "broker": False,
            "order": False,
            "execution": False,
            "trade": False,
        }
        receipt_path = replay_root / f"theme-replay-{trade_date}-{suffix}.json"
        receipt_sha = write_exact(receipt_path, receipt)
        return {
            "command_status": "COMPLETE",
            "dc": dc_summary,
            "fallback_company_count": len(fallback),
            "receipt_path": receipt_path.relative_to(workspace).as_posix(),
            "receipt_sha256": receipt_sha,
            "tdx": tdx_summary,
            "token_hash_recorded": False,
            "token_material_recorded": False,
        }
    finally:
        if token is not None:
            token = None
            os.environ.pop("TUSHARE_TOKEN", None)
            os.environ.pop("TUSHARE_URL", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--expected-import-root", required=True)
    parser.add_argument("--selected-symbols", required=True)
    parser.add_argument("--expected-selected-symbols-sha256", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--allow-live", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        result = run(parse_args())
    except (ThemeCaptureSafetyError, TushareContractError, OSError, ValueError):
        print(
            json.dumps(
                {
                    "blocker": "THEME_DAILY_REPLAY_BLOCKED",
                    "status": "BLOCKED",
                    "token_hash_recorded": False,
                    "token_material_recorded": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
