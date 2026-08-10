from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pytest


def load_script() -> Any:
    path = Path("scripts/run_tushare_vip_fundamental_upgrade.py").resolve()
    spec = importlib.util.spec_from_file_location("vip_upgrade_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(tmp_path: Path, *, execute: bool, allow_live: bool) -> argparse.Namespace:
    return argparse.Namespace(
        allow_live=allow_live,
        as_of="20260807",
        attempt_id="vip-upgrade-20260809-001" if execute else None,
        canonical_root=str(tmp_path / "canonical"),
        execute=execute,
        expected_fundamental_pointer_sha256="a" * 64 if execute else None,
        journal_root=str(tmp_path / "attempts" / "attempt-1") if execute else None,
        policy_path=str(tmp_path / "policy.json"),
        policy_sha256="b" * 64,
        scope_path=str(tmp_path / "scope.json"),
        scope_sha256="c" * 64,
        staging_root=str(tmp_path / "staging"),
    )


def policy() -> dict[str, Any]:
    return {
        "request_plan": {
            "planned_max_network_attempts": 2800,
            "planned_terminal_request_count": 1400,
        }
    }


def test_dry_run_never_reads_pointer_package_token_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    monkeypatch.setattr(
        module,
        "_validate_inputs",
        lambda _args: (policy(), tmp_path / "staging", tmp_path / "canonical"),
    )
    monkeypatch.setattr(
        module,
        "pointer_sha256",
        lambda _root: (_ for _ in ()).throw(AssertionError("pointer read")),
    )
    monkeypatch.setattr(
        module,
        "verify_package",
        lambda: (_ for _ in ()).throw(AssertionError("package read")),
    )
    monkeypatch.setenv("TUSHARE_TOKEN", "SECRET_CANARY_MUST_NOT_BE_READ")

    result = module.run(args(tmp_path, execute=False, allow_live=False))
    assert result == {
        "as_of": "20260807",
        "execute": False,
        "planned_max_network_attempts": 2800,
        "planned_terminal_request_count": 1400,
        "status": "DRY_RUN_VALIDATED",
    }


def test_execute_requires_both_live_authority_and_expected_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    monkeypatch.setattr(
        module,
        "_validate_inputs",
        lambda _args: (policy(), tmp_path / "staging", tmp_path / "canonical"),
    )
    with pytest.raises(module.UpgradeSafetyError, match="AUTHORITY_MISSING"):
        module.run(args(tmp_path, execute=True, allow_live=False))
