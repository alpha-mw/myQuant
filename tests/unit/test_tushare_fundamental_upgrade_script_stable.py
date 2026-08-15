from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_tushare_vip_fundamental_upgrade.py"


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "tushare_fundamental_upgrade_stable",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(tmp_path: Path, *, execute: bool) -> argparse.Namespace:
    return argparse.Namespace(
        as_of="20260807",
        canonical_root=str(tmp_path / "canonical"),
        execute=execute,
        execution_closure_path=str(tmp_path / "execution.json"),
        execution_closure_sha256="a" * 64,
        scope_path=str(tmp_path / "scope.json"),
        scope_sha256="b" * 64,
        staging_root=str(tmp_path / "staging"),
    )


def execution() -> dict[str, Any]:
    return {
        "closure_id": "c" * 64,
        "contract_sha256": "d" * 64,
        "request_plan": {
            "planned_max_network_attempts": 2800,
            "planned_terminal_request_count": 1400,
        },
    }


def test_dry_run_only_reports_stable_preflight_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    staging = tmp_path / "staging"
    canonical = tmp_path / "canonical"
    monkeypatch.setattr(
        module,
        "_validate_inputs",
        lambda _args: (execution(), staging, canonical),
    )

    result = module.run(args(tmp_path, execute=False))

    assert result == {
        "as_of": "20260807",
        "execute": False,
        "execution_closure_id": "c" * 64,
        "execution_contract_sha256": "d" * 64,
        "planned_max_network_attempts": 2800,
        "planned_terminal_request_count": 1400,
        "status": "DRY_RUN_VALIDATED",
    }
    assert not staging.exists()


def test_execute_fails_closed_for_unified_activation_without_pointer_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    monkeypatch.setattr(
        module,
        "_validate_inputs",
        lambda _args: (execution(), tmp_path / "staging", tmp_path / "canonical"),
    )

    with pytest.raises(
        module.UpgradeSafetyError,
        match="VIP_UPGRADE_REQUIRES_UNIFIED_ACTIVATION",
    ):
        module.run(args(tmp_path, execute=True))


def test_market_upgrade_import_boundary_has_no_legacy_or_pointer_promotion() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    retired_modules = tuple(
        ".".join(("quant_investor", name))
        for name in (
            "".join(("intelligence", "_v2")),
            "".join(("v17", "_v4_runtime")),
            "".join(("v17", "_v4_contract")),
            "".join(("v", "17")),
        )
    )
    for forbidden in (
        *retired_modules,
        "pointer_sha256",
        "run_staged_vip_promotion",
    ):
        assert forbidden not in source
