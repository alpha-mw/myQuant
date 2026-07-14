from __future__ import annotations

from quant_investor.config import MAINLINE_ENV_DEFAULTS
from quant_investor.market.dag import context


def test_macro_v2_defaults_are_double_gated_off():
    assert MAINLINE_ENV_DEFAULTS["MACRO_V2_OBSERVER_ENABLED"] == "0"
    assert MAINLINE_ENV_DEFAULTS["MACRO_V2_OBSERVER_KILL_SWITCH"] == "1"
    assert MAINLINE_ENV_DEFAULTS["MACRO_V2_PRODUCTION_ENABLED"] == "0"
    assert MAINLINE_ENV_DEFAULTS["MACRO_V2_PRODUCTION_KILL_SWITCH"] == "1"


def test_macro_v2_dag_metadata_kill_switch_precedes_missing_input(monkeypatch):
    monkeypatch.setattr(context.config, "MACRO_V2_OBSERVER_ENABLED", True)
    monkeypatch.setattr(context.config, "MACRO_V2_OBSERVER_KILL_SWITCH", True)
    monkeypatch.setattr(context.config, "MACRO_V2_OBSERVATIONS_PATH", "/must/not/be/read")

    result = context._macro_v2_observer_metadata(market="CN", as_of="20240510")

    assert result["active"] is False
    assert result["applied"] is False
    assert result["reason"] == "kill_switch_active"


def test_macro_v2_observer_failure_is_diagnostic_only(monkeypatch, tmp_path):
    monkeypatch.setattr(context.config, "MACRO_V2_OBSERVER_ENABLED", True)
    monkeypatch.setattr(context.config, "MACRO_V2_OBSERVER_KILL_SWITCH", False)
    monkeypatch.setattr(context.config, "MACRO_V2_PRODUCTION_ENABLED", True)
    monkeypatch.setattr(context.config, "MACRO_V2_PRODUCTION_KILL_SWITCH", False)
    monkeypatch.setattr(context.config, "MACRO_V2_OBSERVATIONS_PATH", str(tmp_path / "missing.parquet"))

    result = context._macro_v2_observer_metadata(market="CN", as_of="20240510")

    assert result["active"] is False
    assert result["applied"] is False
    assert result["production_eligible"] is False
    assert result["reason"] == "observer_build_failed"
