"""CLI routing guards; native source semantics are tested in the existing compiler suite."""

import hashlib
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.cli.unified import research_compile_daily, CommandError
from quant_investor.factors import production_authority


def request(root, day):
    raw = canonical_json_bytes({})
    (root / "observation.json").write_bytes(raw)
    (root / "observation.json").chmod(0o600)
    sha = hashlib.sha256(raw).hexdigest()
    value = {
        "as_of": "2026-08-28T13:30:00Z",
        "expected_trade_date": day,
        "expected_factor_pointer_sha256": "a" * 64,
        "strategy_id": "aggressive_tech_manufacturing",
        "policy": {},
        "industry_source": None,
        "theme_source": None,
        "low_observation_path": "observation.json",
        "low_observation_sha256": sha,
        "w80_observation_path": "observation.json",
        "w80_observation_sha256": sha,
    }
    raw = canonical_json_bytes(value)
    (root / "request.json").write_bytes(raw)
    (root / "request.json").chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def test_explicit_history_never_falls_back_to_active_head(tmp_path, monkeypatch):
    class Selected(Exception):
        pass

    def historical(self, **kwargs):
        assert kwargs == {"expected_pointer_sha256": "a" * 64, "expected_trade_date": "20260828"}
        raise Selected()

    monkeypatch.setattr(
        production_authority.FactorProductionStore, "read_historical_research_inputs", historical
    )
    monkeypatch.setattr(
        production_authority,
        "read_factor_production_research_inputs",
        lambda *a, **k: pytest.fail("active fallback"),
    )
    sha = request(tmp_path, "20260828")
    with pytest.raises(Selected):
        research_compile_daily(
            workspace_root=str(tmp_path), request_path="request.json", expected_request_sha256=sha
        )


def test_null_historical_date_does_not_select_legacy_mode(tmp_path):
    sha = request(tmp_path, None)
    with pytest.raises(CommandError, match="HISTORICAL_DATE_INVALID"):
        research_compile_daily(
            workspace_root=str(tmp_path), request_path="request.json", expected_request_sha256=sha
        )
