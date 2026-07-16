from __future__ import annotations

import json
from pathlib import Path

from scripts.check_cn_dashboard_export import (
    DASHBOARD_SCHEMA_VERSION,
    validate_dashboard_contract_v3,
)


ROOT = Path(__file__).resolve().parents[2]


def test_v3_sample_has_industries_and_no_theme_surface() -> None:
    payload = json.loads(
        (ROOT / "portfolio_dashboard/sample/dashboard_snapshot.v3.json").read_text()
    )
    assert payload["schema_version"] == "dashboard_contract.v3"
    assert "industries" in payload
    assert "themes" not in payload
    assert "theme_protocol" not in payload


def test_checker_rejects_v2_contract() -> None:
    errors, _ = validate_dashboard_contract_v3({"schema_version": "dashboard_contract.v2"})
    assert DASHBOARD_SCHEMA_VERSION == "dashboard_contract.v3"
    assert any("schema_version" in error for error in errors)
