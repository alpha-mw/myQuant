from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_storage_validate_clean_passes_with_bounded_local_lineage(tmp_path):
    from quant_investor.market.market_data_store import run_storage_validate_clean

    data_root = tmp_path / "data"
    (data_root / "clean").mkdir(parents=True)
    (data_root / "clean" / "README.md").write_text("clean root", encoding="utf-8")
    _write_json(
        data_root
        / "factor_readiness"
        / "tushare"
        / "daily"
        / "full_a_000001.SZ_factor_readiness_report.json",
        {"schema_version": "factor-readiness-test", "status": "pass"},
    )
    _write_json(
        data_root
        / "cleaning_reports"
        / "tushare"
        / "daily"
        / "full_a_000001.SZ_cleaning_report.json",
        {"schema_version": "tushare-clean-test", "status": "pass"},
    )

    result = run_storage_validate_clean(market="CN", data_root=data_root)

    assert result["status"] == "passed"
    assert result["blockers"] == []
    assert result["local_read_only"] is True
    assert result["roots"]["factor_readiness"]["json_validated_count"] == 1
    assert result["roots"]["cleaning_reports"]["json_validated_count"] == 1


def test_storage_validate_clean_fails_closed_on_missing_lineage(tmp_path):
    from quant_investor.market.market_data_store import run_storage_validate_clean

    data_root = tmp_path / "data"
    (data_root / "clean").mkdir(parents=True)

    result = run_storage_validate_clean(market="CN", data_root=data_root)

    assert result["status"] == "failed"
    assert any("factor_readiness" in item for item in result["blockers"])
    assert any("cleaning_reports" in item for item in result["blockers"])
