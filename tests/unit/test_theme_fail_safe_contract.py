from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from quant_investor.market.dag.theme_context import (
    build_theme_portfolio_constraints,
    build_theme_risk_constraints,
    extract_symbol_theme_metadata,
)
from quant_investor.themes.calibration import build_theme_calibration_report
from quant_investor.themes.replay import (
    ThemeCalibrationDataset,
    build_theme_calibration_dataset,
)
from quant_investor.themes.governance import evaluate_theme_governance
from quant_investor.themes.scanner import ThemeScanner
from quant_investor.themes.storage import ThemeSnapshotStore


SYMBOL = "000001.SZ"


def test_theme_scanner_empty_and_malformed_inputs_safe() -> None:
    empty_result = ThemeScanner().scan(
        frames={},
        industry_map={},
        symbol_market_state={},
        market="CN",
        universe_key="empty",
        as_of="20260618",
    )
    malformed_result = ThemeScanner().scan(
        frames={SYMBOL: object()},
        industry_map={SYMBOL: "AI"},
        symbol_market_state={SYMBOL: "bad-state"},
        market="CN",
        universe_key="malformed",
        as_of="20260618",
        min_member_count=1,
    )

    assert empty_result.theme_scores == {}
    assert empty_result.symbol_scores == {}
    assert malformed_result.metadata["no_network"] is True


def test_extract_symbol_theme_metadata_all_bad_inputs_safe() -> None:
    bad_contexts = [
        SimpleNamespace(metadata={"theme_rotation": "not-a-map"}),
        SimpleNamespace(
            metadata={
                "theme_rotation": {
                    "schema_version": "theme_rotation.v1",
                    "status": "success",
                    "symbol_scores": ["bad"],
                    "symbol_primary_theme": "bad",
                    "symbol_phase": 123,
                    "symbol_risk_flags": {SYMBOL: "not-a-list"},
                }
            }
        ),
        SimpleNamespace(metadata="not-a-map"),
    ]

    for context in bad_contexts:
        metadata = extract_symbol_theme_metadata(
            global_context=context,
            symbol=SYMBOL,
        )
        assert metadata["available"] is False


def test_build_theme_risk_constraints_bad_metadata_safe() -> None:
    context = SimpleNamespace(metadata={"theme_rotation": "not-a-map"})

    constraints = build_theme_risk_constraints(
        global_context=context,
        symbols=[SYMBOL],
        enabled=True,
    )

    assert constraints["theme_risk_guard_enabled"] is True
    assert constraints["theme_risk_by_symbol"] == {}
    assert constraints["theme_risk_flags"] == []
    assert constraints["theme_position_limits"] == {}
    assert constraints["theme_gross_exposure_cap"] is None
    assert constraints["diagnostic_notes"] == ["theme_risk_guard_no_theme_risk"]


def test_build_theme_portfolio_constraints_bad_metadata_safe() -> None:
    context = SimpleNamespace(metadata={"theme_rotation": "not-a-map"})

    constraints = build_theme_portfolio_constraints(
        global_context=context,
        symbols=[SYMBOL],
        enabled=True,
    )

    assert constraints["theme_portfolio_cap_enabled"] is True
    assert constraints["theme_exposure_map"] == {}
    assert constraints["theme_caps"] == {}
    assert constraints["theme_names"] == {}
    assert constraints["diagnostic_notes"] == ["theme_portfolio_cap_no_theme_data"]


def test_theme_snapshot_store_bad_json_load_latest_safe(tmp_path: Path) -> None:
    bad_path = tmp_path / "CN" / "20260618" / "full_a_20260618_bad_theme_rotation.json"
    bad_path.parent.mkdir(parents=True)
    bad_path.write_text("{bad json", encoding="utf-8")

    assert ThemeSnapshotStore(tmp_path).load_latest(market="CN", universe_key="full_a") is None


def test_theme_governance_bad_metadata_safe() -> None:
    payload = evaluate_theme_governance(
        {
            "schema_version": "theme_rotation.v1",
            "enabled": True,
            "status": "success",
            "theme_scores": {
                "industry::bad": {
                    "theme_id": "industry::bad",
                    "score": "bad",
                    "confidence": 0.6,
                    "breadth": 0.5,
                    "member_count": 8,
                }
            },
        }
    ).to_dict()

    assert payload["status"] == "success"
    assert payload["decisions"][0]["gate_label"] == "unavailable"
    assert payload["summary_counts"]["unavailable"] == 1


def test_theme_replay_bad_snapshot_safe(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_snapshot.json"
    bad_path.write_text("{bad json", encoding="utf-8")

    dataset = build_theme_calibration_dataset(
        snapshots=[
            {"theme_rotation": "not-a-map"},
            bad_path,
            object(),
        ],
        frames={},
    )

    assert dataset.records == []
    assert dataset.metadata["malformed_snapshot_count"] >= 3


def test_theme_calibration_empty_dataset_safe() -> None:
    report = build_theme_calibration_report(ThemeCalibrationDataset())

    assert report.record_count == 0
    assert report.available_count == 0
    assert "insufficient_calibration_sample" in report.warnings
    assert report.metadata["offline_only"] is True
    assert report.metadata["no_llm"] is True
    assert report.metadata["no_network"] is True
