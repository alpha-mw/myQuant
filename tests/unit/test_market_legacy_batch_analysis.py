"""Legacy market batch-analysis compatibility boundary tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_investor.market.full_report import MarketArtifactContractError
from quant_investor.versioning import (
    ARCHITECTURE_VERSION,
    BRANCH_SCHEMA_VERSION,
    IC_PROTOCOL_VERSION,
    LIKELIHOOD_SCHEMA_VERSION,
    REPORT_PROTOCOL_VERSION,
)


def _branch_result(name: str):
    return SimpleNamespace(
        branch_name=name,
        score=0.1,
        confidence=0.6,
        conclusion=f"{name} conclusion",
        support_drivers=[],
        drag_drivers=[],
        investment_risks=[],
        coverage_notes=[],
        diagnostic_notes=[],
        module_coverage={},
        metadata={},
        symbol_scores={},
    )


def _pipeline_result(**overrides):
    payload = {
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_schema_version": BRANCH_SCHEMA_VERSION,
        "likelihood_schema_version": LIKELIHOOD_SCHEMA_VERSION,
        "ic_protocol_version": IC_PROTOCOL_VERSION,
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
        "branch_results": {
            name: _branch_result(name)
            for name in ("quant", "fundamental", "macro")
        },
        "final_strategy": SimpleNamespace(
            trade_recommendations=[],
            target_exposure=0.3,
            style_bias="均衡",
            candidate_symbols=[],
            position_limits={},
            branch_consensus={},
            risk_summary={},
            execution_notes=[],
            research_mode="production",
        ),
        "execution_log": [],
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _current_batch_payload(**overrides):
    envelope = {
        "architecture_version": ARCHITECTURE_VERSION,
        "branch_schema_version": BRANCH_SCHEMA_VERSION,
        "likelihood_schema_version": LIKELIHOOD_SCHEMA_VERSION,
        "ic_protocol_version": IC_PROTOCOL_VERSION,
        "report_protocol_version": REPORT_PROTOCOL_VERSION,
    }
    payload = {
        **envelope,
        "category": "hs300",
        "batch_id": 1,
        "timestamp": "20260612_122200",
        "stocks": ["000001.SZ"],
        "stock_count": 1,
        "branches": {
            name: {"score": 0.0, "confidence": 0.5}
            for name in ("quant", "fundamental", "macro")
        },
        "analysis_meta": {
            **envelope,
            "market": "CN",
            "universe": "hs300",
        },
    }
    payload.update(overrides)
    return payload


def test_legacy_batch_module_saves_batch_result(tmp_path):
    from quant_investor.market.legacy_batch_analysis import save_batch_result

    output = save_batch_result(
        _current_batch_payload(),
        market="CN",
        output_dir=str(tmp_path),
    )

    payload = json.loads(Path(output).read_text(encoding="utf-8"))

    assert Path(output).name == "batch_hs300_001_20260612_122200.json"
    assert payload["stocks"] == ["000001.SZ"]


def test_current_market_batch_namespaces_are_versioned():
    from quant_investor.market.config import get_market_settings

    assert get_market_settings("CN").analysis_output_dir == (
        "results/v15/cn_analysis_full"
    )
    assert get_market_settings("US").analysis_output_dir == (
        "results/v15/us_analysis_full"
    )


@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    [
        ("unversioned", "architecture_version"),
        ("old", "branch_schema_version"),
        ("intelligence", "analysis_meta.intelligence_snapshot"),
    ],
)
def test_save_batch_result_rejects_noncurrent_artifact_before_writing(
    tmp_path,
    mutation,
    error_pattern,
):
    from quant_investor.market.legacy_batch_analysis import save_batch_result

    batch = _current_batch_payload()
    if mutation == "unversioned":
        batch.pop("architecture_version")
    elif mutation == "old":
        batch["branch_schema_version"] = "branch-schema.v13.four-branch"
    else:
        batch["analysis_meta"]["intelligence_snapshot"] = {"score": 1.0}
    output_dir = tmp_path / "rejected"

    with pytest.raises(MarketArtifactContractError, match=error_pattern):
        save_batch_result(
            batch,
            market="CN",
            output_dir=str(output_dir),
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("market", "legacy_root"),
    [
        ("CN", "results/cn_analysis_full"),
        ("US", "results/us_analysis_full"),
        ("CN", "results/cn_analysis_full/run1"),
        ("US", "results/us_analysis_full/run1"),
    ],
)
def test_save_batch_result_rejects_retired_output_root_before_writing(
    monkeypatch,
    tmp_path,
    market,
    legacy_root,
):
    from quant_investor.market.legacy_batch_analysis import save_batch_result

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="read-only"):
        save_batch_result(
            _current_batch_payload(),
            market=market,
            output_dir=legacy_root,
        )

    assert not (tmp_path / legacy_root).exists()


def test_legacy_batch_producer_emits_current_envelope_and_exact_branches(
    monkeypatch,
):
    import quant_investor.market.legacy_batch_analysis as legacy_batch

    result = _pipeline_result()

    class FakeQuantInvestor:
        def __init__(self, **kwargs):
            pass

        def run(self):
            return result

    monkeypatch.setattr(legacy_batch, "QuantInvestor", FakeQuantInvestor)

    batch = legacy_batch.analyze_batch(
        ["000001.SZ"],
        "hs300",
        1,
        verbose=False,
    )

    assert batch is not None
    assert batch["architecture_version"] == ARCHITECTURE_VERSION
    assert batch["likelihood_schema_version"] == LIKELIHOOD_SCHEMA_VERSION
    assert batch["analysis_meta"]["ic_protocol_version"] == IC_PROTOCOL_VERSION
    assert list(batch["branches"]) == ["quant", "fundamental", "macro"]


def test_legacy_batch_producer_raises_for_unversioned_result(monkeypatch):
    import quant_investor.market.legacy_batch_analysis as legacy_batch

    result = _pipeline_result(architecture_version=None)

    class FakeQuantInvestor:
        def __init__(self, **kwargs):
            pass

        def run(self):
            return result

    monkeypatch.setattr(legacy_batch, "QuantInvestor", FakeQuantInvestor)

    with pytest.raises(MarketArtifactContractError, match="architecture_version"):
        legacy_batch.analyze_batch(
            ["000001.SZ"],
            "hs300",
            1,
            verbose=False,
        )


def test_analyze_category_wrapper_preserves_monkeypatch_compatibility(
    monkeypatch,
    tmp_path,
):
    import quant_investor.market.analyze as analyze

    batch_calls: list[tuple[list[str], int]] = []
    saved_batches: list[int] = []

    monkeypatch.setattr(
        analyze,
        "get_all_local_symbols",
        lambda category, market="CN", data_dir=None: [
            "000001.SZ",
            "000002.SZ",
        ],
    )

    def _fake_analyze_batch(
        symbols,
        category,
        batch_id,
        **kwargs,
    ):
        batch_calls.append((list(symbols), int(batch_id)))
        return {
            "category": category,
            "batch_id": batch_id,
            "timestamp": f"20260612_12220{batch_id}",
            "stocks": list(symbols),
            "stock_count": len(symbols),
        }

    def _fake_save_batch_result(result, **kwargs):
        saved_batches.append(int(result["batch_id"]))
        return str(tmp_path / f"batch_{result['batch_id']}.json")

    monkeypatch.setattr(analyze, "analyze_batch", _fake_analyze_batch)
    monkeypatch.setattr(analyze, "save_batch_result", _fake_save_batch_result)

    results = analyze.analyze_category_full(
        "hs300",
        market="CN",
        batch_size=1,
        output_dir=str(tmp_path),
    )

    assert [batch_id for _, batch_id in batch_calls] == [1, 2]
    assert saved_batches == [1, 2]
    assert [result["stocks"] for result in results] == [
        ["000001.SZ"],
        ["000002.SZ"],
    ]
