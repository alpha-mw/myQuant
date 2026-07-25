from __future__ import annotations

from copy import deepcopy

import pytest

from quant_investor.v17.deep_research import (
    COVERAGE_SECTIONS,
    LAYER_NAMES,
    SEVERE_RED_FLAGS,
    SIGNAL_WEIGHTS,
    TEMPLATE_RESOURCE_SHA256,
    TEMPLATE_SOURCE_SHA256,
    evaluate_deep_research,
    load_deep_research_template,
)
from quant_investor.v17.resources import FROZEN_POLICY_RESOURCE_SHA256S, resource_byte_sha256


def _response() -> dict[str, object]:
    evidence_id = "sealed-1"
    return {
        "symbol": "000001.SZ",
        "layers": {
            layer: [
                {"layer": layer, "content": f"synthetic {layer}", "evidence_ids": [evidence_id]}
            ]
            for layer in LAYER_NAMES
        },
        "coverage": {
            section: {"conclusion": f"synthetic {section}", "evidence_ids": [evidence_id]}
            for section in COVERAGE_SECTIONS
        },
        "signals": {
            dimension: {"signal": 1, "evidence_ids": [evidence_id]} for dimension in SIGNAL_WEIGHTS
        },
        "severe_red_flags": {
            flag: {"triggered": False, "evidence_ids": []} for flag in SEVERE_RED_FLAGS
        },
    }


def test_signal_overlay_multiplies_positive_base_q25() -> None:
    result = evaluate_deep_research(
        _response(),
        sealed_symbol="000001.SZ",
        sealed_evidence_ids=("sealed-1",),
        base_q25_by_horizon={120: 0.02, 252: 0.10, 378: 0.03},
        base_eligible=True,
    )
    assert result.status == "DEEP_RESEARCH_COMPLETE"
    assert result.f_eligible is True
    assert result.weighted_signal == pytest.approx(1.0)
    assert result.delta == pytest.approx(0.10)
    assert result.adjusted_q25_252 == pytest.approx(0.11)


def test_severe_red_flag_only_revokes_buy_permission() -> None:
    response = _response()
    response["severe_red_flags"]["core_thesis_falsified"] = {
        "triggered": True,
        "evidence_ids": ["sealed-1"],
    }
    result = evaluate_deep_research(
        response,
        sealed_symbol="000001.SZ",
        sealed_evidence_ids=("sealed-1",),
        base_q25_by_horizon={120: 0.02, 252: 0.10, 378: 0.03},
        base_eligible=True,
    )
    assert result.status == "DEEP_RESEARCH_COMPLETE_RED_FLAG"
    assert result.research_complete is True
    assert result.f_eligible is False
    assert result.buy_permission_revoked is True
    assert result.adjusted_q25_252 is None
    assert not hasattr(result, "sell_instruction")


def test_unsealed_evidence_fails_closed() -> None:
    response = deepcopy(_response())
    response["signals"]["financial"]["evidence_ids"] = ["invented-source"]
    result = evaluate_deep_research(
        response,
        sealed_symbol="000001.SZ",
        sealed_evidence_ids=("sealed-1",),
        base_q25_by_horizon={120: 0.02, 252: 0.10, 378: 0.03},
        base_eligible=True,
    )
    assert result.status == "DEEP_RESEARCH_INVALID"
    assert result.f_eligible is False
    assert any(item.startswith("unsealed_evidence:signals.financial") for item in result.blockers)


def test_exact_shapes_reject_top_level_and_nested_source_fields() -> None:
    response = deepcopy(_response())
    response["url"] = "https://example.invalid"
    response["signals"]["financial"]["source"] = "invented"
    result = evaluate_deep_research(
        response,
        sealed_symbol="000001.SZ",
        sealed_evidence_ids=("sealed-1",),
        base_q25_by_horizon={120: 0.02, 252: 0.10, 378: 0.03},
        base_eligible=True,
    )
    assert "response_keys_mismatch" in result.blockers
    assert "object_keys_mismatch:signals.financial" in result.blockers


def test_noncanonical_evidence_and_non_strict_base_types_fail_closed() -> None:
    response = deepcopy(_response())
    response["signals"]["financial"]["evidence_ids"] = [1]
    response["signals"]["valuation"]["signal"] = "1"
    result = evaluate_deep_research(
        response,
        sealed_symbol="000001.SZ",
        sealed_evidence_ids=("sealed-1",),
        base_q25_by_horizon={120: True, 252: 0.10, 378: 0.03},
        base_eligible=1,
    )
    assert "invalid_evidence_ids:signals.financial" in result.blockers
    assert "invalid_signal:valuation" in result.blockers
    assert "base_q25_missing:120" in result.blockers
    assert "base_eligible_not_strict_bool" in result.blockers


def test_installed_template_load_does_not_reopen_absolute_source(monkeypatch) -> None:
    def forbidden_file_hash(_path):
        raise AssertionError("runtime template load must be package-contained")

    monkeypatch.setattr("quant_investor.v17.deep_research.file_sha256", forbidden_file_hash)
    template = load_deep_research_template()
    assert template["source"]["sha256"] == TEMPLATE_SOURCE_SHA256


def test_packaged_template_binds_frozen_markdown_source_hash() -> None:
    template = load_deep_research_template()
    assert template["source"]["sha256"] == TEMPLATE_SOURCE_SHA256
    assert (
        TEMPLATE_RESOURCE_SHA256 == FROZEN_POLICY_RESOURCE_SHA256S["deep_research_template.v1.json"]
    )
    assert resource_byte_sha256("deep_research_template.v1.json") == TEMPLATE_RESOURCE_SHA256


def test_non_object_response_fails_closed_without_attribute_error() -> None:
    result = evaluate_deep_research(
        None,  # type: ignore[arg-type]
        sealed_symbol="000001.SZ",
        sealed_evidence_ids=("sealed-1",),
        base_q25_by_horizon={120: 0.02, 252: 0.10, 378: 0.03},
        base_eligible=True,
    )
    assert result.status == "DEEP_RESEARCH_INVALID"
    assert result.blockers == ("response_not_object",)
