from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from quant_investor.macro.contracts import canonical_hash, is_official_source
from quant_investor.macro.registry import (
    INDUSTRY_CHAINS,
    INDUSTRY_COMPONENT_WEIGHTS,
    NATIONAL_INDICATORS,
)
from quant_investor.macro.acquisition import (
    MacroAcquisitionError,
    build_macro_acquisition_plan,
    run_macro_acquisition_plan,
)


def _coverage() -> dict[str, object]:
    authorities = {
        "cn.gdp_yoy": "nbs_official",
        "cn.pmi_manufacturing": "nbs_official",
        "cn.m1_yoy": "pboc_official",
        "cn.m2_yoy": "pboc_official",
        "cn.social_financing_flow": "pboc_official",
        "cn.fiscal_expenditure_yoy": "mof_official",
        "cn.exports_yoy": "customs_official",
        "cn.imports_yoy": "customs_official",
        "market.breadth": "local_strict_parquet",
        "market.volatility_percentile": "local_strict_parquet",
    }
    statuses = {
        "cn.gdp_yoy": "mapped_raw_missing",
        "cn.pmi_manufacturing": "raw_present_pit_evidence_missing",
    }
    national = []
    for definition in NATIONAL_INDICATORS:
        status = statuses.get(
            definition.indicator_id, "mapping_not_implemented"
        )
        national.append(
            {
                "indicator_id": definition.indicator_id,
                "frequency": definition.frequency,
                "unit": definition.unit,
                "expected_authority": authorities.get(
                    definition.indicator_id, "nbs_official"
                ),
                "mapping_endpoint": (
                    "cn_gdp"
                    if definition.indicator_id == "cn.gdp_yoy"
                    else (
                        "cn_pmi"
                        if definition.indicator_id == "cn.pmi_manufacturing"
                        else None
                    )
                ),
                "status": status,
                "blockers": [
                    (
                        "raw_table_missing"
                        if status == "mapped_raw_missing"
                        else (
                            "timestamp_level_availability_evidence_not_bound"
                            if status == "raw_present_pit_evidence_missing"
                            else "reviewed_raw_mapping_missing"
                        )
                    )
                ],
            }
        )
    industry = [
        {
            "indicator_id": f"industry.{chain}.{component}",
            "industry_chain": chain,
            "component": component,
            "status": "mapping_not_implemented",
            "expected_authority": "UNCONFIRMED",
            "blockers": ["official_industry_source_not_mapped"],
        }
        for chain in INDUSTRY_CHAINS
        for component in INDUSTRY_COMPONENT_WEIGHTS
    ]
    semantic: dict[str, object] = {
        "schema_version": "macro-coverage-audit.v1.1",
        "market": "CN",
        "as_of": "2026-07-14",
        "status": "blocked",
        "national": national,
        "industry": industry,
        "observer_only": True,
        "production_eligible": False,
        "activation_authorized": False,
        "applied": False,
    }
    return {**semantic, "audit_hash": canonical_hash(semantic)}


def test_acquisition_plan_classifies_gaps_without_creating_observations():
    plan = build_macro_acquisition_plan(_coverage())
    tasks = {row["indicator_id"]: row for row in plan["national_tasks"]}
    assert len(plan["national_tasks"]) == 16
    assert len(plan["industry_tasks"]) == 96
    assert plan["task_count"] == 112
    assert plan["observation_count"] == 0
    assert tasks["cn.gdp_yoy"]["action"] == (
        "acquire_raw_and_release_evidence"
    )
    assert tasks["cn.pmi_manufacturing"]["action"] == (
        "bind_timestamp_release_evidence"
    )
    assert tasks["cn.exports_yoy"]["action"] == (
        "implement_official_mapping"
    )
    assert tasks["market.breadth"]["action"] == (
        "build_local_strict_parquet_observation"
    )
    assert tasks["market.breadth"]["acceptance_requirements"] == [
        "strict_parquet_snapshot_sha256",
        "trade_date_and_cutoff_bound",
        "source_table_lineage",
        "deterministic_recompute",
        "zero_quarantine_recompile",
    ]
    assert plan["industry_tasks"][0]["action"] == (
        "confirm_authority_and_mapping"
    )
    assert plan["industry_tasks"][0]["authority_status"] == "UNCONFIRMED"
    assert plan["production_eligible"] is False
    assert plan["applied"] is False


def test_acquisition_contract_requires_hash_bound_timestamp_evidence():
    plan = build_macro_acquisition_plan(_coverage())
    pmi = next(
        row
        for row in plan["national_tasks"]
        if row["indicator_id"] == "cn.pmi_manufacturing"
    )
    assert pmi["allowed_domains"] == ["stats.gov.cn"]
    assert pmi["acceptance_requirements"] == [
        "immutable_raw_capture_sha256",
        "issuer_bound_https_url",
        "source_record_id",
        "timezone_release_at",
        "timezone_available_at",
        "captured_at_not_before_available_at",
        "period_value_unit_frequency_exact_match",
        "zero_quarantine_recompile",
    ]
    assert is_official_source("pboc_official") is True


def test_ready_indicator_is_retained_as_satisfied_without_acquisition_action():
    coverage = _coverage()
    pmi_row = next(
        row
        for row in coverage["national"]  # type: ignore[union-attr]
        if row["indicator_id"] == "cn.pmi_manufacturing"
    )
    pmi_row["status"] = "pit_signal_ready"
    pmi_row["blockers"] = []
    semantic = dict(coverage)
    semantic.pop("audit_hash")
    coverage["audit_hash"] = canonical_hash(semantic)
    plan = build_macro_acquisition_plan(coverage)
    pmi = next(
        row
        for row in plan["national_tasks"]
        if row["indicator_id"] == "cn.pmi_manufacturing"
    )
    assert pmi["task_status"] == "satisfied"
    assert pmi["action"] == "none"
    assert plan["open_task_count"] == 111


def test_partial_registry_coverage_fails_closed():
    coverage = _coverage()
    coverage["national"] = coverage["national"][:-1]  # type: ignore[index]
    semantic = dict(coverage)
    semantic.pop("audit_hash")
    coverage["audit_hash"] = canonical_hash(semantic)
    with pytest.raises(MacroAcquisitionError, match="coverage_scope_mismatch"):
        build_macro_acquisition_plan(coverage)


def test_tampered_or_unsupported_coverage_audit_fails_closed():
    tampered = _coverage()
    tampered["as_of"] = "2026-07-15"
    with pytest.raises(MacroAcquisitionError, match="coverage_hash_mismatch"):
        build_macro_acquisition_plan(tampered)
    unsupported = _coverage()
    unsupported["schema_version"] = "macro-coverage-audit.v0"
    semantic = dict(unsupported)
    semantic.pop("audit_hash")
    unsupported["audit_hash"] = canonical_hash(semantic)
    with pytest.raises(
        MacroAcquisitionError, match="coverage_schema_unsupported"
    ):
        build_macro_acquisition_plan(unsupported)


def test_rehashed_authority_or_industry_metadata_drift_fails_closed():
    authority_drift = _coverage()
    pmi = next(
        row
        for row in authority_drift["national"]  # type: ignore[union-attr]
        if row["indicator_id"] == "cn.pmi_manufacturing"
    )
    pmi["expected_authority"] = "mof_official"
    semantic = dict(authority_drift)
    semantic.pop("audit_hash")
    authority_drift["audit_hash"] = canonical_hash(semantic)
    with pytest.raises(
        MacroAcquisitionError, match="coverage_authority_mismatch"
    ):
        build_macro_acquisition_plan(authority_drift)

    industry_drift = _coverage()
    industry_drift["industry"][0][  # type: ignore[index]
        "industry_chain"
    ] = "wrong_chain"
    semantic = dict(industry_drift)
    semantic.pop("audit_hash")
    industry_drift["audit_hash"] = canonical_hash(semantic)
    with pytest.raises(
        MacroAcquisitionError, match="coverage_industry_metadata_mismatch"
    ):
        build_macro_acquisition_plan(industry_drift)


def test_private_plan_is_idempotent_and_tamper_evident(tmp_path: Path):
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(json.dumps(_coverage()), encoding="utf-8")
    kwargs = {
        "coverage_audit": coverage_path,
        "output_root": tmp_path / "plans",
    }
    first = run_macro_acquisition_plan(**kwargs)
    output = Path(first["output_dir"])
    assert first["idempotent"] is False
    assert all(
        os.stat(path).st_mode & 0o777 == 0o600 for path in output.iterdir()
    )
    second = run_macro_acquisition_plan(**kwargs)
    assert second["idempotent"] is True
    report = output / "acquisition_report.md"
    report.write_bytes(report.read_bytes() + b"tamper")
    with pytest.raises(MacroAcquisitionError, match="artifact_mismatch"):
        run_macro_acquisition_plan(**kwargs)


def test_acquisition_planner_does_not_call_network(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        "quant_investor.data._tushare_client.TushareClientPool.query",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network forbidden")
        ),
    )
    plan = build_macro_acquisition_plan(_coverage())
    assert plan["observer_only"] is True
