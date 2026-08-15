from __future__ import annotations

from quant_investor.contracts import get_contract
from quant_investor.factors.governance.bootstrap import (
    BLEND_W75_CONTROL,
    BLEND_W80,
    LOW_DOLLAR_VOLUME,
)
from quant_investor.factors.governance.common import (
    bootstrap_validation_namespace_id,
    prospective_validation_namespace_id,
)
from quant_investor.factors.governance.errors import FactorGovernanceError
from quant_investor.factors.governance.implementations import (
    implementation_code_sha256,
    installed_implementation_rows,
    installed_semantic_row,
)
from quant_investor.system.components import ast_entrypoint_sha256


def _ref(kind: str, artifact_id: str, marker: str) -> dict[str, str]:
    return {
        "kind": kind,
        "contract_sha256": get_contract(kind).contract_sha256,
        "artifact_id": artifact_id,
        "semantic_sha256": marker * 64,
        "byte_sha256": marker * 64,
    }


def test_installed_factor_semantics_and_ast_identity_are_exact() -> None:
    low = installed_semantic_row(LOW_DOLLAR_VOLUME)
    blend = installed_semantic_row(BLEND_W80)
    assert low == {
        "factor_id": LOW_DOLLAR_VOLUME,
        "implementation_id": f"installed-{LOW_DOLLAR_VOLUME}",
        "module_name": "quant_investor.factors.governance.implementations",
        "qualified_name": "_low_dollar_volume",
        "code_sha256": implementation_code_sha256(LOW_DOLLAR_VOLUME),
        "family": "liquidity",
        "primitive": "low_dollar_volume",
        "direction": "HIGHER_IS_BETTER",
        "formula": "-log(mean(amount[t-4:t]))",
        "normalized_expression": (
            '{"input":"amount","operator":"NEGATIVE_LOG_ROLLING_MEAN",' '"window_open_sessions":5}'
        ),
        "parameters_json": '{"window_open_sessions":5}',
        "input_fields": ["amount"],
    }
    assert blend["primitive"] == "volstab_momentum_amihud_blend"
    assert blend["parameters_json"] == (
        '{"amihud_window_open_sessions":5,"inner_amihud_weight":"0.400000000000",'
        '"inner_momentum_weight":"0.600000000000","momentum_window_open_sessions":90,'
        '"outer_volume_stability_weight":"0.800000000000",'
        '"volume_stability_base_open_sessions":19,'
        '"volume_stability_smoothing_open_sessions":2}'
    )
    assert implementation_code_sha256(LOW_DOLLAR_VOLUME) == low["code_sha256"]


def test_installed_factor_ast_identity_matches_system_rederivation() -> None:
    for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80):
        row = installed_semantic_row(factor_id)
        assert implementation_code_sha256(factor_id) == ast_entrypoint_sha256(
            row["module_name"], row["qualified_name"]
        )


def test_w75_has_no_installed_implementation() -> None:
    try:
        installed_semantic_row(BLEND_W75_CONTROL)
    except FactorGovernanceError as exc:
        assert exc.exit_code == 2
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("W75 unexpectedly entered the installed registry")


def test_installed_rows_bind_each_exact_component_ref() -> None:
    refs = {
        LOW_DOLLAR_VOLUME: _ref("system.installed_component_manifest", "component-low", "1"),
        BLEND_W80: _ref("system.installed_component_manifest", "component-w80", "2"),
    }
    rows = installed_implementation_rows(implementation_component_refs=refs)
    assert [row["factor_id"] for row in rows] == sorted(refs)
    assert {row["implementation_component_ref"]["artifact_id"] for row in rows} == {
        "component-low",
        "component-w80",
    }


def test_validation_namespaces_are_deterministic_and_root_bound() -> None:
    calendar = _ref("system.source_object", "calendar", "3")
    implementation = _ref("system.source_object", "implementation", "4")
    manifest = _ref("factor.validator_manifest", "manifest", "5")
    first = prospective_validation_namespace_id(
        exchange_calendar_ref=calendar,
        implementation_manifest_ref=implementation,
        factor_validator_manifest_ref=manifest,
    )
    second = prospective_validation_namespace_id(
        exchange_calendar_ref=calendar,
        implementation_manifest_ref=implementation,
        factor_validator_manifest_ref=manifest,
    )
    changed = prospective_validation_namespace_id(
        exchange_calendar_ref={**calendar, "artifact_id": "another-calendar"},
        implementation_manifest_ref=implementation,
        factor_validator_manifest_ref=manifest,
    )
    assert first == second
    assert first != changed
    assert "202" not in first

    receipt = _ref("factor.validation_receipt", "bootstrap-receipt", "6")
    assert bootstrap_validation_namespace_id(intrinsic_receipt_ref=receipt) == (
        bootstrap_validation_namespace_id(intrinsic_receipt_ref=receipt)
    )
