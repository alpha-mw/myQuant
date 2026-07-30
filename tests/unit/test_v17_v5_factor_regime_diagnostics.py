from __future__ import annotations

import copy
from dataclasses import replace
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import math

import pytest

from quant_investor.v17_v5_contract import (
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v5_contract.validators import (
    FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_ID,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION,
)
from quant_investor.v17_v5_contract.schema_validation import SchemaValidationError
from quant_investor.v17_v5_runtime.factor_regime_diagnostics import (
    FactorRegimeDiagnosticError,
    build_regime_conditioned_factor_diagnostic,
    build_unavailable_regime_conditioned_factor_diagnostic,
    validate_regime_conditioned_factor_diagnostic,
    validate_regime_conditioned_factor_diagnostic_replay,
)
from quant_investor.v17_v5_runtime.factor_regime_origin_inventory import (
    ContentArtifactRef,
    FACTOR_REGIME_ORIGIN_INVENTORY_VERSION,
    FactorRegimeOriginInput,
    RegimeEvidenceSnapshot,
    build_factor_regime_origin_inventory,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _reseal(artifact: dict[str, object]) -> dict[str, object]:
    unsealed = copy.deepcopy(artifact)
    unsealed.pop("semantic_sha256")
    return seal_semantic(unsealed)


def _policy_ref() -> dict[str, str]:
    return {
        "artifact_id": FACTOR_REGIME_DIAGNOSTIC_POLICY_ID,
        "byte_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256,
        "relative_path": FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH,
        "semantic_sha256": FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256,
        "version": FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION,
    }


def _v5_ref(kind: str, semantic_sha256: str | None = None) -> dict[str, str]:
    semantic = semantic_sha256 or _sha(f"{kind}-semantic")
    return {
        "artifact_id": f"{kind}.synthetic-test-only",
        "byte_sha256": _sha(f"{kind}-bytes"),
        "semantic_sha256": semantic,
        "version": f"myquant.v17.v5.{kind}.v1",
    }


def _v4_ref(kind: str, index: int) -> ContentArtifactRef:
    return ContentArtifactRef(
        artifact_id=f"{kind}.{index:03d}",
        byte_sha256=_sha(f"{kind}-{index}-bytes"),
        cutoff=f"2026-01-{index + 2:02d}T08:00:00Z",
        relative_path=f"tests/fixtures/v17_v5_sprint1b/{kind}/{index:03d}.json",
        semantic_sha256=_sha(f"{kind}-{index}-semantic"),
        strategy_id="cn-strategy",
        version=(
            "myquant.v17.v4.regime-evidence.v2"
            if kind == "regime-evidence"
            else f"myquant.v17.v4.{kind}.v1"
        ),
    )


def _regime(index: int, *, state: str) -> RegimeEvidenceSnapshot:
    return RegimeEvidenceSnapshot(
        available_at=f"2026-01-{index + 2:02d}T07:00:00Z",
        calendar_previous_open_session=f"2026-01-{index + 1:02d}",
        cutoff=f"2026-01-{index + 2:02d}T07:30:00Z",
        decision_session=f"2026-01-{index + 2:02d}",
        effective_session=f"2026-01-{index + 2:02d}",
        hard_state_derivation="SEALED_ARGMAX_POLICY_V1",
        inference_kind="FILTERED_CAUSAL",
        no_retroactive_causal_backfill=True,
        observed_through_session=f"2026-01-{index + 1:02d}",
        publication_phase="PRIOR_SESSION_EFFECTIVE_NEXT_SESSION",
        published_at=f"2026-01-{index + 2:02d}T07:45:00Z",
        regime_artifact_ref=replace(
            _v4_ref("regime-evidence", index),
            cutoff=f"2026-01-{index + 2:02d}T07:30:00Z",
        ),
        regime_state=state,
        scope_kind="FULL_MARKET",
        smoothing_used=False,
        source_commit="1da7ffb636a3254940525d746549d15e827f06ba",
        source_version="myquant.v17.v4.regime-evidence.v2",
        state_order=["趋势上涨", "震荡低波", "震荡高波", "趋势下跌", "未知"],
        state_probabilities={
            "趋势上涨": "0.8" if state == "趋势上涨" else "0.0",
            "震荡低波": "0.2" if state == "趋势上涨" else "0.0",
            "震荡高波": "0.0",
            "趋势下跌": "1.0" if state == "趋势下跌" else "0.0",
            "未知": "1.0" if state == "未知" else "0.0",
        },
        strategy_id="cn-strategy",
    )


def _origin(
    index: int, *, rank_ic: str, coverage: str = "0.8", state: str = "趋势上涨"
) -> FactorRegimeOriginInput:
    comparable = int(Decimal(coverage) * Decimal(10))
    return FactorRegimeOriginInput(
        comparable_symbol_count=comparable,
        coverage=coverage,
        decision_session=f"2026-01-{index + 2:02d}",
        eligible_symbol_count=10,
        factor_evidence_ref=_v4_ref("forward-evaluation-receipt", index),
        factor_implementation_sha256=_sha("implementation"),
        factor_name="cn_factor_test",
        factor_observation_ref=_v4_ref("factor-universe-observation", index),
        label_end_session=f"2026-02-{index + 1:02d}",
        label_horizon_sessions=20,
        label_origin_session=f"2026-01-{index + 2:02d}",
        matured_label_ref=_v4_ref("forward-label", index),
        observation_run_ref=_v4_ref("forward-observation-run", index),
        origin_cutoff=f"2026-01-{index + 2:02d}T08:00:00Z",
        origin_id=f"origin.{index:03d}",
        rank_ic=rank_ic,
        regime_evidence=_regime(index, state=state),
        request_ref=_v4_ref("forward-run-request", index),
        source_locator_ref=_v4_ref("forward-source-locator", index),
        strategy_id="cn-strategy",
    )


def _inventory(*origins: FactorRegimeOriginInput) -> dict[str, object]:
    return build_factor_regime_origin_inventory(
        created_at="2026-03-31T00:00:00Z",
        cutoff="2026-03-30T00:00:00Z",
        factor_implementation_sha256=_sha("implementation"),
        factor_name="cn_factor_test",
        origin_rows=list(origins),
        policy_ref=_policy_ref(),
        strategy_id="cn-strategy",
    )


def _inventory_ref(inventory: dict[str, object]) -> dict[str, str]:
    return {
        "artifact_id": str(inventory["inventory_id"]),
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(inventory)).hexdigest(),
        "semantic_sha256": str(inventory["semantic_sha256"]),
        "version": FACTOR_REGIME_ORIGIN_INVENTORY_VERSION,
    }


def _diagnostic(inventory: dict[str, object]) -> dict[str, object]:
    return build_regime_conditioned_factor_diagnostic(
        created_at="2026-04-01T00:00:00Z",
        cutoff="2026-03-31T00:00:00Z",
        factor_evidence_ref=_v5_ref("factor-evidence"),
        origin_inventory=inventory,
        origin_inventory_ref=_inventory_ref(inventory),
        policy_ref=_policy_ref(),
    )


def _render(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        rendered = value.quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN)
    if rendered.is_zero():
        rendered = abs(rendered)
    return format(rendered, ".12f")


def _manual_newey_west(values: list[Decimal]) -> tuple[str, str]:
    average = sum(values, Decimal(0)) / Decimal(len(values))
    centered = [value - average for value in values]
    count = Decimal(len(values))
    with localcontext() as context:
        context.prec = 50
        long_run_variance = sum(value * value for value in centered) / count
        for step in range(1, 20):
            covariance = (
                sum(
                    centered[index] * centered[index - step] for index in range(step, len(centered))
                )
                / count
            )
            long_run_variance += (
                Decimal(2) * (Decimal(1) - Decimal(step) / Decimal(20)) * covariance
            )
        se = (long_run_variance / count).sqrt()
        t_stat = average / se
    return _render(se), _render(t_stat)


def test_unavailable_diagnostic_allows_unknown_implementation_sha_and_has_no_fake_refs() -> None:
    artifact = build_unavailable_regime_conditioned_factor_diagnostic(
        created_at="2026-04-01T00:00:00Z",
        cutoff="2026-03-31T00:00:00Z",
        factor_implementation_sha256=None,
        factor_name="cn_factor_test",
        policy_ref=_policy_ref(),
        strategy_id="cn-strategy",
        unavailable_prerequisites=[
            "v4_factor_evidence_unavailable",
            "v4_regime_evidence_unavailable",
        ],
    )

    assert artifact["status"] == "UNAVAILABLE"
    assert artifact["factor_implementation_sha256"] is None
    assert artifact["factor_evidence_ref"] is None
    assert artifact["origin_inventory_ref"] is None
    assert validate_regime_conditioned_factor_diagnostic(artifact) == artifact
    assert validate_artifact(artifact) == artifact
    assert (
        validate_regime_conditioned_factor_diagnostic_replay(
            artifact,
            created_at="2026-04-01T00:00:00Z",
            cutoff="2026-03-31T00:00:00Z",
            factor_implementation_sha256=None,
            factor_name="cn_factor_test",
            policy_ref=_policy_ref(),
            strategy_id="cn-strategy",
            unavailable_prerequisites=[
                "v4_factor_evidence_unavailable",
                "v4_regime_evidence_unavailable",
            ],
        )
        == artifact
    )


def test_empty_inventory_is_unobserved_not_unavailable() -> None:
    inventory = _inventory()
    artifact = _diagnostic(inventory)

    assert artifact["status"] == "UNOBSERVED"
    assert artifact["unconditional_metrics"] is None
    assert artifact["by_regime"] == []
    assert artifact["regime_occupancy"]["total_origin_count"] == 0
    assert artifact["limitation_codes"] == ["regime_conditioned_no_observed_origins"]


def test_synthetic_test_only_accumulating_metrics_are_deterministic() -> None:
    inventory = _inventory(
        _origin(0, rank_ic="0.1", coverage="0.8", state="趋势上涨"),
        _origin(1, rank_ic="0.2", coverage="0.9", state="趋势上涨"),
        _origin(2, rank_ic="-0.1", coverage="0.7", state="趋势下跌"),
    )
    artifact = _diagnostic(inventory)

    assert artifact["status"] == "ACCUMULATING"
    assert artifact["unconditional_metrics"]["matured_origin_count"] == 3
    assert artifact["unconditional_metrics"]["rank_ic_mean"] == "0.066666666667"
    assert artifact["unconditional_metrics"]["rank_ic_median"] == "0.100000000000"
    assert artifact["unconditional_metrics"]["rank_ic_std"] == "0.152752523165"
    assert artifact["unconditional_metrics"]["rank_icir"] == "0.436435780472"
    assert artifact["unconditional_metrics"]["coverage_p10"] == "0.700000000000"
    up = next(row for row in artifact["by_regime"] if row["regime_state"] == "趋势上涨")
    down = next(row for row in artifact["by_regime"] if row["regime_state"] == "趋势下跌")
    assert up["origin_share"] == "0.666666666667"
    assert up["coverage_p10"] == "0.800000000000"
    assert up["delta_rank_ic_vs_unconditional"] == "0.083333333333"
    assert down["rank_ic_std"] is None
    assert down["rank_icir"] is None
    assert artifact["regime_occupancy"]["regime_concentration"] == "0.555555555556"
    assert artifact["regime_occupancy"]["posterior_confidence_summary"] == {
        "hard_state_probability_mean": "0.866666666667",
        "hard_state_probability_min": "0.800000000000",
        "hard_state_probability_p10": "0.800000000000",
        "posterior_origin_count": 3,
    }
    assert validate_regime_conditioned_factor_diagnostic(artifact) == artifact
    assert validate_artifact(artifact) == artifact


def test_newey_west_lag_19_matches_manual_formula_for_20_origins() -> None:
    values = [Decimal("-0.95") + Decimal(index) * Decimal("0.10") for index in range(20)]
    inventory = _inventory(
        *[
            _origin(index, rank_ic=_render(value), coverage="0.8", state="趋势上涨")
            for index, value in enumerate(values)
        ]
    )
    artifact = _diagnostic(inventory)
    expected_se, expected_t = _manual_newey_west(values)

    assert artifact["unconditional_metrics"]["newey_west_se_lag_19"] == expected_se
    assert artifact["unconditional_metrics"]["newey_west_t_stat"] == expected_t
    assert artifact["unconditional_metrics"]["rank_ic_std"] == "0.591607978310"


def test_v5_content_refs_are_pathless_and_json_contains_no_nonfinite_or_governance_fields() -> None:
    inventory = _inventory(_origin(0, rank_ic="0.1"))
    bad_ref = _inventory_ref(inventory)
    bad_ref["relative_path"] = "data/private/not-allowed.json"

    with pytest.raises(FactorRegimeDiagnosticError, match="pathless"):
        build_regime_conditioned_factor_diagnostic(
            created_at="2026-04-01T00:00:00Z",
            cutoff="2026-03-31T00:00:00Z",
            factor_evidence_ref=_v5_ref("factor-evidence"),
            origin_inventory=inventory,
            origin_inventory_ref=bad_ref,
            policy_ref=_policy_ref(),
        )

    artifact = _diagnostic(inventory)
    encoded = canonical_bytes(artifact)
    assert encoded == canonical_bytes(_diagnostic(inventory))
    assert b"NaN" not in encoded
    assert b"Infinity" not in encoded
    assert b"factor_weight" not in encoded
    assert b"lifecycle_action" not in encoded
    assert b"validity" not in encoded


def test_origin_inventory_ref_mismatch_fails_closed() -> None:
    inventory = _inventory(_origin(0, rank_ic="0.1"))
    for field, value in (
        ("artifact_id", "other-inventory"),
        ("byte_sha256", _sha("other-bytes")),
        ("semantic_sha256", _sha("other-semantic")),
        ("version", "myquant.v17.v5.other-inventory.v1"),
    ):
        bad_ref = _inventory_ref(inventory)
        bad_ref[field] = value
        with pytest.raises(FactorRegimeDiagnosticError, match="does not bind"):
            build_regime_conditioned_factor_diagnostic(
                created_at="2026-04-01T00:00:00Z",
                cutoff="2026-03-31T00:00:00Z",
                factor_evidence_ref=_v5_ref("factor-evidence"),
                origin_inventory=inventory,
                origin_inventory_ref=bad_ref,
                policy_ref=_policy_ref(),
            )


def test_contract_rejects_resealed_diagnostic_identity_drift() -> None:
    diagnostic = copy.deepcopy(_diagnostic(_inventory(_origin(0, rank_ic="0.1"))))
    diagnostic["diagnostic_id"] = "regime-conditioned-factor-diagnostic-forged"
    with pytest.raises(FactorRegimeDiagnosticError, match="diagnostic identity"):
        validate_regime_conditioned_factor_diagnostic(_reseal(diagnostic))


@pytest.mark.parametrize(
    ("section", "expected_error"),
    (
        ("by_regime", "by-regime diagnostic contains an ineligible state"),
        ("regime_origin_counts", "regime occupancy contains an ineligible state"),
    ),
)
def test_contract_rejects_unknown_state_in_public_diagnostic_groups(
    section: str,
    expected_error: str,
) -> None:
    diagnostic = copy.deepcopy(_diagnostic(_inventory(_origin(0, rank_ic="0.1"))))
    if section == "by_regime":
        diagnostic["by_regime"][0]["regime_state"] = "未知"
    else:
        diagnostic["regime_occupancy"]["regime_origin_counts"][0]["regime_state"] = "未知"
    diagnostic = _reseal(diagnostic)
    identity_material = copy.deepcopy(diagnostic)
    identity_material.pop("diagnostic_id")
    identity_material.pop("semantic_sha256")
    diagnostic["diagnostic_id"] = (
        "regime-conditioned-factor-diagnostic-"
        f"{hashlib.sha256(canonical_bytes(identity_material)).hexdigest()[:32]}"
    )
    diagnostic = _reseal(diagnostic)

    with pytest.raises(FactorRegimeDiagnosticError, match=expected_error):
        validate_regime_conditioned_factor_diagnostic(diagnostic)
    with pytest.raises(SchemaValidationError, match="is outside the closed enum"):
        validate_artifact(diagnostic)


def test_all_unknown_origins_are_unobserved_and_not_bucketed() -> None:
    inventory = _inventory(
        _origin(0, rank_ic="0.1", state="未知"),
        _origin(1, rank_ic="0.2", state="未知"),
    )
    artifact = _diagnostic(inventory)

    assert artifact["status"] == "UNOBSERVED"
    assert artifact["by_regime"] == []
    assert artifact["unconditional_metrics"] is None
    assert artifact["regime_occupancy"]["missing_regime_count"] == 2
    assert artifact["regime_occupancy"]["total_origin_count"] == 0
    assert "NO_CONDITIONING_ELIGIBLE_ORIGIN" in artifact["limitation_codes"]
    assert "REGIME_HARD_STATE_UNKNOWN" in artifact["limitation_codes"]
    assert validate_artifact(artifact) == artifact


def test_created_at_before_cutoff_and_bad_unavailable_reason_fail_closed() -> None:
    with pytest.raises(FactorRegimeDiagnosticError, match="created_at"):
        build_unavailable_regime_conditioned_factor_diagnostic(
            created_at="2026-03-30T00:00:00Z",
            cutoff="2026-03-31T00:00:00Z",
            factor_implementation_sha256=None,
            factor_name="cn_factor_test",
            policy_ref=_policy_ref(),
            strategy_id="cn-strategy",
            unavailable_prerequisites=["v4_regime_evidence_unavailable"],
        )
    with pytest.raises(FactorRegimeDiagnosticError, match="unavailable prerequisite"):
        build_unavailable_regime_conditioned_factor_diagnostic(
            created_at="2026-04-01T00:00:00Z",
            cutoff="2026-03-31T00:00:00Z",
            factor_implementation_sha256=None,
            factor_name="cn_factor_test",
            policy_ref=_policy_ref(),
            strategy_id="cn-strategy",
            unavailable_prerequisites=["bad reason"],
        )
