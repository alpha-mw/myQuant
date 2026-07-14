from __future__ import annotations

import ast
import copy
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import pytest

import quant_investor.bayesian.posterior_overlay as overlay_module
import quant_investor.versioning as versioning
from quant_investor.bayesian.calibration_v2 import (
    GROUP_ALL_HORIZONS,
    GROUP_ALL_MARKETS,
    GROUP_ALL_REGIMES,
    TARGET_POSTERIOR_WIN_RATE,
    CalibrationBucket,
    CalibrationCurve,
    CalibrationCurveKey,
    CalibrationModelV2,
)
from quant_investor.bayesian.posterior_overlay import (
    OVERLAY_METADATA_KEY,
    OVERLAY_MODE_SHADOW,
    EdgeCostConfig,
    attach_overlay_metadata,
    bps_to_decimal_return,
    build_calibrated_posterior_overlay,
    build_calibrated_posterior_overlays,
    clamp_probability,
    horizon_label_for_days,
)
from quant_investor.bayesian.types import PosteriorResult
from quant_investor.portfolio_optimizer import build_candidate_from_overlay


DECISION_AS_OF = datetime(2026, 4, 27, tzinfo=timezone.utc)
DECISION_AS_OF_TEXT = "2026-04-27T00:00:00Z"


def _bucket(index: int, probability: float, bucket_count: int = 2) -> CalibrationBucket:
    lower = index / bucket_count
    upper = (index + 1) / bucket_count
    return CalibrationBucket(
        bucket_index=index,
        lower_bound=lower,
        upper_bound=upper,
        center=(lower + upper) / 2.0,
        total_count=10,
        positive_count=int(probability * 10),
        raw_mean=(lower + upper) / 2.0,
        empirical_rate=probability,
        prior_alpha=0.0,
        prior_beta=0.0,
        calibrated_probability=probability,
    )


def _curve(
    key: CalibrationCurveKey,
    *,
    low_probability: float = 0.20,
    high_probability: float = 0.80,
    total_examples: int = 20,
) -> CalibrationCurve:
    return CalibrationCurve(
        key=key,
        bucket_count=2,
        prior_strength=0.0,
        total_examples=total_examples,
        positive_examples=int(total_examples * 0.5),
        base_rate=0.5,
        buckets=[
            _bucket(0, low_probability),
            _bucket(1, high_probability),
        ],
    )


def _model(*curves: CalibrationCurve) -> CalibrationModelV2:
    return CalibrationModelV2(
        model_id="calibration-v2-test",
        trained_at="2026-04-26T00:00:00Z",
        bucket_count=2,
        prior_strength=0.0,
        min_examples_per_curve=1,
        curves=list(curves),
    )


def _posterior(symbol: str = "000001.SZ", win_rate: float = 0.75) -> PosteriorResult:
    return PosteriorResult(
        symbol=symbol,
        company_name="测试公司",
        posterior_win_rate=win_rate,
        posterior_expected_alpha=0.04,
        posterior_confidence=0.70,
        posterior_action_score=0.65,
        posterior_edge_after_costs=0.02,
        posterior_capacity_penalty=0.01,
        action_threshold_used=0.55,
        metadata={"source": "unit"},
    )


def _proof_payload(model: CalibrationModelV2) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "2026-07-14.overlay-cutoff-proof.v1",
        "decision_as_of": DECISION_AS_OF_TEXT,
        "training_input_cutoff": "2026-04-25T00:00:00Z",
        "outcome_resolution_cutoff": "2026-04-24T00:00:00Z",
        "training_examples_sha256": "1" * 64,
        "resolved_outcomes_sha256": "2" * 64,
        "outcome_ledger_sha256": "3" * 64,
        "model_id": model.model_id,
        "model_sha256": overlay_module.calibration_model_sha256(model),
        "model_trained_at": model.trained_at,
    }
    payload["proof_sha256"] = overlay_module.overlay_cutoff_proof_sha256(
        payload
    )
    return payload


def _proof(model: CalibrationModelV2):
    return overlay_module.OverlayCutoffProof.from_dict(_proof_payload(model))


def _run_shadow(
    result: PosteriorResult,
    model: CalibrationModelV2,
    **kwargs: object,
):
    return overlay_module.run_calibrated_posterior_overlay(
        mode=OVERLAY_MODE_SHADOW,
        result=result,
        decision_as_of=DECISION_AS_OF,
        cutoff_proof=_proof(model),
        model_loader=lambda: model,
        **kwargs,
    )


def test_numeric_helpers() -> None:
    assert bps_to_decimal_return(100.0) == pytest.approx(0.01)
    assert clamp_probability(-0.1) == 0.0
    assert clamp_probability(1.1) == 1.0
    assert horizon_label_for_days(20) == "20D"

    with pytest.raises(ValueError, match="finite"):
        clamp_probability(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        bps_to_decimal_return(float("inf"))


def test_edge_cost_config_validation() -> None:
    assert EdgeCostConfig().transaction_cost_bps == 0.0

    with pytest.raises(ValueError, match="non-negative"):
        EdgeCostConfig(transaction_cost_bps=-1.0)
    with pytest.raises(ValueError, match="calibration_blend_weight"):
        EdgeCostConfig(calibration_blend_weight=1.1)
    with pytest.raises(ValueError, match="max_probability_adjustment"):
        EdgeCostConfig(max_probability_adjustment=-0.1)


def test_calibration_overlay_computes_edge_after_costs() -> None:
    model = _model(_curve(CalibrationCurveKey()))
    result = _posterior(win_rate=0.75)
    config = EdgeCostConfig(
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
        market_impact_bps=15.0,
        risk_capital_charge=0.002,
    )

    overlay = _run_shadow(
        result,
        model,
        market="CN",
        horizon_days=20,
        macro_regime="趋势上涨",
        edge_cost_config=config,
    )

    assert 0.0 <= overlay.calibrated_posterior_win_rate <= 1.0
    assert overlay.calibrated_posterior_win_rate == pytest.approx(0.80)
    assert overlay.calibrated_posterior_expected_alpha == pytest.approx((0.80 - 0.50) * 0.10)
    assert overlay.edge_breakdown.total_cost_penalty == pytest.approx(0.015)
    assert overlay.calibrated_edge_after_costs == pytest.approx(0.015)
    assert overlay.diagnostics.model_id == "calibration-v2-test"
    assert overlay.diagnostics.selected_curve_examples == 20
    assert overlay.metadata == {}
    assert overlay.source_sha256 == overlay_module.posterior_result_source_sha256(
        result
    )

    round_trip = type(overlay).from_dict(overlay.to_dict())
    assert round_trip.diagnostics.model_id == overlay.diagnostics.model_id
    assert round_trip.edge_breakdown.total_cost_penalty == pytest.approx(overlay.edge_breakdown.total_cost_penalty)


def test_probability_adjustment_cap() -> None:
    model = _model(_curve(CalibrationCurveKey(), low_probability=0.95, high_probability=0.95))
    config = EdgeCostConfig(max_probability_adjustment=0.05)

    overlay = _run_shadow(
        _posterior(win_rate=0.40),
        model,
        edge_cost_config=config,
    )

    assert overlay.diagnostics.cap_applied is True
    assert overlay.diagnostics.probability_delta_after_cap == pytest.approx(0.05)
    assert overlay.calibrated_posterior_win_rate == pytest.approx(0.45)


def test_blend_behavior_before_cap() -> None:
    model = _model(_curve(CalibrationCurveKey(), high_probability=0.90))
    result = _posterior(win_rate=0.70)

    raw_overlay = _run_shadow(
        result,
        model,
        edge_cost_config=EdgeCostConfig(calibration_blend_weight=0.0),
    )
    half_overlay = _run_shadow(
        result,
        model,
        edge_cost_config=EdgeCostConfig(calibration_blend_weight=0.5),
    )
    full_overlay = _run_shadow(
        result,
        model,
        edge_cost_config=EdgeCostConfig(calibration_blend_weight=1.0),
    )

    assert raw_overlay.calibrated_posterior_win_rate == pytest.approx(0.70)
    assert half_overlay.calibrated_posterior_win_rate == pytest.approx(0.80)
    assert full_overlay.calibrated_posterior_win_rate == pytest.approx(0.90)


def test_batch_overlay_preserves_order_rejects_duplicate_and_does_not_mutate() -> None:
    model = _model(_curve(CalibrationCurveKey()))
    first = _posterior("000001.SZ")
    second = _posterior("000002.SZ", win_rate=0.25)

    overlays = build_calibrated_posterior_overlays(
        [first, second],
        model,
        overlay_mode=OVERLAY_MODE_SHADOW,
        decision_as_of=DECISION_AS_OF,
        cutoff_proof=_proof(model),
    )

    assert [overlay.symbol for overlay in overlays] == ["000001.SZ", "000002.SZ"]
    assert OVERLAY_METADATA_KEY not in first.metadata
    assert first.posterior_win_rate == pytest.approx(0.75)
    model_reads: list[str] = []

    class PoisonModel:
        def __getattribute__(self, name: str):
            model_reads.append(name)
            raise AssertionError(f"duplicate preflight read model: {name}")

    with pytest.raises(ValueError, match="Duplicate"):
        build_calibrated_posterior_overlays(
            [first, _posterior("000001.SZ")],
            PoisonModel(),  # type: ignore[arg-type]
            overlay_mode=OVERLAY_MODE_SHADOW,
            decision_as_of=DECISION_AS_OF,
            cutoff_proof=_proof(model),
        )
    assert model_reads == []


def test_attach_overlay_metadata_never_mutates_core_result() -> None:
    model = _model(_curve(CalibrationCurveKey()))
    result = _posterior()
    overlay = _run_shadow(result, model)

    copied = attach_overlay_metadata(result, overlay, mutate=False)

    assert copied is not result
    assert OVERLAY_METADATA_KEY in copied.metadata
    assert OVERLAY_METADATA_KEY not in result.metadata
    assert copied.posterior_win_rate == result.posterior_win_rate
    assert copied.posterior_expected_alpha == result.posterior_expected_alpha
    assert copied.posterior_action_score == result.posterior_action_score
    assert copied.posterior_edge_after_costs == result.posterior_edge_after_costs

    with pytest.raises(ValueError, match="mutate"):
        attach_overlay_metadata(result, overlay, mutate=True)
    assert OVERLAY_METADATA_KEY not in result.metadata
    assert result.posterior_win_rate == pytest.approx(0.75)


def test_calibration_v2_select_curve_fallback_order_and_calibrate_behavior() -> None:
    exact_key = CalibrationCurveKey(TARGET_POSTERIOR_WIN_RATE, "CN", "20D", "趋势上涨")
    exact = _curve(exact_key, high_probability=0.66, total_examples=40)
    global_curve = _curve(
        CalibrationCurveKey(TARGET_POSTERIOR_WIN_RATE, GROUP_ALL_MARKETS, GROUP_ALL_HORIZONS, GROUP_ALL_REGIMES),
        high_probability=0.80,
        total_examples=20,
    )
    model = _model(global_curve, exact)

    selected_exact = model.select_curve(
        TARGET_POSTERIOR_WIN_RATE,
        market="CN",
        horizon_label="20D",
        macro_regime="趋势上涨",
    )
    selected_fallback = model.select_curve(
        TARGET_POSTERIOR_WIN_RATE,
        market="US",
        horizon_label="20D",
        macro_regime="震荡",
    )

    assert selected_exact is exact
    assert selected_fallback is global_curve
    assert model.select_curve("missing_target", market="CN", horizon_label="20D") is None
    assert model.calibrate(
        TARGET_POSTERIOR_WIN_RATE,
        0.75,
        market="CN",
        horizon_label="20D",
        macro_regime="趋势上涨",
    ) == pytest.approx(0.66)
    assert model.calibrate("missing_target", -0.4) == pytest.approx(0.3)


def test_runner_off_is_lazy_before_all_shadow_inputs() -> None:
    reads: list[str] = []

    class Poison:
        def __getattribute__(self, name: str):
            reads.append(name)
            raise AssertionError(f"poison input read: {name}")

    loader_calls = 0

    def poison_loader():
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("off mode loaded model")

    assert overlay_module.run_calibrated_posterior_overlay(
        mode="off",
        result=Poison(),
        decision_as_of=Poison(),
        cutoff_proof=Poison(),
        model_loader=poison_loader,
        edge_cost_config=Poison(),
        metadata=Poison(),
    ) is None
    assert overlay_module.run_calibrated_posterior_overlay(
        result=Poison(),
        decision_as_of=Poison(),
        cutoff_proof=Poison(),
        model_loader=poison_loader,
    ) is None
    assert loader_calls == 0
    assert reads == []


@pytest.mark.parametrize(
    "mode",
    [None, True, False, "OFF", "SHADOW", "live", "production", "shadow "],
)
def test_runner_rejects_non_exact_modes_before_loader(mode: object) -> None:
    loader_calls = 0

    def poison_loader():
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("invalid mode loaded model")

    with pytest.raises(ValueError, match="mode"):
        overlay_module.run_calibrated_posterior_overlay(
            mode=mode,
            model_loader=poison_loader,
        )
    assert loader_calls == 0


def test_shadow_rejects_missing_or_naive_cutoff_inputs_before_loader() -> None:
    model = _model(_curve(CalibrationCurveKey()))
    loader_calls = 0

    def loader():
        nonlocal loader_calls
        loader_calls += 1
        return model

    with pytest.raises(ValueError, match="decision_as_of"):
        overlay_module.run_calibrated_posterior_overlay(
            mode="shadow",
            result=_posterior(),
            decision_as_of=datetime(2026, 4, 27),
            cutoff_proof=_proof(model),
            model_loader=loader,
        )
    with pytest.raises(ValueError, match="proof"):
        overlay_module.run_calibrated_posterior_overlay(
            mode="shadow",
            result=_posterior(),
            decision_as_of=DECISION_AS_OF,
            cutoff_proof=None,
            model_loader=loader,
        )
    assert loader_calls == 0


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("decision_as_of", "2026-04-28T00:00:00Z"),
        ("outcome_resolution_cutoff", "2026-04-26T00:00:00Z"),
        ("training_input_cutoff", "2026-04-27T01:00:00Z"),
        ("model_trained_at", "2026-04-27T01:00:00Z"),
        ("training_examples_sha256", "not-a-sha"),
        ("proof_sha256", "0" * 64),
    ],
)
def test_shadow_validates_proof_structure_and_time_before_loader(
    field_name: str,
    value: str,
) -> None:
    model = _model(_curve(CalibrationCurveKey()))
    proof = _proof(model)
    object.__setattr__(proof, field_name, value)
    if field_name != "proof_sha256":
        object.__setattr__(
            proof,
            "proof_sha256",
            overlay_module.overlay_cutoff_proof_sha256(proof),
        )
    loader_calls = 0

    def loader():
        nonlocal loader_calls
        loader_calls += 1
        return model

    with pytest.raises(ValueError):
        overlay_module.run_calibrated_posterior_overlay(
            mode="shadow",
            result=_posterior(),
            decision_as_of=DECISION_AS_OF,
            cutoff_proof=proof,
            model_loader=loader,
        )
    assert loader_calls == 0


@pytest.mark.parametrize("mutation", ["model_id", "trained_at", "model_body"])
def test_shadow_rejects_loaded_model_identity_or_hash_drift(
    mutation: str,
) -> None:
    proof_model = _model(_curve(CalibrationCurveKey()))
    proof = _proof(proof_model)
    loaded_model = copy.deepcopy(proof_model)
    if mutation == "model_id":
        loaded_model.model_id = "different-model"
    elif mutation == "trained_at":
        loaded_model.trained_at = "2026-04-25T23:00:00Z"
    else:
        loaded_model.curves[0].buckets[0].calibrated_probability = 0.99
    loader_calls = 0

    def loader():
        nonlocal loader_calls
        loader_calls += 1
        return loaded_model

    with pytest.raises(ValueError, match="model"):
        overlay_module.run_calibrated_posterior_overlay(
            mode="shadow",
            result=_posterior(),
            decision_as_of=DECISION_AS_OF,
            cutoff_proof=proof,
            model_loader=loader,
        )
    assert loader_calls == 1


def _mutate_posterior_field(result: PosteriorResult, field_name: str) -> None:
    value = getattr(result, field_name)
    if field_name == "prior":
        result.prior.metadata["source-attestation"] = "changed"
    elif field_name == "likelihoods":
        result.likelihoods.metadata["source-attestation"] = "changed"
    elif isinstance(value, str):
        setattr(result, field_name, f"{value}-changed")
    elif isinstance(value, bool):
        setattr(result, field_name, not value)
    elif isinstance(value, int):
        setattr(result, field_name, value + 1)
    elif isinstance(value, float):
        setattr(result, field_name, value + 0.01)
    elif isinstance(value, list):
        value.append("changed")
    elif isinstance(value, dict):
        value["source-attestation"] = "changed"
    else:
        raise AssertionError(f"unhandled PosteriorResult field: {field_name}")


@pytest.mark.parametrize(
    "field_name",
    [field.name for field in fields(PosteriorResult)],
)
def test_source_sha_covers_every_posterior_field(field_name: str) -> None:
    result = _posterior()
    baseline = overlay_module.posterior_result_source_sha256(result)
    mutated = copy.deepcopy(result)

    _mutate_posterior_field(mutated, field_name)

    assert overlay_module.posterior_result_source_sha256(mutated) != baseline


def test_source_sha_uses_recursive_asdict_not_posterior_to_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _posterior()
    result.prior.metadata = {"set": {"b", "a"}, "nested": {"value": 1}}
    result.likelihoods.metadata = {"deep": [{"field": "covered"}]}

    def explode_to_dict(_self):
        raise AssertionError("PosteriorResult.to_dict must not be used")

    monkeypatch.setattr(PosteriorResult, "to_dict", explode_to_dict)

    digest = overlay_module.posterior_result_source_sha256(result)
    reordered = copy.deepcopy(result)
    reordered.prior.metadata = {"nested": {"value": 1}, "set": {"a", "b"}}
    assert overlay_module.posterior_result_source_sha256(reordered) == digest


@pytest.mark.parametrize(
    ("container_name", "field_name"),
    [
        *[("prior", field.name) for field in fields(type(_posterior().prior))],
        *[
            ("likelihoods", field.name)
            for field in fields(type(_posterior().likelihoods))
        ],
    ],
)
def test_source_sha_covers_every_nested_posterior_field(
    container_name: str,
    field_name: str,
) -> None:
    result = _posterior()
    baseline = overlay_module.posterior_result_source_sha256(result)
    mutated = copy.deepcopy(result)
    container = getattr(mutated, container_name)
    value = getattr(container, field_name)
    if isinstance(value, float):
        setattr(container, field_name, value + 0.01)
    elif isinstance(value, dict):
        value["nested-source-attestation"] = 1.0
    else:
        raise AssertionError(
            f"unhandled nested field: {container_name}.{field_name}"
        )

    assert overlay_module.posterior_result_source_sha256(mutated) != baseline


def test_source_sha_rejects_non_finite_recursive_payload() -> None:
    result = _posterior()
    result.metadata["bad"] = float("nan")

    with pytest.raises(ValueError, match="non-finite|canonical"):
        overlay_module.posterior_result_source_sha256(result)


def test_shadow_rejects_model_side_effect_on_posterior_source() -> None:
    result = _posterior()
    model = _model(_curve(CalibrationCurveKey()))

    def malicious_calibrate(self, *args, **kwargs):
        result.metadata["mutated_by_model"] = True
        return 0.80

    model.calibrate = MethodType(malicious_calibrate, model)  # type: ignore[method-assign]
    proof = _proof(model)

    with pytest.raises(ValueError, match="source"):
        overlay_module.run_calibrated_posterior_overlay(
            mode="shadow",
            result=result,
            decision_as_of=DECISION_AS_OF,
            cutoff_proof=proof,
            model_loader=lambda: model,
        )


def test_shadow_rejects_model_mutation_during_calibration() -> None:
    result = _posterior()
    model = _model(_curve(CalibrationCurveKey()))
    original_calibrate = model.calibrate

    def malicious_calibrate(self, *args, **kwargs):
        self.metadata["mutated_during_calibration"] = True
        return original_calibrate(*args, **kwargs)

    model.calibrate = MethodType(malicious_calibrate, model)  # type: ignore[method-assign]
    proof = _proof(model)

    with pytest.raises(ValueError, match="model drifted"):
        overlay_module.run_calibrated_posterior_overlay(
            mode="shadow",
            result=result,
            decision_as_of=DECISION_AS_OF,
            cutoff_proof=proof,
            model_loader=lambda: model,
        )


def test_v2_shadow_round_trip_and_provenance_invariants() -> None:
    result = _posterior()
    model = _model(_curve(CalibrationCurveKey()))
    overlay = _run_shadow(result, model, market="CN")

    assert overlay is not None
    assert overlay.schema_version.endswith("posterior-overlay.v2")
    assert overlay.overlay_mode == "shadow"
    assert overlay.report_only is True
    assert overlay.production_eligible is False
    assert overlay.production_weight == 0.0
    assert overlay.decision_as_of == DECISION_AS_OF_TEXT
    assert overlay.source_sha256 == overlay_module.posterior_result_source_sha256(
        result
    )
    assert overlay.model_id == model.model_id
    assert overlay.model_sha256 == overlay_module.calibration_model_sha256(model)
    assert overlay.cutoff_proof_sha256 == overlay.cutoff_proof.proof_sha256
    assert overlay.calibrated_posterior_win_rate == pytest.approx(0.80)

    round_trip = type(overlay).from_dict(overlay.to_dict())
    assert round_trip.to_dict() == overlay.to_dict()


@pytest.mark.parametrize(
    "attack",
    [
        "top_mode",
        "top_boolean",
        "top_weight",
        "diagnostics_schema",
        "diagnostics_mode",
        "proof_schema",
        "proof_hash",
        "reserved_metadata",
    ],
)
def test_direct_overlay_construction_enforces_nested_v2_invariants(
    attack: str,
) -> None:
    model = _model(_curve(CalibrationCurveKey()))
    overlay = _run_shadow(_posterior(), model)

    with pytest.raises((TypeError, ValueError)):
        if attack == "top_mode":
            replace(overlay, overlay_mode="off")
        elif attack == "top_boolean":
            replace(overlay, report_only="true")  # type: ignore[arg-type]
        elif attack == "top_weight":
            replace(overlay, production_weight=1.0)
        elif attack == "diagnostics_schema":
            replace(
                overlay,
                diagnostics=replace(
                    overlay.diagnostics,
                    schema_version="2026-04-26.posterior-overlay.v1",
                ),
            )
        elif attack == "diagnostics_mode":
            replace(
                overlay,
                diagnostics=replace(overlay.diagnostics, overlay_mode="off"),
            )
        elif attack == "proof_schema":
            replace(
                overlay,
                cutoff_proof=replace(
                    overlay.cutoff_proof,
                    schema_version="forged-proof.v1",
                ),
            )
        elif attack == "proof_hash":
            replace(
                overlay,
                cutoff_proof=replace(
                    overlay.cutoff_proof,
                    proof_sha256="0" * 64,
                ),
            )
        else:
            replace(overlay, metadata={"model_id": "forged"})


def test_overlay_serialization_revalidates_nested_objects() -> None:
    model = _model(_curve(CalibrationCurveKey()))
    overlay = _run_shadow(_posterior(), model)
    object.__setattr__(overlay.diagnostics, "overlay_mode", "off")

    with pytest.raises(ValueError, match="overlay_mode"):
        overlay.to_dict()


@pytest.mark.parametrize("attack", ["missing", "unknown", "schema", "hash"])
def test_cutoff_proof_from_dict_is_strict(attack: str) -> None:
    model = _model(_curve(CalibrationCurveKey()))
    payload = _proof_payload(model)
    if attack == "missing":
        payload.pop("outcome_ledger_sha256")
    elif attack == "unknown":
        payload["unknown"] = "field"
    elif attack == "schema":
        payload["schema_version"] = "unknown-proof.v1"
        payload["proof_sha256"] = overlay_module.overlay_cutoff_proof_sha256(
            payload
        )
    else:
        payload["proof_sha256"] = "0" * 64

    with pytest.raises((TypeError, ValueError)):
        overlay_module.OverlayCutoffProof.from_dict(payload)


@pytest.mark.parametrize(
    "attack",
    [
        "v1",
        "missing",
        "unknown",
        "string_boolean",
        "integer_weight",
        "nonzero_weight",
        "nonfinite_weight",
        "reserved_metadata",
    ],
)
def test_v2_overlay_from_dict_rejects_malformed_or_forged_payloads(
    attack: str,
) -> None:
    model = _model(_curve(CalibrationCurveKey()))
    overlay = _run_shadow(_posterior(), model)
    payload = copy.deepcopy(overlay.to_dict())
    if attack == "v1":
        payload["schema_version"] = "2026-04-26.posterior-overlay.v1"
    elif attack == "missing":
        payload.pop("source_sha256")
    elif attack == "unknown":
        payload["unknown"] = "field"
    elif attack == "string_boolean":
        payload["report_only"] = "true"
    elif attack == "integer_weight":
        payload["production_weight"] = 0
    elif attack == "nonzero_weight":
        payload["production_weight"] = 0.01
    elif attack == "nonfinite_weight":
        payload["production_weight"] = float("nan")
    else:
        payload["metadata"] = {"source_sha256": "4" * 64}

    with pytest.raises((TypeError, ValueError)):
        type(overlay).from_dict(payload)


def test_legacy_builders_default_off_without_reading_inputs() -> None:
    reads: list[str] = []

    class Poison:
        def __getattribute__(self, name: str):
            reads.append(name)
            raise AssertionError(f"legacy wrapper read: {name}")

    class PoisonSequence:
        def __iter__(self):
            raise AssertionError("legacy batch iterated in off mode")

    assert build_calibrated_posterior_overlay(Poison(), Poison()) is None
    assert build_calibrated_posterior_overlays(
        PoisonSequence(),  # type: ignore[arg-type]
        Poison(),
    ) == []
    assert reads == []


def test_optimizer_bridge_rejects_legal_v2_shadow_overlay() -> None:
    model = _model(_curve(CalibrationCurveKey()))
    overlay = _run_shadow(_posterior(), model)

    with pytest.raises(ValueError, match="report-only"):
        build_candidate_from_overlay(overlay, current_weight=0.25)


def test_overlay_schema_bump_does_not_change_mainline_version_payload() -> None:
    assert versioning.POSTERIOR_OVERLAY_SCHEMA_VERSION.endswith(
        "posterior-overlay.v2"
    )
    assert versioning.PORTFOLIO_OPTIMIZER_SCHEMA_VERSION.endswith(
        "portfolio-optimizer.v2"
    )
    assert versioning.ARCHITECTURE_VERSION == "13.0.0-stable"
    assert versioning.output_version_payload() == {
        "architecture_version": "13.0.0-stable",
        "branch_schema_version": "branch-schema.v13.four-branch",
        "calibration_schema_version": "2026-03-22.calibration.v2",
        "ic_protocol_version": "ic-protocol.v13.four-branch",
        "report_protocol_version": "report-protocol.v13.four-branch",
    }


def test_production_surfaces_do_not_import_overlay_or_optimizer_bridges() -> None:
    root = Path(__file__).resolve().parents[2]
    restricted_paths = [
        *sorted((root / "quant_investor" / "market" / "dag").rglob("*.py")),
        root / "quant_investor" / "market" / "dag_executor.py",
        root / "quant_investor" / "control_chain.py",
        *sorted((root / "quant_investor" / "pipeline").rglob("*.py")),
        root / "quant_investor" / "agents" / "orchestrator.py",
        root / "quant_investor" / "agents" / "risk_guard.py",
        root / "quant_investor" / "agents" / "portfolio_constructor.py",
    ]
    forbidden_modules = {"posterior_overlay", "portfolio_optimizer"}
    forbidden_symbols = {
        "run_calibrated_posterior_overlay",
        "build_calibrated_posterior_overlay",
        "build_calibrated_posterior_overlays",
        "attach_overlay_metadata",
        "build_candidate_from_overlay",
        "build_candidates_from_overlays",
    }
    violations: list[str] = []
    for path in restricted_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(alias.name for alias in node.names)]
            else:
                names = []
            if any(
                any(part in name for part in forbidden_modules)
                or name in forbidden_symbols
                for name in names
            ):
                violations.append(f"{path}:{getattr(node, 'lineno', 0)}")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                else:
                    call_name = ""
                if call_name not in {"__import__", "import_module"}:
                    continue
                dynamic_args = [
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                ]
                if any(
                    any(part in value for part in forbidden_modules)
                    for value in dynamic_args
                ):
                    violations.append(
                        f"{path}:{getattr(node, 'lineno', 0)}:dynamic"
                    )

    assert violations == []
