from __future__ import annotations

import copy
from dataclasses import replace
import hashlib

import pytest

from quant_investor.v17_v5_contract import canonical_bytes, seal_semantic, validate_artifact
from quant_investor.v17_v5_contract.validators import ArtifactContractError
from quant_investor.v17_v5_contract.validators import (
    FACTOR_REGIME_DIAGNOSTIC_POLICY_BYTE_SHA256,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_ID,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_SEMANTIC_SHA256,
    FACTOR_REGIME_DIAGNOSTIC_POLICY_VERSION,
)
from quant_investor.v17_v5_runtime.factor_regime_origin_inventory import (
    ContentArtifactRef,
    FactorRegimeOriginInput,
    FactorRegimeOriginInventoryError,
    RegimeEvidenceSnapshot,
    build_factor_regime_origin_inventory,
    validate_factor_regime_origin_inventory,
    validate_factor_regime_origin_inventory_replay,
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


def _v4_ref(kind: str, index: int, *, strategy_id: str = "cn-strategy") -> ContentArtifactRef:
    return ContentArtifactRef(
        artifact_id=f"{kind}.{index:03d}",
        byte_sha256=_sha(f"{kind}-{index}-bytes"),
        cutoff=f"2026-01-{index + 1:02d}T08:00:00Z",
        relative_path=f"tests/fixtures/v17_v5_sprint1b/{kind}/{index:03d}.json",
        semantic_sha256=_sha(f"{kind}-{index}-semantic"),
        strategy_id=strategy_id,
        version=f"myquant.v17.v4.{kind}.v1",
    )


def _regime(index: int, *, state: str = "趋势上涨") -> RegimeEvidenceSnapshot:
    return RegimeEvidenceSnapshot(
        available_at=f"2026-01-{index + 1:02d}T07:00:00Z",
        cutoff=f"2026-01-{index + 1:02d}T07:30:00Z",
        decision_session=f"2026-01-{index + 1:02d}",
        effective_session=f"2026-01-{index + 1:02d}",
        published_at=f"2026-01-{index + 1:02d}T07:45:00Z",
        regime_artifact_ref=replace(
            _v4_ref("regime-evidence", index),
            cutoff=f"2026-01-{index + 1:02d}T07:30:00Z",
        ),
        regime_state=state,
        source_version="myquant.v17.v4.regime-evidence.v1",
        state_probabilities={state: "0.8", "震荡低波": "0.2"},
        strategy_id="cn-strategy",
    )


def _origin(
    index: int, *, state: str = "趋势上涨", rank_ic: str | None = "0.1"
) -> FactorRegimeOriginInput:
    return FactorRegimeOriginInput(
        comparable_symbol_count=8,
        coverage="0.8",
        decision_session=f"2026-01-{index + 1:02d}",
        eligible_symbol_count=10,
        factor_evidence_ref=_v4_ref("forward-evaluation-receipt", index),
        factor_implementation_sha256=_sha("implementation"),
        factor_name="cn_factor_test",
        factor_observation_ref=_v4_ref("factor-universe-observation", index),
        label_end_session=f"2026-02-{index + 1:02d}",
        label_horizon_sessions=20,
        label_origin_session=f"2026-01-{index + 1:02d}",
        matured_label_ref=_v4_ref("forward-label", index),
        observation_run_ref=_v4_ref("forward-observation-run", index),
        origin_cutoff=f"2026-01-{index + 1:02d}T08:00:00Z",
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


def test_synthetic_test_only_inventory_is_deterministic_and_preserves_regime_enum() -> None:
    first = _inventory(_origin(1, state="趋势下跌"), _origin(0, state="趋势上涨"))
    second = _inventory(_origin(0, state="趋势上涨"), _origin(1, state="趋势下跌"))

    assert first == second
    assert first["origin_count"] == 2
    assert first["regime_counts"] == [
        {"origin_count": 1, "regime_state": "趋势上涨"},
        {"origin_count": 1, "regime_state": "趋势下跌"},
    ]
    row = first["origin_rows"][0]
    assert row["regime_state"] == "趋势上涨"
    for field in (
        "factor_evidence_ref",
        "factor_observation_ref",
        "matured_label_ref",
        "observation_run_ref",
        "request_ref",
        "source_locator_ref",
        "regime_evidence_ref",
    ):
        assert row[field]["relative_path"].startswith("tests/fixtures/v17_v5_sprint1b/")
    assert row["state_probabilities"] == [
        {"probability": "0.800000000000", "regime_state": "趋势上涨"},
        {"probability": "0.200000000000", "regime_state": "震荡低波"},
    ]
    assert validate_factor_regime_origin_inventory(first) == first
    assert validate_artifact(first) == first
    assert (
        validate_factor_regime_origin_inventory_replay(
            first,
            created_at="2026-03-31T00:00:00Z",
            cutoff="2026-03-30T00:00:00Z",
            factor_implementation_sha256=_sha("implementation"),
            factor_name="cn_factor_test",
            origin_rows=[_origin(1, state="趋势下跌"), _origin(0, state="趋势上涨")],
            policy_ref=_policy_ref(),
            strategy_id="cn-strategy",
        )
        == first
    )


def test_regime_publication_or_effective_after_origin_fails_closed() -> None:
    late_publication = replace(
        _origin(0),
        regime_evidence=replace(_regime(0), published_at="2026-01-01T08:00:01Z"),
    )
    late_effective = replace(
        _origin(0),
        regime_evidence=replace(_regime(0), effective_session="2026-01-02"),
    )

    with pytest.raises(FactorRegimeOriginInventoryError, match="published_at"):
        _inventory(late_publication)
    with pytest.raises(FactorRegimeOriginInventoryError, match="effective_session"):
        _inventory(late_effective)


def test_duplicate_origin_strategy_and_implementation_mismatch_fail_closed() -> None:
    duplicate = replace(_origin(0), origin_id="origin.alias")
    strategy_mismatch = replace(
        _origin(0),
        regime_evidence=replace(_regime(0), strategy_id="cn-other"),
    )
    implementation_mismatch = replace(_origin(0), factor_implementation_sha256=_sha("other"))

    with pytest.raises(FactorRegimeOriginInventoryError, match="duplicate origin"):
        _inventory(_origin(0), duplicate)
    with pytest.raises(FactorRegimeOriginInventoryError, match="strategy_id mismatch"):
        _inventory(strategy_mismatch)
    with pytest.raises(FactorRegimeOriginInventoryError, match="stratum mismatch"):
        _inventory(implementation_mismatch)
    with pytest.raises(FactorRegimeOriginInventoryError, match="strategy_id mismatch"):
        _inventory(
            replace(
                _origin(0),
                factor_observation_ref=_v4_ref(
                    "factor-universe-observation",
                    0,
                    strategy_id="other-strategy",
                ),
            )
        )


def test_regime_ref_version_cutoff_and_hard_state_posterior_must_close() -> None:
    origin = _origin(0)
    regime = origin.regime_evidence
    with pytest.raises(FactorRegimeOriginInventoryError, match="source version"):
        _inventory(replace(origin, regime_evidence=replace(regime, source_version="other.v1")))
    with pytest.raises(FactorRegimeOriginInventoryError, match="cutoff does not match"):
        _inventory(
            replace(
                origin,
                regime_evidence=replace(regime, cutoff="2026-01-01T07:00:00Z"),
            )
        )
    with pytest.raises(FactorRegimeOriginInventoryError, match="sealed hard regime state"):
        _inventory(
            replace(
                origin,
                regime_evidence=replace(
                    regime,
                    state_probabilities={"震荡低波": "1.0"},
                ),
            )
        )


def test_non_20_session_label_bad_coverage_and_v5_path_ref_fail_closed() -> None:
    bad_horizon = replace(_origin(0), label_horizon_sessions=5)
    bad_coverage = replace(_origin(0), coverage="0.7")
    pathless_v4 = replace(
        _origin(0),
        factor_observation_ref=replace(
            _v4_ref("factor-universe-observation", 0), relative_path=None
        ),
    )

    with pytest.raises(FactorRegimeOriginInventoryError, match="horizon"):
        _inventory(bad_horizon)
    with pytest.raises(FactorRegimeOriginInventoryError, match="coverage"):
        _inventory(bad_coverage)
    with pytest.raises(FactorRegimeOriginInventoryError, match="relative_path"):
        _inventory(pathless_v4)


def test_artifact_bytes_are_deterministic_and_contain_no_governance_action_fields() -> None:
    artifact = _inventory(_origin(0))
    encoded = canonical_bytes(artifact)

    assert encoded == canonical_bytes(_inventory(_origin(0)))
    assert b"factor_weight" not in encoded
    assert b"lifecycle_action" not in encoded
    assert b"validity" not in encoded
    assert b"NaN" not in encoded
    assert b"Infinity" not in encoded


def test_public_contract_rejects_resealed_identity_coverage_and_ref_drift() -> None:
    artifact = _inventory(_origin(0))

    bad_row_identity = copy.deepcopy(artifact)
    bad_row_identity["origin_rows"][0]["row_identity_sha256"] = _sha("forged-row")
    with pytest.raises(ArtifactContractError, match="row identity"):
        validate_artifact(_reseal(bad_row_identity))

    bad_inventory_id = copy.deepcopy(artifact)
    bad_inventory_id["inventory_id"] = "factor-regime-origin-inventory-forged"
    with pytest.raises(ArtifactContractError, match="inventory identity"):
        validate_artifact(_reseal(bad_inventory_id))

    bad_coverage = copy.deepcopy(artifact)
    row = bad_coverage["origin_rows"][0]
    row["coverage"] = "0.700000000000"
    identity_row = dict(row)
    identity_row.pop("row_identity_sha256")
    row["row_identity_sha256"] = hashlib.sha256(canonical_bytes(identity_row)).hexdigest()
    with pytest.raises(ArtifactContractError, match="coverage"):
        validate_artifact(_reseal(bad_coverage))

    bad_ref = copy.deepcopy(artifact)
    row = bad_ref["origin_rows"][0]
    row["factor_observation_ref"]["strategy_id"] = "other-strategy"
    identity_row = dict(row)
    identity_row.pop("row_identity_sha256")
    row["row_identity_sha256"] = hashlib.sha256(canonical_bytes(identity_row)).hexdigest()
    with pytest.raises(ArtifactContractError, match="ref strategy"):
        validate_artifact(_reseal(bad_ref))
