from __future__ import annotations

import copy
from dataclasses import replace
import hashlib

import pytest

from quant_investor.v17_v5_contract import canonical_bytes, seal_semantic, validate_artifact
from quant_investor.v17_v5_contract.validators import (
    ArtifactContractError,
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
        cutoff=f"2026-01-{index + 2:02d}T08:00:00Z",
        relative_path=f"tests/fixtures/v17_v5_sprint1b/{kind}/{index:03d}.json",
        semantic_sha256=_sha(f"{kind}-{index}-semantic"),
        strategy_id=strategy_id,
        version=(
            "myquant.v17.v4.regime-evidence.v3"
            if kind == "regime-evidence"
            else (
                "myquant.v17.v4.regime-state-checkpoint.v1"
                if kind == "regime-state-checkpoint"
                else f"myquant.v17.v4.{kind}.v1"
            )
        ),
    )


def _regime(
    index: int,
    *,
    state: str = "趋势上涨",
    continuity: str = "CONTIGUOUS",
) -> RegimeEvidenceSnapshot:
    segment_position = 0 if continuity in {"GENESIS", "RECOVERY", "ROLLOVER"} else (index % 63) + 1
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
        source_commit="6a2fa23dec68d87eb686464a86d8ba8997416310",
        source_version="myquant.v17.v4.regime-evidence.v3",
        state_order=["趋势上涨", "震荡低波", "震荡高波", "趋势下跌", "未知"],
        state_probabilities={
            "趋势上涨": "0.8" if state == "趋势上涨" else "0.0",
            "震荡低波": "0.2" if state == "趋势上涨" else "0.0",
            "震荡高波": "0.0",
            "趋势下跌": "1.0" if state == "趋势下跌" else "0.0",
            "未知": "1.0" if state == "未知" else "0.0",
        },
        strategy_id="cn-strategy",
        checkpoint_ref=replace(
            _v4_ref("regime-state-checkpoint", index),
            cutoff=f"2026-01-{index + 2:02d}T07:30:00Z",
        ),
        finalized=True,
        continuity_kind=continuity,
        segment_id=_sha(f"segment-{index // 64}"),
        segment_index=index // 64,
        segment_position=segment_position,
        transition_commitment_sha256=_sha(f"record-{index}"),
        chain_digest_sha256=_sha(f"chain-{index}"),
        segment_accumulator_sha256=_sha(f"segment-accumulator-{index}"),
    )


def _origin(
    index: int, *, state: str = "趋势上涨", rank_ic: str | None = "0.1"
) -> FactorRegimeOriginInput:
    return FactorRegimeOriginInput(
        comparable_symbol_count=8,
        coverage="0.8",
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
        {"probability": "0.000000000000", "regime_state": "震荡高波"},
        {"probability": "0.000000000000", "regime_state": "趋势下跌"},
        {"probability": "0.000000000000", "regime_state": "未知"},
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
        regime_evidence=replace(_regime(0), published_at="2026-01-02T08:00:01Z"),
    )
    late_effective = replace(
        _origin(0),
        regime_evidence=replace(_regime(0), effective_session="2026-01-03"),
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


def test_regime_ref_version_cutoff_and_sealed_hard_state_are_preserved() -> None:
    origin = _origin(0)
    regime = origin.regime_evidence
    with pytest.raises(FactorRegimeOriginInventoryError, match="source version"):
        _inventory(replace(origin, regime_evidence=replace(regime, source_version="other.v2")))
    with pytest.raises(FactorRegimeOriginInventoryError, match="cutoff does not match"):
        _inventory(
            replace(
                origin,
                regime_evidence=replace(regime, cutoff="2026-01-01T07:00:00Z"),
            )
        )
    posterior_argmax_differs = replace(
        origin,
        regime_evidence=replace(
            regime,
            state_probabilities={
                "趋势上涨": "0.0",
                "震荡低波": "1.0",
                "震荡高波": "0.0",
                "趋势下跌": "0.0",
                "未知": "0.0",
            },
        ),
    )
    artifact = _inventory(posterior_argmax_differs)
    assert artifact["origin_rows"][0]["regime_state"] == "趋势上涨"


def test_v3_origin_binding_rejects_stale_future_smoothed_and_subset_regime() -> None:
    stale = replace(
        _origin(0),
        regime_evidence=replace(_regime(0), decision_session="2026-01-01"),
    )
    wrong_observed = replace(
        _origin(0),
        regime_evidence=replace(_regime(0), observed_through_session="2026-01-02"),
    )
    smoothed = replace(_origin(0), regime_evidence=replace(_regime(0), smoothing_used=True))
    subset = replace(_origin(0), regime_evidence=replace(_regime(0), scope_kind="SUBSET"))

    with pytest.raises(FactorRegimeOriginInventoryError, match="decision_session"):
        _inventory(stale)
    with pytest.raises(FactorRegimeOriginInventoryError, match="previous open"):
        _inventory(wrong_observed)
    with pytest.raises(FactorRegimeOriginInventoryError, match="smoothing_used"):
        _inventory(smoothed)
    with pytest.raises(FactorRegimeOriginInventoryError, match="scope_kind"):
        _inventory(subset)


def test_unknown_regime_is_excluded_and_counted_without_conditionable_bucket() -> None:
    artifact = _inventory(_origin(0, state="未知"))

    assert artifact["origin_count"] == 0
    assert artifact["excluded_origin_count"] == 1
    assert artifact["regime_counts"] == []
    assert artifact["limitation_codes"] == ["REGIME_HARD_STATE_UNKNOWN"]
    assert artifact["excluded_origin_rows"][0]["regime_state"] == "未知"
    assert validate_factor_regime_origin_inventory(artifact) == artifact
    assert validate_artifact(artifact) == artifact


@pytest.mark.parametrize(
    ("continuity", "eligible", "limitation"),
    (
        ("GENESIS", False, "REGIME_CONTINUITY_GENESIS"),
        ("RECOVERY", False, "REGIME_CONTINUITY_RECOVERY"),
        ("CONTIGUOUS", True, None),
        ("ROLLOVER", True, None),
    ),
)
def test_v3_continuity_conditioning_eligibility_is_sealed(
    continuity: str,
    eligible: bool,
    limitation: str | None,
) -> None:
    origin = replace(
        _origin(0),
        regime_evidence=_regime(0, continuity=continuity),
    )
    artifact = _inventory(origin)

    assert artifact["origin_count"] == (1 if eligible else 0)
    assert artifact["excluded_origin_count"] == (0 if eligible else 1)
    if limitation is not None:
        assert artifact["limitation_codes"] == [limitation]


def test_v3_not_finalized_fails_closed() -> None:
    origin = _origin(0)
    with pytest.raises(
        FactorRegimeOriginInventoryError,
        match="REGIME_EVIDENCE_V3_NOT_FINALIZED",
    ):
        _inventory(
            replace(
                origin,
                regime_evidence=replace(origin.regime_evidence, finalized=False),
            )
        )


def test_multi_segment_conditioning_uses_only_current_sealed_continuity() -> None:
    continuities = (
        "GENESIS",
        "CONTIGUOUS",
        "CONTIGUOUS",
        "ROLLOVER",
        "RECOVERY",
        "CONTIGUOUS",
    )
    origins = [
        replace(
            _origin(index),
            regime_evidence=_regime(index, continuity=continuity),
        )
        for index, continuity in enumerate(continuities)
    ]

    artifact = _inventory(*origins)

    assert artifact["origin_count"] == 4
    assert artifact["excluded_origin_count"] == 2
    assert [row["regime_continuity_kind"] for row in artifact["origin_rows"]] == [
        "CONTIGUOUS",
        "CONTIGUOUS",
        "ROLLOVER",
        "CONTIGUOUS",
    ]
    assert {row["regime_continuity_kind"] for row in artifact["excluded_origin_rows"]} == {
        "GENESIS",
        "RECOVERY",
    }
    assert artifact["limitation_codes"] == [
        "REGIME_CONTINUITY_GENESIS",
        "REGIME_CONTINUITY_RECOVERY",
    ]


def test_two_distinct_unknown_regime_evidence_refs_for_one_origin_fail_closed() -> None:
    first = _origin(0, state="未知")
    second_regime = replace(
        _regime(0, state="未知"),
        regime_artifact_ref=replace(
            _regime(0, state="未知").regime_artifact_ref,
            artifact_id="regime-evidence.alternate",
            byte_sha256=_sha("alternate-regime-bytes"),
            semantic_sha256=_sha("alternate-regime-semantic"),
        ),
    )
    second = replace(
        _origin(0, state="未知"),
        origin_id="origin.unknown-alternate",
        regime_evidence=second_regime,
    )

    with pytest.raises(FactorRegimeOriginInventoryError, match="duplicate origin"):
        _inventory(first, second)


def test_regime_source_commit_must_match_active_predecessor_pin() -> None:
    origin = _origin(0)
    wrong_source = replace(
        origin,
        regime_evidence=replace(
            origin.regime_evidence,
            source_commit="ec1370553fdf7ca0951ec4b03ea9fc426a872b4e",
        ),
    )

    with pytest.raises(FactorRegimeOriginInventoryError, match="pinned V4 predecessor"):
        _inventory(wrong_source)


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
    with pytest.raises(FactorRegimeOriginInventoryError, match="row identity"):
        validate_factor_regime_origin_inventory(_reseal(bad_row_identity))

    bad_inventory_id = copy.deepcopy(artifact)
    bad_inventory_id["inventory_id"] = "factor-regime-origin-inventory-forged"
    with pytest.raises(FactorRegimeOriginInventoryError, match="inventory identity"):
        validate_factor_regime_origin_inventory(_reseal(bad_inventory_id))

    bad_coverage = copy.deepcopy(artifact)
    row = bad_coverage["origin_rows"][0]
    row["coverage"] = "0.700000000000"
    identity_row = dict(row)
    identity_row.pop("row_identity_sha256")
    row["row_identity_sha256"] = hashlib.sha256(canonical_bytes(identity_row)).hexdigest()
    with pytest.raises(FactorRegimeOriginInventoryError, match="identity"):
        validate_factor_regime_origin_inventory(_reseal(bad_coverage))

    bad_ref = copy.deepcopy(artifact)
    row = bad_ref["origin_rows"][0]
    row["factor_observation_ref"]["strategy_id"] = "other-strategy"
    identity_row = dict(row)
    identity_row.pop("row_identity_sha256")
    row["row_identity_sha256"] = hashlib.sha256(canonical_bytes(identity_row)).hexdigest()
    with pytest.raises(FactorRegimeOriginInventoryError, match="identity"):
        validate_factor_regime_origin_inventory(_reseal(bad_ref))


@pytest.mark.parametrize(
    ("continuity", "state", "limitation_codes"),
    [
        ("CONTIGUOUS", "趋势上涨", ["REGIME_HARD_STATE_UNKNOWN"]),
        ("GENESIS", "趋势上涨", ["REGIME_CONTINUITY_RECOVERY"]),
        ("RECOVERY", "未知", ["REGIME_CONTINUITY_RECOVERY"]),
    ],
)
def test_resealed_excluded_origin_must_match_exact_exclusion_facts(
    continuity: str,
    state: str,
    limitation_codes: list[str],
) -> None:
    artifact = _inventory(
        replace(
            _origin(0, state="未知"),
            regime_evidence=_regime(0, state="未知", continuity="GENESIS"),
        )
    )
    tampered = copy.deepcopy(artifact)
    row = tampered["excluded_origin_rows"][0]
    row["regime_continuity_kind"] = continuity
    row["regime_state"] = state
    row["row_limitation_codes"] = limitation_codes
    identity_row = dict(row)
    identity_row.pop("row_identity_sha256")
    row["row_identity_sha256"] = hashlib.sha256(canonical_bytes(identity_row)).hexdigest()
    tampered["limitation_codes"] = sorted(limitation_codes)
    resealed = _reseal(tampered)

    with pytest.raises(FactorRegimeOriginInventoryError, match="excluded origin"):
        validate_factor_regime_origin_inventory(resealed)
    with pytest.raises(ArtifactContractError, match="excluded origin"):
        validate_artifact(resealed)
