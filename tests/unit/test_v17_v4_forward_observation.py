from __future__ import annotations

from datetime import date, timedelta
import hashlib
from typing import Any

import pytest

from quant_investor.v17_v4_contract import canonical_resource_bytes
from quant_investor.v17_v4_runtime.factor_observation import (
    LABEL_HORIZONS,
    FactorObservationError,
    build_factor_forward_label,
    build_factor_observation,
    factor_observation_ref,
    validate_factor_forward_label,
    validate_factor_observation,
)
from quant_investor.v17_v4_runtime.forward_evaluation_receipt import (
    ForwardEvaluationReceiptError,
    build_existing_factor_inventory,
    build_factor_evaluation_receipt,
    build_forward_evidence_origin_inventory,
    build_industry_evaluation_receipt,
    build_strategy_evaluation_receipt,
    validate_evaluation_receipt,
    validate_existing_factor_inventory,
    validate_forward_evidence_origin_inventory,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
STRATEGY_ID = "cn-forward-research"
DECISION_SESSION = "2026-01-02"
OBSERVATION_CUTOFF = "2026-01-02T07:00:00Z"
FACTOR_NAME = "factor-a"
FACTOR_SET_SHA = SHA_C
QUANT_POLICY_SHA = SHA_D


def _ref(
    artifact_id: str,
    artifact_version: str,
    relative_path: str,
    *,
    sha: str = SHA_A,
    cutoff: str = OBSERVATION_CUTOFF,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "byte_sha256": sha,
        "cutoff": cutoff,
        "relative_path": relative_path,
        "semantic_sha256": sha,
        "strategy_id": STRATEGY_ID,
    }


def _request_ref() -> dict[str, str]:
    return _ref(
        "request-2026-01-02",
        "myquant.v17.v4.forward-run-request.v1",
        "data/private/v17_v4_runs/forward_requests/request-2026-01-02.json",
    )


def _factor_ref() -> dict[str, str]:
    return _ref(
        FACTOR_NAME,
        "myquant.v17.v4.factor-definition.v1",
        "data/private/v17_v4_sources/factors/factor-a.json",
        sha=SHA_B,
    )


def _observation_run_ref() -> dict[str, str]:
    return _ref(
        "observation-run-2026-01-02",
        "myquant.v17.v4.forward-observation-run.v1",
        "results/v17_v4_shadow/forward_evidence/observations/run.json",
    )


def _observation(
    *,
    source_refs: list[dict[str, str]] | None = None,
    observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return build_factor_observation(
        observation_id="factor-observation-2026-01-02",
        strategy_id=STRATEGY_ID,
        decision_session=DECISION_SESSION,
        cutoff=OBSERVATION_CUTOFF,
        factor_ref=_factor_ref(),
        request_ref=_request_ref(),
        source_refs=(
            [
                _ref(
                    "market",
                    "myquant.v17.v4.market-source.v1",
                    "data/private/v17_v4_sources/market.json",
                ),
                _ref(
                    "pit",
                    "myquant.v17.v4.pit-source.v1",
                    "data/private/v17_v4_sources/pit.json",
                    sha=SHA_C,
                ),
            ]
            if source_refs is None
            else source_refs
        ),
        observations=(
            [
                {"status": "AVAILABLE", "symbol": "600000.SH", "value": "0.2"},
                {"status": "AVAILABLE", "symbol": "000001.SZ", "value": "0.8"},
            ]
            if observations is None
            else observations
        ),
    )


def _open_sessions(count: int = 61) -> list[str]:
    sessions: list[str] = []
    current = date.fromisoformat(DECISION_SESSION)
    while len(sessions) < count:
        if current.weekday() < 5:
            sessions.append(current.isoformat())
        current += timedelta(days=1)
    return sessions


def _label(
    horizon: int = 1,
    *,
    matured_at: str | None = None,
) -> dict[str, Any]:
    sessions = _open_sessions()
    maturity = matured_at or f"{sessions[horizon]}T07:00:00Z"
    return build_factor_forward_label(
        strategy_id=STRATEGY_ID,
        decision_session=DECISION_SESSION,
        observation_run_ref=_observation_run_ref(),
        horizon_sessions=horizon,
        shanghai_open_sessions=sessions,
        origin_adjusted_closes={"000001.SZ": "10", "600000.SH": "20"},
        end_adjusted_closes={"000001.SZ": "11", "600000.SH": "18"},
        market_origin_adjusted_close="100",
        market_end_adjusted_close="105",
        industry_by_symbol={
            "000001.SZ": "banks",
            "600000.SH": "technology",
        },
        industry_origin_adjusted_closes={"banks": "100", "technology": "200"},
        industry_end_adjusted_closes={"banks": "95", "technology": "220"},
        evidence_refs=[
            _ref(
                "label-market",
                "myquant.v17.v4.label-market.v1",
                "data/private/v17_v4_sources/label-market.json",
                cutoff=maturity,
            )
        ],
        matured_at=maturity,
    )


def _artifact_ref(
    artifact: dict[str, Any],
    *,
    identity_field: str,
    relative_path: str,
) -> dict[str, str]:
    return {
        "artifact_id": str(artifact[identity_field]),
        "artifact_version": str(artifact["version"]),
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(artifact)).hexdigest(),
        "cutoff": str(artifact["cutoff"]),
        "relative_path": relative_path,
        "semantic_sha256": str(artifact["semantic_sha256"]),
        "strategy_id": str(artifact["strategy_id"]),
    }


def _lineage(label: dict[str, Any]) -> dict[str, Any]:
    return {
        "factor_definition_sha256": SHA_B,
        "factor_name": FACTOR_NAME,
        "factor_set_sha256": FACTOR_SET_SHA,
        "horizon_sessions": int(label["horizon_sessions"]),
        "quant_policy_sha256": QUANT_POLICY_SHA,
        "source_lineage_sha256": str(label["source_lineage_sha256"]),
    }


def _inventories() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    observation = _observation()
    label = _label()
    label_ref = _artifact_ref(
        label,
        identity_field="label_id",
        relative_path="results/v17_v4_shadow/forward_labels/label-1d.json",
    )
    origin_inventory = build_forward_evidence_origin_inventory(
        inventory_id="forward-origins-1d",
        strategy_id=STRATEGY_ID,
        decision_session=DECISION_SESSION,
        cutoff=str(label["cutoff"]),
        request_ref=_request_ref(),
        origins=[
            {
                "evidence_ref": label_ref,
                "lineage_key": _lineage(label),
                "origin": DECISION_SESSION,
            }
        ],
    )
    observation_ref = factor_observation_ref(
        observation,
        relative_path=("results/v17_v4_shadow/forward_observations/factor-a.json"),
    )
    existing_inventory = build_existing_factor_inventory(
        inventory_id="existing-factors-1",
        strategy_id=STRATEGY_ID,
        decision_session=DECISION_SESSION,
        cutoff=str(label["cutoff"]),
        request_ref=_request_ref(),
        source_refs=[observation_ref],
        factors=[
            {
                "definition_sha256": SHA_B,
                "exposure_observation_refs": [observation_ref],
                "factor_name": FACTOR_NAME,
                "factor_ref": _factor_ref(),
                "lifecycle": "ACTIVE",
            }
        ],
    )
    return origin_inventory, existing_inventory, label


def test_full_universe_observation_is_deterministic_and_sorted() -> None:
    first = _observation()
    second = _observation(
        source_refs=list(reversed(first["source_refs"])),
        observations=list(reversed(first["observations"])),
    )

    assert first == second
    assert [row["symbol"] for row in first["observations"]] == [
        "000001.SZ",
        "600000.SH",
    ]
    assert validate_factor_observation(first) == first


def test_observation_rejects_evidence_available_after_cutoff() -> None:
    with pytest.raises(FactorObservationError, match="source_refs\\[0\\]_after_cutoff"):
        _observation(
            source_refs=[
                _ref(
                    "future-market",
                    "myquant.v17.v4.market-source.v1",
                    "data/private/v17_v4_sources/future-market.json",
                    cutoff="2026-01-02T07:00:01Z",
                )
            ]
        )


@pytest.mark.parametrize("horizon", LABEL_HORIZONS)
def test_exact_forward_horizons_use_adjusted_close_arithmetic(
    horizon: int,
) -> None:
    label = _label(horizon)
    validated = validate_factor_forward_label(
        label,
        observation_run_ref=_observation_run_ref(),
    )

    assert validated["horizon_sessions"] == horizon
    assert len(validated["shanghai_open_sessions"]) == horizon + 1
    first = validated["label_rows"][0]
    assert first["total_return"] == "0.1"
    assert first["market_adjusted_return"] == "0.05"
    assert first["industry_adjusted_return"] == "0.15"
    assert first["cost_adjusted_return"] == "0.098"


def test_label_rejects_maturity_before_end_session_close() -> None:
    sessions = _open_sessions()
    with pytest.raises(
        FactorObservationError,
        match="label_not_matured_at_end_session_close",
    ):
        _label(matured_at=f"{sessions[1]}T06:59:59Z")


def test_origin_inventory_blocks_duplicate_origin_conflict() -> None:
    label = _label()
    label_ref = _artifact_ref(
        label,
        identity_field="label_id",
        relative_path="results/v17_v4_shadow/forward_labels/label-1d.json",
    )
    conflicting = {**label_ref, "byte_sha256": SHA_D}
    origin = {
        "evidence_ref": label_ref,
        "lineage_key": _lineage(label),
        "origin": DECISION_SESSION,
    }

    with pytest.raises(
        ForwardEvaluationReceiptError,
        match="DUPLICATE_ORIGIN_CONFLICT",
    ):
        build_forward_evidence_origin_inventory(
            inventory_id="conflicting-origins",
            strategy_id=STRATEGY_ID,
            decision_session=DECISION_SESSION,
            cutoff=str(label["cutoff"]),
            request_ref=_request_ref(),
            origins=[origin, {**origin, "evidence_ref": conflicting}],
        )


@pytest.mark.parametrize(
    ("builder", "subject"),
    [
        (build_factor_evaluation_receipt, {"factor_name": FACTOR_NAME}),
        (
            build_industry_evaluation_receipt,
            {"factor_name": FACTOR_NAME, "industry_id": "banks"},
        ),
        (
            build_strategy_evaluation_receipt,
            {"factor_name": FACTOR_NAME, "strategy_id": STRATEGY_ID},
        ),
    ],
)
def test_evaluation_receipts_are_sealed_and_have_no_authority(
    builder: Any,
    subject: dict[str, str],
) -> None:
    origins, existing, label = _inventories()
    receipt = builder(
        receipt_id=f"{builder.__name__}-1",
        **subject,
        factor_definition_sha256=SHA_B,
        factor_set_sha256=FACTOR_SET_SHA,
        quant_policy_sha256=QUANT_POLICY_SHA,
        horizon_sessions=1,
        source_lineage_sha256=str(label["source_lineage_sha256"]),
        cutoff=str(origins["cutoff"]),
        created_at=str(origins["cutoff"]),
        metric_state="COMPLETE",
        metrics={"rank_ic": "0.08", "sample_count": 1},
        observation_run_ref=_observation_run_ref(),
        forward_evidence_origin_inventory=origins,
        forward_evidence_origin_inventory_path=("results/v17_v4_shadow/evaluation/origins.json"),
        existing_factor_inventory=existing,
        existing_factor_inventory_path=("results/v17_v4_shadow/evaluation/existing.json"),
    )

    assert (
        validate_evaluation_receipt(
            receipt,
            forward_evidence_origin_inventory=origins,
            existing_factor_inventory=existing,
        )
        == receipt
    )
    assert not any(receipt["authority"].values())
    assert receipt["shadow_only"] is True
    assert receipt["promotion_eligible"] is False


def test_inventory_and_unavailable_metric_state_replay() -> None:
    origins, existing, label = _inventories()
    assert validate_forward_evidence_origin_inventory(origins) == origins
    assert validate_existing_factor_inventory(existing) == existing
    receipt = build_factor_evaluation_receipt(
        receipt_id="factor-a-unavailable",
        factor_name=FACTOR_NAME,
        factor_definition_sha256=SHA_B,
        factor_set_sha256=FACTOR_SET_SHA,
        quant_policy_sha256=QUANT_POLICY_SHA,
        horizon_sessions=1,
        source_lineage_sha256=str(label["source_lineage_sha256"]),
        cutoff=str(origins["cutoff"]),
        created_at=str(origins["cutoff"]),
        metric_state="UNAVAILABLE",
        metrics=None,
        unavailable_reason="insufficient-matured-origins",
        observation_run_ref=_observation_run_ref(),
        forward_evidence_origin_inventory=origins,
        forward_evidence_origin_inventory_path=("results/v17_v4_shadow/evaluation/origins.json"),
        existing_factor_inventory=existing,
        existing_factor_inventory_path=("results/v17_v4_shadow/evaluation/existing.json"),
    )

    assert receipt["completeness"] == "UNAVAILABLE"
    assert receipt["metric_rows"] == []
    assert receipt["blockers"] == ["insufficient-matured-origins"]
