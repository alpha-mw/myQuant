from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import pytest

from quant_investor.v17_v4_contract import (
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v4_contract.schema_validation import (
    SchemaValidationError,
    artifact_identity_field,
    schema_versions,
    validate_artifact,
)
from quant_investor.v17_v4_contract.validators import (
    ArtifactContractError,
    regime_artifact_identity,
    validate_regime_evidence_v2,
)

STATE_ORDER = ["趋势上涨", "震荡低波", "震荡高波", "趋势下跌", "未知"]
NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
SHADOW_FLAGS = {
    "formal_activation_eligible": False,
    "performance_evidence_eligible": False,
    "promotion_eligible": False,
    "shadow_only": True,
}
POLICY_REF = {
    "byte_sha256": "1" * 64,
    "relative_path": "resources/regime_inference_policy.v1.json",
    "semantic_sha256": "2" * 64,
    "version": "myquant.v17.v4.regime-inference-policy.v1",
}
TRANSITION_MATRIX = {
    "趋势上涨": {
        "趋势上涨": "0.620000000000",
        "震荡低波": "0.220000000000",
        "震荡高波": "0.100000000000",
        "趋势下跌": "0.040000000000",
        "未知": "0.020000000000",
    },
    "震荡低波": {
        "趋势上涨": "0.220000000000",
        "震荡低波": "0.460000000000",
        "震荡高波": "0.200000000000",
        "趋势下跌": "0.070000000000",
        "未知": "0.050000000000",
    },
    "震荡高波": {
        "趋势上涨": "0.120000000000",
        "震荡低波": "0.240000000000",
        "震荡高波": "0.420000000000",
        "趋势下跌": "0.170000000000",
        "未知": "0.050000000000",
    },
    "趋势下跌": {
        "趋势上涨": "0.070000000000",
        "震荡低波": "0.180000000000",
        "震荡高波": "0.250000000000",
        "趋势下跌": "0.460000000000",
        "未知": "0.040000000000",
    },
    "未知": {
        "趋势上涨": "0.200000000000",
        "震荡低波": "0.250000000000",
        "震荡高波": "0.250000000000",
        "趋势下跌": "0.200000000000",
        "未知": "0.100000000000",
    },
}


def _seal_identity(body: dict[str, Any], identity_field: str) -> dict[str, Any]:
    document = dict(body)
    document[identity_field] = regime_artifact_identity(
        document,
        identity_field=identity_field,
    )
    return seal_semantic(document)


def _reseal(document: Mapping[str, Any], identity_field: str) -> dict[str, Any]:
    body = dict(document)
    body.pop("semantic_sha256", None)
    body.pop(identity_field, None)
    return _seal_identity(body, identity_field)


def _reference(document: Mapping[str, Any], relative_path: str) -> dict[str, str]:
    raw = canonical_resource_bytes(document)
    try:
        identity_field = artifact_identity_field(document["version"])
    except SchemaValidationError:
        identity_field = next(
            key for key in document if key.endswith("_id") and key != "strategy_id"
        )
    return {
        "artifact_id": str(document[identity_field]),
        "artifact_version": str(document["version"]),
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": str(document["cutoff"]),
        "relative_path": relative_path,
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": str(document["strategy_id"]),
    }


def _terminal(
    *,
    version: str,
    identity_field: str,
    terminal_role: str,
    role_fields: Mapping[str, Any],
) -> dict[str, Any]:
    return _seal_identity(
        {
            "authority": dict(NO_AUTHORITY),
            "available_at": "2026-07-29T07:05:00Z",
            "created_at": "2026-07-29T07:05:00Z",
            "cutoff": "2026-07-30T07:00:00Z",
            "protocol_version": "myquant.v17.v4",
            "strategy_id": "synthetic-cn-strategy",
            "terminal_role": terminal_role,
            "version": version,
            **dict(role_fields),
        },
        identity_field,
    )


def _fixtures() -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    symbols = [f"{900000 + index:06d}.SH" for index in range(30)]
    calendar = _terminal(
        version="myquant.v17.v4.regime-calendar-terminal.v1",
        identity_field="calendar_terminal_id",
        terminal_role="SHANGHAI_OPEN_CALENDAR",
        role_fields={"open_sessions": ["2026-07-29", "2026-07-30"]},
    )
    membership = _terminal(
        version="myquant.v17.v4.regime-pit-membership-terminal.v1",
        identity_field="pit_membership_terminal_id",
        terminal_role="ACTIVE_PIT_MEMBERSHIP",
        role_fields={
            "active_symbols": symbols,
            "observed_through_session": "2026-07-29",
        },
    )
    market = _terminal(
        version="myquant.v17.v4.regime-market-terminal.v1",
        identity_field="market_terminal_id",
        terminal_role="SEALED_MARKET_SYMBOL_INVENTORY",
        role_fields={
            "observed_through_session": "2026-07-29",
            "symbols": symbols,
        },
    )
    calendar_ref = _reference(calendar, "synthetic/regime/calendar.json")
    membership_ref = _reference(membership, "synthetic/regime/membership.json")
    market_ref = _reference(market, "synthetic/regime/market.json")
    locator = _terminal(
        version="myquant.v17.v4.regime-source-locator-terminal.v1",
        identity_field="source_locator_terminal_id",
        terminal_role="REGIME_SOURCE_LOCATOR",
        role_fields={
            "calendar_ref": calendar_ref,
            "market_source_refs": [market_ref],
            "pit_membership_ref": membership_ref,
        },
    )
    locator_ref = _reference(locator, "synthetic/regime/source-locator.json")

    feature = _seal_identity(
        {
            "authority": dict(NO_AUTHORITY),
            "available_at": "2026-07-29T07:05:00Z",
            "average_liquidity": "0.700000000000",
            "average_return": "0.010000000000",
            "average_volatility": "0.020000000000",
            "breadth": "0.600000000000",
            "calendar_ref": calendar_ref,
            "coverage_ratio": "1.000000000000",
            "created_at": "2026-07-29T07:05:00Z",
            "cutoff": "2026-07-30T07:00:00Z",
            "effective_session": "2026-07-30",
            "fake_breakout_share": "0.100000000000",
            "full_market_symbols": symbols,
            "macro_score": "0.200000000000",
            "market_sample_count": 30,
            "market_source_refs": [market_ref],
            "median_drawdown": "0.080000000000",
            "minimum_market_sample": 30,
            "momentum_share": "0.650000000000",
            "observed_through_session": "2026-07-29",
            "open_sessions": ["2026-07-29", "2026-07-30"],
            "pit_active_symbol_count": 30,
            "pit_membership_ref": membership_ref,
            "pressure_score": "0.250000000000",
            "protocol_version": "myquant.v17.v4",
            "risk_on_score": "0.650000000000",
            "sampled": False,
            "scope_kind": "FULL_MARKET",
            "source_locator_ref": locator_ref,
            "source_scope": "FULL_PIT_MARKET",
            "state_likelihoods": {
                "趋势上涨": "0.350000000000",
                "震荡低波": "0.250000000000",
                "震荡高波": "0.150000000000",
                "趋势下跌": "0.150000000000",
                "未知": "0.100000000000",
            },
            "state_order": list(STATE_ORDER),
            "strategy_id": "synthetic-cn-strategy",
            "version": "myquant.v17.v4.regime-feature-snapshot.v1",
            "volatility_score": "0.300000000000",
            **SHADOW_FLAGS,
        },
        "feature_snapshot_id",
    )
    transition = _seal_identity(
        {
            "authority": dict(NO_AUTHORITY),
            "available_at": "2026-07-29T07:05:00Z",
            "created_at": "2026-07-29T07:05:00Z",
            "cutoff": "2026-07-30T07:00:00Z",
            "effective_session": "2026-07-30",
            "inference_policy_ref": dict(POLICY_REF),
            "observed_through_session": "2026-07-29",
            "protocol_version": "myquant.v17.v4",
            "source_evidence_refs": [],
            "state_order": list(STATE_ORDER),
            "strategy_id": "synthetic-cn-strategy",
            "transition_matrix": deepcopy(TRANSITION_MATRIX),
            "transition_source": "PINNED_NATIVE_DEFAULT_V1",
            "version": "myquant.v17.v4.regime-transition-matrix-snapshot.v1",
            **SHADOW_FLAGS,
        },
        "transition_snapshot_id",
    )
    transition_ref = _reference(transition, "synthetic/regime/transition.json")
    model = _seal_identity(
        {
            "authority": dict(NO_AUTHORITY),
            "available_at": "2026-07-29T07:05:00Z",
            "created_at": "2026-07-29T07:05:00Z",
            "cutoff": "2026-07-30T07:00:00Z",
            "effective_session": "2026-07-30",
            "formula_version": "NATIVE_HEURISTIC_LIKELIHOOD_DECIMAL_V1",
            "inference_policy_ref": dict(POLICY_REF),
            "model_id": "v17-v4-native-regime-filtered-model",
            "model_implementation_sha256": "3" * 64,
            "model_kind": "PINNED_RULE_BASED_NO_TRAINING",
            "model_training_end_session": None,
            "model_version": "PINNED_RULE_BASED_NO_TRAINING_V1",
            "observed_through_session": "2026-07-29",
            "predecessor_evidence_ref": None,
            "protocol_version": "myquant.v17.v4",
            "state_order": list(STATE_ORDER),
            "strategy_id": "synthetic-cn-strategy",
            "training_source_refs": [],
            "transition_matrix_ref": transition_ref,
            "version": "myquant.v17.v4.regime-model-snapshot.v1",
            **SHADOW_FLAGS,
        },
        "model_snapshot_id",
    )
    feature_ref = _reference(feature, "synthetic/regime/feature.json")
    model_ref = _reference(model, "synthetic/regime/model.json")
    source_refs = sorted(
        [calendar_ref, locator_ref, market_ref, membership_ref],
        key=lambda item: (
            item["relative_path"],
            item["byte_sha256"],
            item["artifact_id"],
        ),
    )
    evidence = _seal_identity(
        {
            "authority": dict(NO_AUTHORITY),
            "available_at": "2026-07-29T07:05:00Z",
            "blocker_codes": [],
            "computed_at": "2026-07-29T07:05:00Z",
            "coverage_ratio": "1.000000000000",
            "created_at": "2026-07-29T07:05:00Z",
            "cutoff": "2026-07-30T07:00:00Z",
            "decision_session": "2026-07-30",
            "effective_session": "2026-07-30",
            "feature_cutoff": "2026-07-30T07:00:00Z",
            "feature_snapshot_ref": feature_ref,
            "hard_state": "趋势上涨",
            "hard_state_derivation": "SEALED_ARGMAX_POLICY_V1",
            "inference_kind": "FILTERED_CAUSAL",
            "inference_policy_ref": dict(POLICY_REF),
            "lineage_phase": "BOOTSTRAP",
            "market_sample_count": 30,
            "minimum_market_sample": 30,
            "model_id": "v17-v4-native-regime-filtered-model",
            "model_implementation_sha256": "3" * 64,
            "model_snapshot_ref": model_ref,
            "model_training_end_session": None,
            "model_version": "PINNED_RULE_BASED_NO_TRAINING_V1",
            "no_retroactive_causal_backfill": True,
            "observed_through_session": "2026-07-29",
            "protocol_version": "myquant.v17.v4",
            "publication_phase": "PRIOR_SESSION_EFFECTIVE_NEXT_SESSION",
            "published_at": "2026-07-29T07:05:00Z",
            "same_session_execution_eligible": False,
            "scope_kind": "FULL_MARKET",
            "scope_ref": membership_ref,
            "smoothing_used": False,
            "source_refs": source_refs,
            "state_order": list(STATE_ORDER),
            "state_probabilities": {
                "趋势上涨": "0.400000000000",
                "震荡低波": "0.200000000000",
                "震荡高波": "0.150000000000",
                "趋势下跌": "0.150000000000",
                "未知": "0.100000000000",
            },
            "status": "AVAILABLE",
            "strategy_id": "synthetic-cn-strategy",
            "transition_matrix_ref": transition_ref,
            "version": "myquant.v17.v4.regime-evidence.v2",
            **SHADOW_FLAGS,
        },
        "evidence_id",
    )
    documents = {
        "calendar": calendar,
        "evidence": evidence,
        "feature": feature,
        "locator": locator,
        "market": market,
        "membership": membership,
        "model": model,
        "transition": transition,
    }
    artifacts = {
        hashlib.sha256(raw).hexdigest(): raw
        for raw in (canonical_resource_bytes(document) for document in documents.values())
    }
    return documents, artifacts


def test_regime_v2_schema_registration_identity_and_v1_bytes_are_frozen() -> None:
    expected = {
        "myquant.v17.v4.regime-calendar-terminal.v1": "calendar_terminal_id",
        "myquant.v17.v4.regime-evidence.v2": "evidence_id",
        "myquant.v17.v4.regime-feature-snapshot.v1": "feature_snapshot_id",
        "myquant.v17.v4.regime-market-terminal.v1": "market_terminal_id",
        "myquant.v17.v4.regime-model-snapshot.v1": "model_snapshot_id",
        "myquant.v17.v4.regime-pit-membership-terminal.v1": ("pit_membership_terminal_id"),
        "myquant.v17.v4.regime-source-locator-terminal.v1": ("source_locator_terminal_id"),
        "myquant.v17.v4.regime-transition-matrix-snapshot.v1": "transition_snapshot_id",
    }
    assert set(expected) <= set(schema_versions())
    for version, identity_field in expected.items():
        assert artifact_identity_field(version) == identity_field
    v1_path = (
        Path(__file__).parents[2]
        / "quant_investor/v17_v4_contract/schemas/regime_evidence.v1.schema.json"
    )
    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == (
        "49d006413465d7304c621f74f9732cc0d5636c400989d25b170ee4337ff229a3"
    )


def test_regime_snapshots_and_evidence_validate_with_recursive_typed_readback() -> None:
    documents, artifacts = _fixtures()
    calls: list[str] = []

    def loader(reference: Mapping[str, str]) -> bytes:
        calls.append(reference["byte_sha256"])
        return artifacts[reference["byte_sha256"]]

    for key in ("feature", "transition", "model", "evidence"):
        validated = validate_artifact(documents[key], artifact_loader=loader)
        assert validated.semantic_sha256 == documents[key]["semantic_sha256"]
    for key in ("feature", "transition", "model"):
        assert hashlib.sha256(canonical_resource_bytes(documents[key])).hexdigest() in calls
    assert all(documents[key]["authority"] == NO_AUTHORITY for key in documents)


def test_regime_terminals_are_schema_closed_identity_bound_and_recursively_loaded() -> None:
    documents, artifacts = _fixtures()
    calls: list[str] = []

    def loader(reference: Mapping[str, str]) -> bytes:
        calls.append(reference["artifact_version"])
        return artifacts[reference["byte_sha256"]]

    for key in ("calendar", "membership", "market", "locator"):
        validate_artifact(documents[key], artifact_loader=loader)
    assert {
        "myquant.v17.v4.regime-calendar-terminal.v1",
        "myquant.v17.v4.regime-market-terminal.v1",
        "myquant.v17.v4.regime-pit-membership-terminal.v1",
    } <= set(calls)

    drift = dict(documents["calendar"])
    drift.pop("semantic_sha256")
    drift["calendar_terminal_id"] = "f" * 64
    with pytest.raises(ArtifactContractError, match="calendar_terminal_id identity mismatch"):
        validate_artifact(seal_semantic(drift))

    open_schema = dict(documents["market"])
    open_schema.pop("semantic_sha256")
    open_schema["unregistered_scope_hint"] = "FULL"
    with pytest.raises(SchemaValidationError, match="additional properties"):
        validate_artifact(seal_semantic(open_schema))


def test_regime_evidence_rejects_semantic_and_identity_drift() -> None:
    documents, _ = _fixtures()
    semantic_drift = deepcopy(documents["evidence"])
    semantic_drift["hard_state"] = "趋势下跌"
    with pytest.raises(ArtifactContractError, match="semantic_sha256 mismatch"):
        validate_artifact(semantic_drift)

    identity_drift = dict(documents["evidence"])
    identity_drift.pop("semantic_sha256")
    identity_drift["evidence_id"] = "f" * 64
    with pytest.raises(ArtifactContractError, match="evidence_id identity mismatch"):
        validate_artifact(seal_semantic(identity_drift))


def test_regime_evidence_rejects_cutoff_after_effective_shanghai_session() -> None:
    documents, _ = _fixtures()
    body = deepcopy(documents["evidence"])
    body.pop("semantic_sha256")
    body.pop("evidence_id")
    body["cutoff"] = "2026-08-01T07:00:00Z"
    candidate = _seal_identity(body, "evidence_id")
    with pytest.raises(ArtifactContractError, match="effective Shanghai session"):
        validate_artifact(candidate)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda body: body["state_probabilities"].pop("未知"),
            "missing required properties",
        ),
        (
            lambda body: body["state_probabilities"].update({"趋势上涨": "0.500000000000"}),
            "sum exactly",
        ),
        (
            lambda body: body["state_probabilities"].update(
                {
                    "趋势上涨": "1.100000000000",
                    "震荡低波": "-0.100000000000",
                }
            ),
            "in \\[0, 1\\]",
        ),
        (
            lambda body: body.update({"hard_state": "趋势下跌"}),
            "sealed native-order argmax",
        ),
        (
            lambda body: body.update({"smoothing_used": True}),
            "does not match const",
        ),
        (
            lambda body: body["authority"].update({"execution": True}),
            "does not match const",
        ),
    ],
)
def test_regime_evidence_rejects_invalid_posterior_state_smoothing_and_authority(
    mutation: Any,
    error: str,
) -> None:
    documents, _ = _fixtures()
    body = deepcopy(documents["evidence"])
    body.pop("semantic_sha256")
    body.pop("evidence_id")
    mutation(body)
    candidate = _seal_identity(body, "evidence_id")
    with pytest.raises((ArtifactContractError, SchemaValidationError), match=error):
        validate_artifact(candidate)


def test_regime_transition_and_feature_semantics_are_not_relabelable() -> None:
    documents, _ = _fixtures()
    transition = deepcopy(documents["transition"])
    transition.pop("semantic_sha256")
    transition.pop("transition_snapshot_id")
    transition["transition_matrix"]["趋势上涨"]["趋势上涨"] = "0.610000000000"
    transition["transition_matrix"]["趋势上涨"]["震荡低波"] = "0.230000000000"
    with pytest.raises(ArtifactContractError, match="PINNED_NATIVE_DEFAULT_V1"):
        validate_artifact(_seal_identity(transition, "transition_snapshot_id"))

    feature = deepcopy(documents["feature"])
    feature.pop("semantic_sha256")
    feature.pop("feature_snapshot_id")
    feature["state_likelihoods"]["未知"] = "0.110000000000"
    with pytest.raises(ArtifactContractError, match="sum exactly"):
        validate_artifact(_seal_identity(feature, "feature_snapshot_id"))


def test_regime_typed_reference_byte_drift_and_recursive_forbidden_keys_fail_closed() -> None:
    documents, artifacts = _fixtures()
    evidence = deepcopy(documents["evidence"])

    def loader(reference: Mapping[str, str]) -> bytes:
        if reference["artifact_version"] == ("myquant.v17.v4.regime-feature-snapshot.v1"):
            return canonical_resource_bytes(documents["transition"])
        return artifacts[reference["byte_sha256"]]

    with pytest.raises(ArtifactContractError, match="feature_snapshot_ref byte SHA mismatch"):
        validate_artifact(evidence, artifact_loader=loader)

    forbidden = dict(evidence)
    forbidden.pop("semantic_sha256")
    forbidden.pop("evidence_id")
    forbidden["source_refs"] = deepcopy(forbidden["source_refs"])
    forbidden["source_refs"][0]["portfolio_weight"] = "0.100000000000"
    forbidden = _seal_identity(forbidden, "evidence_id")
    with pytest.raises(ArtifactContractError, match="forbidden causal regime field"):
        validate_regime_evidence_v2(forbidden, schema_checked=True)
