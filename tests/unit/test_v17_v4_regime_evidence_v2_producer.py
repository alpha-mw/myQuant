from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v4_contract.canonical import load_canonical_resource
from quant_investor.v17_v4_contract.resources import (
    load_packaged_json,
    read_packaged_asset,
)
from quant_investor.v17_v4_contract.validators import (
    regime_artifact_identity,
)
import quant_investor.v17_v4_runtime.regime_evidence_v2 as subject
from quant_investor.v17_v4_runtime.source_storage import SourceStore

STRATEGY = "cn-regime-shadow"
CUTOFF = "2026-07-30T07:00:00Z"
BOOTSTRAP_CREATED = "2026-07-29T15:21:00Z"
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


@dataclass
class RegimeFixture:
    workspace: Path
    kwargs: dict[str, Any]
    feature: dict[str, Any]
    feature_path: str
    market_path: str
    output_path: str


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _store_document(
    store: SourceStore,
    *,
    path: str,
    document: Mapping[str, Any],
) -> tuple[bytes, dict[str, str]]:
    raw = canonical_resource_bytes(document)
    store.write_exact_once(path, raw)
    identity = next(
        value for key, value in document.items() if key.endswith("_id") and key != "strategy_id"
    )
    return raw, {
        "artifact_id": str(identity),
        "artifact_version": str(document["version"]),
        "byte_sha256": _sha(raw),
        "cutoff": str(document["cutoff"]),
        "relative_path": path,
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": str(document["strategy_id"]),
    }


def _terminal(
    *,
    version: str,
    identity_field: str,
    identity: str,
    role: str,
    created_at: str,
    cutoff: str,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    del identity
    body = {
        "authority": dict(NO_AUTHORITY),
        "available_at": created_at,
        "created_at": created_at,
        "cutoff": cutoff,
        identity_field: "0" * 64,
        "protocol_version": PROTOCOL_VERSION,
        "strategy_id": STRATEGY,
        "terminal_role": role,
        "version": version,
        **dict(fields),
    }
    body[identity_field] = regime_artifact_identity(
        body,
        identity_field=identity_field,
    )
    return seal_semantic(body)


def _snapshot(
    body: Mapping[str, Any],
    *,
    identity_field: str,
) -> dict[str, Any]:
    provisional = dict(body)
    provisional[identity_field] = "0" * 64
    provisional[identity_field] = regime_artifact_identity(
        provisional,
        identity_field=identity_field,
    )
    return seal_semantic(provisional)


def _symbols(count: int = 30) -> list[str]:
    return [f"{index:06d}.SZ" for index in range(1, count + 1)]


def _policy() -> tuple[dict[str, Any], bytes, dict[str, str]]:
    raw = read_packaged_asset(subject.INFERENCE_POLICY_PATH)
    policy = load_packaged_json(subject.INFERENCE_POLICY_PATH)
    return (
        policy,
        raw,
        {
            "byte_sha256": _sha(raw),
            "relative_path": subject.INFERENCE_POLICY_PATH,
            "semantic_sha256": policy["semantic_sha256"],
            "version": subject.INFERENCE_POLICY_VERSION,
        },
    )


def make_regime_fixture(
    workspace: Path,
    *,
    observed: str = subject.BOOTSTRAP_OBSERVED_SESSION,
    effective: str = subject.BOOTSTRAP_DECISION_SESSION,
    created_at: str = BOOTSTRAP_CREATED,
    cutoff: str | None = None,
    open_sessions: list[str] | None = None,
    market_symbols: list[str] | None = None,
    prior: tuple[dict[str, Any], str, str] | None = None,
    model_implementation_sha256: str | None = None,
) -> RegimeFixture:
    workspace.mkdir(parents=True, exist_ok=True)
    store = SourceStore(workspace)
    store.initialize()
    policy, policy_raw, policy_ref = _policy()
    all_sessions = open_sessions or [
        "2026-07-29",
        "2026-07-30",
        "2026-07-31",
    ]
    symbols = _symbols()
    observed_market_symbols = symbols if market_symbols is None else market_symbols
    decision_cutoff = cutoff or f"{effective}T07:00:00Z"
    root = "data/private/v17_v4_sources/regime_inputs"

    calendar = _terminal(
        version=subject.CALENDAR_TERMINAL_VERSION,
        identity_field="calendar_terminal_id",
        identity=f"calendar-{effective}",
        role="SHANGHAI_OPEN_CALENDAR",
        created_at=created_at,
        cutoff=decision_cutoff,
        fields={"open_sessions": all_sessions},
    )
    _, calendar_ref = _store_document(
        store,
        path=f"{root}/{effective}-calendar.json",
        document=calendar,
    )
    pit = _terminal(
        version=subject.PIT_TERMINAL_VERSION,
        identity_field="pit_membership_terminal_id",
        identity=f"pit-{effective}",
        role="ACTIVE_PIT_MEMBERSHIP",
        created_at=created_at,
        cutoff=decision_cutoff,
        fields={
            "active_symbols": symbols,
            "observed_through_session": observed,
        },
    )
    _, pit_ref = _store_document(
        store,
        path=f"{root}/{effective}-pit.json",
        document=pit,
    )
    market = _terminal(
        version=subject.MARKET_TERMINAL_VERSION,
        identity_field="market_terminal_id",
        identity=f"market-{effective}",
        role="SEALED_MARKET_SYMBOL_INVENTORY",
        created_at=created_at,
        cutoff=decision_cutoff,
        fields={
            "observed_through_session": observed,
            "symbols": observed_market_symbols,
        },
    )
    market_path = f"{root}/{effective}-market.json"
    _, market_ref = _store_document(
        store,
        path=market_path,
        document=market,
    )
    locator = _terminal(
        version=subject.LOCATOR_TERMINAL_VERSION,
        identity_field="source_locator_terminal_id",
        identity=f"locator-{effective}",
        role="REGIME_SOURCE_LOCATOR",
        created_at=created_at,
        cutoff=decision_cutoff,
        fields={
            "calendar_ref": calendar_ref,
            "market_source_refs": [market_ref],
            "pit_membership_ref": pit_ref,
        },
    )
    _, locator_ref = _store_document(
        store,
        path=f"{root}/{effective}-locator.json",
        document=locator,
    )

    feature_body: dict[str, Any] = {
        "authority": dict(NO_AUTHORITY),
        "available_at": created_at,
        "average_liquidity": "0.800000000000",
        "average_return": "0.010000000000",
        "average_volatility": "0.020000000000",
        "breadth": "0.600000000000",
        "calendar_ref": calendar_ref,
        "coverage_ratio": "1.000000000000",
        "created_at": created_at,
        "cutoff": decision_cutoff,
        "effective_session": effective,
        "fake_breakout_share": "0.200000000000",
        "full_market_symbols": symbols,
        "macro_score": "0.100000000000",
        "market_sample_count": len(symbols),
        "market_source_refs": [market_ref],
        "median_drawdown": "0.100000000000",
        "minimum_market_sample": 30,
        "momentum_share": "0.550000000000",
        "observed_through_session": observed,
        "open_sessions": all_sessions,
        "pit_active_symbol_count": len(symbols),
        "pit_membership_ref": pit_ref,
        "protocol_version": PROTOCOL_VERSION,
        "sampled": False,
        "scope_kind": "FULL_MARKET",
        "source_locator_ref": locator_ref,
        "source_scope": "FULL_PIT_MARKET",
        "state_order": list(subject.STATE_ORDER),
        "strategy_id": STRATEGY,
        "version": subject.FEATURE_SNAPSHOT_VERSION,
        **dict(SHADOW_FLAGS),
    }
    risk_on, volatility, pressure, likelihoods = subject._derive_scores_and_likelihoods(
        feature_body
    )
    feature_body.update(
        {
            "pressure_score": pressure,
            "risk_on_score": risk_on,
            "state_likelihoods": likelihoods,
            "volatility_score": volatility,
        }
    )
    feature = _snapshot(
        feature_body,
        identity_field="feature_snapshot_id",
    )
    feature_path = f"{root}/{effective}-feature.json"
    feature_raw, feature_ref = _store_document(
        store,
        path=feature_path,
        document=feature,
    )

    transition = _snapshot(
        {
            "authority": dict(NO_AUTHORITY),
            "available_at": created_at,
            "created_at": created_at,
            "cutoff": decision_cutoff,
            "effective_session": effective,
            "inference_policy_ref": policy_ref,
            "observed_through_session": observed,
            "protocol_version": PROTOCOL_VERSION,
            "source_evidence_refs": [],
            "state_order": list(subject.STATE_ORDER),
            "strategy_id": STRATEGY,
            "transition_matrix": policy["transition_matrix"],
            "transition_source": subject.TRANSITION_SOURCE,
            "version": subject.TRANSITION_SNAPSHOT_VERSION,
            **dict(SHADOW_FLAGS),
        },
        identity_field="transition_snapshot_id",
    )
    transition_path = f"{root}/{effective}-transition.json"
    transition_raw, transition_ref = _store_document(
        store,
        path=transition_path,
        document=transition,
    )

    prior_ref: dict[str, str] | None = None
    prior_path: str | None = None
    prior_sha: str | None = None
    if prior is not None:
        prior_document, prior_path, prior_sha = prior
        prior_raw = store.read(prior_path, prior_sha)
        prior_ref = subject._artifact_ref(
            prior_document,
            prior_raw,
            relative_path=prior_path,
        )
    model = _snapshot(
        {
            "authority": dict(NO_AUTHORITY),
            "available_at": created_at,
            "created_at": created_at,
            "cutoff": decision_cutoff,
            "effective_session": effective,
            "formula_version": subject.FORMULA_VERSION,
            "inference_policy_ref": policy_ref,
            "model_id": subject.MODEL_ID,
            "model_implementation_sha256": (
                model_implementation_sha256 or subject.implementation_sha256()
            ),
            "model_kind": subject.MODEL_KIND,
            "model_training_end_session": None,
            "model_version": subject.MODEL_VERSION,
            "observed_through_session": observed,
            "predecessor_evidence_ref": prior_ref,
            "protocol_version": PROTOCOL_VERSION,
            "state_order": list(subject.STATE_ORDER),
            "strategy_id": STRATEGY,
            "training_source_refs": [],
            "transition_matrix_ref": transition_ref,
            "version": subject.MODEL_SNAPSHOT_VERSION,
            **dict(SHADOW_FLAGS),
        },
        identity_field="model_snapshot_id",
    )
    model_path = f"{root}/{effective}-model.json"
    model_raw, _ = _store_document(
        store,
        path=model_path,
        document=model,
    )
    output_path = subject.regime_evidence_v2_path(
        strategy_id=STRATEGY,
        effective_session=effective,
    )
    return RegimeFixture(
        workspace=workspace,
        kwargs={
            "workspace_root": workspace,
            "evidence_id": "0" * 64,
            "strategy_id": STRATEGY,
            "decision_session": effective,
            "cutoff": decision_cutoff,
            "created_at": created_at,
            "inference_policy_path": subject.INFERENCE_POLICY_PATH,
            "inference_policy_sha256": _sha(policy_raw),
            "model_snapshot_path": model_path,
            "model_snapshot_sha256": _sha(model_raw),
            "transition_matrix_path": transition_path,
            "transition_matrix_sha256": _sha(transition_raw),
            "feature_snapshot_path": feature_path,
            "feature_snapshot_sha256": _sha(feature_raw),
            "prior_evidence_path": prior_path,
            "prior_evidence_sha256": prior_sha,
        },
        feature=feature,
        feature_path=feature_path,
        market_path=market_path,
        output_path=output_path,
    )


def _clock(value: str) -> Any:
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    return lambda: parsed


def build_fixture(fixture: RegimeFixture) -> subject.RegimeEvidenceV2BuildResult:
    try:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock(fixture.kwargs["created_at"]),
        )
    except subject.RegimeEvidenceV2Error as exc:
        marker = "evidence_id mismatch; expected "
        assert exc.blocker_code == subject.IDENTITY_BLOCKER
        assert exc.detail.startswith(marker)
        fixture.kwargs["evidence_id"] = exc.detail.removeprefix(marker)
    return subject.build_regime_evidence_v2(
        **fixture.kwargs,
        _now_fn=_clock(fixture.kwargs["created_at"]),
    )


def _replace_snapshot(
    fixture: RegimeFixture,
    *,
    argument_prefix: str,
    identity_field: str,
    mutation: Any,
) -> dict[str, Any]:
    path_key = f"{argument_prefix}_path"
    sha_key = f"{argument_prefix}_sha256"
    original_path = fixture.workspace / fixture.kwargs[path_key]
    document = dict(
        load_canonical_resource(
            original_path.read_bytes(),
            label=argument_prefix,
        )
    )
    document.pop("semantic_sha256")
    document.pop(identity_field)
    mutation(document)
    replaced = _snapshot(document, identity_field=identity_field)
    replacement_path = fixture.kwargs[path_key].replace(
        ".json",
        "-replacement.json",
    )
    raw, _ = _store_document(
        SourceStore(fixture.workspace),
        path=replacement_path,
        document=replaced,
    )
    fixture.kwargs[path_key] = replacement_path
    fixture.kwargs[sha_key] = _sha(raw)
    return replaced


def test_bootstrap_inference_is_causal_decimal_and_authority_false(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(tmp_path)
    result = build_fixture(fixture)
    document = result.document

    assert result.created is True
    assert result.reused is False
    assert result.evidence_path == fixture.output_path
    assert document["lineage_phase"] == "BOOTSTRAP"
    assert document["publication_phase"] == subject.TEMPORAL_MODE
    assert document["observed_through_session"] == "2026-07-29"
    assert document["decision_session"] == "2026-07-30"
    assert document["effective_session"] == "2026-07-30"
    assert document["created_at"] == document["computed_at"]
    assert document["created_at"] == document["available_at"]
    assert document["created_at"] == document["published_at"]
    assert (
        sum(
            map(
                subject.Decimal,
                document["state_probabilities"].values(),
            ),
            subject.Decimal(0),
        )
        == subject.DECIMAL_ONE
    )
    assert document["hard_state"] == subject._hard_state(document["state_probabilities"])
    assert document["model_training_end_session"] is None
    assert document["authority"] == NO_AUTHORITY
    assert all(
        document[field] is expected
        for field, expected in {
            **SHADOW_FLAGS,
            "no_retroactive_causal_backfill": True,
            "same_session_execution_eligible": False,
        }.items()
    )
    raw = (tmp_path / result.evidence_path).read_bytes()
    assert b"NaN" not in raw and b"Infinity" not in raw


def test_exact_once_delayed_identical_retry_skips_wall_clock(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(tmp_path)
    first = build_fixture(fixture)
    before = (tmp_path / first.evidence_path).read_bytes()

    def forbidden_clock() -> datetime:
        raise AssertionError("existing exact slot must precede freshness")

    second = subject.build_regime_evidence_v2(
        **fixture.kwargs,
        _now_fn=forbidden_clock,
    )
    assert second.reused is True
    assert second.created is False
    assert second.evidence_sha256 == first.evidence_sha256
    assert (tmp_path / first.evidence_path).read_bytes() == before


def test_existing_slot_with_different_stable_arguments_is_conflict(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(tmp_path)
    build_fixture(fixture)
    changed = dict(fixture.kwargs)
    changed["created_at"] = "2026-07-29T15:21:01Z"
    with pytest.raises(subject.RegimeEvidenceV2Conflict):
        subject.build_regime_evidence_v2(
            **changed,
            _now_fn=lambda: (_ for _ in ()).throw(AssertionError("clock must not run")),
        )


def test_missing_owner_input_is_typed_gap_and_writes_no_completion(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(tmp_path)
    fixture.kwargs["feature_snapshot_path"] = (
        "data/private/v17_v4_sources/regime_inputs/missing.json"
    )
    with pytest.raises(subject.RegimeEvidenceV2InputGap) as raised:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock(BOOTSTRAP_CREATED),
        )
    assert raised.value.status == subject.TRUE_CURRENT_CANONICAL_INPUT_GAP
    assert not (tmp_path / fixture.output_path).exists()


@pytest.mark.parametrize(
    ("created_at", "now"),
    [
        ("2026-07-29T06:59:59Z", "2026-07-29T06:59:59Z"),
        ("2026-07-29T15:21:00Z", "2026-07-29T15:30:00Z"),
        ("2026-08-01T07:00:01Z", "2026-08-01T07:00:01Z"),
    ],
)
def test_future_late_or_preclose_new_publication_fails_closed(
    tmp_path: Path,
    created_at: str,
    now: str,
) -> None:
    fixture = make_regime_fixture(tmp_path, created_at=created_at)
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock(now),
        )
    assert raised.value.blocker_code in {
        subject.TEMPORAL_BLOCKER,
        subject.IDENTITY_BLOCKER,
    }
    assert not (tmp_path / fixture.output_path).exists()


def test_later_caller_cutoff_cannot_backfill_an_earlier_decision_session(
    tmp_path: Path,
) -> None:
    created_at = "2026-07-31T07:00:00Z"
    fixture = make_regime_fixture(
        tmp_path,
        created_at=created_at,
        cutoff="2026-08-01T07:00:00Z",
    )
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock(created_at),
        )
    assert raised.value.blocker_code == subject.TEMPORAL_BLOCKER
    assert "decision session" in raised.value.detail
    assert not (tmp_path / fixture.output_path).exists()


def test_postclose_but_prepolicy_publication_is_rejected(
    tmp_path: Path,
) -> None:
    created_at = "2026-07-29T08:00:00Z"
    fixture = make_regime_fixture(tmp_path, created_at=created_at)
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock(created_at),
        )
    assert raised.value.blocker_code == subject.TEMPORAL_BLOCKER
    assert "predates" in raised.value.detail
    assert not (tmp_path / fixture.output_path).exists()


def test_market_subset_cannot_support_full_market_claim(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(
        tmp_path,
        market_symbols=_symbols(29),
    )
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock(BOOTSTRAP_CREATED),
        )
    assert raised.value.blocker_code in {
        subject.SCOPE_BLOCKER,
        subject.INPUT_TAMPER_BLOCKER,
    }
    assert not (tmp_path / fixture.output_path).exists()


def test_largest_remainder_and_argmax_use_native_state_order() -> None:
    normalized = subject._normalize_largest_remainder(
        {
            "趋势上涨": subject.Decimal(1),
            "震荡低波": subject.Decimal(1),
            "震荡高波": subject.Decimal(1),
            "趋势下跌": subject.Decimal(0),
            "未知": subject.Decimal(0),
        },
        label="tie",
    )
    assert normalized == {
        "趋势上涨": "0.333333333334",
        "震荡低波": "0.333333333333",
        "震荡高波": "0.333333333333",
        "趋势下跌": "0.000000000000",
        "未知": "0.000000000000",
    }
    assert subject._hard_state(normalized) == "趋势上涨"


def test_independent_workspaces_publish_identical_bytes(
    tmp_path: Path,
) -> None:
    first_fixture = make_regime_fixture(tmp_path / "first")
    second_fixture = make_regime_fixture(tmp_path / "second")
    first = build_fixture(first_fixture)
    second = build_fixture(second_fixture)
    assert first.evidence_sha256 == second.evidence_sha256
    assert (first_fixture.workspace / first.evidence_path).read_bytes() == (
        second_fixture.workspace / second.evidence_path
    ).read_bytes()


def test_model_implementation_sha_mismatch_is_hard_failure(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(
        tmp_path,
        model_implementation_sha256="f" * 64,
    )
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock(BOOTSTRAP_CREATED),
        )
    assert raised.value.blocker_code == subject.IMPLEMENTATION_BLOCKER


@pytest.mark.parametrize(
    "argument",
    [
        "feature_snapshot_sha256",
        "transition_matrix_sha256",
        "model_snapshot_sha256",
        "inference_policy_sha256",
    ],
)
def test_each_explicit_input_sha_mismatch_fails_before_publication(
    tmp_path: Path,
    argument: str,
) -> None:
    fixture = make_regime_fixture(tmp_path)
    fixture.kwargs[argument] = "f" * 64
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock(BOOTSTRAP_CREATED),
        )
    assert raised.value.blocker_code == subject.INPUT_TAMPER_BLOCKER
    assert not (tmp_path / fixture.output_path).exists()


def test_normal_publication_requires_and_binds_contiguous_predecessor(
    tmp_path: Path,
) -> None:
    first_fixture = make_regime_fixture(tmp_path)
    first = build_fixture(first_fixture)
    second_fixture = make_regime_fixture(
        tmp_path,
        observed="2026-07-30",
        effective="2026-07-31",
        created_at="2026-07-30T15:21:00Z",
        prior=(
            first.document,
            first.evidence_path,
            first.evidence_sha256,
        ),
    )
    second = build_fixture(second_fixture)
    assert second.document["lineage_phase"] == "NORMAL"
    assert (
        second.document["model_snapshot_ref"]["artifact_id"]
        == load_canonical_resource(
            (tmp_path / second_fixture.kwargs["model_snapshot_path"]).read_bytes(),
            label="normal model",
        )["model_snapshot_id"]
    )
    assert second.document["observed_through_session"] == "2026-07-30"
    assert second.document["effective_session"] == "2026-07-31"


def test_nonbootstrap_publication_without_predecessor_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(
        tmp_path,
        observed="2026-07-30",
        effective="2026-07-31",
        created_at="2026-07-30T15:21:00Z",
    )
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock("2026-07-30T15:21:00Z"),
        )
    assert raised.value.blocker_code == subject.TEMPORAL_BLOCKER
    assert not (tmp_path / fixture.output_path).exists()


def test_noncontiguous_predecessor_is_rejected(
    tmp_path: Path,
) -> None:
    first_fixture = make_regime_fixture(tmp_path)
    first = build_fixture(first_fixture)
    fixture = make_regime_fixture(
        tmp_path,
        observed="2026-07-31",
        effective="2026-08-03",
        created_at="2026-07-31T15:21:00Z",
        open_sessions=[
            "2026-07-29",
            "2026-07-30",
            "2026-07-31",
            "2026-08-03",
        ],
        prior=(
            first.document,
            first.evidence_path,
            first.evidence_sha256,
        ),
    )
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock("2026-07-31T15:21:00Z"),
        )
    assert raised.value.blocker_code == subject.TEMPORAL_BLOCKER
    assert not (tmp_path / fixture.output_path).exists()


def test_calendar_terminal_rejects_weekend_open_session(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(
        tmp_path,
        open_sessions=[
            "2026-07-29",
            "2026-07-30",
            "2026-08-01",
        ],
    )
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock(BOOTSTRAP_CREATED),
        )
    assert raised.value.blocker_code == subject.INPUT_TAMPER_BLOCKER
    assert not (tmp_path / fixture.output_path).exists()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_feature_input_link_aliases_fail_closed(
    tmp_path: Path,
    link_kind: str,
) -> None:
    fixture = make_regime_fixture(tmp_path)
    feature_path = tmp_path / fixture.feature_path
    alias_path = tmp_path / "feature-alias.json"
    if link_kind == "symlink":
        raw = feature_path.read_bytes()
        feature_path.unlink()
        alias_path.write_bytes(raw)
        feature_path.symlink_to(alias_path)
    else:
        os.link(feature_path, alias_path)
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock(BOOTSTRAP_CREATED),
        )
    assert raised.value.blocker_code == subject.INPUT_TAMPER_BLOCKER
    assert not (tmp_path / fixture.output_path).exists()


def test_feature_path_escape_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(tmp_path)
    fixture.kwargs["feature_snapshot_path"] = "../feature.json"
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock(BOOTSTRAP_CREATED),
        )
    assert raised.value.blocker_code == subject.INPUT_TAMPER_BLOCKER
    assert not (tmp_path / fixture.output_path).exists()


def test_non_null_model_training_end_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(tmp_path)
    _replace_snapshot(
        fixture,
        argument_prefix="model_snapshot",
        identity_field="model_snapshot_id",
        mutation=lambda document: document.update({"model_training_end_session": "2026-07-29"}),
    )
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.build_regime_evidence_v2(
            **fixture.kwargs,
            _now_fn=_clock(BOOTSTRAP_CREATED),
        )
    assert raised.value.blocker_code == subject.INPUT_TAMPER_BLOCKER
    assert not (tmp_path / fixture.output_path).exists()


def test_replay_detects_transitive_feature_byte_drift(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(tmp_path)
    result = build_fixture(fixture)
    feature_path = tmp_path / fixture.feature_path
    feature_path.write_bytes(feature_path.read_bytes() + b"\n")
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        subject.replay_regime_evidence_v2(
            workspace_root=tmp_path,
            evidence_path=result.evidence_path,
            evidence_sha256=result.evidence_sha256,
        )
    assert raised.value.blocker_code == subject.INPUT_TAMPER_BLOCKER


def test_replay_and_status_never_reclassify_the_sealed_hard_state(
    tmp_path: Path,
) -> None:
    fixture = make_regime_fixture(tmp_path)
    result = build_fixture(fixture)
    replayed = subject.replay_regime_evidence_v2(
        workspace_root=tmp_path,
        evidence_path=result.evidence_path,
        evidence_sha256=result.evidence_sha256,
    )
    status = subject.regime_evidence_v2_status(
        workspace_root=tmp_path,
        evidence_path=result.evidence_path,
        evidence_sha256=result.evidence_sha256,
    )
    assert replayed == result.document
    assert status["hard_state"] == result.document["hard_state"]
    assert status["status"] == "AVAILABLE"


def test_bounded_closure_rejects_cycle_before_recursing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    reference = {
        "artifact_id": "cycle-node",
        "artifact_version": "myquant.v17.v4.regime-test-cycle.v1",
        "byte_sha256": "a" * 64,
        "cutoff": CUTOFF,
        "relative_path": ("data/private/v17_v4_sources/regime_inputs/cycle.json"),
        "semantic_sha256": "0" * 64,
        "strategy_id": STRATEGY,
    }

    class FakeStore:
        def read(self, relative_path: str, expected_sha256: str) -> bytes:
            del relative_path, expected_sha256
            raise AssertionError("cycle must be rejected before readback")

    loader = subject._ClosureLoader(FakeStore())
    loader.active.append(
        (
            reference["artifact_version"],
            reference["artifact_id"],
            reference["byte_sha256"],
        )
    )
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        loader(reference)
    assert raised.value.blocker_code == subject.INPUT_TAMPER_BLOCKER
    assert "cycle" in raised.value.detail


@pytest.mark.parametrize(
    ("constant", "value"),
    [
        ("MAX_CLOSURE_NODES", 0),
        ("MAX_REFERENCES", 0),
        ("MAX_JSON_BYTES_PER_NODE", 8),
        ("MAX_RAW_BYTES_PER_FILE", 8),
    ],
)
def test_closure_resource_limits_are_hard(
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    value: int,
) -> None:
    document = seal_semantic(
        {
            "cutoff": CUTOFF,
            "node_id": "node",
            "protocol_version": PROTOCOL_VERSION,
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.regime-test-node.v1",
        }
    )
    raw = canonical_resource_bytes(document)
    reference = {
        "artifact_id": "node",
        "artifact_version": document["version"],
        "byte_sha256": "b" * 64,
        "cutoff": CUTOFF,
        "relative_path": ("data/private/v17_v4_sources/regime_inputs/node.json"),
        "semantic_sha256": document["semantic_sha256"],
        "strategy_id": STRATEGY,
    }

    class FakeStore:
        def read(self, relative_path: str, expected_sha256: str) -> bytes:
            del relative_path, expected_sha256
            return raw

    monkeypatch.setattr(subject, constant, value)
    loader = subject._ClosureLoader(FakeStore())
    with pytest.raises(subject.RegimeEvidenceV2Error) as raised:
        loader(reference)
    assert raised.value.blocker_code == subject.INPUT_TAMPER_BLOCKER


def test_nonfinite_decimal_never_reaches_json() -> None:
    with pytest.raises(subject.RegimeEvidenceV2Error):
        subject._normalize_largest_remainder(
            {
                state: (
                    subject.Decimal("NaN")
                    if state == subject.STATE_ORDER[0]
                    else subject.Decimal(1)
                )
                for state in subject.STATE_ORDER
            },
            label="nonfinite",
        )
