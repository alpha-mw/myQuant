from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.v17_v4_contract import canonical_resource_bytes
from quant_investor.v17_v4_contract.schema_validation import (
    validate_instance_against_schema,
)
from quant_investor.v17_v4_runtime.forward_fusion import (
    FORWARD_ACCUMULATION_STATE,
    FUSION_TOP24_V2_VERSION,
    ForwardFusionError,
    ForwardFusionWriter,
    SHADOW_FUSION_MATURED_LABEL_VERSION,
    SHADOW_FUSION_OBSERVATION_VERSION,
    SHADOW_FUSION_POLICY_VERSION,
    build_forward_fusion_top24,
    build_shadow_fusion_matured_label,
    build_shadow_fusion_observation,
    build_shadow_fusion_policy,
    publish_forward_fusion_prediction,
    publish_shadow_fusion_matured_label,
)
from quant_investor.v17_v4_runtime.source_storage import (
    SourceStorageSecurityError,
)

STRATEGY = "quant-first"
SESSION = "2026-07-29"
CUTOFF = "2026-07-29T07:00:00Z"
SHA_A = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64
NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}


def _schema(name: str) -> dict[str, object]:
    path = Path(__file__).parents[2] / "quant_investor" / "v17_v4_contract" / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _pool(size: int = 40) -> list[str]:
    return [f"{index:06d}.SZ" for index in range(1, size + 1)]


def _policy(
    *,
    policy_id: str = "zero-history-v1",
    effective_from_session: str = SESSION,
    created_at: str = CUTOFF,
) -> dict[str, object]:
    return build_shadow_fusion_policy(
        policy_id=policy_id,
        strategy_id=STRATEGY,
        effective_from_session=effective_from_session,
        created_at=created_at,
    )


def _fusion(
    *,
    session: str = SESSION,
    cutoff: str = CUTOFF,
    policy: dict[str, object] | None = None,
    factor_sha: str = SHA_C,
    bundle_sha: str = SHA_B,
    size: int = 40,
    fundamental_count: int = 8,
) -> dict[str, object]:
    symbols = _pool(size)
    return build_forward_fusion_top24(
        output_id=f"forward-fusion-{session}",
        strategy_id=STRATEGY,
        decision_session=session,
        cutoff=cutoff,
        pool_symbols=symbols,
        quant_scores={symbol: str(index % 7) for index, symbol in enumerate(symbols)},
        fundamental_scores={
            symbol: str(index) for index, symbol in enumerate(symbols[:fundamental_count])
        },
        source_locator_semantic_sha256=SHA_A,
        input_bundle_sha256=bundle_sha,
        factor_set_byte_sha256=factor_sha,
        policy=policy or _policy(),
    )


def _observation(
    *,
    fusion: dict[str, object] | None = None,
    policy: dict[str, object] | None = None,
    session: str = SESSION,
    cutoff: str = CUTOFF,
    factor_sha: str = SHA_C,
    bundle_sha: str = SHA_B,
) -> dict[str, object]:
    active_policy = policy or _policy()
    active_fusion = fusion or _fusion(
        session=session,
        cutoff=cutoff,
        policy=active_policy,
        factor_sha=factor_sha,
        bundle_sha=bundle_sha,
    )
    return build_shadow_fusion_observation(
        strategy_id=STRATEGY,
        decision_session=session,
        cutoff=cutoff,
        created_at=cutoff,
        source_locator_semantic_sha256=SHA_A,
        input_bundle_sha256=bundle_sha,
        factor_set_byte_sha256=factor_sha,
        policy=active_policy,
        fusion_top24=active_fusion,
        fusion_relative_path=(
            "results/v17_v4_shadow/forward_fusion/strategies/" f"{STRATEGY}/fusions/{session}.json"
        ),
    )


def _future_sessions(origin: str, horizon: int) -> list[str]:
    sessions = [date.fromisoformat(origin)]
    cursor = sessions[0]
    while len(sessions) <= horizon:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            sessions.append(cursor)
    return [value.isoformat() for value in sessions]


def _closes(
    fusion: dict[str, object],
    *,
    scale: int,
) -> dict[str, str]:
    return {
        str(row["symbol"]): str(scale + int(row["rank"]))
        for row in fusion["rows"]  # type: ignore[index]
    }


def _matured_at(end_session: str) -> str:
    return f"{end_session}T07:00:00Z"


def _seed_prediction_files(
    root: Path,
    *,
    observation: dict[str, object],
    fusion: dict[str, object],
) -> str:
    writer = ForwardFusionWriter(root.resolve())
    writer.initialize()
    observation_path = (
        "results/v17_v4_shadow/forward_fusion/strategies/"
        f"{STRATEGY}/observations/{observation['decision_session']}.json"
    )
    fusion_ref = observation["fusion_top24_ref"]
    assert isinstance(fusion_ref, dict)
    writer.write_exact_once(
        str(fusion_ref["relative_path"]),
        canonical_resource_bytes(fusion),
    )
    writer.write_exact_once(
        observation_path,
        canonical_resource_bytes(observation),
    )
    return observation_path


def test_zero_history_policy_is_exact_and_schema_valid() -> None:
    policy = _policy()
    validate_instance_against_schema(
        policy,
        _schema("shadow_fusion_policy.v1.schema.json"),
    )
    assert policy["version"] == SHADOW_FUSION_POLICY_VERSION
    assert policy["state"] == FORWARD_ACCUMULATION_STATE
    assert policy["quant_weight"] == "0.5"
    assert policy["fundamental_weight"] == "0.5"
    assert policy["authority"] == NO_AUTHORITY
    for field in (
        "canary_evidence_eligible",
        "formal_activation_eligible",
        "formal_research_publication_eligible",
        "performance_evidence_eligible",
        "promotion_eligible",
    ):
        assert policy[field] is False


def test_fusion_uses_weak_percentiles_and_zero_for_missing_fundamental() -> None:
    symbols = _pool(30)
    fusion = build_forward_fusion_top24(
        output_id="weak-percentile-fusion",
        strategy_id=STRATEGY,
        decision_session=SESSION,
        cutoff=CUTOFF,
        pool_symbols=symbols,
        quant_scores={symbol: "7" for symbol in symbols},
        fundamental_scores={
            symbols[0]: "1",
            symbols[1]: "1",
            symbols[2]: "4",
        },
        source_locator_semantic_sha256=SHA_A,
        input_bundle_sha256=SHA_B,
        factor_set_byte_sha256=SHA_C,
        policy=_policy(),
    )
    validate_instance_against_schema(
        fusion,
        _schema("fusion_top24.v2.schema.json"),
    )
    assert fusion["version"] == FUSION_TOP24_V2_VERSION
    assert fusion["fundamental_available_count"] == 3
    assert fusion["fundamental_unavailable_count"] == 27
    rows = {row["symbol"]: row for row in fusion["rows"]}
    assert rows[symbols[0]]["quant_percentile"] == "1"
    assert rows[symbols[0]]["fundamental_percentile"] == (
        "0.6666666666666666666666666666666666666667"
    )
    assert rows[symbols[2]]["fundamental_percentile"] == "1"
    missing = next(row for row in fusion["rows"] if row["fundamental_available"] is False)
    assert missing["fundamental_score"] is None
    assert missing["fundamental_percentile"] == "0"
    assert missing["fused_score"] == "0.5"


def test_large_missing_cohort_and_ascii_symbol_tie_break() -> None:
    symbols = _pool(200)
    fusion = build_forward_fusion_top24(
        output_id="large-missing-fusion",
        strategy_id=STRATEGY,
        decision_session=SESSION,
        cutoff=CUTOFF,
        pool_symbols=symbols,
        quant_scores={symbol: "1" for symbol in symbols},
        fundamental_scores={},
        source_locator_semantic_sha256=SHA_A,
        input_bundle_sha256=SHA_B,
        factor_set_byte_sha256=SHA_C,
        policy=_policy(),
    )
    assert fusion["fundamental_available_count"] == 0
    assert fusion["fundamental_unavailable_count"] == 200
    assert [row["symbol"] for row in fusion["rows"]] == symbols[:24]
    assert all(row["base_target"] == "0.03" for row in fusion["rows"])
    assert all(row["fused_score"] == "0.5" for row in fusion["rows"])


def test_fusion_rejects_cross_pool_branch_rows() -> None:
    symbols = _pool()
    with pytest.raises(ForwardFusionError, match="quant_scores_same_pool"):
        build_forward_fusion_top24(
            output_id="bad-pool-fusion",
            strategy_id=STRATEGY,
            decision_session=SESSION,
            cutoff=CUTOFF,
            pool_symbols=symbols,
            quant_scores={symbol: "1" for symbol in symbols[:-1]},
            fundamental_scores={},
            source_locator_semantic_sha256=SHA_A,
            input_bundle_sha256=SHA_B,
            factor_set_byte_sha256=SHA_C,
            policy=_policy(),
        )
    with pytest.raises(
        ForwardFusionError,
        match="fundamental_scores_same_pool",
    ):
        build_forward_fusion_top24(
            output_id="bad-fundamental-pool",
            strategy_id=STRATEGY,
            decision_session=SESSION,
            cutoff=CUTOFF,
            pool_symbols=symbols,
            quant_scores={symbol: "1" for symbol in symbols},
            fundamental_scores={"999999.SZ": "1"},
            source_locator_semantic_sha256=SHA_A,
            input_bundle_sha256=SHA_B,
            factor_set_byte_sha256=SHA_C,
            policy=_policy(),
        )


def test_observation_identity_binds_all_prediction_dimensions() -> None:
    policy = _policy()
    fusion = _fusion(policy=policy)
    observation = _observation(fusion=fusion, policy=policy)
    validate_instance_against_schema(
        observation,
        _schema("shadow_fusion_observation.v1.schema.json"),
    )
    assert observation["version"] == SHADOW_FUSION_OBSERVATION_VERSION
    assert observation["factor_set_byte_sha256"] == SHA_C
    assert observation["input_bundle_sha256"] == SHA_B
    assert observation["source_locator_semantic_sha256"] == SHA_A
    assert observation["policy_semantic_sha256"] == policy["semantic_sha256"]
    changed_fusion = _fusion(
        policy=policy,
        bundle_sha="4" * 64,
    )
    changed = _observation(
        fusion=changed_fusion,
        policy=policy,
        bundle_sha="4" * 64,
    )
    assert changed["observation_id"] != observation["observation_id"]
    assert changed["prediction_identity_sha256"] != observation["prediction_identity_sha256"]
    assert changed["evidence_group_sha256"] == observation["evidence_group_sha256"]


def test_fusion_golden_replay_bytes_are_stable() -> None:
    raw = canonical_resource_bytes(_fusion())
    assert hashlib.sha256(raw).hexdigest() == (
        "26bd161124a556247cf857e8b69a5523f8f3d184fddbb3cc6a84ce1967cd44f2"
    )
    assert raw == canonical_resource_bytes(_fusion())


def test_prediction_publish_is_idempotent_after_restart_and_conflicts_fail(
    tmp_path: Path,
) -> None:
    root = str(tmp_path.resolve())
    policy = _policy()
    kwargs = {
        "policy": policy,
        "output_id": "prediction-publish-fusion",
        "strategy_id": STRATEGY,
        "decision_session": SESSION,
        "cutoff": CUTOFF,
        "pool_symbols": _pool(),
        "quant_scores": {symbol: str(index) for index, symbol in enumerate(_pool())},
        "fundamental_scores": {symbol: str(index) for index, symbol in enumerate(_pool()[:10])},
        "source_locator_semantic_sha256": SHA_A,
        "input_bundle_sha256": SHA_B,
        "factor_set_byte_sha256": SHA_C,
    }
    first = publish_forward_fusion_prediction(root, **kwargs)
    restarted = publish_forward_fusion_prediction(root, **kwargs)
    assert first["created"] is True
    assert restarted["created"] is False
    assert first["observation"] == restarted["observation"]
    assert Path(root, first["observation_path"]).read_bytes() == (
        canonical_resource_bytes(first["observation"])
    )
    with pytest.raises(
        ForwardFusionError,
        match="conflicting_duplicate_session",
    ):
        publish_forward_fusion_prediction(
            root,
            **{
                **kwargs,
                "input_bundle_sha256": "4" * 64,
            },
        )


def test_second_session_and_set_policy_rotation_do_not_pool(
    tmp_path: Path,
) -> None:
    root = str(tmp_path.resolve())
    first_policy = _policy()
    first = publish_forward_fusion_prediction(
        root,
        policy=first_policy,
        output_id="session-one-fusion",
        strategy_id=STRATEGY,
        decision_session=SESSION,
        cutoff=CUTOFF,
        pool_symbols=_pool(),
        quant_scores={symbol: "1" for symbol in _pool()},
        fundamental_scores={},
        source_locator_semantic_sha256=SHA_A,
        input_bundle_sha256=SHA_B,
        factor_set_byte_sha256=SHA_C,
    )
    second_session = "2026-07-30"
    second_cutoff = "2026-07-30T07:00:00Z"
    rotated_policy = _policy(
        policy_id="zero-history-v2",
        effective_from_session=second_session,
        created_at=second_cutoff,
    )
    second = publish_forward_fusion_prediction(
        root,
        policy=rotated_policy,
        output_id="session-two-fusion",
        strategy_id=STRATEGY,
        decision_session=second_session,
        cutoff=second_cutoff,
        pool_symbols=_pool(),
        quant_scores={symbol: "1" for symbol in _pool()},
        fundamental_scores={},
        source_locator_semantic_sha256=SHA_A,
        input_bundle_sha256="4" * 64,
        factor_set_byte_sha256="5" * 64,
    )
    assert first["observation_path"] != second["observation_path"]
    assert (
        first["observation"]["evidence_group_sha256"]
        != second["observation"]["evidence_group_sha256"]
    )
    assert (
        first["observation"]["policy_semantic_sha256"]
        != second["observation"]["policy_semantic_sha256"]
    )


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_matured_labels_are_separate_exact_future_horizon_artifacts(
    tmp_path: Path,
    horizon: int,
) -> None:
    policy = _policy()
    fusion = _fusion(policy=policy)
    observation = _observation(fusion=fusion, policy=policy)
    sessions = _future_sessions(SESSION, horizon)
    origin_closes = _closes(fusion, scale=100)
    end_closes = _closes(fusion, scale=110)
    label = build_shadow_fusion_matured_label(
        observation=observation,
        fusion_top24=fusion,
        observation_relative_path=(
            "results/v17_v4_shadow/forward_fusion/strategies/"
            f"{STRATEGY}/observations/{SESSION}.json"
        ),
        horizon_sessions=horizon,
        shanghai_sessions=sessions,
        origin_closes=origin_closes,
        end_closes=end_closes,
        matured_at=_matured_at(sessions[-1]),
    )
    validate_instance_against_schema(
        label,
        _schema("shadow_fusion_matured_label.v1.schema.json"),
    )
    assert label["version"] == SHADOW_FUSION_MATURED_LABEL_VERSION
    assert label["horizon_sessions"] == horizon
    assert label["label_end_session"] == sessions[-1]
    assert label["evidence_group_sha256"] == observation["evidence_group_sha256"]
    observation_path = _seed_prediction_files(
        tmp_path,
        observation=observation,
        fusion=fusion,
    )
    published = publish_shadow_fusion_matured_label(
        str(tmp_path.resolve()),
        observation=observation,
        fusion_top24=fusion,
        observation_relative_path=observation_path,
        horizon_sessions=horizon,
        shanghai_sessions=sessions,
        origin_closes=origin_closes,
        end_closes=end_closes,
        matured_at=_matured_at(sessions[-1]),
    )
    restarted = publish_shadow_fusion_matured_label(
        str(tmp_path.resolve()),
        observation=observation,
        fusion_top24=fusion,
        observation_relative_path=observation_path,
        horizon_sessions=horizon,
        shanghai_sessions=sessions,
        origin_closes=origin_closes,
        end_closes=end_closes,
        matured_at=_matured_at(sessions[-1]),
    )
    assert published["created"] is True
    assert restarted["created"] is False
    assert published["label_path"].endswith(f"-h{horizon}.json")


def test_matured_label_rejects_preclose_backfill_and_session_gaps() -> None:
    policy = _policy()
    fusion = _fusion(policy=policy)
    observation = _observation(fusion=fusion, policy=policy)
    sessions = _future_sessions(SESSION, 5)
    args = {
        "observation": observation,
        "fusion_top24": fusion,
        "observation_relative_path": (
            "results/v17_v4_shadow/forward_fusion/strategies/"
            f"{STRATEGY}/observations/{SESSION}.json"
        ),
        "horizon_sessions": 5,
        "shanghai_sessions": sessions,
        "origin_closes": _closes(fusion, scale=100),
        "end_closes": _closes(fusion, scale=110),
    }
    with pytest.raises(
        ForwardFusionError,
        match="label_not_exact_future_close",
    ):
        build_shadow_fusion_matured_label(
            **args,
            matured_at=f"{sessions[-1]}T06:59:59Z",
        )
    next_day = (date.fromisoformat(sessions[-1]) + timedelta(days=1)).isoformat()
    with pytest.raises(
        ForwardFusionError,
        match="label_not_exact_future_close",
    ):
        build_shadow_fusion_matured_label(
            **args,
            matured_at=f"{next_day}T07:00:00Z",
        )
    broken_sessions = list(sessions)
    broken_sessions[2] = broken_sessions[1]
    with pytest.raises(ForwardFusionError, match="label_session_sequence"):
        build_shadow_fusion_matured_label(
            **{
                **args,
                "shanghai_sessions": broken_sessions,
                "matured_at": _matured_at(broken_sessions[-1]),
            },
        )


def test_label_conflict_fails_and_writer_cannot_escape_narrow_root(
    tmp_path: Path,
) -> None:
    policy = _policy()
    fusion = _fusion(policy=policy)
    observation = _observation(fusion=fusion, policy=policy)
    sessions = _future_sessions(SESSION, 1)
    observation_path = _seed_prediction_files(
        tmp_path,
        observation=observation,
        fusion=fusion,
    )
    kwargs = {
        "observation": observation,
        "fusion_top24": fusion,
        "observation_relative_path": observation_path,
        "horizon_sessions": 1,
        "shanghai_sessions": sessions,
        "origin_closes": _closes(fusion, scale=100),
        "end_closes": _closes(fusion, scale=110),
        "matured_at": _matured_at(sessions[-1]),
    }
    publish_shadow_fusion_matured_label(
        str(tmp_path.resolve()),
        **kwargs,
    )
    with pytest.raises(
        ForwardFusionError,
        match="conflicting_duplicate_label",
    ):
        publish_shadow_fusion_matured_label(
            str(tmp_path.resolve()),
            **{
                **kwargs,
                "end_closes": _closes(fusion, scale=120),
            },
        )
    writer = ForwardFusionWriter(tmp_path.resolve())
    with pytest.raises(SourceStorageSecurityError):
        writer.write_exact_once(
            "results/v17_v4_shadow/strategies/quant-first/" "runs/escape.json",
            b"{}\n",
        )
