from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quant_investor.factors.production_control_v1 import (
    ACTIVE_SET_SCHEMA_VERSION,
    ARTIFACT_REF_SCHEMA_VERSION,
    PROTOCOL_ID as FACTOR_PROTOCOL_ID,
    READINESS_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
    build_artifact_ref as build_factor_artifact_ref,
    validate_active_set_pointer,
)
from quant_investor.factors.runtime import production_factor_set_sha256
from quant_investor.v17_v4_contract import (
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.validators import (
    ArtifactContractError,
)
from quant_investor.v17_v4_runtime.calibration import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CALIBRATION_RECEIPT_VERSION,
    CLOSURE_MONTHS,
    FUSION_PROMOTION_RECEIPT_VERSION,
    ORIGIN_INVENTORY_VERSION,
    CalibrationClosureError,
    OriginCalibrationInput,
    artifact_ref,
    bootstrap_matrix_sha256,
    build_calibration_receipt,
    build_fusion_promotion_receipt,
    run_calibration_closure,
)
from quant_investor.v17_v4_runtime.pit_admission import (
    NATURAL_KEYS,
    REQUIRED_ROLES,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
STRATEGY = "quant-first"
RUN_ID = "calibration-run-1"
NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sessions(start: date, count: int) -> tuple[date, ...]:
    result: list[date] = []
    cursor = start
    while len(result) < count:
        if cursor.weekday() < 5:
            result.append(cursor)
        cursor += timedelta(days=1)
    return tuple(result)


def _month_ends(sessions: tuple[date, ...]) -> tuple[date, ...]:
    values: dict[str, date] = {}
    for session in sessions:
        values[session.strftime("%Y-%m")] = session
    return tuple(values.values())


def _utc_at_close(session: date) -> str:
    return (
        datetime.combine(session, time(15, 0), tzinfo=SHANGHAI)
        .astimezone(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _raw_source_ref(
    *,
    role: str,
    kind: str,
    origin_id: str,
    cutoff: str,
    artifacts: dict[str, bytes],
) -> dict[str, str]:
    raw = f"{kind}:{role}:{origin_id}\n".encode()
    version = f"myquant.v17.v4.{kind}.{role}.v1"
    artifact_id = f"{kind}-{role}-{origin_id}"
    reference = {
        "artifact_id": artifact_id,
        "artifact_version": version,
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": cutoff,
        "relative_path": (
            "data/private/v17_v4_sources/"
            f"{role}/{origin_id}/{kind}.bin"
        ),
        "semantic_sha256": _sha(
            f"{kind}-semantic:{role}:{origin_id}"
        ),
        "strategy_id": STRATEGY,
    }
    artifacts[reference["byte_sha256"]] = raw
    return reference


def _pit_catalog_ref(
    *,
    origin: date,
    history_start: date,
    origin_id: str,
    cutoff: str,
    dataset_refs: dict[str, dict[str, str]],
    artifacts: dict[str, bytes],
) -> dict[str, str]:
    complete_dataset_refs = dict(dataset_refs)
    expected_refs: dict[str, dict[str, str]] = {}
    for role in REQUIRED_ROLES:
        if role not in complete_dataset_refs:
            complete_dataset_refs[role] = _raw_source_ref(
                role=role,
                kind="dataset",
                origin_id=origin_id,
                cutoff=cutoff,
                artifacts=artifacts,
            )
        expected_refs[role] = _raw_source_ref(
            role=role,
            kind="expected-keys",
            origin_id=origin_id,
            cutoff=cutoff,
            artifacts=artifacts,
        )
    summaries = [
        {
            "expected_keys_sha256": _sha(
                f"keys:{role}:{origin_id}"
            ),
            "latest_available_at": cutoff,
            "natural_key_fields": list(NATURAL_KEYS[role]),
            "observed_keys_sha256": _sha(
                f"keys:{role}:{origin_id}"
            ),
            "role": role,
            "row_count": 1,
            "row_set_sha256": _sha(f"rows:{role}:{origin_id}"),
        }
        for role in REQUIRED_ROLES
    ]
    admission_payload = {
        "history_start": history_start.isoformat(),
        "decision_session": origin.isoformat(),
        "decision_cutoff": cutoff,
        "datasets": summaries,
    }
    admission_sha = hashlib.sha256(
        canonical_bytes(admission_payload)
    ).hexdigest()
    source_payload = {
        "admission_closure_sha256": admission_sha,
        "dataset_refs": complete_dataset_refs,
        "expected_key_inventory_refs": expected_refs,
    }
    catalog = seal_semantic(
        {
            "admission_closure_sha256": admission_sha,
            "authority": dict(NO_AUTHORITY),
            "catalog_id": f"pit-catalog-{origin_id}",
            "cutoff": cutoff,
            "dataset_refs": complete_dataset_refs,
            "dataset_summaries": summaries,
            "decision_session": origin.isoformat(),
            "expected_key_inventory_refs": expected_refs,
            "generation_id": f"pit-generation-{origin_id}",
            "history_start": history_start.isoformat(),
            "protocol_version": "myquant.v17.v4",
            "source_closure_sha256": hashlib.sha256(
                canonical_bytes(source_payload)
            ).hexdigest(),
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.pit-generation-catalog.v1",
        }
    )
    return _store_v4_artifact(
        catalog,
        path=(
            "data/private/v17_v4_sources/pit_catalog/generations/"
            f"{origin_id}.json"
        ),
        artifacts=artifacts,
    )


def _factor_nested_ref(
    *,
    schema: str,
    identity: str,
    path: str,
) -> dict[str, str]:
    return {
        "artifact_schema": schema,
        "byte_sha256": _sha(f"factor-bytes:{identity}"),
        "relative_path": path,
        "schema_version": ARTIFACT_REF_SCHEMA_VERSION,
        "semantic_sha256": _sha(f"factor-semantic:{identity}"),
    }


def _factor_active_set_ref(
    *,
    origin: date,
    origin_id: str,
    origin_cutoff: str,
    artifacts: dict[str, bytes],
    activated_at: str | None = None,
) -> tuple[str, dict[str, str]]:
    names = [f"factor_{index}" for index in range(5)]
    factor_set_sha = production_factor_set_sha256(names)
    active_set = seal_semantic(
        {
            "activated_at": activated_at or origin_cutoff,
            "active_set_id": f"active-factor-{origin_id}",
            "as_of": origin.isoformat(),
            "authority": {
                "account_new_risk": False,
                "active_for_production_research": True,
                "broker": False,
                "execution": False,
                "order": False,
                "trade": False,
            },
            "production_factor_names": names,
            "production_factor_set_sha256": factor_set_sha,
            "protocol_id": FACTOR_PROTOCOL_ID,
            "readiness_ref": _factor_nested_ref(
                schema=READINESS_SCHEMA_VERSION,
                identity=f"readiness-{origin_id}",
                path=(
                    "data/private/factor_governance_history/"
                    f"{origin_id}/readiness.json"
                ),
            ),
            "registry_ref": _factor_nested_ref(
                schema=REGISTRY_SCHEMA_VERSION,
                identity=f"registry-{origin_id}",
                path=(
                    "data/private/factor_governance_history/"
                    f"{origin_id}/registry.json"
                ),
            ),
            "runtime_contracts_sha256": _sha(
                f"runtime-contracts:{origin_id}"
            ),
            "schema_version": ACTIVE_SET_SCHEMA_VERSION,
            "transaction_id": f"transaction-{origin_id}",
            "v4_activation_receipt_ref": _factor_nested_ref(
                schema="factor-governance-activation-receipt.v4",
                identity=f"v4-activation-{origin_id}",
                path=(
                    "data/private/factor_governance_history/"
                    f"{origin_id}/v4-activation.json"
                ),
            ),
        }
    )
    validate_active_set_pointer(active_set)
    reference = build_factor_artifact_ref(
        active_set,
        relative_path=(
            "data/private/factor_governance_history/"
            f"{origin_id}/active-set.json"
        ),
    )
    artifacts[reference["byte_sha256"]] = canonical_resource_bytes(
        active_set
    )
    return factor_set_sha, reference


def _store_v4_artifact(
    artifact: dict[str, Any],
    *,
    path: str,
    artifacts: dict[str, bytes],
) -> dict[str, str]:
    reference = artifact_ref(artifact, relative_path=path)
    artifacts[reference["byte_sha256"]] = canonical_resource_bytes(
        artifact
    )
    return reference


def _fixture(
    *,
    origin_count: int = 138,
    positive: bool = True,
) -> tuple[
    list[OriginCalibrationInput],
    tuple[date, ...],
    date,
    str,
    dict[str, bytes],
]:
    sessions = _sessions(date(1999, 1, 4), 6_200)
    session_index = {
        session: index for index, session in enumerate(sessions)
    }
    origins = tuple(
        origin
        for origin in _month_ends(sessions)
        if (
            session_index[origin] >= 2_519
            and session_index[origin] + 252 < len(sessions)
        )
    )[:origin_count]
    assert len(origins) == origin_count
    pool = tuple(f"{index:06d}.SZ" for index in range(1, 25))
    rows: list[OriginCalibrationInput] = []
    artifacts: dict[str, bytes] = {}
    for month_index, origin in enumerate(origins):
        index = session_index[origin]
        origin_cutoff = _utc_at_close(origin)
        label_60 = sessions[index + 60]
        label_252 = sessions[index + 252]
        origin_id = origin.strftime("%Y%m%d")
        quant_scores = {
            symbol: str(index)
            for index, symbol in enumerate(pool, start=1)
        }
        fundamental_scores = {
            symbol: str(
                ((index * 7 + month_index) % len(pool)) + 1
            )
            for index, symbol in enumerate(pool, start=1)
        }
        forward60 = {
            symbol: (
                "0.02"
                if positive
                else (
                    "0.02"
                    if int(symbol[:6]) % 2 == 0
                    else "-0.02"
                )
            )
            for symbol in pool
        }
        forward252 = {
            symbol: "0.03" if positive else "-0.01"
            for symbol in pool
        }
        false_map = {symbol: False for symbol in pool}
        source_refs: dict[str, dict[str, str]] = {
            role: _raw_source_ref(
                role=role,
                kind="dataset",
                origin_id=origin_id,
                cutoff=origin_cutoff,
                artifacts=artifacts,
            )
            for role in (
                "benchmark_total_return",
                "corporate_actions",
                "official_delisting_cash",
            )
        }
        source_refs["pit_catalog"] = _pit_catalog_ref(
            origin=origin,
            history_start=sessions[index - 2_519],
            origin_id=origin_id,
            cutoff=origin_cutoff,
            dataset_refs={
                role: source_refs[role]
                for role in (
                    "benchmark_total_return",
                    "corporate_actions",
                    "official_delisting_cash",
                )
            },
            artifacts=artifacts,
        )
        preselect = seal_semantic(
            {
                "authority": dict(NO_AUTHORITY),
                "cutoff": origin_cutoff,
                "locator_id": f"preselect-{origin_id}",
                "origin": origin.isoformat(),
                "pit_catalog_ref": source_refs["pit_catalog"],
                "protocol_version": "myquant.v17.v4",
                "strategy_id": STRATEGY,
                "version": "myquant.v17.v4.preselect-locator.v1",
            }
        )
        source_refs["preselect_locator"] = _store_v4_artifact(
            preselect,
            path=(
                "data/private/v17_v4_calibration/"
                f"{origin_id}/preselect.json"
            ),
            artifacts=artifacts,
        )
        initial_pool = seal_semantic(
            {
                "authority": dict(NO_AUTHORITY),
                "cutoff": origin_cutoff,
                "ordered_pool": list(pool),
                "origin": origin.isoformat(),
                "output_id": f"initial-pool-{origin_id}",
                "preselect_locator_ref": source_refs[
                    "preselect_locator"
                ],
                "protocol_version": "myquant.v17.v4",
                "strategy_id": STRATEGY,
                "version": "myquant.v17.v4.initial-pool-output.v1",
            }
        )
        source_refs["initial_pool"] = _store_v4_artifact(
            initial_pool,
            path=(
                "data/private/v17_v4_calibration/"
                f"{origin_id}/initial-pool.json"
            ),
            artifacts=artifacts,
        )
        for role, branch_kind, scores in (
            ("quant_branch", "QUANT", quant_scores),
            (
                "fundamental_branch",
                "FUNDAMENTAL",
                fundamental_scores,
            ),
        ):
            branch = seal_semantic(
                {
                    "authority": dict(NO_AUTHORITY),
                    "branch_kind": branch_kind,
                    "cutoff": origin_cutoff,
                    "initial_pool_ref": source_refs["initial_pool"],
                    "origin": origin.isoformat(),
                    "output_id": f"{role}-{origin_id}",
                    "protocol_version": "myquant.v17.v4",
                    "score_rows": [
                        {
                            "score": scores[symbol],
                            "symbol": symbol,
                        }
                        for symbol in pool
                    ],
                    "strategy_id": STRATEGY,
                    "version": "myquant.v17.v4.branch-output.v1",
                }
            )
            source_refs[role] = _store_v4_artifact(
                branch,
                path=(
                    "data/private/v17_v4_calibration/"
                    f"{origin_id}/{role}.json"
                ),
                artifacts=artifacts,
            )
        label_refs = {
            "label_60": _store_v4_artifact(
                seal_semantic(
                    {
                        "authority": dict(NO_AUTHORITY),
                        "cutoff": _utc_at_close(label_60),
                        "label_end_session": label_60.isoformat(),
                        "label_id": f"label-60-{origin_id}",
                        "label_kind": "LABEL_60",
                        "origin": origin.isoformat(),
                        "protocol_version": "myquant.v17.v4",
                        "rows": [
                            {
                                "delisted": False,
                                "forward_return": forward60[symbol],
                                "official_terminal_cash": False,
                                "symbol": symbol,
                            }
                            for symbol in pool
                        ],
                        "strategy_id": STRATEGY,
                        "version": (
                            "myquant.v17.v4.total-return-labels.v1"
                        ),
                    }
                ),
                path=(
                    "data/private/v17_v4_calibration/"
                    f"{origin_id}/label-60.json"
                ),
                artifacts=artifacts,
            ),
            "label_252": _store_v4_artifact(
                seal_semantic(
                    {
                        "authority": dict(NO_AUTHORITY),
                        "cutoff": _utc_at_close(label_252),
                        "label_end_session": label_252.isoformat(),
                        "label_id": f"label-252-{origin_id}",
                        "label_kind": "LABEL_252",
                        "origin": origin.isoformat(),
                        "protocol_version": "myquant.v17.v4",
                        "rows": [
                            {
                                "delisted": False,
                                "forward_return": forward252[symbol],
                                "official_terminal_cash": False,
                                "symbol": symbol,
                            }
                            for symbol in pool
                        ],
                        "strategy_id": STRATEGY,
                        "version": (
                            "myquant.v17.v4.total-return-labels.v1"
                        ),
                    }
                ),
                path=(
                    "data/private/v17_v4_calibration/"
                    f"{origin_id}/label-252.json"
                ),
                artifacts=artifacts,
            ),
        }
        factor_set_sha, factor_ref = _factor_active_set_ref(
            origin=origin,
            origin_id=origin_id,
            origin_cutoff=origin_cutoff,
            artifacts=artifacts,
        )
        rows.append(
            OriginCalibrationInput(
                origin=origin,
                quant_history_start_session=sessions[index - 1_259],
                fundamental_history_start_session=sessions[index - 2_519],
                label_60_end_session=label_60,
                label_252_end_session=label_252,
                ordered_pool=pool,
                quant_scores=quant_scores,
                fundamental_scores=fundamental_scores,
                forward_return_60=forward60,
                forward_return_252=forward252,
                delisted_60=dict(false_map),
                delisted_252=dict(false_map),
                official_terminal_cash_60=dict(false_map),
                official_terminal_cash_252=dict(false_map),
                source_refs=source_refs,
                label_refs=label_refs,
                factor_active_set_ref=factor_ref,
                factor_effective_from_session=origin,
                factor_effective_to_session=origin,
                factor_set_sha256=factor_set_sha,
            )
        )
    active_cutoff = sessions[session_index[origins[-1]] + 252]
    cutoff = _utc_at_close(active_cutoff)
    return rows, sessions, active_cutoff, cutoff, artifacts


def _run(
    rows: list[OriginCalibrationInput],
    sessions: tuple[date, ...],
    active_cutoff: date,
    cutoff: str,
    artifacts: dict[str, bytes],
):
    return run_calibration_closure(
        rows,
        canonical_sessions=sessions,
        active_cutoff=active_cutoff,
        strategy_id=STRATEGY,
        run_id=RUN_ID,
        cutoff=cutoff,
        created_at=cutoff,
        artifact_loader=lambda reference: artifacts[
            reference["byte_sha256"]
        ],
    )


def test_120_month_closure_with_antecedents_promotes() -> None:
    rows, sessions, active_cutoff, cutoff, artifacts = _fixture()
    result = _run(rows, sessions, active_cutoff, cutoff, artifacts)

    for reference in (
        rows[0].source_refs["pit_catalog"],
        rows[0].source_refs["preselect_locator"],
        rows[0].source_refs["initial_pool"],
        rows[0].source_refs["quant_branch"],
        rows[0].source_refs["fundamental_branch"],
        rows[0].label_refs["label_60"],
        rows[0].label_refs["label_252"],
    ):
        validate_artifact(
            json.loads(
                artifacts[reference["byte_sha256"]].decode("utf-8")
            )
        )

    assert result.promoted is True
    assert result.blockers == ()
    assert result.input_origin_count == 138
    assert len(result.closure_origins) == CLOSURE_MONTHS
    assert len(result.folds) == 5
    assert all(len(fold.training_origins) == 60 for fold in result.folds)
    assert all(len(fold.oos_origins) == 12 for fold in result.folds)
    assert result.oos_hit60_lower_95 > 0.50
    assert result.oos_q25_252_lower_95 > 0
    assert result.origin_inventory["version"] == ORIGIN_INVENTORY_VERSION
    assert result.origin_inventory["authority"] == {
        "broker": False,
        "execution": False,
        "formal_research_publication": False,
        "order": False,
        "research_runtime_default": False,
        "trade": False,
    }

    inventory_ref = _store_v4_artifact(
        result.origin_inventory,
        path=(
            "data/private/v17_v4_calibration/"
            "calibration-run-1/origins.json"
        ),
        artifacts=artifacts,
    )
    loader = lambda reference: artifacts[reference["byte_sha256"]]
    quant = build_calibration_receipt(
        result,
        calibration_kind="QUANT_TIMING",
        receipt_id="quant-calibration-1",
        created_at=cutoff,
        origin_inventory_ref=inventory_ref,
        artifact_loader=loader,
    )
    fundamental = build_calibration_receipt(
        result,
        calibration_kind="FUNDAMENTAL_FORWARD",
        receipt_id="fundamental-calibration-1",
        created_at=cutoff,
        origin_inventory_ref=inventory_ref,
        artifact_loader=loader,
    )
    quant_ref = _store_v4_artifact(
        quant,
        path=(
            "data/private/v17_v4_calibration/"
            "calibration-run-1/quant.json"
        ),
        artifacts=artifacts,
    )
    fundamental_ref = _store_v4_artifact(
        fundamental,
        path=(
            "data/private/v17_v4_calibration/"
            "calibration-run-1/fundamental.json"
        ),
        artifacts=artifacts,
    )
    promotion = build_fusion_promotion_receipt(
        result,
        receipt_id="fusion-promotion-1",
        created_at=cutoff,
        origin_inventory_ref=inventory_ref,
        quant_calibration_receipt_ref=quant_ref,
        fundamental_calibration_receipt_ref=fundamental_ref,
        artifact_loader=loader,
    )
    assert quant["version"] == CALIBRATION_RECEIPT_VERSION
    assert quant["minimum_open_session_span"] == 1260
    assert fundamental["minimum_open_session_span"] == 2520
    assert promotion["version"] == FUSION_PROMOTION_RECEIPT_VERSION
    assert promotion["status"] == "PROMOTED"
    assert promotion["accepted"] is True
    assert promotion["bootstrap"]["replicates"] == BOOTSTRAP_REPLICATES
    assert promotion["bootstrap"]["seed"] == BOOTSTRAP_SEED
    validate_artifact(result.origin_inventory)
    validate_artifact(quant, artifact_loader=loader)
    validate_artifact(fundamental, artifact_loader=loader)
    validate_artifact(promotion, artifact_loader=loader)

    tampered = dict(promotion)
    tampered["oos_hit60_lower_95"] = "0.50"
    tampered.pop("semantic_sha256")
    tampered = seal_semantic(tampered)
    with pytest.raises(
        ArtifactContractError,
        match="threshold closure mismatch",
    ):
        validate_artifact(tampered, artifact_loader=loader)

    forged_inventory_ref = dict(inventory_ref)
    forged_inventory_ref["byte_sha256"] = _sha(
        "forged-inventory-bytes"
    )
    with pytest.raises(
        CalibrationClosureError,
        match="origin_inventory_read_failed",
    ):
        build_calibration_receipt(
            result,
            calibration_kind="QUANT_TIMING",
            receipt_id="forged-quant-calibration-1",
            created_at=cutoff,
            origin_inventory_ref=forged_inventory_ref,
            artifact_loader=loader,
        )

    second_fundamental = build_calibration_receipt(
        result,
        calibration_kind="FUNDAMENTAL_FORWARD",
        receipt_id="fundamental-calibration-2",
        created_at=cutoff,
        origin_inventory_ref=inventory_ref,
        artifact_loader=loader,
    )
    second_fundamental_ref = _store_v4_artifact(
        second_fundamental,
        path=(
            "data/private/v17_v4_calibration/"
            "calibration-run-1/fundamental-2.json"
        ),
        artifacts=artifacts,
    )
    with pytest.raises(
        CalibrationClosureError,
        match="calibration_receipt_readback_mismatch",
    ):
        build_fusion_promotion_receipt(
            result,
            receipt_id="wrong-kind-fusion-promotion-1",
            created_at=cutoff,
            origin_inventory_ref=inventory_ref,
            quant_calibration_receipt_ref=fundamental_ref,
            fundamental_calibration_receipt_ref=(
                second_fundamental_ref
            ),
            artifact_loader=loader,
        )


def test_bootstrap_matrix_identity_is_v4_frozen() -> None:
    assert bootstrap_matrix_sha256() == (
        "8e4467cf152ca8de71c94ed1a20715a18ba8eefa19428217541e0baa17df9458"
    )


def test_exact_spans_labels_cash_and_historical_factor_are_fail_closed() -> None:
    rows, sessions, active_cutoff, cutoff, artifacts = _fixture()
    session_index = {
        session: index for index, session in enumerate(sessions)
    }

    short_quant = deepcopy(rows)
    origin = date.fromisoformat(str(short_quant[0].origin))
    short_quant[0] = OriginCalibrationInput(
        **{
            **short_quant[0].__dict__,
            "quant_history_start_session": sessions[
                session_index[origin] - 1_258
            ],
        }
    )
    with pytest.raises(CalibrationClosureError, match="quant_span_below_1260"):
        _run(short_quant, sessions, active_cutoff, cutoff, artifacts)

    wrong_label = deepcopy(rows)
    wrong_label[0] = OriginCalibrationInput(
        **{
            **wrong_label[0].__dict__,
            "label_60_end_session": sessions[
                session_index[origin] + 59
            ],
        }
    )
    with pytest.raises(CalibrationClosureError, match="label_60_offset"):
        _run(wrong_label, sessions, active_cutoff, cutoff, artifacts)

    no_cash = deepcopy(rows)
    symbol = no_cash[0].ordered_pool[0]
    delisted_252 = dict(no_cash[0].delisted_252)
    delisted_252[symbol] = True
    no_cash[0] = replace(no_cash[0], delisted_252=delisted_252)
    with pytest.raises(
        CalibrationClosureError,
        match="delisting_without_official_terminal_cash",
    ):
        _run(no_cash, sessions, active_cutoff, cutoff, artifacts)

    retroactive = deepcopy(rows)
    origin_id = origin.strftime("%Y%m%d")
    _, factor_ref = _factor_active_set_ref(
        origin=origin,
        origin_id=f"{origin_id}-retroactive",
        origin_cutoff=_utc_at_close(origin),
        artifacts=artifacts,
        activated_at=cutoff,
    )
    retroactive[0] = replace(
        retroactive[0],
        factor_active_set_ref=factor_ref,
    )
    with pytest.raises(
        CalibrationClosureError,
        match="historical_factor_set_readback",
    ):
        _run(retroactive, sessions, active_cutoff, cutoff, artifacts)

    score_drift = deepcopy(rows)
    quant_scores = dict(score_drift[0].quant_scores)
    quant_scores[symbol] = "999"
    score_drift[0] = replace(
        score_drift[0],
        quant_scores=quant_scores,
    )
    with pytest.raises(
        CalibrationClosureError,
        match="quant_branch_native_scores",
    ):
        _run(score_drift, sessions, active_cutoff, cutoff, artifacts)

    byte_drift = dict(artifacts)
    pool_ref = rows[0].source_refs["initial_pool"]
    byte_drift[pool_ref["byte_sha256"]] = (
        byte_drift[pool_ref["byte_sha256"]] + b" "
    )
    with pytest.raises(
        CalibrationClosureError,
        match="initial_pool_byte_sha",
    ):
        _run(rows, sessions, active_cutoff, cutoff, byte_drift)

    relabelled = deepcopy(rows)
    relabelled_artifacts = dict(artifacts)
    origin_cutoff = _utc_at_close(origin)
    synthetic = seal_semantic(
        {
            "artifact_id": "synthetic-initial-pool",
            "calibration_binding": {
                "ordered_pool_sha256": _sha("caller-binding"),
            },
            "cutoff": origin_cutoff,
            "protocol_version": "myquant.v17.v4",
            "strategy_id": STRATEGY,
            "untyped_payload": {
                "this_is_not_an_initial_pool": True,
            },
            "version": "myquant.v17.v4.initial-pool-output.v1",
        }
    )
    synthetic_raw = canonical_resource_bytes(synthetic)
    synthetic_ref = {
        "artifact_id": synthetic["artifact_id"],
        "artifact_version": synthetic["version"],
        "byte_sha256": hashlib.sha256(synthetic_raw).hexdigest(),
        "cutoff": origin_cutoff,
        "relative_path": (
            "data/private/v17_v4_calibration/"
            "synthetic/initial-pool.json"
        ),
        "semantic_sha256": synthetic["semantic_sha256"],
        "strategy_id": STRATEGY,
    }
    relabelled_artifacts[synthetic_ref["byte_sha256"]] = synthetic_raw
    source_refs = dict(relabelled[0].source_refs)
    source_refs["initial_pool"] = synthetic_ref
    relabelled[0] = replace(
        relabelled[0],
        source_refs=source_refs,
    )
    with pytest.raises(
        CalibrationClosureError,
        match="initial_pool_native_schema",
    ):
        _run(
            relabelled,
            sessions,
            active_cutoff,
            cutoff,
            relabelled_artifacts,
        )


def test_skipped_month_and_120_only_leakage_are_blocked() -> None:
    rows, sessions, active_cutoff, cutoff, artifacts = _fixture()
    skipped = [*rows[:50], *rows[51:]]
    with pytest.raises(CalibrationClosureError, match="scheduled_month_skipped"):
        _run(skipped, sessions, active_cutoff, cutoff, artifacts)

    only_120 = rows[-120:]
    with pytest.raises(CalibrationClosureError, match="training_below_60"):
        _run(only_120, sessions, active_cutoff, cutoff, artifacts)


def test_promotion_lower_bounds_are_strict() -> None:
    rows, sessions, active_cutoff, cutoff, artifacts = _fixture(
        positive=False
    )
    result = _run(rows, sessions, active_cutoff, cutoff, artifacts)

    assert result.promoted is False
    assert result.blockers == (
        "oos_hit60_lower_95_not_above_0.50",
        "oos_q25_252_lower_95_not_above_zero",
    )
