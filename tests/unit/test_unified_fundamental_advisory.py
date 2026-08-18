from __future__ import annotations

import hashlib

import pytest

from quant_investor.contracts import canonical_json_bytes, get_contract, seal_artifact
from quant_investor.system import SystemContractError, SystemPreconditionError
from quant_investor.system.fundamental_advisory import (
    build_fundamental_advisory,
    build_fundamental_operator_veto,
    build_fundamental_veto_subject,
    factor_dependency_sha256,
    require_fundamental_proceed,
    validate_factor_dependency_rows,
    validate_fundamental_advisory,
    validate_fundamental_operator_veto,
)
from quant_investor.system.store import OBJECT_REF_SORT_FIELDS, object_ref_for_artifact

STAMP = "2026-08-18T00:00:00Z"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _ref(label: str) -> dict[str, str]:
    return {
        "kind": "system.source_object",
        "contract_sha256": get_contract("system.source_object").contract_sha256,
        "artifact_id": label,
        "semantic_sha256": _sha(f"semantic:{label}"),
        "byte_sha256": _sha(f"bytes:{label}"),
    }


def _sorted_refs(*labels: str) -> list[dict[str, str]]:
    return sorted(
        [_ref(label) for label in labels],
        key=lambda row: tuple(row[field] for field in OBJECT_REF_SORT_FIELDS),
    )


def _dependencies() -> list[dict[str, object]]:
    return validate_factor_dependency_rows(
        [
            {
                "factor_id": "pv_blend_volstab19x2_mom90_amihud5_w75",
                "required_source_roles": [
                    "EXCHANGE_CALENDAR",
                    "MARKET",
                    "PIT_MEMBERSHIP",
                ],
            },
            {
                "factor_id": "pv_blend_volstab19x2_mom90_amihud5_w80",
                "required_source_roles": [
                    "EXCHANGE_CALENDAR",
                    "MARKET",
                    "PIT_MEMBERSHIP",
                ],
            },
            {
                "factor_id": "pv_low_dollar_volume_5d",
                "required_source_roles": [
                    "EXCHANGE_CALENDAR",
                    "MARKET",
                    "PIT_MEMBERSHIP",
                ],
            },
        ]
    )


def _subject() -> dict[str, object]:
    dependencies = _dependencies()
    return build_fundamental_veto_subject(
        bootstrap_admission_intent_sha256=_sha("intent"),
        deployed_release_ref=_ref("release"),
        release_code_manifest_sha256=_sha("code"),
        system_as_of_date="2026-08-17",
        calendar_compilation_ref=_ref("calendar-compilation"),
        exchange_calendar_ref=_ref("calendar"),
        current_market_pointer_ref=_ref("market"),
        current_pit_pointer_ref=_ref("pit-pointer"),
        current_pit_membership_ref=_ref("pit-membership"),
        fundamental_pointer_ref=_ref("fundamental-pointer"),
        fundamental_manifest_ref=_ref("fundamental-manifest"),
        fundamental_table_refs=_sorted_refs("daily", "period", "quarantine"),
        fundamental_evidence_refs=_sorted_refs("evidence-a", "evidence-b"),
        fundamental_provenance_binding_sha256=_sha("provenance"),
        fundamental_target_bindings_sha256=_sha("target-bindings"),
        fundamental_snapshot_cutoff_date="2025-08-17",
        factor_set_sha256=_sha("factor-set"),
        factor_dependency_rows=dependencies,
        factor_dependency_sha256=factor_dependency_sha256(dependencies),
        created_at=STAMP,
    )


def _advisory(operator_veto_ref: dict[str, str] | None) -> dict[str, object]:
    dependencies = _dependencies()
    return build_fundamental_advisory(
        veto_subject_ref=object_ref_for_artifact(_subject()),
        operator_veto_ref=operator_veto_ref,
        integrity_status="VERIFIED",
        required_by_active_factor_set=False,
        system_as_of_date="2026-08-17",
        fundamental_snapshot_cutoff_date="2025-08-17",
        calendar_age_days=365,
        open_session_age=244,
        latest_admitted_available_at="2025-08-17",
        last_refresh_basis="SNAPSHOT_CUTOFF_DATE",
        disclosure_check="PASS",
        freshness_policy="ADVISORY_NO_FIXED_MAXIMUM",
        default_action="PROCEED",
        operator_veto_present=operator_veto_ref is not None,
        effective_action="BLOCK" if operator_veto_ref is not None else "PROCEED",
        factor_dependency_rows=dependencies,
        factor_dependency_sha256=factor_dependency_sha256(dependencies),
        fundamental_machine_states={
            "mixed": True,
            "legacy_direct_reader_provenance": "limited",
            "binding_aware_research_ready": True,
            "homogeneous_history_ready": False,
        },
        source_limitations=[
            "FUNDAMENTAL_HISTORY_MIXED",
            "FUNDAMENTAL_HISTORY_NOT_HOMOGENEOUS",
            "FUNDAMENTAL_LEGACY_DIRECT_READER_PROVENANCE_LIMITED",
        ],
        generic_json_max_bytes=64 * 1024**2,
        predecessor_manifest_max_bytes=128 * 1024**2,
        fundamental_parquet_max_bytes=512 * 1024**2,
        generic_replay_max_cells=100_000_000,
        daily_replay_max_cells=256_000_000,
        fundamental_table_source_rows=[
            {
                "table_name": "fundamental_daily",
                "source_ref": _ref("daily"),
                "row_count": 10,
                "column_count": 27,
                "observed_cells": 270,
                "cell_limit": 256_000_000,
            },
            {
                "table_name": "fundamental_period",
                "source_ref": _ref("period"),
                "row_count": 4,
                "column_count": 13,
                "observed_cells": 52,
                "cell_limit": 100_000_000,
            },
            {
                "table_name": "fundamental_quarantine",
                "source_ref": _ref("quarantine"),
                "row_count": 0,
                "column_count": 0,
                "observed_cells": 0,
                "cell_limit": 100_000_000,
            },
        ],
        predecessor_manifest_source_ref=_ref("predecessor-manifest"),
        ordinary_json_source_refs=_sorted_refs("ordinary-a", "ordinary-b"),
        created_at=STAMP,
    )


def test_arbitrary_age_is_disclosed_but_never_becomes_a_threshold() -> None:
    advisory = _advisory(None)
    assert validate_fundamental_advisory(advisory) == advisory
    assert require_fundamental_proceed(advisory) == advisory
    assert advisory["payload"]["calendar_age_days"] == 365
    assert advisory["payload"]["open_session_age"] == 244
    assert advisory["payload"]["freshness_policy"] == "ADVISORY_NO_FIXED_MAXIMUM"
    assert advisory["payload"]["effective_action"] == "PROCEED"
    encoded = canonical_json_bytes(advisory)
    assert b'"fresh"' not in encoded and b'"stale"' not in encoded


def test_veto_is_blocking_only_and_carries_zero_downstream_authority() -> None:
    subject = _subject()
    veto = build_fundamental_operator_veto(
        veto_subject_ref=object_ref_for_artifact(subject),
        reason_codes=["FUNDAMENTAL_OPERATOR_HOLD"],
        issued_at=STAMP,
        actor_uid=501,
    )
    assert validate_fundamental_operator_veto(veto) == veto
    veto_payload = veto["payload"]
    assert veto_payload["state"] == "VETO"
    assert all(
        veto_payload[field] is False
        for field in (
            "human_signature_claimed",
            "system_activation_authorized",
            "factor_activation_authorized",
            "portfolio_authority",
            "strategy_record_authority",
            "broker_authority",
            "order_authority",
            "trade_authority",
            "funds_transfer_authority",
        )
    )
    advisory = _advisory(object_ref_for_artifact(veto))
    assert advisory["payload"]["operator_veto_present"] is True
    assert advisory["payload"]["effective_action"] == "BLOCK"
    with pytest.raises(SystemPreconditionError, match="veto blocks"):
        require_fundamental_proceed(advisory)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("state", "ALLOW"),
        lambda payload: payload.__setitem__("human_signature_claimed", True),
        lambda payload: payload.__setitem__("trade_authority", True),
        lambda payload: payload.__setitem__("reason_codes", ["ARBITRARY_FREE_TEXT"]),
    ],
)
def test_veto_cannot_elevate_or_claim_unproved_authority(mutation: object) -> None:
    veto = build_fundamental_operator_veto(
        veto_subject_ref=object_ref_for_artifact(_subject()),
        reason_codes=["FUNDAMENTAL_OPERATOR_HOLD"],
        issued_at=STAMP,
        actor_uid=501,
    )
    payload = dict(veto["payload"])
    mutation(payload)  # type: ignore[operator]
    tampered = seal_artifact("system.fundamental_operator_veto", payload, created_at=STAMP)
    with pytest.raises(SystemContractError):
        validate_fundamental_operator_veto(tampered)


def test_dependency_rows_reject_fundamental_injection_and_reordering() -> None:
    dependencies = _dependencies()
    injected = [dict(row) for row in dependencies]
    injected[0] = {
        **injected[0],
        "required_source_roles": [
            "EXCHANGE_CALENDAR",
            "FUNDAMENTAL",
            "MARKET",
            "PIT_MEMBERSHIP",
        ],
    }
    assert factor_dependency_sha256(injected) != factor_dependency_sha256(dependencies)
    reordered = list(reversed(dependencies))
    with pytest.raises(SystemContractError, match="not sorted"):
        validate_factor_dependency_rows(reordered)
