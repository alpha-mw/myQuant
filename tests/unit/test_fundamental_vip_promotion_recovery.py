from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import pytest

from quant_investor.intelligence_v2.sources.tushare.fundamental_v4 import (
    FundamentalV4ContractError,
    append_promotion_journal_event,
    build_promotion_event,
    classify_promotion_recovery,
    create_promotion_journal,
    read_promotion_journal,
    validate_promotion_event,
    validate_promotion_event_chain,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4.promotion import (
    ZERO_SHA256,
)
import quant_investor.intelligence_v2.sources.tushare.fundamental_v4.upgrade as upgrade

NOW = "2026-08-09T08:00:00Z"
OLD = "a" * 64
NEW = "b" * 64


def intent() -> dict[str, Any]:
    return build_promotion_event(
        attempt_id="vip-upgrade-20260809-001",
        event_type="INTENT",
        ordinal=1,
        previous_event_sha256=ZERO_SHA256,
        evidence={
            "as_of": "20260807",
            "authorized_arguments_sha256": "1" * 64,
            "candidate_generation_id": "cn-fundamental-vip-20260807-v1",
            "candidate_pointer_sha256": NEW,
            "expected_old_pointer_sha256": OLD,
            "implementation_sha256": "2" * 64,
            "manifest_sha256": "3" * 64,
            "package_sha256": "4" * 64,
            "reconciliation_sha256": "5" * 64,
            "scope_sha256": "6" * 64,
        },
        event_at=NOW,
    )


def append_precas(root: Path) -> None:
    append_promotion_journal_event(
        root,
        event_type="PRECAS_VALIDATED",
        evidence={
            "expected_pointer_sha256": OLD,
            "generation_aggregate_sha256": "7" * 64,
            "generation_id": "cn-fundamental-vip-20260807-v1",
            "scope_sha256": "6" * 64,
        },
        event_at=NOW,
    )


def append_cas(root: Path) -> None:
    append_promotion_journal_event(
        root,
        event_type="CAS_COMMITTED",
        evidence={
            "generation_id": "cn-fundamental-vip-20260807-v1",
            "pointer_sha256": NEW,
            "previous_pointer_sha256": OLD,
        },
        event_at=NOW,
    )


def test_journal_exact_write_replay_and_promoted_recovery(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    create_promotion_journal(root, intent=intent())
    append_precas(root)
    append_cas(root)
    append_promotion_journal_event(
        root,
        event_type="POSTCHECK_PASSED",
        evidence={
            "generation_id": "cn-fundamental-vip-20260807-v1",
            "pointer_sha256": NEW,
            "scope_sha256": "6" * 64,
        },
        event_at=NOW,
    )
    events = read_promotion_journal(root)

    assert validate_promotion_event_chain(events) == events
    assert (
        classify_promotion_recovery(
            events,
            observed_pointer_sha256_first=NEW,
            observed_pointer_sha256_second=NEW,
            candidate_generation_valid=True,
            old_generation_valid=True,
        )
        == "PROMOTED"
    )
    assert stat_mode(root) == 0o700
    assert all(stat_mode(root / name) == 0o600 for name in os.listdir(root))

    append_promotion_journal_event(
        root,
        event_type="TERMINAL",
        evidence={"observed_pointer_sha256": NEW, "state": "PROMOTED"},
        event_at=NOW,
    )
    terminal = read_promotion_journal(root)
    assert terminal[-1]["evidence"]["state"] == "PROMOTED"


def stat_mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode & 0o777


def test_not_promoted_and_rollback_are_evidence_classified(tmp_path: Path) -> None:
    not_promoted = tmp_path / "not-promoted"
    create_promotion_journal(not_promoted, intent=intent())
    assert (
        classify_promotion_recovery(
            read_promotion_journal(not_promoted),
            observed_pointer_sha256_first=OLD,
            observed_pointer_sha256_second=OLD,
            candidate_generation_valid=False,
            old_generation_valid=True,
        )
        == "NOT_PROMOTED"
    )

    rolled_back = tmp_path / "rolled-back"
    create_promotion_journal(rolled_back, intent=intent())
    append_precas(rolled_back)
    append_cas(rolled_back)
    append_promotion_journal_event(
        rolled_back,
        event_type="ROLLBACK_COMMITTED",
        evidence={
            "generation_id": "cn-fundamental-vip-20260807-v1",
            "pointer_sha256": OLD,
            "rolled_back_from_sha256": NEW,
        },
        event_at=NOW,
    )
    assert (
        classify_promotion_recovery(
            read_promotion_journal(rolled_back),
            observed_pointer_sha256_first=OLD,
            observed_pointer_sha256_second=OLD,
            candidate_generation_valid=False,
            old_generation_valid=True,
        )
        == "ROLLED_BACK"
    )


@pytest.mark.parametrize(
    ("first", "second", "candidate_valid", "old_valid"),
    [
        (OLD, NEW, True, True),
        ("c" * 64, "c" * 64, True, True),
        (NEW, NEW, False, True),
        (OLD, OLD, True, False),
    ],
)
def test_uncertain_recovery_never_infers_success(
    tmp_path: Path,
    first: str,
    second: str,
    candidate_valid: bool,
    old_valid: bool,
) -> None:
    root = tmp_path / f"attempt-{first[0]}-{second[0]}-{candidate_valid}-{old_valid}"
    create_promotion_journal(root, intent=intent())
    append_precas(root)
    append_cas(root)
    assert (
        classify_promotion_recovery(
            read_promotion_journal(root),
            observed_pointer_sha256_first=first,
            observed_pointer_sha256_second=second,
            candidate_generation_valid=candidate_valid,
            old_generation_valid=old_valid,
        )
        == "PROMOTION_UNCERTAIN"
    )


def test_resealed_forgery_invalid_transition_and_terminal_append_rejected(
    tmp_path: Path,
) -> None:
    document = intent()
    assert validate_promotion_event(document) == document
    forged = copy.deepcopy(document)
    forged["evidence"]["as_of"] = "20260808"
    with pytest.raises(FundamentalV4ContractError):
        validate_promotion_event(forged)

    root = tmp_path / "attempt"
    create_promotion_journal(root, intent=document)
    with pytest.raises(FundamentalV4ContractError):
        append_cas(root)
    append_promotion_journal_event(
        root,
        event_type="TERMINAL",
        evidence={"observed_pointer_sha256": OLD, "state": "NOT_PROMOTED"},
        event_at=NOW,
    )
    with pytest.raises(FundamentalV4ContractError):
        append_precas(root)


def test_journal_rejects_mode_and_hardlink_tamper(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    create_promotion_journal(root, intent=intent())
    event_path = root / "01_INTENT.json"
    os.chmod(event_path, 0o644)
    with pytest.raises(FundamentalV4ContractError):
        read_promotion_journal(root)

    os.chmod(event_path, 0o600)
    linked = root / "02_PRECAS_VALIDATED.json"
    os.link(event_path, linked)
    with pytest.raises(FundamentalV4ContractError):
        read_promotion_journal(root)


def preflight() -> dict[str, Any]:
    return {
        "candidate_generation_id": "cn-fundamental-vip-20260807-v1",
        "candidate_pointer": {
            "metadata": {
                "provider_manifest": {
                    "as_of": "20260807",
                    "performance_gate_passed": True,
                    "schema_version": "cn-fundamental-provider-manifest.v4",
                }
            }
        },
        "candidate_pointer_sha256": NEW,
        "expected_pointer_sha256": OLD,
        "generation_aggregate_sha256": "7" * 64,
        "manifest_sha256": "3" * 64,
        "provider_evidence": {
            "implementation_sha256": "2" * 64,
            "reconciliation_sha256": "5" * 64,
        },
        "scope_sha256": "6" * 64,
    }


def run_upgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    promote: Any,
    observed_sha: str,
    observed_generation: str,
) -> dict[str, Any]:
    monkeypatch.setattr(
        upgrade,
        "preflight_staged_fundamental_promotion",
        lambda **_kwargs: preflight(),
    )
    monkeypatch.setattr(upgrade, "promote_staged_fundamental_generation", promote)
    monkeypatch.setattr(upgrade, "pointer_sha256", lambda _root: observed_sha)
    monkeypatch.setattr(
        upgrade,
        "load_fundamental_pointer",
        lambda _root: {"generation_id": observed_generation},
    )
    return upgrade.run_staged_vip_promotion(
        staging_root=tmp_path / "staging",
        canonical_root=tmp_path / "canonical",
        journal_root=tmp_path / "attempt",
        attempt_id="vip-upgrade-20260809-002",
        as_of="20260807",
        expected_pointer_sha256=OLD,
        package_sha256="4" * 64,
        authorized_arguments={"execute": True},
        clock=lambda: NOW,
    )


def test_upgrade_orchestration_records_success_without_reimplementing_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def promote(**kwargs: Any) -> dict[str, Any]:
        recorder = kwargs["phase_recorder"]
        recorder(
            "PRECAS_VALIDATED",
            {
                "expected_pointer_sha256": OLD,
                "generation_aggregate_sha256": "7" * 64,
                "generation_id": "cn-fundamental-vip-20260807-v1",
                "scope_sha256": "6" * 64,
            },
        )
        recorder(
            "CAS_COMMITTED",
            {
                "generation_id": "cn-fundamental-vip-20260807-v1",
                "pointer_sha256": NEW,
                "previous_pointer_sha256": OLD,
            },
        )
        recorder(
            "POSTCHECK_PASSED",
            {
                "generation_id": "cn-fundamental-vip-20260807-v1",
                "pointer_sha256": NEW,
                "scope_sha256": "6" * 64,
            },
        )
        return {"pointer_sha256": NEW}

    result = run_upgrade(
        tmp_path,
        monkeypatch,
        promote=promote,
        observed_sha=NEW,
        observed_generation="cn-fundamental-vip-20260807-v1",
    )
    assert result["state"] == "PROMOTED"
    assert result["terminal_recorded"] is True
    assert [row["event_type"] for row in read_promotion_journal(tmp_path / "attempt")] == [
        "INTENT",
        "PRECAS_VALIDATED",
        "CAS_COMMITTED",
        "POSTCHECK_PASSED",
        "TERMINAL",
    ]


def test_upgrade_interruption_uses_pointer_evidence_not_exception_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_before_cas(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("contains words promoted and rollback but proves nothing")

    result = run_upgrade(
        tmp_path,
        monkeypatch,
        promote=fail_before_cas,
        observed_sha=OLD,
        observed_generation="canonical-old",
    )
    assert result["state"] == "NOT_PROMOTED"
    assert read_promotion_journal(tmp_path / "attempt")[-1]["evidence"]["state"] == ("NOT_PROMOTED")
