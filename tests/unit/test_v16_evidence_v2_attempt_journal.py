from __future__ import annotations

import os
from pathlib import Path

import pytest

from quant_investor.v16.evidence_v2 import attempt_journal as journal_module
from quant_investor.v16.evidence_v2.attempt_journal import (
    GLOBAL_AUTHORITY_BLOCKERS,
    ProvisionalAttemptJournal,
    replay_provisional_events,
)
from quant_investor.v16.evidence_v2.contracts import (
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    seal_semantic,
    semantic_sha256,
    sha256_bytes,
)
from quant_investor.v16.evidence_v2.schedule import (
    ATTEMPT_GENESIS_SCHEMA,
    PRIVATE_ROOT_POLICY,
    SCHEDULE_DECLARATION_SCHEMA,
    ScheduleAnchorBinding,
    build_attempt_genesis,
)
from quant_investor.v16.evidence_v2.timestamp import TimestampAnchorBinding


def _ref(name: str, *, schema: str = "synthetic.v1") -> EvidenceRef:
    return EvidenceRef(
        schema_version="v16.evidence-ref.v2",
        artifact_schema=schema,
        absolute_path=f"/private/evidence/{name}",
        byte_sha256=sha256_bytes(f"bytes:{name}".encode("ascii")),
        semantic_sha256=sha256_bytes(f"semantic:{name}".encode("ascii")),
        root_policy=PRIVATE_ROOT_POLICY,
    )


def _bound(name: str, payload: dict[str, object]) -> BoundCanonicalArtifact:
    raw = canonical_json_bytes(payload)
    return BoundCanonicalArtifact(
        reference=EvidenceRef(
            schema_version="v16.evidence-ref.v2",
            artifact_schema=str(payload["schema_version"]),
            absolute_path=f"/private/evidence/{name}",
            byte_sha256=sha256_bytes(raw),
            semantic_sha256=semantic_sha256(payload),
            root_policy=PRIVATE_ROOT_POLICY,
        ),
        payload=raw,
    )


def _genesis() -> BoundCanonicalArtifact:
    payload = build_attempt_genesis(
        protocol_attempt_id="attempt-v16-001",
        runtime_capsule=_ref("runtime.json", schema="v16.hermetic-runtime-capsule.v2"),
        proposed_factor_graph=_ref(
            "factor-graph.json",
            schema="factor-v4.transition-graph.v2",
        ),
        open_session_calendar=_ref(
            "calendar.json",
            schema="v16.open-session-calendar.v2",
        ),
    )
    assert payload["schema_version"] == ATTEMPT_GENESIS_SCHEMA
    return _bound("attempt-genesis.json", payload)


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "journal"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


def _store(tmp_path: Path) -> ProvisionalAttemptJournal:
    return ProvisionalAttemptJournal(
        _root(tmp_path),
        _test_acl_checker=lambda _fd, _label: True,
    )


def _schedule_binding(epoch: str) -> tuple[ScheduleAnchorBinding, dict[str, object]]:
    genesis = _genesis().read()
    schedule_payload = seal_semantic(
        {
            "schema_version": SCHEDULE_DECLARATION_SCHEMA,
            "protocol_attempt_id": "attempt-v16-001",
            "epoch": epoch,
            "runtime_capsule": genesis["runtime_capsule"],
            "open_session_calendar": genesis["open_session_calendar"],
        }
    )
    schedule = _bound(f"schedule-{epoch}.json", schedule_payload)
    attempt = _bound(
        f"timestamp-attempt-{epoch}.json",
        seal_semantic(
            {
                "schema_version": "v16.rfc3161-attempt-state.v2",
                "fixture": f"attempt-{epoch}",
            }
        ),
    )
    receipt = _bound(
        f"timestamp-receipt-{epoch}.json",
        seal_semantic(
            {
                "schema_version": "v16.rfc3161-validation-receipt.v2",
                "fixture": f"receipt-{epoch}",
            }
        ),
    )
    timestamp = TimestampAnchorBinding(
        attempt=attempt,
        validation_receipt=receipt,
        verification_bundle=None,  # type: ignore[arg-type]
    )
    return ScheduleAnchorBinding(schedule=schedule, timestamp=timestamp), schedule_payload


def test_empty_and_initialized_state_are_permanently_nonauthorizing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    empty = store.read_state()
    assert empty["state"] == "empty"
    assert set(GLOBAL_AUTHORITY_BLOCKERS).issubset(empty["blockers"])
    assert empty["new_risk_authorized"] is False

    initialized = store.initialize(_genesis())
    assert initialized["state"] == "genesis_registered"
    assert initialized["event_count"] == 1
    assert initialized["external_anti_rollback_checkpoint_bound"] is False
    assert initialized["new_risk_authorized"] is False
    event_path = store.root / "00000000000000000000.event.json"
    assert event_path.stat().st_mode & 0o777 == 0o600

    with pytest.raises(EvidenceV2Error, match="already initialized"):
        store.initialize(_genesis())


def test_ordered_abc_journal_never_becomes_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    state = store.initialize(_genesis())

    for epoch in ("A", "B", "C"):
        if epoch == "C":
            state = store.record_factor_activation_bound(
                activation_receipt_ref=_ref(
                    "factor-activation-receipt.json",
                    schema="factor-governance-activation-receipt.v4",
                ),
                expected_head_sha256=state["head_event_byte_sha256"],
            )
        binding, schedule_payload = _schedule_binding(epoch)
        monkeypatch.setattr(
            journal_module,
            "validate_schedule_anchor_binding",
            lambda _binding, payload=schedule_payload: payload,
        )
        state = store.append_schedule(
            binding,
            expected_head_sha256=state["head_event_byte_sha256"],
        )
        assert state["state"] == f"epoch_{epoch.lower()}_scheduled"
        state = store.record_epoch_evidence_completed(
            epoch=epoch,
            evidence_refs=[_ref(f"epoch-{epoch}-evidence.json")],
            expected_head_sha256=state["head_event_byte_sha256"],
        )

    state = store.record_attempt_evidence_completed(
        evidence_refs=[_ref("attempt-complete-evidence.json")],
        expected_head_sha256=state["head_event_byte_sha256"],
    )
    assert state["state"] == "evidence_complete"
    assert state["new_risk_authorized"] is False
    assert "global_attempt_registry_authority_not_integrated" in state["blockers"]

    with pytest.raises(EvidenceV2Error, match="terminal state is absorbing"):
        store.record_terminal_failure(
            blockers=["too_late"],
            expected_head_sha256=state["head_event_byte_sha256"],
        )


def test_head_cas_and_terminal_failure_are_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    initialized = store.initialize(_genesis())
    stale_head = "0" * 64
    with pytest.raises(EvidenceV2Error, match="head CAS mismatch"):
        store.record_terminal_failure(
            blockers=["synthetic_failure"],
            expected_head_sha256=stale_head,
        )

    failed = store.record_terminal_failure(
        blockers=["synthetic_failure"],
        evidence_refs=[_ref("failure-evidence.json")],
        expected_head_sha256=initialized["head_event_byte_sha256"],
    )
    assert failed["state"] == "failed_terminal"
    assert failed["terminal_blockers"] == ["synthetic_failure"]
    assert "attempt_terminal:synthetic_failure" in failed["blockers"]
    assert failed["new_risk_authorized"] is False


def test_partial_event_write_remains_terminally_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    initialized = store.initialize(_genesis())
    real_write = os.write
    calls = 0

    def partial_write(descriptor: int, payload: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            view = memoryview(payload)  # type: ignore[arg-type]
            return real_write(descriptor, view[:8])
        raise OSError("synthetic interrupted persistence")

    monkeypatch.setattr(journal_module.os, "write", partial_write)
    with pytest.raises(EvidenceV2Error, match="terminally incomplete"):
        store.record_terminal_failure(
            blockers=["synthetic_failure"],
            expected_head_sha256=initialized["head_event_byte_sha256"],
        )
    monkeypatch.setattr(journal_module.os, "write", real_write)

    partial_path = store.root / "00000000000000000001.event.json"
    assert partial_path.exists()
    assert partial_path.stat().st_size == 8
    with pytest.raises(EvidenceV2Error):
        store.read_state()


def test_unknown_entry_and_hash_tamper_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize(_genesis())
    unknown = store.root / "unexpected.json"
    unknown.write_text("{}", encoding="utf-8")
    unknown.chmod(0o600)
    with pytest.raises(EvidenceV2Error, match="unexpected provisional journal entry"):
        store.read_state()
    unknown.unlink()

    event_path = store.root / "00000000000000000000.event.json"
    raw = event_path.read_bytes()
    event_path.write_bytes(raw.replace(b"genesis_registered", b"genesis_tampered__", 1))
    event_path.chmod(0o600)
    with pytest.raises(EvidenceV2Error):
        store.read_state()


def test_local_deletion_is_not_misrepresented_as_anti_rollback(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.initialize(_genesis())
    (store.root / "00000000000000000000.event.json").unlink()

    state = store.read_state()
    assert state["state"] == "empty"
    assert state["external_anti_rollback_checkpoint_bound"] is False
    assert "global_attempt_registry_authority_not_integrated" in state["blockers"]
    assert state["new_risk_authorized"] is False


def test_pure_replay_rejects_event_reorder() -> None:
    genesis = _genesis()
    payload = genesis.read()
    event = journal_module.build_provisional_event(
        sequence=0,
        event_type="attempt_genesis_registered",
        protocol_attempt_id="attempt-v16-001",
        previous_event_byte_sha256=None,
        state_before="empty",
        epoch=None,
        lineage={
            field: payload[field]
            for field in (
                "runtime_capsule",
                "proposed_factor_graph",
                "open_session_calendar",
            )
        },
        subject_refs=[genesis.reference],
    )
    changed = dict(event)
    changed.pop("semantic_sha256")
    changed["sequence"] = 1
    changed["previous_event_byte_sha256"] = "1" * 64
    changed = seal_semantic(changed)
    with pytest.raises(EvidenceV2Error):
        replay_provisional_events([changed])
