"""Local provisional one-attempt journal for disconnected evidence-v2.

The journal provides crash-visible, append-only process coordination.  It is
not a global anti-rollback authority: the current OS user can still delete or
restore local files.  Consequently every projection remains nonauthorizing and
retains the global-authority migration blocker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterator

from .contracts import (
    ARTIFACT_MAX_BYTES,
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    parse_canonical_json_bytes,
    require_sha256,
    seal_semantic,
    sha256_bytes,
    validate_semantic_seal,
)
from .schedule import (
    ATTEMPT_GENESIS_SCHEMA,
    EPOCH_ORDER,
    PRIVATE_ROOT_POLICY,
    SCHEDULE_DECLARATION_SCHEMA,
    ScheduleAnchorBinding,
    validate_attempt_genesis,
    validate_schedule_anchor_binding,
)
from .secure_io import _AclChecker, _platform_ancestor_acl_safe, platform_acl_absent

PROVISIONAL_EVENT_SCHEMA = "v16.provisional-attempt-journal-event.v1"
PROVISIONAL_STATE_SCHEMA = "v16.provisional-attempt-journal-state.v1"
FACTOR_ACTIVATION_RECEIPT_SCHEMA = "factor-governance-activation-receipt.v4"
LOCK_FILENAME = ".v16-provisional-attempt-journal.lock"
MAX_EVENT_COUNT = 10_000

GLOBAL_AUTHORITY_BLOCKERS = (
    "evidence_v2_disconnected_from_authorizing_consumers",
    "global_attempt_registry_authority_not_integrated",
    "provisional_journal_head_not_bound_to_external_anti_rollback_authority",
)

_EVENT_NAME = re.compile(r"([0-9]{20})\.event\.json")
_TERMINAL_STATES = frozenset({"evidence_complete", "failed_terminal"})
_LINEAGE_FIELDS = {
    "runtime_capsule",
    "proposed_factor_graph",
    "open_session_calendar",
}


def _safe_id(value: Any, *, label: str) -> str:
    text = str(value or "")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if (
        not text
        or text != text.strip()
        or len(text) > 128
        or any(character not in allowed for character in text)
    ):
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    return text


def _reference(value: Any, *, label: str) -> EvidenceRef:
    if not isinstance(value, EvidenceRef):
        raise EvidenceV2Error(f"{label} must be an EvidenceRef")
    return value


def _lineage_from_genesis(genesis: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    return {
        field: EvidenceRef.from_dict(genesis[field]).to_dict()
        for field in sorted(_LINEAGE_FIELDS)
    }


def _normalize_lineage(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != _LINEAGE_FIELDS:
        raise EvidenceV2Error("provisional journal lineage fields mismatch")
    return {
        field: EvidenceRef.from_dict(value[field]).to_dict()
        for field in sorted(_LINEAGE_FIELDS)
    }


def _normalize_refs(values: Sequence[EvidenceRef], *, allow_empty: bool) -> list[dict[str, str]]:
    refs = [_reference(item, label="journal subject reference") for item in values]
    if not allow_empty and not refs:
        raise EvidenceV2Error("provisional journal event requires subject references")
    identities = [item.byte_sha256 for item in refs]
    if len(identities) != len(set(identities)):
        raise EvidenceV2Error("provisional journal subject references must be byte-distinct")
    return [item.to_dict() for item in refs]


def _transition(
    *,
    event_type: str,
    epoch: str | None,
    state_before: str,
) -> str:
    if state_before in _TERMINAL_STATES:
        raise EvidenceV2Error("provisional journal terminal state is absorbing")
    if event_type == "attempt_genesis_registered":
        if epoch is not None or state_before != "empty":
            raise EvidenceV2Error("attempt genesis must be the first journal event")
        return "genesis_registered"
    if event_type == "epoch_schedule_registered":
        expected = {
            "A": "genesis_registered",
            "B": "epoch_a_complete",
            "C": "factor_activation_bound",
        }
        if epoch not in expected or state_before != expected[epoch]:
            raise EvidenceV2Error("epoch schedule is out of order")
        return f"epoch_{epoch.lower()}_scheduled"
    if event_type == "epoch_evidence_completed":
        if epoch not in EPOCH_ORDER or state_before != f"epoch_{epoch.lower()}_scheduled":
            raise EvidenceV2Error("epoch completion is out of order")
        return f"epoch_{epoch.lower()}_complete"
    if event_type == "factor_activation_bound":
        if epoch is not None or state_before != "epoch_b_complete":
            raise EvidenceV2Error("Factor activation binding requires completed epoch B")
        return "factor_activation_bound"
    if event_type == "attempt_evidence_completed":
        if epoch is not None or state_before != "epoch_c_complete":
            raise EvidenceV2Error("attempt evidence completion requires completed epoch C")
        return "evidence_complete"
    if event_type == "attempt_failed_terminal":
        if epoch is not None:
            raise EvidenceV2Error("terminal failure is protocol-wide, not epoch-scoped")
        return "failed_terminal"
    raise EvidenceV2Error("unsupported provisional journal event type")


def build_provisional_event(
    *,
    sequence: int,
    event_type: str,
    protocol_attempt_id: str,
    previous_event_byte_sha256: str | None,
    state_before: str,
    epoch: str | None,
    lineage: Mapping[str, Any],
    subject_refs: Sequence[EvidenceRef],
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or not 0 <= sequence < MAX_EVENT_COUNT
    ):
        raise EvidenceV2Error("provisional journal sequence is outside its bound")
    if sequence == 0:
        if previous_event_byte_sha256 is not None:
            raise EvidenceV2Error("first provisional event cannot have a predecessor")
    else:
        require_sha256(previous_event_byte_sha256, label="previous event byte SHA")
    normalized_blockers = sorted(set(str(item) for item in blockers if str(item)))
    if event_type == "attempt_failed_terminal":
        if not normalized_blockers:
            raise EvidenceV2Error("terminal failure requires exact blockers")
    elif normalized_blockers:
        raise EvidenceV2Error("non-failure journal events cannot carry blockers")
    allow_empty = event_type == "attempt_failed_terminal"
    state_after = _transition(
        event_type=event_type,
        epoch=epoch,
        state_before=state_before,
    )
    return seal_semantic(
        {
            "schema_version": PROVISIONAL_EVENT_SCHEMA,
            "protocol_version": "v16",
            "sequence": sequence,
            "event_type": event_type,
            "protocol_attempt_id": _safe_id(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "previous_event_byte_sha256": previous_event_byte_sha256,
            "state_before": state_before,
            "state_after": state_after,
            "epoch": epoch,
            "lineage": _normalize_lineage(lineage),
            "subject_refs": _normalize_refs(subject_refs, allow_empty=allow_empty),
            "blockers": normalized_blockers,
            "authority_scope": "local_provisional_no_anti_rollback",
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def validate_provisional_event(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_version",
        "sequence",
        "event_type",
        "protocol_attempt_id",
        "previous_event_byte_sha256",
        "state_before",
        "state_after",
        "epoch",
        "lineage",
        "subject_refs",
        "blockers",
        "authority_scope",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    if set(payload) != fields or payload["schema_version"] != PROVISIONAL_EVENT_SCHEMA:
        raise EvidenceV2Error("provisional journal event envelope mismatch")
    if payload["protocol_version"] != "v16":
        raise EvidenceV2Error("provisional journal protocol mismatch")
    if payload["authority_scope"] != "local_provisional_no_anti_rollback":
        raise EvidenceV2Error("provisional journal authority scope drift")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("provisional journal events must be nonauthorizing")
    subject_refs = payload["subject_refs"]
    if not isinstance(subject_refs, list):
        raise EvidenceV2Error("provisional journal subject refs must be a list")
    rebuilt = build_provisional_event(
        sequence=payload["sequence"],
        event_type=str(payload["event_type"]),
        protocol_attempt_id=str(payload["protocol_attempt_id"]),
        previous_event_byte_sha256=payload["previous_event_byte_sha256"],
        state_before=str(payload["state_before"]),
        epoch=(None if payload["epoch"] is None else str(payload["epoch"])),
        lineage=payload["lineage"],
        subject_refs=[EvidenceRef.from_dict(item) for item in subject_refs],
        blockers=payload["blockers"],
    )
    if rebuilt != payload:
        raise EvidenceV2Error("provisional journal event is not canonical")
    return payload


def replay_provisional_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [validate_provisional_event(item) for item in events]
    state = "empty"
    attempt_id: str | None = None
    lineage: dict[str, dict[str, str]] | None = None
    previous_sha: str | None = None
    terminal_blockers: list[str] = []
    for sequence, event in enumerate(normalized):
        if event["sequence"] != sequence:
            raise EvidenceV2Error("provisional journal event sequence has a gap or reorder")
        if event["previous_event_byte_sha256"] != previous_sha:
            raise EvidenceV2Error("provisional journal predecessor hash mismatch")
        if event["state_before"] != state:
            raise EvidenceV2Error("provisional journal state predecessor mismatch")
        expected_after = _transition(
            event_type=str(event["event_type"]),
            epoch=(None if event["epoch"] is None else str(event["epoch"])),
            state_before=state,
        )
        if event["state_after"] != expected_after:
            raise EvidenceV2Error("provisional journal state transition mismatch")
        if sequence == 0:
            if event["event_type"] != "attempt_genesis_registered":
                raise EvidenceV2Error("provisional journal lacks a genesis event")
            attempt_id = str(event["protocol_attempt_id"])
            lineage = _normalize_lineage(event["lineage"])
        elif event["protocol_attempt_id"] != attempt_id or event["lineage"] != lineage:
            raise EvidenceV2Error("provisional journal immutable lineage drift")
        state = expected_after
        if state == "failed_terminal":
            terminal_blockers = list(event["blockers"])
        previous_sha = sha256_bytes(canonical_json_bytes(event))
    blockers = list(GLOBAL_AUTHORITY_BLOCKERS)
    blockers.extend(f"attempt_terminal:{item}" for item in terminal_blockers)
    return seal_semantic(
        {
            "schema_version": PROVISIONAL_STATE_SCHEMA,
            "protocol_version": "v16",
            "protocol_attempt_id": attempt_id,
            "state": state,
            "event_count": len(normalized),
            "head_event_byte_sha256": previous_sha,
            "lineage": lineage,
            "terminal_blockers": terminal_blockers,
            "authority_scope": "local_provisional_no_anti_rollback",
            "external_anti_rollback_checkpoint_bound": False,
            "readiness_status": "no_new_risk",
            "blockers": sorted(set(blockers)),
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def _signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


class ProvisionalAttemptJournal:
    """Append-only local journal with no default path and no reset API."""

    def __init__(
        self,
        root: str | Path,
        *,
        _test_acl_checker: _AclChecker | None = None,
    ) -> None:
        root_text = os.fspath(root)
        if (
            not root_text.startswith("/")
            or "\x00" in root_text
            or os.path.normpath(root_text) != root_text
            or root_text.startswith("//")
        ):
            raise EvidenceV2Error("provisional journal root must be canonical and absolute")
        self.root = Path(root_text)
        self._acl_checker = _test_acl_checker or platform_acl_absent
        self._ancestor_acl_checker = (
            _test_acl_checker or _platform_ancestor_acl_safe
        )

    def _open_root(self) -> int:
        descriptors: list[int] = []
        try:
            descriptor = os.open(
                "/",
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            descriptors.append(descriptor)
            ancestor_path = Path("/")
            for part in self.root.parts[1:]:
                parent = os.fstat(descriptor)
                if not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode) & 0o022:
                    raise EvidenceV2Error(
                        "provisional journal ancestor is not a trusted directory"
                    )
                if (
                    self._ancestor_acl_checker(descriptor, str(ancestor_path))
                    is not True
                ):
                    raise EvidenceV2Error(
                        "provisional journal ancestor has an extended allow ACL"
                    )
                descriptor = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptors[-1],
                )
                descriptors.append(descriptor)
                ancestor_path /= part
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or metadata.st_uid != os.getuid()
                or self._acl_checker(descriptor, str(self.root)) is not True
            ):
                raise EvidenceV2Error("provisional journal root owner/mode/ACL mismatch")
        except OSError as exc:
            for opened in reversed(descriptors):
                os.close(opened)
            raise EvidenceV2Error("provisional journal root open failed") from exc
        except Exception:
            for opened in reversed(descriptors):
                os.close(opened)
            raise
        for ancestor in descriptors[:-1]:
            os.close(ancestor)
        return descriptor

    def _check_regular(self, descriptor: int, *, label: str) -> os.stat_result:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or self._acl_checker(descriptor, label) is not True
        ):
            raise EvidenceV2Error(f"{label} owner/mode/link/ACL mismatch")
        return metadata

    @contextmanager
    def _locked(self) -> Iterator[int]:
        root_fd = self._open_root()
        lock_fd: int | None = None
        try:
            lock_fd = os.open(
                LOCK_FILENAME,
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=root_fd,
            )
            os.fchmod(lock_fd, 0o600)
            self._check_regular(lock_fd, label="provisional journal lock")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield root_fd
            path_metadata = os.stat(self.root, follow_symlinks=False)
            if (path_metadata.st_dev, path_metadata.st_ino) != (
                os.fstat(root_fd).st_dev,
                os.fstat(root_fd).st_ino,
            ):
                raise EvidenceV2Error("provisional journal root was replaced during operation")
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
            os.close(root_fd)

    def _read_event(self, root_fd: int, name: str) -> dict[str, Any]:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=root_fd,
            )
        except OSError as exc:
            raise EvidenceV2Error("provisional journal event open failed") from exc
        try:
            before = self._check_regular(descriptor, label=f"provisional journal event {name}")
            if before.st_size <= 0 or before.st_size > ARTIFACT_MAX_BYTES:
                raise EvidenceV2Error("provisional journal event size is invalid")
            path_before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, ARTIFACT_MAX_BYTES - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > ARTIFACT_MAX_BYTES:
                    raise EvidenceV2Error("provisional journal event exceeded its read bound")
                chunks.append(chunk)
            after = os.fstat(descriptor)
            path_after = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if (
                _signature(before) != _signature(after)
                or _signature(path_before) != _signature(path_after)
                or (after.st_dev, after.st_ino) != (path_after.st_dev, path_after.st_ino)
            ):
                raise EvidenceV2Error("provisional journal event changed during read")
        finally:
            os.close(descriptor)
        value = parse_canonical_json_bytes(b"".join(chunks))
        if not isinstance(value, Mapping):
            raise EvidenceV2Error("provisional journal event must be a JSON object")
        return validate_provisional_event(value)

    def _read_events(self, root_fd: int) -> list[dict[str, Any]]:
        try:
            entries = os.listdir(root_fd)
        except OSError as exc:
            raise EvidenceV2Error("provisional journal inventory failed") from exc
        event_entries: list[tuple[int, str]] = []
        for name in entries:
            if name == LOCK_FILENAME:
                continue
            match = _EVENT_NAME.fullmatch(name)
            if match is None:
                raise EvidenceV2Error(f"unexpected provisional journal entry: {name}")
            event_entries.append((int(match.group(1)), name))
        event_entries.sort()
        if [sequence for sequence, _ in event_entries] != list(range(len(event_entries))):
            raise EvidenceV2Error("provisional journal event files have a gap or reorder")
        return [self._read_event(root_fd, name) for _, name in event_entries]

    def _write_event(self, root_fd: int, event: Mapping[str, Any]) -> None:
        payload = validate_provisional_event(event)
        raw = canonical_json_bytes(payload)
        name = f"{payload['sequence']:020d}.event.json"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=root_fd,
            )
            try:
                remaining = memoryview(raw)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("provisional journal write made no progress")
                    remaining = remaining[written:]
                os.fsync(descriptor)
                metadata = self._check_regular(
                    descriptor,
                    label=f"provisional journal event {name}",
                )
                if metadata.st_size != len(raw):
                    raise OSError("provisional journal event size readback mismatch")
                os.fsync(root_fd)
            except Exception as exc:
                try:
                    os.fsync(descriptor)
                    os.fsync(root_fd)
                except OSError:
                    pass
                raise EvidenceV2Error(
                    "provisional journal event persistence is terminally incomplete"
                ) from exc
        except FileExistsError as exc:
            raise EvidenceV2Error("provisional journal event sequence already exists") from exc
        except OSError as exc:
            raise EvidenceV2Error("provisional journal event create failed") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def read_state(self) -> dict[str, Any]:
        with self._locked() as root_fd:
            return replay_provisional_events(self._read_events(root_fd))

    def initialize(self, genesis: BoundCanonicalArtifact) -> dict[str, Any]:
        if not isinstance(genesis, BoundCanonicalArtifact):
            raise EvidenceV2Error("provisional journal genesis must be byte-bound")
        if (
            genesis.reference.artifact_schema != ATTEMPT_GENESIS_SCHEMA
            or genesis.reference.root_policy != PRIVATE_ROOT_POLICY
        ):
            raise EvidenceV2Error("provisional journal genesis reference is invalid")
        genesis_payload = validate_attempt_genesis(genesis.read())
        with self._locked() as root_fd:
            state = replay_provisional_events(self._read_events(root_fd))
            if state["event_count"] != 0:
                raise EvidenceV2Error("provisional journal is already initialized")
            event = build_provisional_event(
                sequence=0,
                event_type="attempt_genesis_registered",
                protocol_attempt_id=str(genesis_payload["protocol_attempt_id"]),
                previous_event_byte_sha256=None,
                state_before="empty",
                epoch=None,
                lineage=_lineage_from_genesis(genesis_payload),
                subject_refs=[genesis.reference],
            )
            self._write_event(root_fd, event)
            return replay_provisional_events(self._read_events(root_fd))

    def _append(
        self,
        *,
        expected_head_sha256: str,
        event_type: str,
        epoch: str | None,
        subject_refs: Sequence[EvidenceRef],
        blockers: Sequence[str] = (),
    ) -> dict[str, Any]:
        expected = require_sha256(expected_head_sha256, label="expected journal head SHA")
        with self._locked() as root_fd:
            events = self._read_events(root_fd)
            state = replay_provisional_events(events)
            if state["head_event_byte_sha256"] != expected:
                raise EvidenceV2Error("provisional journal head CAS mismatch")
            if state["protocol_attempt_id"] is None or state["lineage"] is None:
                raise EvidenceV2Error("provisional journal is not initialized")
            event = build_provisional_event(
                sequence=len(events),
                event_type=event_type,
                protocol_attempt_id=str(state["protocol_attempt_id"]),
                previous_event_byte_sha256=expected,
                state_before=str(state["state"]),
                epoch=epoch,
                lineage=state["lineage"],
                subject_refs=subject_refs,
                blockers=blockers,
            )
            self._write_event(root_fd, event)
            updated = replay_provisional_events(self._read_events(root_fd))
            if updated["head_event_byte_sha256"] != sha256_bytes(canonical_json_bytes(event)):
                raise EvidenceV2Error("provisional journal post-append head mismatch")
            return updated

    def append_schedule(
        self,
        binding: ScheduleAnchorBinding,
        *,
        expected_head_sha256: str,
    ) -> dict[str, Any]:
        if not isinstance(binding, ScheduleAnchorBinding):
            raise EvidenceV2Error("provisional journal schedule binding has the wrong type")
        schedule = validate_schedule_anchor_binding(binding)
        state = self.read_state()
        if schedule["protocol_attempt_id"] != state["protocol_attempt_id"]:
            raise EvidenceV2Error("provisional journal schedule attempt ID drift")
        if (
            schedule["runtime_capsule"] != state["lineage"]["runtime_capsule"]
            or schedule["open_session_calendar"]
            != state["lineage"]["open_session_calendar"]
        ):
            raise EvidenceV2Error("provisional journal schedule immutable lineage drift")
        return self._append(
            expected_head_sha256=expected_head_sha256,
            event_type="epoch_schedule_registered",
            epoch=str(schedule["epoch"]),
            subject_refs=[
                binding.schedule.reference,
                binding.timestamp.attempt.reference,
                binding.timestamp.validation_receipt.reference,
            ],
        )

    def record_epoch_evidence_completed(
        self,
        *,
        epoch: str,
        evidence_refs: Sequence[EvidenceRef],
        expected_head_sha256: str,
    ) -> dict[str, Any]:
        return self._append(
            expected_head_sha256=expected_head_sha256,
            event_type="epoch_evidence_completed",
            epoch=epoch,
            subject_refs=evidence_refs,
        )

    def record_factor_activation_bound(
        self,
        *,
        activation_receipt_ref: EvidenceRef,
        expected_head_sha256: str,
    ) -> dict[str, Any]:
        reference = _reference(
            activation_receipt_ref,
            label="Factor activation receipt reference",
        )
        if reference.artifact_schema != FACTOR_ACTIVATION_RECEIPT_SCHEMA:
            raise EvidenceV2Error("Factor activation receipt schema is not v4")
        return self._append(
            expected_head_sha256=expected_head_sha256,
            event_type="factor_activation_bound",
            epoch=None,
            subject_refs=[reference],
        )

    def record_attempt_evidence_completed(
        self,
        *,
        evidence_refs: Sequence[EvidenceRef],
        expected_head_sha256: str,
    ) -> dict[str, Any]:
        return self._append(
            expected_head_sha256=expected_head_sha256,
            event_type="attempt_evidence_completed",
            epoch=None,
            subject_refs=evidence_refs,
        )

    def record_terminal_failure(
        self,
        *,
        blockers: Sequence[str],
        evidence_refs: Sequence[EvidenceRef] = (),
        expected_head_sha256: str,
    ) -> dict[str, Any]:
        return self._append(
            expected_head_sha256=expected_head_sha256,
            event_type="attempt_failed_terminal",
            epoch=None,
            subject_refs=evidence_refs,
            blockers=blockers,
        )


__all__ = [
    "GLOBAL_AUTHORITY_BLOCKERS",
    "PROVISIONAL_EVENT_SCHEMA",
    "PROVISIONAL_STATE_SCHEMA",
    "ProvisionalAttemptJournal",
    "build_provisional_event",
    "replay_provisional_events",
    "validate_provisional_event",
]
