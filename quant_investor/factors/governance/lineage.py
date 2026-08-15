"""Immutable observation lineage with a durable non-authorizing append CAS."""

from __future__ import annotations

from collections.abc import Mapping
import errno
import fcntl
import os
import secrets
import stat
from typing import Any, Final

from quant_investor.contracts import (
    MAX_CANONICAL_JSON_BYTES,
    canonical_json_bytes,
    seal_artifact,
)

from .common import (
    SIGNAL_OPEN_SESSIONS,
    artifact_ref,
    business_identity,
    canonical_identifier,
    exact_payload,
    observation_lineage_identity,
    validate_artifact_ref,
    validate_governance_artifact,
)
from .errors import FactorGovernanceError
from .prospective import (
    OBSERVATION_KIND,
    validate_configuration_selection,
    validate_observation,
    validate_preregistration,
)

OBSERVATION_HEAD_KIND: Final = "factor.observation_head"

_DIRECTORY_FLAGS: Final = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_FLAGS: Final = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_CREATE_FLAGS: Final = (
    os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)

_HEAD_FIELDS: Final = {
    "observation_head_id",
    "observation_lineage_id",
    "preregistration_id",
    "selection_id",
    "observation_count",
    "previous_head_ref",
    "head_observation_ref",
    "authority",
}


def _head_identity_inputs(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observation_lineage_id": payload["observation_lineage_id"],
        "preregistration_id": payload["preregistration_id"],
        "selection_id": payload["selection_id"],
        "observation_count": payload["observation_count"],
        "previous_head_ref": payload["previous_head_ref"],
        "head_observation_ref": payload["head_observation_ref"],
    }


def validate_observation_head(document: Mapping[str, Any] | bytes) -> dict[str, Any]:
    """Validate an immutable non-authorizing append head."""

    envelope, payload = exact_payload(document, kind=OBSERVATION_HEAD_KIND, fields=_HEAD_FIELDS)
    expected_lineage = observation_lineage_identity(
        payload["preregistration_id"], payload["selection_id"]
    )
    count = payload["observation_count"]
    previous = payload["previous_head_ref"]
    if (
        payload["observation_lineage_id"] != expected_lineage
        or type(count) is not int
        or not 1 <= count <= SIGNAL_OPEN_SESSIONS
        or payload["authority"] != "NON_AUTHORIZING"
        or (count == 1) is not (previous is None)
    ):
        raise FactorGovernanceError("observation head policy binding differs")
    if previous is not None:
        validate_artifact_ref(
            previous,
            label="previous_head_ref",
            expected_kind=OBSERVATION_HEAD_KIND,
        )
    validate_artifact_ref(
        payload["head_observation_ref"],
        label="head_observation_ref",
        expected_kind=OBSERVATION_KIND,
    )
    expected_id = business_identity("observation-head", _head_identity_inputs(payload))
    if payload["observation_head_id"] != expected_id:
        raise FactorGovernanceError("observation head business identity differs")
    return envelope


def _build_observation_head(
    observation: Mapping[str, Any], previous_head: Mapping[str, Any] | None
) -> dict[str, Any]:
    observation_payload = observation["payload"]
    ordinal = observation_payload["ordinal"]
    previous_ref = artifact_ref(previous_head) if previous_head is not None else None
    if previous_head is None:
        if ordinal != 0 or observation_payload["previous_observation_ref"] is not None:
            raise FactorGovernanceError("first append lacks the observation lineage root")
    else:
        previous_payload = previous_head["payload"]
        if (
            previous_payload["observation_count"] != ordinal
            or previous_payload["observation_lineage_id"]
            != observation_payload["observation_lineage_id"]
            or previous_payload["head_observation_ref"]
            != observation_payload["previous_observation_ref"]
        ):
            raise FactorGovernanceError("observation does not extend the expected head")
    payload: dict[str, Any] = {
        "observation_lineage_id": observation_payload["observation_lineage_id"],
        "preregistration_id": observation_payload["preregistration_id"],
        "selection_id": observation_payload["selection_id"],
        "observation_count": ordinal + 1,
        "previous_head_ref": previous_ref,
        "head_observation_ref": artifact_ref(observation),
        "authority": "NON_AUTHORIZING",
    }
    payload["observation_head_id"] = business_identity(
        "observation-head", _head_identity_inputs(payload)
    )
    return seal_artifact(OBSERVATION_HEAD_KIND, payload, created_at=observation["created_at"])


def _storage_error(detail: str) -> FactorGovernanceError:
    return FactorGovernanceError(detail, code="FACTOR_OBSERVATION_STORAGE_UNSAFE")


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _verify_directory(value: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(value.st_mode)
        or value.st_uid != os.geteuid()
        or stat.S_IMODE(value.st_mode) != 0o700
    ):
        raise _storage_error("observation lineage directory must be current-UID mode 0700")


def _verify_file(value: os.stat_result) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.geteuid()
        or stat.S_IMODE(value.st_mode) != 0o600
        or value.st_nlink != 1
    ):
        raise _storage_error("observation lineage file must be current-UID mode 0600")


def _absolute_root_parts(storage_root: str | os.PathLike[str]) -> tuple[str, ...]:
    try:
        raw = os.fspath(storage_root)
    except TypeError as exc:
        raise _storage_error("observation lineage root is invalid") from exc
    if type(raw) is not str or not raw or "\x00" in raw:
        raise _storage_error("observation lineage root is invalid")
    absolute = os.path.abspath(raw)
    parts = tuple(part for part in absolute.split(os.sep) if part)
    if not parts:
        raise _storage_error("observation lineage root cannot be the filesystem root")
    return parts


def _open_storage_root(storage_root: str | os.PathLike[str], *, optional: bool) -> int | None:
    descriptor: int | None = None
    try:
        descriptor = os.open(os.sep, _DIRECTORY_FLAGS)
        for part in _absolute_root_parts(storage_root):
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if optional:
                    os.close(descriptor)
                    return None
                raise _storage_error("observation lineage root is absent") from None
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise _storage_error("observation lineage root contains a symlink") from exc
                raise _storage_error("observation lineage root cannot be opened") from exc
            os.close(descriptor)
            descriptor = child
        _verify_directory(os.fstat(descriptor))
        return descriptor
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _reject_casefold_alias(parent_fd: int, leaf: str) -> None:
    try:
        with os.scandir(parent_fd) as entries:
            for entry in entries:
                if entry.name != leaf and entry.name.casefold() == leaf.casefold():
                    raise _storage_error("observation lineage path has a casefold collision")
    except FactorGovernanceError:
        raise
    except OSError as exc:
        raise _storage_error("observation lineage directory cannot be enumerated") from exc


def _open_directory_leaf(parent_fd: int, leaf: str) -> int:
    try:
        return os.open(leaf, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise _storage_error("observation lineage directory symlink rejected") from exc
        raise _storage_error("observation lineage directory cannot be opened") from exc


def _create_directory(parent_fd: int, leaf: str) -> int:
    created = False
    try:
        os.mkdir(leaf, mode=0o700, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    except OSError as exc:
        raise _storage_error("observation lineage directory cannot be created") from exc
    descriptor = _open_directory_leaf(parent_fd, leaf)
    try:
        if created:
            os.fchmod(descriptor, 0o700)
            os.fsync(parent_fd)
        _verify_directory(os.fstat(descriptor))
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory(parent_fd: int, leaf: str, *, create: bool) -> int | None:
    canonical_identifier(leaf, label="observation lineage directory")
    _reject_casefold_alias(parent_fd, leaf)
    try:
        descriptor = _open_directory_leaf(parent_fd, leaf)
    except FileNotFoundError:
        if not create:
            return None
        return _create_directory(parent_fd, leaf)
    try:
        _verify_directory(os.fstat(descriptor))
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_lineage_directory(
    storage_root: str | os.PathLike[str], lineage_id: str, *, create: bool
) -> int | None:
    canonical_identifier(lineage_id, label="observation_lineage_id")
    root_fd = _open_storage_root(storage_root, optional=not create)
    if root_fd is None:
        return None
    lineages_fd: int | None = None
    try:
        lineages_fd = _open_directory(root_fd, "factor_observation_lineage", create=create)
        if lineages_fd is None:
            return None
        return _open_directory(lineages_fd, lineage_id, create=create)
    finally:
        if lineages_fd is not None:
            os.close(lineages_fd)
        os.close(root_fd)


def _open_file(parent_fd: int, leaf: str, *, optional: bool) -> int | None:
    try:
        return os.open(leaf, _READ_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        if optional:
            return None
        raise _storage_error("observation lineage artifact is absent") from None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise _storage_error("observation lineage artifact symlink rejected") from exc
        raise _storage_error("observation lineage artifact cannot be opened") from exc


def _read_descriptor(descriptor: int, expected_size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _verify_exact_read(
    parent_fd: int,
    leaf: str,
    descriptor: int,
    before: os.stat_result,
    raw: bytes,
) -> None:
    after = os.fstat(descriptor)
    _verify_file(after)
    try:
        path_after = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise _storage_error("observation lineage artifact path changed") from exc
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(path_after)
        or len(raw) != after.st_size
    ):
        raise _storage_error("observation lineage artifact changed during read")


def _read_file(parent_fd: int, leaf: str, *, optional: bool) -> bytes | None:
    descriptor = _open_file(parent_fd, leaf, optional=optional)
    if descriptor is None:
        return None
    try:
        before = os.fstat(descriptor)
        _verify_file(before)
        if before.st_size <= 0 or before.st_size > MAX_CANONICAL_JSON_BYTES:
            raise _storage_error("observation lineage artifact size is invalid")
        raw = _read_descriptor(descriptor, before.st_size)
        _verify_exact_read(parent_fd, leaf, descriptor, before, raw)
        return raw
    finally:
        os.close(descriptor)


def _read_head(directory_fd: int) -> tuple[dict[str, Any] | None, bytes | None]:
    raw = _read_file(directory_fd, "head.json", optional=True)
    if raw is None:
        return None, None
    return validate_observation_head(raw), raw


def read_observation_head(
    storage_root: str | os.PathLike[str], observation_lineage_id: str
) -> dict[str, Any] | None:
    """Read the current non-authorizing append head, if one exists."""

    directory_fd = _open_lineage_directory(storage_root, observation_lineage_id, create=False)
    if directory_fd is None:
        return None
    try:
        head, raw = _read_head(directory_fd)
        if head is not None and raw is not None:
            if head["payload"]["observation_lineage_id"] != observation_lineage_id:
                raise FactorGovernanceError(
                    "stored observation head differs from its lineage directory"
                )
            _validate_stored_head_closure(directory_fd, head, raw)
        return head
    finally:
        os.close(directory_fd)


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise _storage_error("observation lineage artifact write was short")
        view = view[written:]


def _write_temporary(parent_fd: int, leaf: str, raw: bytes) -> int:
    descriptor = os.open(leaf, _CREATE_FLAGS, 0o600, dir_fd=parent_fd)
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        _verify_file(os.fstat(descriptor))
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_exact_once(parent_fd: int, leaf: str, raw: bytes) -> None:
    if type(raw) is not bytes or not raw or len(raw) > MAX_CANONICAL_JSON_BYTES:
        raise _storage_error("immutable observation lineage bytes are invalid")
    existing = _read_file(parent_fd, leaf, optional=True)
    if existing is not None:
        if existing != raw:
            raise FactorGovernanceError("immutable observation lineage artifact conflicts")
        return
    temporary = f".{leaf}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor: int | None = None
    try:
        descriptor = _write_temporary(parent_fd, temporary, raw)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(
                temporary,
                leaf,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            existing = _read_file(parent_fd, leaf, optional=False)
            if existing != raw:
                raise FactorGovernanceError("immutable observation lineage artifact conflicts")
        os.unlink(temporary, dir_fd=parent_fd)
        os.fsync(parent_fd)
        if _read_file(parent_fd, leaf, optional=False) != raw:
            raise _storage_error("immutable observation lineage readback differs")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass


def _replace_head(directory_fd: int, raw: bytes, *, expected_current_raw: bytes | None) -> None:
    if _read_file(directory_fd, "head.json", optional=True) != expected_current_raw:
        raise FactorGovernanceError(
            "observation append head changed",
            code="FACTOR_OBSERVATION_CAS_MISMATCH",
        )
    temporary = f".head.cas-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor: int | None = None
    try:
        descriptor = _write_temporary(directory_fd, temporary, raw)
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary,
            "head.json",
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
        if _read_file(directory_fd, "head.json", optional=False) != raw:
            raise _storage_error("observation append head readback differs")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _head_matches(current: Mapping[str, Any] | None, expected: Mapping[str, Any] | None) -> bool:
    if current is None or expected is None:
        return current is None and expected is None
    return artifact_ref(current) == artifact_ref(expected)


def _read_stored_artifact(
    directory_fd: int, directory_name: str, reference: Mapping[str, Any], *, kind: str
) -> tuple[dict[str, Any], bytes]:
    artifact_id = canonical_identifier(reference["artifact_id"], label="stored artifact id")
    artifacts_fd = _open_directory(directory_fd, directory_name, create=False)
    if artifacts_fd is None:
        raise _storage_error("observation lineage immutable directory is absent")
    try:
        raw = _read_file(artifacts_fd, f"{artifact_id}.json", optional=False)
    finally:
        os.close(artifacts_fd)
    if raw is None:  # pragma: no cover - optional=False is exhaustive
        raise _storage_error("observation lineage immutable artifact is absent")
    artifact = validate_governance_artifact(raw, expected_kind=kind)
    if artifact_ref(artifact) != reference:
        raise FactorGovernanceError("stored observation lineage reference differs")
    return artifact, raw


def _validate_stored_head_closure(
    directory_fd: int, current: Mapping[str, Any], current_raw: bytes
) -> dict[str, Any]:
    stored_head, stored_head_raw = _read_stored_artifact(
        directory_fd,
        "heads",
        artifact_ref(current),
        kind=OBSERVATION_HEAD_KIND,
    )
    if stored_head != current or stored_head_raw != current_raw:
        raise FactorGovernanceError("mutable observation head differs from immutable head")
    latest_observation: dict[str, Any] | None = None
    head = stored_head
    seen: set[str] = set()
    while True:
        validated_head = validate_observation_head(head)
        head_id = validated_head["payload"]["observation_head_id"]
        if head_id in seen:
            raise FactorGovernanceError("observation head history contains a cycle")
        seen.add(head_id)
        observation, _ = _read_stored_artifact(
            directory_fd,
            "observations",
            validated_head["payload"]["head_observation_ref"],
            kind=OBSERVATION_KIND,
        )
        _validate_head_observation_binding(validated_head, observation)
        if latest_observation is None:
            latest_observation = observation
        previous_ref = validated_head["payload"]["previous_head_ref"]
        if previous_ref is None:
            break
        previous_head, _ = _read_stored_artifact(
            directory_fd,
            "heads",
            previous_ref,
            kind=OBSERVATION_HEAD_KIND,
        )
        _validate_head_predecessor(validated_head, observation, previous_head)
        head = previous_head
    if latest_observation is None:  # pragma: no cover - every valid head has an observation
        raise FactorGovernanceError("observation head history is empty")
    return latest_observation


def _validate_head_observation_binding(
    head: Mapping[str, Any], observation: Mapping[str, Any]
) -> None:
    head_payload = head["payload"]
    observation_payload = observation["payload"]
    count = head_payload["observation_count"]
    if (
        head["created_at"] != observation["created_at"]
        or observation_payload["ordinal"] != count - 1
        or observation_payload["observation_lineage_id"] != head_payload["observation_lineage_id"]
        or observation_payload["preregistration_id"] != head_payload["preregistration_id"]
        or observation_payload["selection_id"] != head_payload["selection_id"]
        or (count == 1) is not (observation_payload["previous_observation_ref"] is None)
    ):
        raise FactorGovernanceError("observation head does not bind its observation")


def _validate_head_predecessor(
    current_head: Mapping[str, Any],
    current_observation: Mapping[str, Any],
    previous_head: Mapping[str, Any],
) -> None:
    current = current_head["payload"]
    previous = validate_observation_head(previous_head)["payload"]
    if (
        previous["observation_count"] != current["observation_count"] - 1
        or previous["observation_lineage_id"] != current["observation_lineage_id"]
        or previous["preregistration_id"] != current["preregistration_id"]
        or previous["selection_id"] != current["selection_id"]
        or current_observation["payload"]["previous_observation_ref"]
        != previous["head_observation_ref"]
        or previous_head["created_at"] >= current_head["created_at"]
    ):
        raise FactorGovernanceError("observation head predecessor binding differs")


def _previous_observation(
    directory_fd: int, current: Mapping[str, Any] | None, current_raw: bytes | None
) -> dict[str, Any] | None:
    if current is None:
        return None
    if current_raw is None:  # pragma: no cover - current/raw are paired
        raise _storage_error("observation append head bytes are absent")
    return _validate_stored_head_closure(directory_fd, current, current_raw)


def _open_lock(directory_fd: int) -> int:
    try:
        descriptor = os.open("head.lock", _CREATE_FLAGS, 0o600, dir_fd=directory_fd)
        os.fchmod(descriptor, 0o600)
        os.fsync(directory_fd)
    except FileExistsError:
        try:
            descriptor = os.open(
                "head.lock",
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise _storage_error("observation append lock cannot be opened") from exc
    except OSError as exc:
        raise _storage_error("observation append lock cannot be created") from exc
    try:
        _verify_file(os.fstat(descriptor))
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        path_stat = os.stat("head.lock", dir_fd=directory_fd, follow_symlinks=False)
        if _stat_identity(os.fstat(descriptor)) != _stat_identity(path_stat):
            raise _storage_error("observation append lock path changed")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def append_observation_cas(
    *,
    storage_root: str | os.PathLike[str],
    expected_head: Mapping[str, Any] | bytes | None,
    observation: Mapping[str, Any] | bytes,
    preregistration: Mapping[str, Any] | bytes,
    selection: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Append once; concurrent writers from one expected head cannot both win."""

    prereg = validate_preregistration(preregistration)
    selected = validate_configuration_selection(selection, preregistration=prereg)
    expected = validate_observation_head(expected_head) if expected_head is not None else None
    lineage_id = observation_lineage_identity(
        prereg["payload"]["preregistration_id"], selected["payload"]["selection_id"]
    )
    directory_fd = _open_lineage_directory(storage_root, lineage_id, create=True)
    if directory_fd is None:  # pragma: no cover - create=True is exhaustive
        raise _storage_error("observation lineage directory is absent")
    lock_fd: int | None = None
    observations_fd: int | None = None
    heads_fd: int | None = None
    try:
        lock_fd = _open_lock(directory_fd)
        current, current_raw = _read_head(directory_fd)
        if not _head_matches(current, expected):
            raise FactorGovernanceError(
                "observation append head changed",
                code="FACTOR_OBSERVATION_CAS_MISMATCH",
            )
        observed = validate_observation(
            observation,
            preregistration=prereg,
            selection=selected,
            previous_observation=_previous_observation(directory_fd, current, current_raw),
        )
        if observed["payload"]["observation_lineage_id"] != lineage_id:
            raise FactorGovernanceError("observation lineage differs from its cycle")
        head = validate_observation_head(_build_observation_head(observed, current))
        observations_fd = _open_directory(directory_fd, "observations", create=True)
        heads_fd = _open_directory(directory_fd, "heads", create=True)
        if observations_fd is None or heads_fd is None:  # pragma: no cover
            raise _storage_error("observation lineage immutable directory is absent")
        _write_exact_once(
            observations_fd,
            f"{observed['artifact_id']}.json",
            canonical_json_bytes(observed),
        )
        _write_exact_once(
            heads_fd,
            f"{head['artifact_id']}.json",
            canonical_json_bytes(head),
        )
        _replace_head(
            directory_fd,
            canonical_json_bytes(head),
            expected_current_raw=current_raw,
        )
        return head
    finally:
        if heads_fd is not None:
            os.close(heads_fd)
        if observations_fd is not None:
            os.close(observations_fd)
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(directory_fd)


__all__ = [
    "OBSERVATION_HEAD_KIND",
    "append_observation_cas",
    "read_observation_head",
    "validate_observation_head",
]
