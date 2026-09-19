"""Confined automatic-run metadata storage and nonblocking execution capability."""

from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import os
from pathlib import Path, PurePosixPath
import secrets

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.errors import SystemSecurityError, SystemStorageError
from quant_investor.system.storage import _verify_file
from .automatic_catchup_contract import (
    PREFIX,
    run_path,
    document_ref,
    digest,
    AutomaticCatchupError,
)
from .daily_contract import ContractError, validate_ref
from .journal_storage import JournalStorage

LOCK = PREFIX + "/.run.lock"
PENDING = PREFIX + "/pending-run.v1.json"
PENDING_SCHEMA = "cn-daily-catchup-pending.v1"
_CURRENT = ContextVar("myquant_automatic_catchup", default=None)
_KEY = object()


class AutomaticRunStorage(JournalStorage):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = str(Path(workspace).resolve(strict=True))
        self._lock_state = None

    @staticmethod
    def _path(value):
        validate_ref({"path": value, "sha256": "0" * 64})
        path = PurePosixPath(value)
        if value in {LOCK, PENDING}:
            return path
        try:
            parts = path.relative_to(PREFIX).parts
        except ValueError as exc:
            raise SystemSecurityError("AUTO_STORAGE_PATH_INVALID") from exc
        if (
            len(parts) != 3
            or parts[0] != "runs"
            or parts[2]
            not in {"resolution.v1.json", "collection.json", "request.json", "closure.v1.json"}
        ):
            raise SystemSecurityError("AUTO_STORAGE_PATH_INVALID")
        validate_ref({"path": "run", "sha256": parts[1]})
        return path

    def require_lock(self):
        if self._lock_state is None:
            raise ContractError("AUTO_RUN_LOCK_REQUIRED")
        parent, fd, identity = self._lock_state
        _verify_file(os.fstat(fd))
        named = os.stat(".run.lock", dir_fd=parent, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != identity:
            raise ContractError("AUTO_RUN_LOCK_CHANGED")
        current, _, _ = self._parent(LOCK, create=False)
        try:
            a, b = os.fstat(current), os.fstat(parent)
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                raise ContractError("AUTO_RUN_ROOT_CHANGED")
        finally:
            os.close(current)

    @contextmanager
    def locked(self):
        if self._lock_state is not None:
            raise ContractError("AUTO_RUN_LOCK_REENTRANT")
        parent, leaf, _ = self._parent(LOCK, create=True)
        fd = None
        try:
            try:
                fd = os.open(
                    leaf, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent
                )
                os.fsync(parent)
            except FileExistsError:
                fd = os.open(leaf, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            metadata = os.fstat(fd)
            _verify_file(metadata)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AutomaticCatchupError("AUTO_RUN_BUSY") from exc
            self._lock_state = (parent, fd, (metadata.st_dev, metadata.st_ino))
            self.require_lock()
            yield self
            self.require_lock()
        finally:
            self._lock_state = None
            if fd is not None:
                os.close(fd)
            os.close(parent)

    def write(self, path, raw, *, projection=False):
        self.require_lock()
        if projection:
            raise SystemSecurityError("AUTO_PROJECTION_NOT_PUBLIC")
        if path == LOCK:
            raise SystemSecurityError("AUTO_LOCK_REWRITE_FORBIDDEN")
        if path == PENDING:
            raise SystemSecurityError("AUTO_PENDING_REQUIRES_LEASE_WRITER")
        result = super().write(path, raw)
        self.require_lock()
        return result

    def pending(self):
        stored = self.read(PENDING)
        if stored is None:
            return None
        value = parse_canonical_json_bytes(stored.data)
        return validate_pending(value)

    def set_pending(self, value):
        self.require_lock()
        value = validate_pending(value)
        self.pending()  # Corrupt prior state is never overwritten.
        raw = canonical_json_bytes(value)
        parent, leaf, relative = self._parent(PENDING, create=True)
        temporary = ".automatic-" + secrets.token_hex(12)
        try:
            previous = self._io._read_leaf(parent, leaf, relative_path=relative, optional=True)
            if previous is not None and previous.data == raw:
                return previous
            descriptor = self._io._write_temporary_file(parent, temporary, raw)
            os.close(descriptor)
            os.replace(temporary, leaf, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
            result = self._io._read_leaf(parent, leaf, relative_path=relative, optional=False)
            if result is None or result.data != raw:
                raise SystemStorageError("AUTO_PENDING_READBACK_FAILED")
            self.require_lock()
            return result
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)


def validate_pending(value):
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "state", "auto_request_ref", "resolution_ref"}
        or value["schema_version"] != PENDING_SCHEMA
        or type(value["state"]) is not str
        or value["state"] not in {"ACTIVE", "IDLE"}
    ):
        raise ContractError("AUTO_PENDING_INVALID")
    validate_ref(value["auto_request_ref"])
    validate_ref(value["resolution_ref"])
    if value["resolution_ref"]["path"] != run_path(value["auto_request_ref"], "resolution.v1.json"):
        raise ContractError("AUTO_PENDING_RESOLUTION_PATH_INVALID")
    return value


class _Capability:
    def __init__(self, storage, resolution_ref, resolution, key):
        if key is not _KEY:
            raise ContractError("AUTO_CAPABILITY_INVALID")
        self.storage = storage
        self.raw = canonical_json_bytes(
            {
                "workspace": storage.workspace,
                "resolution_ref": resolution_ref,
                "request_ref": resolution["derived_request_ref"],
            }
        )
        self.active = True
        self._fingerprint = (id(storage), digest(self.raw))

    def __reduce__(self):
        raise TypeError("automatic execution capabilities cannot be serialized")

    def require(self, workspace, request_ref):
        if not self.active or _CURRENT.get() is not self:
            raise ContractError("AUTO_CAPABILITY_EXPIRED")
        if self._fingerprint != (id(self.storage), digest(self.raw)):
            raise ContractError("AUTO_CAPABILITY_CHANGED")
        self.storage.require_lock()
        binding = parse_canonical_json_bytes(self.raw)
        if (
            binding["workspace"] != str(Path(workspace).resolve(strict=True))
            or binding["request_ref"] != request_ref
        ):
            raise ContractError("AUTO_CAPABILITY_BINDING_INVALID")


@contextmanager
def automatic_execution(
    storage, *, resolution_ref, resolution, completed_context=None, synthetic=False
):
    storage.require_lock()
    if _CURRENT.get() is not None:
        raise ContractError("AUTO_CAPABILITY_REENTRANT")
    path = run_path(resolution["auto_request_ref"], "resolution.v1.json")
    stored = storage.read(path)
    if (
        resolution_ref != document_ref(path, resolution)
        or stored is None
        or stored.data != canonical_json_bytes(resolution)
    ):
        raise ContractError("AUTO_CAPABILITY_RESOLUTION_INVALID")
    if storage.read(run_path(resolution["auto_request_ref"], "closure.v1.json")) is not None:
        raise ContractError("AUTO_RESOLUTION_CLOSED")
    pending = storage.pending()
    if pending is None or pending["resolution_ref"] != resolution_ref:
        raise ContractError("AUTO_CAPABILITY_PENDING_MISMATCH")
    if pending["state"] == "IDLE":
        from .automatic_catchup_closure import completion_state

        if completed_context is None or completed_context["resolution"] != resolution:
            raise ContractError("AUTO_IDLE_RUN_NOT_COMPLETED")
        _, unresolved, _ = completion_state(completed_context, synthetic=synthetic)
        if unresolved:
            raise ContractError("AUTO_IDLE_RUN_NOT_COMPLETED")
    capability = _Capability(storage, resolution_ref, resolution, _KEY)
    token = _CURRENT.set(capability)
    try:
        yield capability
    finally:
        capability.active = False
        _CURRENT.reset(token)


def require_dispatch_ownership(*, workspace, request_ref, request):
    """Public automatic roots and generated EXECUTE refs remain controller-private."""
    paths = [request_ref["path"]]
    if request.get("recipe_ref") is not None:
        paths.append(request["recipe_ref"]["path"])
    if any(PurePosixPath(PREFIX) in PurePosixPath(path).parents for path in paths):
        capability = _CURRENT.get()
        if capability is None:
            raise ContractError("AUTO_OWNED_REQUEST_INTERNAL_ONLY")
        capability.require(workspace, request_ref)
    if request["action"] == "EXECUTE":
        for path in paths:
            parts = PurePosixPath(path).parts
            if (
                len(parts) == 8
                and parts[:4] == ("results", "operations", "daily_production", "CN")
                and parts[5] == "catchup"
            ):
                raise ContractError("CATCHUP_GENERATED_EXECUTE_INTERNAL_ONLY")


def current_automatic_origin(*, workspace, derived_request_ref, synthetic=False):
    """Private forward origin accessor; valid only inside the active auto lock."""
    capability = _CURRENT.get()
    if capability is None:
        return None
    capability.require(workspace, derived_request_ref)
    bound = parse_canonical_json_bytes(capability.raw)
    from .automatic_catchup_resolution import read_automatic_resolution

    storage = capability.storage
    saved = storage.read(bound["resolution_ref"]["path"])
    if saved is None or saved.byte_sha256 != bound["resolution_ref"]["sha256"]:
        raise ContractError("AUTO_ORIGIN_RESOLUTION_CHANGED")
    value = parse_canonical_json_bytes(saved.data)
    context = read_automatic_resolution(
        workspace=workspace,
        resolution_ref=bound["resolution_ref"],
        release_install_ref=value["release_install_ref"],
        synthetic=synthetic,
    )
    pending = storage.pending()
    if (
        context["resolution"] != value
        or value["derived_request_ref"] != derived_request_ref
        or pending is None
        or pending["state"] != "ACTIVE"
        or pending["resolution_ref"] != bound["resolution_ref"]
        or pending["auto_request_ref"] != value["auto_request_ref"]
    ):
        raise ContractError("AUTO_ORIGIN_ACTIVE_BINDING_INVALID")
    capability.require(workspace, derived_request_ref)
    context["sources"].recheck()
    return {
        "auto_request_ref": value["auto_request_ref"],
        "resolution_ref": bound["resolution_ref"],
        "derived_request_ref": derived_request_ref,
    }
