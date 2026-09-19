"""One configured request locator, with exact history and borrowed automatic lock."""

import os
from pathlib import PurePosixPath
import secrets

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.errors import SystemSecurityError
from .journal_storage import JournalStorage
from .automatic_catchup_storage import AutomaticRunStorage
from .daily_contract import validate_ref
from .source_slot_contract import (
    PREFIX,
    SourceSlotError,
    seal,
    validate_locator,
    validate_transition,
)


class SourceLocatorStorage(JournalStorage):
    def __init__(self, workspace, config_ref):
        super().__init__(workspace)
        self.config_ref = dict(validate_ref(config_ref))
        self.prefix = f"{PREFIX}/{config_ref['sha256']}"
        self.current = self.prefix + "/request.v1.json"

    def _path(self, value):
        path = PurePosixPath(value)
        if str(path) != value:
            raise SystemSecurityError("SOURCE_LOCATOR_STORAGE_PATH_INVALID")
        if value == self.current:
            return path
        if path.parent != PurePosixPath(self.prefix) / "history" or path.suffix != ".json":
            raise SystemSecurityError("SOURCE_LOCATOR_STORAGE_PATH_INVALID")
        validate_ref({"path": "history", "sha256": path.stem})
        return path

    def chain(self):
        stored = self.read(self.current)
        chain, seen = [], set()
        for _ in range(4096):
            if stored is None:
                return chain
            value = validate_locator(
                parse_canonical_json_bytes(stored.data), config_ref=self.config_ref
            )
            if stored.byte_sha256 in seen:
                raise SourceSlotError("SOURCE_LOCATOR_HISTORY_CYCLE")
            seen.add(stored.byte_sha256)
            chain.append((value, stored))
            previous = value["previous_locator_sha256"]
            if previous is None:
                validate_transition(None, value)
                return chain
            prior = self.read(self.prefix + f"/history/{previous}.json")
            if prior is None or prior.byte_sha256 != previous:
                raise SourceSlotError("SOURCE_LOCATOR_HISTORY_MISSING_OR_CHANGED")
            old = validate_locator(
                parse_canonical_json_bytes(prior.data), config_ref=self.config_ref
            )
            validate_transition(old, value)
            stored = prior
        raise SourceSlotError("SOURCE_LOCATOR_HISTORY_BOUND_EXCEEDED")

    def _lock(self, lock):
        if type(lock) is not AutomaticRunStorage or lock.workspace != str(self._io.workspace_root):
            raise SourceSlotError("SOURCE_LOCATOR_LOCK_WORKSPACE_MISMATCH")
        lock.require_lock()

    def write(self, path, raw, *, projection=False):
        raise SystemSecurityError("SOURCE_LOCATOR_REQUIRES_CAS")

    def publish(self, value, *, expected_sha256, lock):
        self._lock(lock)
        chain = self.chain()
        old, stored = chain[0] if chain else (None, None)
        raw = canonical_json_bytes(validate_locator(value, config_ref=self.config_ref))
        if stored is not None and stored.data == raw:
            return stored
        actual = None if stored is None else stored.byte_sha256
        if actual != expected_sha256 or value["previous_locator_sha256"] != actual:
            raise SourceSlotError("SOURCE_LOCATOR_CAS_CONFLICT")
        validate_transition(old, value)
        if stored is not None:
            JournalStorage.write(self, self.prefix + f"/history/{actual}.json", stored.data)
        parent, leaf, relative = self._parent(self.current, create=True)
        temporary = ".source-locator-" + secrets.token_hex(12)
        try:
            current = self._io._read_leaf(parent, leaf, relative_path=relative, optional=True)
            if (None if current is None else current.byte_sha256) != actual:
                raise SourceSlotError("SOURCE_LOCATOR_CAS_CONFLICT")
            descriptor = self._io._write_temporary_file(parent, temporary, raw)
            os.close(descriptor)
            self._lock(lock)
            os.replace(temporary, leaf, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
            result = self._io._read_leaf(parent, leaf, relative_path=relative, optional=False)
            if result is None or result.data != raw:
                raise SourceSlotError("SOURCE_LOCATOR_READBACK_FAILED")
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)
        self._lock(lock)
        self.chain()
        return result


def publish_prepared_locator(
    *, workspace, config_ref, prior_sha256, commitment_ref, commitment, request_ref, lock
):
    """Fixed preparation hook; it cannot be supplied as executable input data."""
    storage = SourceLocatorStorage(workspace, config_ref)
    chain = storage.chain()
    if not chain:
        raise SourceSlotError("SOURCE_PREPARING_LOCATOR_REQUIRED")
    old, stored = chain[0]
    if old["state"] == "REQUEST_AVAILABLE":
        if old["request_ref"] != request_ref or old["preparation_commitment_ref"] != commitment_ref:
            raise SourceSlotError("SOURCE_PREPARED_LOCATOR_CONFLICT")
        return
    if stored.byte_sha256 != prior_sha256 or any(
        old[k] != commitment[k]
        for k in ("config_ref", "trade_date", "calendar_ref", "raw_calendar_ref")
    ):
        raise SourceSlotError("SOURCE_PREPARATION_LOCATOR_CHANGED")
    value = seal(
        {
            **old,
            "state": "REQUEST_AVAILABLE",
            "request_ref": request_ref,
            "preparation_commitment_ref": commitment_ref,
            "previous_locator_sha256": prior_sha256,
        }
    )
    storage.publish(value, expected_sha256=prior_sha256, lock=lock)
