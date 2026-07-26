"""Immutable byte-chain ledger storage with run-level flock CAS."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
from typing import Any

from quant_investor.v17_v2_contract.canonical import (
    CanonicalContractError,
    load_canonical_resource,
)
from quant_investor.v17_v2_contract.identities import (
    IdentityContractError,
    require_path_id,
    require_sha256,
)
from quant_investor.v17_v2_contract.validators import (
    V17V2ValidationError,
    validate_shadow_ledger,
    validate_shadow_ledger_chain,
    validate_shadow_ledger_successor,
)

from .storage import (
    CASMismatchError,
    EMPTY_SHA,
    ExactOnceConflictError,
    SecureStore,
    StorageError,
    StorageNotFoundError,
)


class LedgerStoreError(StorageError):
    """Raised when stored ledger bytes violate the immutable chain."""


@dataclass(frozen=True)
class LedgerCommit:
    run_id: str
    sequence: int
    byte_sha256: str
    committed: bool
    idempotent: bool


def _run_id(value: str) -> str:
    try:
        return require_path_id(value, label="run_id")
    except IdentityContractError as exc:
        raise LedgerStoreError(str(exc)) from exc


def _run_root(run_id: str) -> PurePosixPath:
    return PurePosixPath("results/v17_shadow/protocol-v2/runs") / run_id


def _ledger_path(run_id: str) -> PurePosixPath:
    return _run_root(run_id) / "ledger.json"


def _lock_path(run_id: str) -> PurePosixPath:
    return _run_root(run_id) / ".ledger.lock"


def _event_path(run_id: str, sequence: int) -> PurePosixPath:
    return _run_root(run_id) / "events" / f"ledger-{sequence:06d}.json"


def _sha256(value: str) -> str:
    try:
        return require_sha256(value, label="expected ledger SHA-256")
    except IdentityContractError as exc:
        raise LedgerStoreError(str(exc)) from exc


def _load_ledger(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise LedgerStoreError("ledger payload must be bytes")
    try:
        payload = load_canonical_resource(raw, label="stored shadow ledger")
        if type(payload) is not dict:
            raise LedgerStoreError("shadow ledger root must be an object")
        return validate_shadow_ledger(payload)
    except (CanonicalContractError, V17V2ValidationError) as exc:
        raise LedgerStoreError(f"shadow ledger validation failed: {exc}") from exc


@dataclass
class LedgerStore:
    """Preserve every ledger generation from sequence zero."""

    store: SecureStore

    def _read_current_optional(self, run_id: str) -> bytes | None:
        try:
            return self.store.read(_ledger_path(run_id))
        except StorageNotFoundError:
            return None

    def _read_chain_locked(self, run_id: str) -> tuple[bytes, ...]:
        current = self._read_current_optional(run_id)
        if current is None:
            raise LedgerStoreError(f"run ledger does not exist: {run_id}")
        current_doc = _load_ledger(current)
        sequence = current_doc.get("sequence")
        if type(sequence) is not int:
            raise LedgerStoreError("stored ledger sequence is invalid")
        chain: list[bytes] = []
        for index in range(sequence + 1):
            try:
                chain.append(self.store.read(_event_path(run_id, index)))
            except StorageNotFoundError as exc:
                raise LedgerStoreError(
                    f"immutable ledger event is missing at sequence {index}"
                ) from exc
        try:
            validate_shadow_ledger_chain(chain)
        except V17V2ValidationError as exc:
            raise LedgerStoreError(f"stored ledger chain is invalid: {exc}") from exc
        if chain[-1] != current:
            raise LedgerStoreError("current ledger bytes do not match immutable chain tip")
        return tuple(chain)

    def initialize(self, run_id: str, ledger_bytes: bytes) -> LedgerCommit:
        """Commit canonical sequence zero, retaining its immutable event bytes."""

        canonical_run_id = _run_id(run_id)
        ledger = _load_ledger(ledger_bytes)
        if ledger.get("run_id") != canonical_run_id or ledger.get("sequence") != 0:
            raise LedgerStoreError("initial ledger must be sequence zero for the exact run_id")
        preliminary = self._read_current_optional(canonical_run_id)
        if preliminary is not None and preliminary != ledger_bytes:
            raise ExactOnceConflictError(
                f"run {canonical_run_id} already has different initial bytes"
            )
        lock_path = _lock_path(canonical_run_id)
        with self.store.locked(lock_path):
            current = self._read_current_optional(canonical_run_id)
            if current is not None and current != ledger_bytes:
                raise ExactOnceConflictError(
                    f"run {canonical_run_id} already has different initial bytes"
                )
            event_result = self.store.write_exact_once(
                _event_path(canonical_run_id, 0),
                ledger_bytes,
            )
            ledger_result = self.store.write_exact_once(
                _ledger_path(canonical_run_id),
                ledger_bytes,
            )
            chain = self._read_chain_locked(canonical_run_id)
            if chain != (ledger_bytes,):
                raise LedgerStoreError("sequence-zero readback is not byte exact")
            digest = hashlib.sha256(ledger_bytes).hexdigest()
            created = event_result.created or ledger_result.created
            return LedgerCommit(
                canonical_run_id,
                0,
                digest,
                created,
                not created,
            )

    def append(
        self,
        run_id: str,
        expected_sha: str,
        successor_bytes: bytes,
    ) -> LedgerCommit:
        """Append one validated successor under a run flock and byte CAS."""

        canonical_run_id = _run_id(run_id)
        expected = _sha256(expected_sha)
        successor = _load_ledger(successor_bytes)
        if successor.get("run_id") != canonical_run_id:
            raise LedgerStoreError("successor run_id mismatch")
        sequence = successor.get("sequence")
        if type(sequence) is not int or sequence <= 0:
            raise LedgerStoreError("successor ledger sequence must be positive")
        preliminary = self._read_current_optional(canonical_run_id)
        if preliminary is None:
            raise CASMismatchError(expected, EMPTY_SHA)
        if preliminary != successor_bytes:
            preliminary_sha = hashlib.sha256(preliminary).hexdigest()
            if preliminary_sha != expected:
                raise CASMismatchError(expected, preliminary_sha)
        lock_path = _lock_path(canonical_run_id)
        with self.store.locked(lock_path):
            current = self._read_current_optional(canonical_run_id)
            if current is None:
                raise CASMismatchError(expected, EMPTY_SHA)
            if current == successor_bytes:
                chain = self._read_chain_locked(canonical_run_id)
                if chain[-1] != successor_bytes:
                    raise LedgerStoreError("idempotent ledger tip lacks immutable chain bytes")
                return LedgerCommit(
                    canonical_run_id,
                    sequence,
                    hashlib.sha256(successor_bytes).hexdigest(),
                    False,
                    True,
                )
            observed_sha = hashlib.sha256(current).hexdigest()
            if observed_sha != expected:
                raise CASMismatchError(expected, observed_sha)
            chain = self._read_chain_locked(canonical_run_id)
            if len(chain) != sequence:
                raise LedgerStoreError("successor sequence is not contiguous with stored chain")
            try:
                validate_shadow_ledger_successor(
                    predecessor_ledger_bytes=current,
                    successor_ledger=successor,
                )
            except V17V2ValidationError as exc:
                raise LedgerStoreError(f"ledger successor validation failed: {exc}") from exc
            self.store.write_exact_once(
                _event_path(canonical_run_id, sequence),
                successor_bytes,
            )
            result = self.store.replace_cas(
                _ledger_path(canonical_run_id),
                expected,
                successor_bytes,
            )
            readback_chain = self._read_chain_locked(canonical_run_id)
            if readback_chain != (*chain, successor_bytes):
                raise LedgerStoreError("successor chain readback is not byte exact")
            return LedgerCommit(
                canonical_run_id,
                sequence,
                result.byte_sha256,
                result.replaced,
                not result.replaced,
            )

    def read_chain(self, run_id: str) -> tuple[bytes, ...]:
        """Read and validate the complete immutable chain under its run flock."""

        canonical_run_id = _run_id(run_id)
        if self._read_current_optional(canonical_run_id) is None:
            raise LedgerStoreError(f"run ledger does not exist: {canonical_run_id}")
        with self.store.locked(_lock_path(canonical_run_id)):
            return self._read_chain_locked(canonical_run_id)


__all__ = [
    "LedgerCommit",
    "LedgerStore",
    "LedgerStoreError",
]
