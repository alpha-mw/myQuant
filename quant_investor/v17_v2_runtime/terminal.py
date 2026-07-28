"""Exact-once terminal outcome publication with explicit latest repair."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
from typing import Any

from quant_investor.v17_v2_contract.canonical import (
    CanonicalContractError,
    canonical_resource_bytes,
    load_canonical_resource,
)
from quant_investor.v17_v2_contract.identities import (
    IdentityContractError,
    require_path_id,
    require_sha256,
)
from quant_investor.v17_v2_contract.schema_validation import (
    SchemaValidationError,
    validate_mapping_against_packaged_schema,
)
from quant_investor.v17_v2_contract.resources import PackageResourceError
from quant_investor.v17_v2_contract.validators import (
    SHADOW_LATEST_POINTER_VERSION,
    SHADOW_OUTPUT_VERSION,
    V17V2ValidationError,
    validate_semantic_seal,
    validate_shadow_ledger,
)

from .storage import (
    CASMismatchError,
    EMPTY_SHA,
    LATEST_LOCK_PATH,
    LATEST_PATH,
    SecureStore,
    StorageCommitError,
    StorageError,
    StorageNotFoundError,
)

_TERMINAL_STATES = frozenset(
    {
        "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION",
        "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        "SHADOW_PORTFOLIO_INFEASIBLE",
        "HARD_STOP_SNAPSHOT_DRIFT",
        "HARD_STOP_INVALID_EVIDENCE",
    }
)


class TerminalPublicationError(StorageError):
    """Diagnosable terminal boundary failure; no silent state rewrite."""

    def __init__(
        self,
        detail: str,
        *,
        phase: str,
        outcome_committed: bool,
        latest_committed: bool,
    ) -> None:
        super().__init__(detail)
        self.phase = phase
        self.outcome_committed = outcome_committed
        self.latest_committed = latest_committed


@dataclass(frozen=True)
class TerminalPublication:
    run_id: str
    outcome_path: str
    outcome_sha256: str
    latest_sha256: str
    outcome_created: bool
    latest_replaced: bool
    repaired: bool


def _canonical_run_id(value: str) -> str:
    try:
        return require_path_id(value, label="run_id")
    except IdentityContractError as exc:
        raise TerminalPublicationError(
            str(exc),
            phase="PRECHECK",
            outcome_committed=False,
            latest_committed=False,
        ) from exc


def _expected_sha(value: str, *, label: str) -> str:
    if value == EMPTY_SHA:
        return value
    try:
        return require_sha256(value, label=label)
    except IdentityContractError as exc:
        raise TerminalPublicationError(
            str(exc),
            phase="PRECHECK",
            outcome_committed=False,
            latest_committed=False,
        ) from exc


def _load_contract(raw: bytes, *, expected_version: str, label: str) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise TerminalPublicationError(
            f"{label} must be bytes",
            phase="PRECHECK",
            outcome_committed=False,
            latest_committed=False,
        )
    try:
        document = load_canonical_resource(raw, label=label)
        if type(document) is not dict:
            raise TerminalPublicationError(
                f"{label} root must be an object",
                phase="PRECHECK",
                outcome_committed=False,
                latest_committed=False,
            )
        validate_mapping_against_packaged_schema(
            document,
            expected_version=expected_version,
        )
        validate_semantic_seal(document)
    except (
        CanonicalContractError,
        PackageResourceError,
        SchemaValidationError,
        V17V2ValidationError,
    ) as exc:
        raise TerminalPublicationError(
            f"{label} validation failed: {exc}",
            phase="PRECHECK",
            outcome_committed=False,
            latest_committed=False,
        ) from exc
    return document


def _ref_matches(
    reference: Any,
    *,
    document: dict[str, Any],
    relative_path: str,
    id_field: str,
) -> bool:
    return type(reference) is dict and reference == {
        "artifact_id": document[id_field],
        "artifact_version": document["version"],
        "relative_path": relative_path,
        "byte_sha256": hashlib.sha256(
            _canonical_bytes_for_validated_document(document)
        ).hexdigest(),
        "semantic_sha256": document["semantic_sha256"],
    }


def _canonical_bytes_for_validated_document(document: dict[str, Any]) -> bytes:
    return canonical_resource_bytes(document)


@dataclass
class TerminalPublisher:
    """Publish one immutable outcome and advance latest only by explicit CAS."""

    store: SecureStore

    def _run_lock(self, run_id: str) -> PurePosixPath:
        return PurePosixPath("results/v17_shadow/protocol-v2/runs") / run_id / ".ledger.lock"

    def _ledger_path(self, run_id: str) -> PurePosixPath:
        return PurePosixPath("results/v17_shadow/protocol-v2/runs") / run_id / "ledger.json"

    def _precheck_documents(
        self,
        *,
        run_id: str,
        expected_ledger_sha: str,
        outcome_path: str | PurePosixPath,
        outcome_bytes: bytes,
        latest_expected_sha: str,
        latest_bytes: bytes,
        repair: bool,
    ) -> tuple[PurePosixPath, dict[str, Any], dict[str, Any]]:
        canonical_run_id = _canonical_run_id(run_id)
        _expected_sha(expected_ledger_sha, label="expected_ledger_sha")
        latest_expected = _expected_sha(
            latest_expected_sha,
            label="latest_expected_sha",
        )
        expected_outcome_path = PurePosixPath(
            f"results/v17_shadow/protocol-v2/outcomes/{canonical_run_id}.json"
        )
        if str(outcome_path) != str(expected_outcome_path):
            raise TerminalPublicationError(
                "outcome_path must be the exact run-scoped protocol-v2 outcome path",
                phase="PRECHECK",
                outcome_committed=False,
                latest_committed=False,
            )
        output = _load_contract(
            outcome_bytes,
            expected_version=SHADOW_OUTPUT_VERSION,
            label="terminal outcome",
        )
        latest = _load_contract(
            latest_bytes,
            expected_version=SHADOW_LATEST_POINTER_VERSION,
            label="latest pointer",
        )
        mode = "REPAIR" if repair else "NORMAL"
        if (
            output.get("run_id") != canonical_run_id
            or latest.get("run_id") != canonical_run_id
            or output.get("terminal_state") not in _TERMINAL_STATES
            or latest.get("terminal_state") != output.get("terminal_state")
            or latest.get("pointer_path") != str(LATEST_PATH)
            or latest.get("publication_mode") != mode
            or latest.get("previous_pointer_byte_sha256") != latest_expected
        ):
            raise TerminalPublicationError(
                "terminal outcome/latest identity or publication mode mismatch",
                phase="PRECHECK",
                outcome_committed=False,
                latest_committed=False,
            )
        if repair and latest_expected == EMPTY_SHA:
            raise TerminalPublicationError(
                "explicit latest repair requires a predecessor pointer",
                phase="PRECHECK",
                outcome_committed=False,
                latest_committed=False,
            )
        if not _ref_matches(
            latest.get("terminal_output_ref"),
            document=output,
            relative_path=str(expected_outcome_path),
            id_field="run_id",
        ):
            raise TerminalPublicationError(
                "latest pointer does not bind exact terminal outcome bytes",
                phase="PRECHECK",
                outcome_committed=False,
                latest_committed=False,
            )
        return expected_outcome_path, output, latest

    def _check_locked_state(
        self,
        *,
        run_id: str,
        expected_ledger_sha: str,
        outcome: dict[str, Any],
        latest: dict[str, Any],
        latest_bytes: bytes,
        latest_expected_sha: str,
    ) -> bytes:
        try:
            ledger_bytes = self.store.read(self._ledger_path(run_id))
        except StorageNotFoundError as exc:
            raise CASMismatchError(expected_ledger_sha, EMPTY_SHA) from exc
        observed_ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
        if observed_ledger_sha != expected_ledger_sha:
            raise CASMismatchError(expected_ledger_sha, observed_ledger_sha)
        try:
            ledger_obj = load_canonical_resource(ledger_bytes, label="terminal ledger")
            if type(ledger_obj) is not dict:
                raise TerminalPublicationError(
                    "terminal ledger root must be an object",
                    phase="PRECHECK",
                    outcome_committed=False,
                    latest_committed=False,
                )
            ledger = validate_shadow_ledger(ledger_obj)
        except (CanonicalContractError, V17V2ValidationError) as exc:
            raise TerminalPublicationError(
                f"terminal ledger validation failed: {exc}",
                phase="PRECHECK",
                outcome_committed=False,
                latest_committed=False,
            ) from exc
        if ledger.get("state") not in _TERMINAL_STATES:
            raise TerminalPublicationError(
                "ledger is not terminal",
                phase="PRECHECK",
                outcome_committed=False,
                latest_committed=False,
            )
        ledger_path = str(self._ledger_path(run_id))
        if (
            outcome.get("terminal_state") != ledger.get("state")
            or not _ref_matches(
                outcome.get("ledger_ref"),
                document=ledger,
                relative_path=ledger_path,
                id_field="run_id",
            )
            or not _ref_matches(
                latest.get("ledger_ref"),
                document=ledger,
                relative_path=ledger_path,
                id_field="run_id",
            )
        ):
            raise TerminalPublicationError(
                "terminal outcome does not bind exact ledger bytes",
                phase="PRECHECK",
                outcome_committed=False,
                latest_committed=False,
            )
        latest_observed = self.store._read_optional(LATEST_PATH)
        latest_observed_sha = EMPTY_SHA if latest_observed is None else latest_observed.byte_sha256
        if latest_observed is not None and latest_observed.data == latest_bytes:
            return ledger_bytes
        if latest_observed_sha != latest_expected_sha:
            raise CASMismatchError(latest_expected_sha, latest_observed_sha)
        return ledger_bytes

    def _publish(
        self,
        *,
        run_id: str,
        expected_ledger_sha: str,
        outcome_path: str | PurePosixPath,
        outcome_bytes: bytes,
        latest_expected_sha: str,
        latest_bytes: bytes,
        repair: bool,
    ) -> TerminalPublication:
        expected_outcome_path, output, latest = self._precheck_documents(
            run_id=run_id,
            expected_ledger_sha=expected_ledger_sha,
            outcome_path=outcome_path,
            outcome_bytes=outcome_bytes,
            latest_expected_sha=latest_expected_sha,
            latest_bytes=latest_bytes,
            repair=repair,
        )
        canonical_run_id = str(output["run_id"])
        outcome_created = False
        try:
            preliminary_ledger = self.store.read(self._ledger_path(canonical_run_id))
        except StorageNotFoundError as exc:
            raise CASMismatchError(expected_ledger_sha, EMPTY_SHA) from exc
        preliminary_ledger_sha = hashlib.sha256(preliminary_ledger).hexdigest()
        if preliminary_ledger_sha != expected_ledger_sha:
            raise CASMismatchError(expected_ledger_sha, preliminary_ledger_sha)
        try:
            preliminary_latest = self.store._read_optional(LATEST_PATH)
        except StorageNotFoundError:
            preliminary_latest = None
        preliminary_latest_sha = (
            EMPTY_SHA if preliminary_latest is None else preliminary_latest.byte_sha256
        )
        if (
            preliminary_latest is None or preliminary_latest.data != latest_bytes
        ) and preliminary_latest_sha != latest_expected_sha:
            raise CASMismatchError(latest_expected_sha, preliminary_latest_sha)
        with self.store.locked(self._run_lock(canonical_run_id)):
            with self.store.locked(LATEST_LOCK_PATH):
                self._check_locked_state(
                    run_id=canonical_run_id,
                    expected_ledger_sha=expected_ledger_sha,
                    outcome=output,
                    latest=latest,
                    latest_bytes=latest_bytes,
                    latest_expected_sha=latest_expected_sha,
                )
                if repair:
                    try:
                        existing = self.store.read(expected_outcome_path)
                    except StorageNotFoundError as exc:
                        raise TerminalPublicationError(
                            "explicit latest repair requires a committed outcome",
                            phase="OUTCOME",
                            outcome_committed=False,
                            latest_committed=False,
                        ) from exc
                    if existing != outcome_bytes:
                        raise TerminalPublicationError(
                            "explicit latest repair outcome bytes differ from committed bytes",
                            phase="OUTCOME",
                            outcome_committed=True,
                            latest_committed=False,
                        )
                else:
                    try:
                        outcome_result = self.store.write_exact_once(
                            expected_outcome_path,
                            outcome_bytes,
                        )
                    except StorageCommitError as exc:
                        raise TerminalPublicationError(
                            f"outcome publication boundary failed: {exc}",
                            phase="OUTCOME",
                            outcome_committed=exc.possibly_committed,
                            latest_committed=False,
                        ) from exc
                    outcome_created = outcome_result.created
                try:
                    latest_result = self.store.replace_cas(
                        LATEST_PATH,
                        latest_expected_sha,
                        latest_bytes,
                    )
                except StorageCommitError as exc:
                    raise TerminalPublicationError(
                        f"latest publication boundary failed: {exc}",
                        phase="LATEST",
                        outcome_committed=True,
                        latest_committed=exc.possibly_committed,
                    ) from exc
        return TerminalPublication(
            canonical_run_id,
            str(expected_outcome_path),
            hashlib.sha256(outcome_bytes).hexdigest(),
            hashlib.sha256(latest_bytes).hexdigest(),
            outcome_created,
            latest_result.replaced,
            repair,
        )

    def publish(
        self,
        run_id: str,
        expected_ledger_sha: str,
        outcome_path: str | PurePosixPath,
        outcome_bytes: bytes,
        latest_expected_sha: str,
        latest_bytes: bytes,
    ) -> TerminalPublication:
        """Normal exact-once terminal publication."""

        return self._publish(
            run_id=run_id,
            expected_ledger_sha=expected_ledger_sha,
            outcome_path=outcome_path,
            outcome_bytes=outcome_bytes,
            latest_expected_sha=latest_expected_sha,
            latest_bytes=latest_bytes,
            repair=False,
        )

    def repair_latest(
        self,
        run_id: str,
        expected_ledger_sha: str,
        outcome_path: str | PurePosixPath,
        outcome_bytes: bytes,
        latest_expected_sha: str,
        latest_bytes: bytes,
    ) -> TerminalPublication:
        """Explicitly repair latest after verifying the committed outcome."""

        return self._publish(
            run_id=run_id,
            expected_ledger_sha=expected_ledger_sha,
            outcome_path=outcome_path,
            outcome_bytes=outcome_bytes,
            latest_expected_sha=latest_expected_sha,
            latest_bytes=latest_bytes,
            repair=True,
        )


__all__ = [
    "TerminalPublication",
    "TerminalPublicationError",
    "TerminalPublisher",
]
