"""Read-only V17 mainline authority derivation and public DTO projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .constants import (
    ACTIVE_POINTER_SCHEMA_ID,
    ACTIVE_STATE,
    AUTHORITY_FLAGS,
    BLOCKED_PREFIX,
    MAINLINE_RUN_SCHEMA_ID,
    MainlineBlocker,
    PROTOCOL,
    PUBLIC_RUN_SCHEMA_ID,
    SUPPORTED_CAPABILITY,
    SUPPORTED_MARKET,
    UNINITIALIZED_STATE,
)
from .contracts import (
    MainlineContractError,
    build_ref,
    contains_forbidden_authority,
    parse_canonical,
    require_identifier,
    require_sha256,
    seal_document,
    validate_active_pointer,
    validate_formal_output,
    validate_mainline_run,
    validate_portfolio_output,
    validate_source_closure,
)
from .storage import (
    MainlineCASMismatch,
    MainlineNotFound,
    MainlineStorageError,
    MainlineStorageSecurityError,
    MainlineStore,
    StoredBytes,
)


class V17MainlineError(RuntimeError):
    def __init__(self, code: str, detail: str | None = None) -> None:
        if type(code) is not str or not code:
            raise ValueError("V17 mainline error code must be non-empty text")
        self.code = code
        self.detail = detail
        super().__init__(code if detail is None else f"{code}:{detail}")


class MainlineUnavailable(V17MainlineError):
    def __init__(self, derived_state: str, blocker: MainlineBlocker) -> None:
        self.derived_state = derived_state
        self.blocker = blocker
        super().__init__(derived_state)


@dataclass(frozen=True)
class MainlineResolution:
    derived_state: str
    blocker: MainlineBlocker | None
    public_run: dict[str, Any] | None

    @property
    def is_active(self) -> bool:
        return self.derived_state == ACTIVE_STATE and self.public_run is not None


def active_pointer_path(strategy_id: str) -> str:
    strategy = require_identifier(strategy_id, label="strategy_id")
    return f"results/v17_mainline/strategies/{strategy}/_active.json"


def mainline_run_path(strategy_id: str, run_id: str) -> str:
    strategy = require_identifier(strategy_id, label="strategy_id")
    run = require_identifier(run_id, label="run_id")
    return f"results/v17_mainline/strategies/{strategy}/runs/{run}/run.json"


def _blocked(blocker: MainlineBlocker) -> MainlineResolution:
    return MainlineResolution(f"{BLOCKED_PREFIX}{blocker.value}", blocker, None)


def _strategy_argument(
    strategy_id: str | None,
    canonical_strategy_id: str | None,
) -> str:
    if strategy_id is None and canonical_strategy_id is None:
        raise V17MainlineError("V17_MAINLINE_ARGUMENTS_INVALID", "strategy_id is required")
    if (
        strategy_id is not None
        and canonical_strategy_id is not None
        and strategy_id != canonical_strategy_id
    ):
        raise V17MainlineError("V17_MAINLINE_ARGUMENTS_INVALID", "strategy identifiers disagree")
    try:
        return require_identifier(
            strategy_id if strategy_id is not None else canonical_strategy_id,
            label="strategy_id",
        )
    except MainlineContractError as exc:
        raise V17MainlineError("V17_MAINLINE_ARGUMENTS_INVALID", str(exc)) from exc


def _read_document(
    store: MainlineStore, reference: Mapping[str, str]
) -> tuple[StoredBytes, dict[str, Any]]:
    stored = store.read(reference["relative_path"], reference["byte_sha256"])
    return stored, parse_canonical(stored.data)


def _authority_failure(document: Mapping[str, Any]) -> MainlineBlocker | None:
    if contains_forbidden_authority(document):
        return MainlineBlocker.SHADOW_AUTHORITY_FORBIDDEN
    return None


def _build_public_run(
    *,
    strategy_id: str,
    pointer: Mapping[str, Any],
    pointer_bytes: StoredBytes,
    run: Mapping[str, Any],
    run_bytes: StoredBytes,
    portfolio: Mapping[str, Any],
) -> dict[str, Any]:
    return seal_document(
        {
            "schema_id": PUBLIC_RUN_SCHEMA_ID,
            "protocol": PROTOCOL,
            "canonical_strategy_id": strategy_id,
            "run_id": run["run_id"],
            "state": ACTIVE_STATE,
            "market": SUPPORTED_MARKET,
            "capability": SUPPORTED_CAPABILITY,
            "authority_source": "FORMAL_V17_V4",
            "authority_flags": dict(AUTHORITY_FLAGS),
            "read_only": True,
            "selector_used": False,
            "fallback_used": False,
            "active_pointer_ref": build_ref(
                schema_id=ACTIVE_POINTER_SCHEMA_ID,
                relative_path=pointer_bytes.relative_path,
                raw=pointer_bytes.data,
            ),
            "mainline_run_ref": build_ref(
                schema_id=MAINLINE_RUN_SCHEMA_ID,
                relative_path=run_bytes.relative_path,
                raw=run_bytes.data,
            ),
            "formal_output_ref": dict(run["formal_output_ref"]),
            "portfolio_output_ref": dict(run["portfolio_output_ref"]),
            "source_closure_ref": dict(run["source_closure_ref"]),
            "cash_weight": portfolio["cash_weight"],
            "gross_weight": portfolio["gross_weight"],
            "targets": [dict(row) for row in portfolio["targets"]],
        }
    )


def derive_mainline_state(
    workspace_root: str | Path,
    strategy_id: str | None = None,
    *,
    canonical_strategy_id: str | None = None,
    expected_pointer_sha256: str | None = None,
) -> MainlineResolution:
    """Resolve exactly one fixed pointer; never scan, select, or fall back."""

    strategy = _strategy_argument(strategy_id, canonical_strategy_id)
    if expected_pointer_sha256 is not None:
        try:
            require_sha256(expected_pointer_sha256, label="expected_pointer_sha256")
        except MainlineContractError as exc:
            raise V17MainlineError("V17_MAINLINE_ARGUMENTS_INVALID", str(exc)) from exc
    try:
        store = MainlineStore(workspace_root)
        pointer_path = active_pointer_path(strategy)
        try:
            pointer_bytes = store.read(pointer_path, expected_pointer_sha256)
        except MainlineNotFound:
            return MainlineResolution(
                UNINITIALIZED_STATE,
                MainlineBlocker.ACTIVE_POINTER_ABSENT,
                None,
            )
        except MainlineCASMismatch:
            return _blocked(MainlineBlocker.ACTIVE_POINTER_SHA_MISMATCH)

        try:
            pointer_document = parse_canonical(pointer_bytes.data)
            pointer = validate_active_pointer(pointer_document)
        except MainlineContractError:
            return _blocked(MainlineBlocker.ACTIVE_POINTER_INVALID)
        if pointer["canonical_strategy_id"] != strategy:
            return _blocked(MainlineBlocker.ACTIVE_POINTER_INVALID)

        try:
            run_bytes, run_document = _read_document(store, pointer["run_ref"])
        except MainlineNotFound:
            return _blocked(MainlineBlocker.ACTIVE_RUN_MISSING)
        except MainlineCASMismatch:
            return _blocked(MainlineBlocker.ACTIVE_RUN_SHA_MISMATCH)
        except MainlineContractError:
            return _blocked(MainlineBlocker.ACTIVE_RUN_INVALID)

        forbidden = _authority_failure(run_document)
        if forbidden is not None:
            return _blocked(forbidden)
        if run_document.get("market") != SUPPORTED_MARKET:
            return _blocked(MainlineBlocker.UNSUPPORTED_MARKET)
        if run_document.get("capabilities") != [SUPPORTED_CAPABILITY]:
            return _blocked(MainlineBlocker.UNSUPPORTED_CAPABILITY)
        try:
            run = validate_mainline_run(run_document)
        except MainlineContractError:
            return _blocked(MainlineBlocker.ACTIVE_RUN_INVALID)
        if run["canonical_strategy_id"] != strategy or run["run_id"] != pointer["run_id"]:
            return _blocked(MainlineBlocker.ACTIVE_RUN_INVALID)

        try:
            _, formal_document = _read_document(store, run["formal_output_ref"])
            if _authority_failure(formal_document) is not None:
                return _blocked(MainlineBlocker.SHADOW_AUTHORITY_FORBIDDEN)
            formal = validate_formal_output(formal_document, strategy_id=strategy)
        except MainlineStorageSecurityError:
            raise
        except (MainlineStorageError, MainlineContractError):
            return _blocked(MainlineBlocker.FORMAL_OUTPUT_INVALID)

        try:
            _, portfolio_document = _read_document(store, run["portfolio_output_ref"])
            if _authority_failure(portfolio_document) is not None:
                return _blocked(MainlineBlocker.SHADOW_AUTHORITY_FORBIDDEN)
            portfolio = validate_portfolio_output(
                portfolio_document,
                strategy_id=strategy,
                run_id=run["run_id"],
            )
        except MainlineStorageSecurityError:
            raise
        except (MainlineStorageError, MainlineContractError):
            return _blocked(MainlineBlocker.PORTFOLIO_OUTPUT_INVALID)

        evidence_refs = formal.get("evidence_refs")
        if type(evidence_refs) is not list or not any(
            type(ref) is dict
            and ref.get("relative_path") == run["portfolio_output_ref"]["relative_path"]
            and ref.get("byte_sha256") == run["portfolio_output_ref"]["byte_sha256"]
            for ref in evidence_refs
        ):
            return _blocked(MainlineBlocker.FORMAL_OUTPUT_INVALID)

        try:
            _, source_document = _read_document(store, run["source_closure_ref"])
            if _authority_failure(source_document) is not None:
                return _blocked(MainlineBlocker.SHADOW_AUTHORITY_FORBIDDEN)
            validate_source_closure(source_document, strategy_id=strategy)
        except MainlineStorageSecurityError:
            raise
        except (MainlineStorageError, MainlineContractError):
            return _blocked(MainlineBlocker.SOURCE_CLOSURE_INVALID)

        public_run = _build_public_run(
            strategy_id=strategy,
            pointer=pointer,
            pointer_bytes=pointer_bytes,
            run=run,
            run_bytes=run_bytes,
            portfolio=portfolio,
        )
        return MainlineResolution(ACTIVE_STATE, None, public_run)
    except MainlineStorageSecurityError:
        return _blocked(MainlineBlocker.STORAGE_SECURITY_VIOLATION)
    except (OSError, MainlineStorageError):
        return _blocked(MainlineBlocker.STORAGE_SECURITY_VIOLATION)


def read_public_run(
    workspace_root: str | Path,
    strategy_id: str | None = None,
    *,
    canonical_strategy_id: str | None = None,
    expected_pointer_sha256: str | None = None,
) -> dict[str, Any]:
    resolution = derive_mainline_state(
        workspace_root,
        strategy_id,
        canonical_strategy_id=canonical_strategy_id,
        expected_pointer_sha256=expected_pointer_sha256,
    )
    if not resolution.is_active:
        assert resolution.blocker is not None
        raise MainlineUnavailable(resolution.derived_state, resolution.blocker)
    assert resolution.public_run is not None
    return resolution.public_run


__all__ = [
    "MainlineResolution",
    "MainlineUnavailable",
    "V17MainlineError",
    "active_pointer_path",
    "derive_mainline_state",
    "mainline_run_path",
    "read_public_run",
]
