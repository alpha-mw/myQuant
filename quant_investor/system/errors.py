"""Stable fail-closed errors for unified system storage and activation."""

from __future__ import annotations

from typing import Final


class SystemError(RuntimeError):
    """Base error safe for CLI consumption without internal paths."""

    default_code: str = "SYSTEM_ERROR"
    exit_code: int = 3

    def __init__(self, detail: str, *, code: str | None = None) -> None:
        self.code = code or self.default_code
        self.public_fields: dict[str, str] = {}
        super().__init__(f"{self.code}:{detail}")


class SystemContractError(SystemError):
    default_code = "SYSTEM_CONTRACT_INVALID"
    exit_code = 2


class SystemPreconditionError(SystemError):
    default_code = "SYSTEM_PRECONDITION_FAILED"
    exit_code = 2


class SystemStorageError(SystemError):
    default_code = "SYSTEM_STORAGE_ERROR"


class SystemSecurityError(SystemStorageError):
    default_code = "SYSTEM_STORAGE_SECURITY"
    exit_code = 2


class SystemNotFound(SystemStorageError):
    default_code = "SYSTEM_NOT_FOUND"
    exit_code = 2


class SystemImmutableConflict(SystemStorageError):
    default_code = "SYSTEM_IMMUTABLE_CONFLICT"
    exit_code = 2


class SystemCASMismatch(SystemStorageError):
    default_code = "SYSTEM_CAS_MISMATCH"
    exit_code = 2

    def __init__(self, expected: str, observed: str) -> None:
        self.expected_pointer_sha256 = expected
        self.observed_pointer_sha256 = observed
        super().__init__("pointer compare-and-swap mismatch")
        self.public_fields = {
            "expected_pointer_sha256": expected,
            "observed_pointer_sha256": observed,
        }


class SystemActivationAuthorizationError(SystemPreconditionError):
    default_code = "SYSTEM_ACTIVATION_AUTHORIZATION_INVALID"


class SystemMigrationMarkerAbsent(SystemPreconditionError):
    default_code = "SYSTEM_MIGRATION_MARKER_ABSENT"


class SystemMigrationClosureError(SystemPreconditionError):
    default_code = "SYSTEM_MIGRATION_CLOSURE_INVALID"


class SystemActivationIncomplete(SystemStorageError):
    default_code = "SYSTEM_ACTIVATION_INCOMPLETE"

    def __init__(self, pointer_sha256: str) -> None:
        super().__init__("authorized pointer exists but marker publication is incomplete")
        self.public_fields = {"pointer_sha256": pointer_sha256}


SYSTEM_ACTIVE_POINTER_ABSENT: Final = "SYSTEM_ACTIVE_POINTER_ABSENT"


__all__ = [
    "SYSTEM_ACTIVE_POINTER_ABSENT",
    "SystemCASMismatch",
    "SystemActivationAuthorizationError",
    "SystemActivationIncomplete",
    "SystemContractError",
    "SystemError",
    "SystemImmutableConflict",
    "SystemMigrationClosureError",
    "SystemMigrationMarkerAbsent",
    "SystemNotFound",
    "SystemPreconditionError",
    "SystemSecurityError",
    "SystemStorageError",
]
