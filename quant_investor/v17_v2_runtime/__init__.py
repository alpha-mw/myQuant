"""Isolated, offline runtime infrastructure for myQuant protocol v2.

This package never imports the retired :mod:`quant_investor.v17` runtime and
does not wire itself into production, CLI, service, pipeline, or providers.
"""

from __future__ import annotations

from .gate import GateDecision, RuntimeGate, RuntimeGateError
from .ledger import LedgerCommit, LedgerStore, LedgerStoreError
from .storage import (
    CASMismatchError,
    EMPTY_SHA,
    ExactOnceConflictError,
    LockUnavailableError,
    SecureStore,
    StorageCommitError,
    StorageError,
    StorageNotFoundError,
    StorageSecurityError,
    StoredBytes,
    WriteResult,
)
from .terminal import (
    TerminalPublication,
    TerminalPublicationError,
    TerminalPublisher,
)

PROTOCOL_VERSION = "myquant.v17.v2"
RUNTIME_AUTHORITY = False

__all__ = [
    "CASMismatchError",
    "EMPTY_SHA",
    "ExactOnceConflictError",
    "GateDecision",
    "LedgerCommit",
    "LedgerStore",
    "LedgerStoreError",
    "LockUnavailableError",
    "PROTOCOL_VERSION",
    "RUNTIME_AUTHORITY",
    "RuntimeGate",
    "RuntimeGateError",
    "SecureStore",
    "StorageCommitError",
    "StorageError",
    "StorageNotFoundError",
    "StorageSecurityError",
    "StoredBytes",
    "TerminalPublication",
    "TerminalPublicationError",
    "TerminalPublisher",
    "WriteResult",
]
