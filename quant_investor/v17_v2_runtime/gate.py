"""Read-only admission gate for the isolated protocol-v2 runtime.

The gate intentionally performs no filesystem writes.  Runtime storage is
allowed only below the two frozen protocol-v2 roots.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Final

from quant_investor.v17_v2_contract.action_matrix import (
    ActionMatrixError,
    ActionOutcome,
    decide_action,
)
from quant_investor.v17_v2_contract.identities import (
    IdentityContractError,
    require_path_id,
)

PROTOCOL_VERSION: Final = "myquant.v17.v2"
RESULTS_ROOT: Final = PurePosixPath("results/v17_shadow/protocol-v2")
SOURCES_ROOT: Final = PurePosixPath("data/private/v17_sources/protocol-v2")

_ACTIONS: Final = frozenset(
    {
        "SOURCE_MAINTAIN",
        "RISK_POLICY_SEAL",
        "SHADOW_PREPARE",
        "SHADOW_RECEIVE",
        "SHADOW_FINALIZE",
        "READ_STATUS",
        "READ_ARTIFACT",
        "REPAIR_LATEST",
    }
)


class RuntimeGateError(ValueError):
    """Raised when a gate request itself is not canonical."""

    exit_code = 2


@dataclass(frozen=True)
class GateDecision:
    """Read-only classification result suitable for service admission."""

    action: str
    run_id: str
    allowed: bool
    reason_code: str
    detail: str
    checked_roots: tuple[str, str]
    matrix_rule: str
    read_only: bool
    allowed_write_namespaces: tuple[str, ...]
    retry_cas: str
    outcomes: tuple[ActionOutcome, ...]


def _workspace_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeGateError("workspace_root must be absolute")
    return path


def _legacy_runtime_loaded() -> bool:
    return any(
        name == "quant_investor.v17" or name.startswith("quant_investor.v17.")
        for name in sys.modules
    )


def _classify_root(workspace_root: Path, relative_root: PurePosixPath) -> str | None:
    current = workspace_root
    for index, part in enumerate(relative_root.parts):
        current = current / part
        try:
            observed = current.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            return f"lstat_failed:{relative_root}:{exc.errno}"
        if stat.S_ISLNK(observed.st_mode):
            return f"symlink_component:{current}"
        if not stat.S_ISDIR(observed.st_mode):
            return f"non_directory_component:{current}"
        if index == len(relative_root.parts) - 1:
            mode = stat.S_IMODE(observed.st_mode)
            if mode != 0o700:
                return f"unsafe_root_mode:{current}:{mode:04o}"
    return None


def _classify_governed_target(
    workspace_root: Path,
    target: PurePosixPath,
) -> str | None:
    current = workspace_root
    governed = False
    for part in target.parts:
        current = current / part
        relative = PurePosixPath(*current.relative_to(workspace_root).parts)
        governed = governed or relative in {RESULTS_ROOT, SOURCES_ROOT}
        try:
            observed = current.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            return f"lstat_failed:{relative}:{exc.errno}"
        if stat.S_ISLNK(observed.st_mode):
            return f"symlink_component:{current}"
        if not stat.S_ISDIR(observed.st_mode):
            return f"non_directory_component:{current}"
        if governed and stat.S_IMODE(observed.st_mode) != 0o700:
            return f"unsafe_directory_mode:{current}:{stat.S_IMODE(observed.st_mode):04o}"
    return None


@dataclass(frozen=True)
class RuntimeGate:
    """Zero-write classifier for protocol-v2 runtime actions."""

    workspace_root: Path

    def __init__(self, workspace_root: str | os.PathLike[str]) -> None:
        object.__setattr__(self, "workspace_root", _workspace_path(workspace_root))

    def classify(
        self,
        action: str,
        run_id: str,
        *,
        version: str = "ABSENT",
        state: str = "MISSING",
        checkpoint: str = "PRE_IMPORT",
    ) -> GateDecision:
        """Classify one action without creating roots, locks, or receipts."""

        if type(action) is not str or action not in _ACTIONS:
            raise RuntimeGateError("action is not registered for protocol v2")
        try:
            canonical_run_id = require_path_id(run_id, label="run_id")
        except IdentityContractError as exc:
            raise RuntimeGateError(str(exc)) from exc

        try:
            matrix = decide_action(
                version=version,
                action=action,
                state=state,
                checkpoint=checkpoint,
            )
        except ActionMatrixError as exc:
            raise RuntimeGateError(str(exc)) from exc
        roots = (str(RESULTS_ROOT), str(SOURCES_ROOT))
        matrix_fields = (
            matrix.rule_id,
            matrix.read_only,
            matrix.allowed_write_namespaces,
            matrix.retry_cas,
            matrix.outcomes,
        )
        if _legacy_runtime_loaded():
            return GateDecision(
                action,
                canonical_run_id,
                False,
                "legacy_v17_runtime_loaded",
                "legacy quant_investor.v17 is already loaded in this process",
                roots,
                *matrix_fields,
            )
        try:
            workspace_stat = self.workspace_root.lstat()
        except OSError as exc:
            return GateDecision(
                action,
                canonical_run_id,
                False,
                "workspace_unavailable",
                f"workspace root is unavailable: errno={exc.errno}",
                roots,
                *matrix_fields,
            )
        if stat.S_ISLNK(workspace_stat.st_mode) or not stat.S_ISDIR(workspace_stat.st_mode):
            return GateDecision(
                action,
                canonical_run_id,
                False,
                "workspace_not_physical_directory",
                "workspace root must be a physical directory",
                roots,
                *matrix_fields,
            )
        for relative_root in (RESULTS_ROOT, SOURCES_ROOT):
            collision = _classify_root(self.workspace_root, relative_root)
            if collision is not None:
                return GateDecision(
                    action,
                    canonical_run_id,
                    False,
                    "namespace_collision",
                    collision,
                    roots,
                    *matrix_fields,
                )
        governed_targets = (
            RESULTS_ROOT / "runs" / canonical_run_id,
            RESULTS_ROOT / "_latest",
        )
        if action == "SOURCE_MAINTAIN":
            governed_targets = (
                SOURCES_ROOT / "objects",
                SOURCES_ROOT / "manifests",
                SOURCES_ROOT / "locators",
            )
        for target in governed_targets:
            collision = _classify_governed_target(self.workspace_root, target)
            if collision is not None:
                return GateDecision(
                    action,
                    canonical_run_id,
                    False,
                    "namespace_collision",
                    collision,
                    roots,
                    *matrix_fields,
                )
        if not matrix.allowed:
            return GateDecision(
                action,
                canonical_run_id,
                False,
                matrix.reason,
                f"frozen action matrix rejected rule {matrix.rule_id}",
                roots,
                *matrix_fields,
            )
        return GateDecision(
            action,
            canonical_run_id,
            True,
            matrix.reason,
            f"frozen action matrix admitted rule {matrix.rule_id}",
            roots,
            *matrix_fields,
        )


__all__ = [
    "GateDecision",
    "PROTOCOL_VERSION",
    "RESULTS_ROOT",
    "RuntimeGate",
    "RuntimeGateError",
    "SOURCES_ROOT",
]
