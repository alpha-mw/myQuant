"""Cycle-free canonical paths and sentinels owned by the unified System."""

from pathlib import PurePosixPath
from typing import Final

SYSTEM_ROOT: Final = PurePosixPath("results/system")
ACTIVE_POINTER_PATH: Final = SYSTEM_ROOT / "_active.json"
MIGRATION_MARKER_PATH: Final = SYSTEM_ROOT / "_migration_complete.json"
EMPTY_POINTER_SHA256: Final = "EMPTY"

__all__ = [
    "ACTIVE_POINTER_PATH",
    "EMPTY_POINTER_SHA256",
    "MIGRATION_MARKER_PATH",
    "SYSTEM_ROOT",
]
