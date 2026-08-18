"""Stable byte-size policy for Fundamental JSON and evidence reads."""

from __future__ import annotations

from typing import Final

FUNDAMENTAL_GENERIC_JSON_MAX_BYTES: Final[int] = 64 * 1024 * 1024
FUNDAMENTAL_PREDECESSOR_MANIFEST_MAX_BYTES: Final[int] = 128 * 1024 * 1024
FUNDAMENTAL_PREDECESSOR_MANIFEST_ROLE: Final[str] = "predecessor_manifest"
FUNDAMENTAL_PARQUET_MAX_BYTES: Final[int] = 512 * 1024 * 1024
FUNDAMENTAL_PARQUET_TABLE_ROLES: Final[frozenset[str]] = frozenset(
    {"fundamental_daily", "fundamental_period", "fundamental_quarantine"}
)
FUNDAMENTAL_GENERIC_REPLAY_MAX_CELLS: Final[int] = 100_000_000
FUNDAMENTAL_DAILY_REPLAY_MAX_CELLS: Final[int] = 256_000_000


class FundamentalSizePolicyViolation(ValueError):
    """One file does not satisfy its exact Fundamental semantic-role bound."""


def validate_fundamental_parquet_size_policy(*, table_role: str, observed_bytes: int) -> None:
    """Enforce the compressed-byte ceiling only for canonical table roles."""

    if table_role not in FUNDAMENTAL_PARQUET_TABLE_ROLES:
        raise ValueError("Fundamental Parquet table role is invalid")
    if (
        type(observed_bytes) is not int
        or observed_bytes <= 0
        or observed_bytes > FUNDAMENTAL_PARQUET_MAX_BYTES
    ):
        raise FundamentalSizePolicyViolation(
            f"{table_role} bytes exceed the Fundamental Parquet bound"
        )


def fundamental_parquet_max_cells(table_role: str) -> int:
    """Return the table-cardinality ceiling for one authenticated table role."""

    if table_role not in FUNDAMENTAL_PARQUET_TABLE_ROLES:
        raise ValueError("Fundamental Parquet table role is invalid")
    if table_role == "fundamental_daily":
        return FUNDAMENTAL_DAILY_REPLAY_MAX_CELLS
    return FUNDAMENTAL_GENERIC_REPLAY_MAX_CELLS


def fundamental_json_maximum_bytes(semantic_role: str) -> int:
    """Return the byte bound for one declared Fundamental semantic role."""

    if type(semantic_role) is not str or not semantic_role:
        raise ValueError("Fundamental semantic role is invalid")
    if semantic_role == FUNDAMENTAL_PREDECESSOR_MANIFEST_ROLE:
        return FUNDAMENTAL_PREDECESSOR_MANIFEST_MAX_BYTES
    return FUNDAMENTAL_GENERIC_JSON_MAX_BYTES


def validate_fundamental_json_size_policy(
    *,
    semantic_role: str,
    maximum_bytes: int,
    observed_bytes: int | None = None,
) -> None:
    """Require the caller's bound and size to match the stable policy."""

    expected = fundamental_json_maximum_bytes(semantic_role)
    if type(maximum_bytes) is not int or maximum_bytes != expected:
        raise ValueError("Fundamental JSON byte bound differs from semantic-role policy")
    if observed_bytes is not None and (
        type(observed_bytes) is not int or observed_bytes <= 0 or observed_bytes > maximum_bytes
    ):
        raise FundamentalSizePolicyViolation(
            f"{semantic_role} bytes exceed the permitted Fundamental bound"
        )


__all__ = [
    "FUNDAMENTAL_GENERIC_JSON_MAX_BYTES",
    "FUNDAMENTAL_PREDECESSOR_MANIFEST_MAX_BYTES",
    "FUNDAMENTAL_PREDECESSOR_MANIFEST_ROLE",
    "FUNDAMENTAL_PARQUET_MAX_BYTES",
    "FUNDAMENTAL_PARQUET_TABLE_ROLES",
    "FUNDAMENTAL_DAILY_REPLAY_MAX_CELLS",
    "FUNDAMENTAL_GENERIC_REPLAY_MAX_CELLS",
    "FundamentalSizePolicyViolation",
    "fundamental_json_maximum_bytes",
    "fundamental_parquet_max_cells",
    "validate_fundamental_json_size_policy",
    "validate_fundamental_parquet_size_policy",
]
