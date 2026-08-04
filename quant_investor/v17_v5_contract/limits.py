"""Hard limits for the V17 v5 Phase-0 contract and compatibility reader."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

LIMITS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "general_json_bytes": 8_388_608,
        "max_array_items": 10_000,
        "max_container_members": 512,
        "max_depth": 32,
        "max_integer_digits": 64,
        "max_key_utf8_bytes": 256,
        "max_string_utf8_bytes": 1_048_576,
        "max_total_nodes": 250_000,
        "compat_max_artifact_bytes": 134_217_728,
        "compat_max_closure_bytes": 536_870_912,
        "compat_max_closure_depth": 32,
        "compat_max_closure_nodes": 512,
        "compat_max_parquet_row_groups": 4_096,
        "compat_max_parquet_rows": 10_000_000,
    }
)


class ContractLimitError(ValueError):
    """Raised when a closed V17 v5 input exceeds a declared bound."""

    exit_code = 2


__all__ = ["ContractLimitError", "LIMITS"]
