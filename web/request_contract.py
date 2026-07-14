"""Fail-closed validation shared by Web payload boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def reject_intelligence_named_keys(value: Any, *, path: str = "request") -> None:
    """Reject any nested mapping key containing the retired branch name."""

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        reject_intelligence_named_keys(model_dump(), path=path)
        return

    if isinstance(value, Mapping):
        for raw_key, nested_value in value.items():
            key = str(raw_key)
            nested_path = f"{path}.{key}"
            if "intelligence" in key.casefold():
                raise ValueError(f"retired Intelligence key is not supported: {nested_path}")
            reject_intelligence_named_keys(nested_value, path=nested_path)
        return

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, nested_value in enumerate(value):
            reject_intelligence_named_keys(
                nested_value,
                path=f"{path}[{index}]",
            )
