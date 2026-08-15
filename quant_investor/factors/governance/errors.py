"""Fail-closed errors for stable Factor governance."""

from __future__ import annotations

import re
from typing import Final

_ERROR_CODE_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")


class FactorGovernanceError(ValueError):
    """Expected validation failure safe for the stable CLI boundary."""

    default_code = "FACTOR_VALIDATION_FAILED"
    exit_code = 2

    def __init__(self, detail: str, *, code: str | None = None) -> None:
        selected = code or self.default_code
        if type(selected) is not str or _ERROR_CODE_RE.fullmatch(selected) is None:
            selected = self.default_code
        self.code = selected
        self.public_fields: dict[str, str] = {}
        super().__init__(f"{self.code}:{detail}")


__all__ = ["FactorGovernanceError"]
