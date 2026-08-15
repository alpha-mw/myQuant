"""Fail-closed public errors for stable Mainline reads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

MAINLINE_UNINITIALIZED: Final = "MAINLINE_UNINITIALIZED"
MAINLINE_BLOCKED: Final = "MAINLINE_BLOCKED"
MAINLINE_ARGUMENTS_INVALID: Final = "MAINLINE_ARGUMENTS_INVALID"
BACKTEST_UNAVAILABLE: Final = "BACKTEST_UNAVAILABLE"


class MainlineError(RuntimeError):
    """A path-free Mainline error suitable for the CLI and Python surface."""

    exit_code = 2

    def __init__(
        self,
        code: str,
        *,
        blockers: Sequence[str] = (),
        public_state: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.blockers = tuple(blockers)
        state = (
            {
                "active_generation_id": None,
                "blockers": list(self.blockers),
                "investment_state": "BLOCKED",
                "mainline_state": "BLOCKED",
                "result": None,
                "status": "BLOCKED",
            }
            if public_state is None
            else dict(public_state)
        )
        if set(state) != {
            "active_generation_id",
            "blockers",
            "investment_state",
            "mainline_state",
            "result",
            "status",
        }:
            raise ValueError("Mainline public error state fields are not exact")
        if state["status"] != "BLOCKED" or state["result"] is not None:
            raise ValueError("Mainline public error state is inconsistent")
        if state["blockers"] != list(self.blockers):
            raise ValueError("Mainline public error blockers are inconsistent")
        self.public_fields = state
        super().__init__(code)


__all__ = [
    "BACKTEST_UNAVAILABLE",
    "MAINLINE_ARGUMENTS_INVALID",
    "MAINLINE_BLOCKED",
    "MAINLINE_UNINITIALIZED",
    "MainlineError",
]
