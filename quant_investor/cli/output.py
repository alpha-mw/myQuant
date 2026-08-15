"""Canonical machine-output and fail-closed CLI error boundaries.

The public command line is an automation surface.  Every non-help response is
therefore one compact, key-sorted JSON object on stdout.  Expected
unavailability and validation failures use exit code 2; unexpected exceptions
use exit code 3 without disclosing local paths or tracebacks.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
import sys
from typing import Any, NoReturn, TypeVar

_T = TypeVar("_T")


class CommandError(RuntimeError):
    """A safe, expected public-command failure."""

    def __init__(
        self,
        blocker_code: str,
        *,
        status: str = "BLOCKED",
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(blocker_code)
        self.blocker_code = str(blocker_code)
        self.status = str(status)
        self.fields = dict(fields or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "blocker_code": self.blocker_code,
            **self.fields,
        }


class MachineArgumentParser(argparse.ArgumentParser):
    """Argument parser with the canonical machine error contract."""

    def error(self, message: str) -> NoReturn:
        del message
        fail_expected(CommandError("ARGUMENTS_INVALID"))


def canonical_json_line(payload: Mapping[str, Any]) -> str:
    """Return the only machine JSON representation emitted by the CLI."""

    if type(payload) is not dict:
        raise TypeError("CLI payload must be a JSON object")
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def emit_json(payload: Mapping[str, Any]) -> None:
    """Write exactly one canonical JSON line to stdout."""

    sys.stdout.write(canonical_json_line(payload) + "\n")


def fail_expected(error: CommandError) -> NoReturn:
    """Emit a validation/precondition response and terminate with code 2."""

    emit_json(error.to_dict())
    raise SystemExit(2) from None


def fail_internal(blocker_code: str = "INTERNAL_ERROR") -> NoReturn:
    """Emit a non-disclosing internal error and terminate with code 3."""

    emit_json({"status": "ERROR", "blocker_code": blocker_code})
    sys.stderr.write("quant-investor encountered an internal error\n")
    raise SystemExit(3) from None


def command_boundary(action: Callable[[], _T]) -> _T:
    """Run one command under the stable 0/2/3 exit-code contract."""

    try:
        return action()
    except CommandError as exc:
        fail_expected(exc)
    except SystemExit:
        raise
    except Exception as exc:
        if getattr(exc, "exit_code", None) == 2:
            code = getattr(exc, "code", None)
            fields = getattr(exc, "public_fields", None)
            if type(code) is str and code:
                fail_expected(
                    CommandError(
                        code,
                        fields=fields if isinstance(fields, Mapping) else None,
                    )
                )
        fail_internal()


__all__ = [
    "CommandError",
    "MachineArgumentParser",
    "canonical_json_line",
    "command_boundary",
    "emit_json",
    "fail_expected",
    "fail_internal",
]
