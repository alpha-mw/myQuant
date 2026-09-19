"""Pure observational node metadata from already-validated journal evidence."""

from copy import deepcopy

from .daily_contract import ContractError, FAILURES, NodeState
from .dependency_diagnostics import dependency_details

_ACTIONS = {
    "EXECUTED": "NATIVE_EXECUTED",
    "NO_ACTION": "NATIVE_REPLAY",
    "ADOPTED": "NATIVE_RECOVERY",
}


def project_node_status(
    row: dict, *, recorded_request: dict | None = None, invocation_command: str | None = None
) -> dict:
    """No I/O, native replay, hash, clock or permission decision occurs here.

    The caller owns validation. A constructed candidate request must not be
    supplied as recorded_request. Historical readers never supply an invocation
    command from a stored mutable projection.
    """
    state = NodeState(row["state"])
    if invocation_command is not None and invocation_command not in _ACTIONS:
        raise ContractError("STATUS_PROJECTION_COMMAND_INVALID")
    result = deepcopy(row)
    dependency = dependency_details(row.get("dependency_error"))
    invalid = state == NodeState.STALE
    start = None if invalid else row.get("start")
    terminal = None if invalid else row.get("terminal")
    detail = row.get("failure") or (terminal or {}).get("failure")
    code = (
        ("VALIDATION_FAILED" if dependency is None else dependency["failure_code"])
        if invalid
        else None if detail is None else detail["code"]
    )
    if code is not None and code not in FAILURES:
        raise ContractError("STATUS_PROJECTION_FAILURE_INVALID")

    attempt = row.get("attempt")
    exact_absence = (
        not invalid
        and type(attempt) is int
        and attempt == 0
        and type(row.get("request_key")) is str
        and start is None
        and terminal is None
    )
    if invalid or not (
        exact_absence or (type(attempt) is int and attempt > 0 and start is not None)
    ):
        attempt = None

    if invalid:
        trigger = "EVIDENCE_INVALID"
    elif invocation_command is not None:
        trigger = _ACTIONS[invocation_command]
    elif state == NodeState.SKIPPED and code == "UPSTREAM_INCOMPLETE":
        trigger = "WAITING_UPSTREAM"
    elif code is not None:
        trigger = "BLOCKING_FAILURE"
    elif start is not None and terminal is None:
        trigger = "ATTEMPT_RUNNING"
    elif terminal is not None:
        trigger = "RECORDED_TERMINAL"
    elif exact_absence:
        trigger = "NOT_STARTED"
    else:
        trigger = None

    upstream = {}
    if not invalid and recorded_request is not None:
        upstream = {
            name: deepcopy(ref)
            for name, ref in recorded_request["input_refs"].items()
            if name.startswith("upstream.")
        }
    result.update(
        trigger_reason=trigger,
        blocking_reason=code,
        reason_code=None if dependency is None else dependency["reason_code"],
        retryable=False if code is None else FAILURES[code].retryable,
        upstream_refs=upstream,
        output_refs=(
            {} if terminal is None or dependency is not None else deepcopy(terminal["output_refs"])
        ),
        started_at=None if start is None else start["started_at"],
        finished_at=None if terminal is None else terminal["finished_at"],
        attempt=attempt,
    )
    return result
