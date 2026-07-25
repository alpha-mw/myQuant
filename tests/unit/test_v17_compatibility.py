from __future__ import annotations

from pathlib import Path

import pytest

from quant_investor.v17.compatibility import (
    ADVANCE_RUN,
    CREATE_RUN,
    FINALIZE_RUN,
    READ_ARTIFACT,
    READ_STATUS,
    RECEIVE_RESPONSE,
    REPAIR_LATEST,
    V1_ACTIONS,
    V1_LEDGER_VERSION,
    V1_MUTATING_ACTIONS,
    V1_NONTERMINAL_STATES,
    V1_READ_ONLY_ACTIONS,
    V1_TERMINAL_STATES,
    V17CompatibilityError,
    decide_v1_compatibility,
    require_v1_compatibility,
)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_v1_compatibility_matrix_constants_are_frozen() -> None:
    assert V1_LEDGER_VERSION == "myquant.v17.shadow-ledger.v1"
    assert V1_TERMINAL_STATES == (
        "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION",
        "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        "SHADOW_PORTFOLIO_INFEASIBLE",
        "HARD_STOP_SNAPSHOT_DRIFT",
        "HARD_STOP_INVALID_EVIDENCE",
    )
    assert V1_NONTERMINAL_STATES == (
        "PREPARED",
        "DETERMINISTIC_COMPLETE",
        "DEEP_REQUEST_READY",
        "DEEP_RESPONSE_RECEIVED",
        "PORTFOLIO_COMPLETE",
    )
    assert V1_READ_ONLY_ACTIONS == (READ_STATUS, READ_ARTIFACT)
    assert V1_MUTATING_ACTIONS == (
        CREATE_RUN,
        ADVANCE_RUN,
        RECEIVE_RESPONSE,
        FINALIZE_RUN,
        REPAIR_LATEST,
    )
    assert V1_ACTIONS == V1_READ_ONLY_ACTIONS + V1_MUTATING_ACTIONS


@pytest.mark.parametrize("state", V1_TERMINAL_STATES)
@pytest.mark.parametrize("action", V1_READ_ONLY_ACTIONS)
def test_v1_terminal_runs_are_read_only(state: str, action: str) -> None:
    decision = require_v1_compatibility(action=action, state=state)
    assert decision.allowed is True
    assert decision.read_only is True
    assert decision.reason == "v1_terminal_read_only"
    assert decision.exit_code == 0


@pytest.mark.parametrize("state", V1_NONTERMINAL_STATES)
@pytest.mark.parametrize("action", V1_ACTIONS)
def test_every_v1_nonterminal_operation_is_retired(state: str, action: str) -> None:
    decision = decide_v1_compatibility(action=action, state=state)
    assert decision.allowed is False
    assert decision.read_only is False
    assert decision.exit_code == 2
    assert decision.reason in {"v1_nonterminal_retired", "v1_new_run_retired"}
    with pytest.raises(V17CompatibilityError) as captured:
        require_v1_compatibility(action=action, state=state)
    assert captured.value.exit_code == 2


@pytest.mark.parametrize("state", V1_TERMINAL_STATES)
@pytest.mark.parametrize("action", V1_MUTATING_ACTIONS)
def test_v1_terminal_mutations_and_latest_repair_are_retired(
    state: str,
    action: str,
) -> None:
    with pytest.raises(V17CompatibilityError) as captured:
        require_v1_compatibility(action=action, state=state)
    assert captured.value.exit_code == 2
    assert captured.value.decision.read_only is False


def test_v1_new_run_and_all_rejections_are_zero_write(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    before = _tree_bytes(tmp_path)

    with pytest.raises(V17CompatibilityError) as captured:
        require_v1_compatibility(action=CREATE_RUN, state=None)

    assert captured.value.exit_code == 2
    assert captured.value.decision.reason == "v1_new_run_retired"
    assert _tree_bytes(tmp_path) == before


@pytest.mark.parametrize(
    ("action", "state"),
    [
        (READ_STATUS, V1_NONTERMINAL_STATES[0]),
        (ADVANCE_RUN, V1_NONTERMINAL_STATES[-1]),
        (REPAIR_LATEST, V1_TERMINAL_STATES[0]),
    ],
)
def test_v1_nonterminal_and_repair_rejections_are_zero_write(
    tmp_path: Path,
    action: str,
    state: str,
) -> None:
    sentinel = tmp_path / "sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    before = _tree_bytes(tmp_path)

    with pytest.raises(V17CompatibilityError) as captured:
        require_v1_compatibility(action=action, state=state)

    assert captured.value.exit_code == 2
    assert _tree_bytes(tmp_path) == before


@pytest.mark.parametrize(
    ("action", "state", "version", "reason"),
    [
        ("UNKNOWN", V1_TERMINAL_STATES[0], V1_LEDGER_VERSION, "unknown_v1_compatibility_action"),
        (READ_STATUS, "UNKNOWN", V1_LEDGER_VERSION, "unknown_v1_state"),
        (READ_STATUS, None, V1_LEDGER_VERSION, "v1_state_required"),
        (
            READ_STATUS,
            V1_TERMINAL_STATES[0],
            "myquant.v17.shadow-ledger.v2",
            "unsupported_v1_ledger_version",
        ),
    ],
)
def test_unknown_v1_dispatch_inputs_fail_closed(
    action: str,
    state: str | None,
    version: str,
    reason: str,
) -> None:
    with pytest.raises(V17CompatibilityError) as captured:
        decide_v1_compatibility(action=action, state=state, ledger_version=version)
    assert captured.value.exit_code == 2
    assert captured.value.decision.reason == reason
