from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from quant_investor.v17.latest import (
    V17LatestError,
    publish_terminal_latest,
)
from quant_investor.v17.semantic import seal_semantic
from quant_investor.v17.state_machine import (
    EMPTY_SHA,
    TERMINAL_OUTPUT_VERSION,
    V17LedgerCASMismatch,
    V17StateMachineError,
    advance_run_state,
    initialize_run,
    load_run_ledger,
    terminal_states,
)
from quant_investor.v17.storage import read_json

TIMES = (
    "2026-07-22T07:00:00Z",
    "2026-07-22T07:01:00Z",
    "2026-07-22T07:02:00Z",
    "2026-07-22T07:03:00Z",
)


def _artifact(version: str) -> dict[str, object]:
    return seal_semantic(
        {
            "version": version,
            "value": "synthetic",
            "authority": False,
        }
    )


def _initialize(repo: Path, run_id: str = "cn-v17-state-test") -> tuple[dict, str]:
    repo.mkdir()
    return initialize_run(
        repo,
        run_id=run_id,
        strategy_id="cn-shadow",
        cutoff=TIMES[0],
        prepared_at=TIMES[0],
        input_bindings={
            "source_manifest_sha256": "a" * 64,
            "source_manifest_path": "data/private/v17_sources/manifests/test.json",
        },
        expected_ledger_sha256=EMPTY_SHA,
    )


def _tree_snapshot(root: Path) -> tuple[tuple[str, int, str], ...]:
    rows: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        entry = os.lstat(path)
        digest = ""
        if stat.S_ISREG(entry.st_mode):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            (
                path.relative_to(root).as_posix(),
                stat.S_IMODE(entry.st_mode),
                digest,
            )
        )
    return tuple(rows)


def _no_portfolio_output(ledger: dict, predecessor: str) -> dict[str, object]:
    return seal_semantic(
        {
            "version": TERMINAL_OUTPUT_VERSION,
            "run_id": ledger["run_id"],
            "strategy_id": ledger["strategy_id"],
            "market": "CN",
            "cutoff": ledger["cutoff"],
            "terminal_state": "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
            "rank_output": {"ranked_symbols": ["000001.SZ"]},
            "portfolio_output": None,
            "blockers": ["source_unavailable:risk_policy"],
            "source_manifest_sha256": "a" * 64,
            "ledger_predecessor_sha256": predecessor,
            "generated_at": TIMES[3],
            "authority": False,
        }
    )


def test_state_machine_has_exact_five_immutable_terminals() -> None:
    assert terminal_states() == frozenset(
        {
            "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION",
            "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
            "SHADOW_PORTFOLIO_INFEASIBLE",
            "HARD_STOP_SNAPSHOT_DRIFT",
            "HARD_STOP_INVALID_EVIDENCE",
        }
    )


def test_initialize_and_cas_conflict_are_private_and_zero_write(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    ledger, ledger_sha = _initialize(repo)
    assert ledger["state"] == "PREPARED"
    run = repo / "results/v17_shadow/runs/cn-v17-state-test"
    assert stat.S_IMODE(run.stat().st_mode) == 0o700
    assert stat.S_IMODE((run / "ledger.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((run / ".ledger.lock").stat().st_mode) == 0o600

    before = _tree_snapshot(repo)
    with pytest.raises(V17LedgerCASMismatch, match="zero writes"):
        advance_run_state(
            repo,
            run_id="cn-v17-state-test",
            expected_ledger_sha256="f" * 64,
            next_state="DETERMINISTIC_COMPLETE",
            transitioned_at=TIMES[1],
            artifacts={"deterministic_result": _artifact("test.deterministic.v1")},
        )
    assert _tree_snapshot(repo) == before
    assert load_run_ledger(repo, "cn-v17-state-test")[1] == ledger_sha


def test_nonterminal_cannot_publish_latest_and_terminal_is_immutable(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    ledger, ledger_sha = _initialize(repo)
    with pytest.raises(V17LatestError, match="terminal"):
        publish_terminal_latest(
            repo,
            run_id=ledger["run_id"],
            expected_ledger_sha256=ledger_sha,
            expected_latest_sha256=EMPTY_SHA,
            published_at=TIMES[1],
        )
    assert not (repo / "results/v17_shadow/_latest/shadow.json").exists()

    for state, role, timestamp in (
        ("DETERMINISTIC_COMPLETE", "deterministic_result", TIMES[1]),
        ("DEEP_REQUEST_READY", "deep_request", TIMES[2]),
        ("DEEP_RESPONSE_RECEIVED", "deep_response", TIMES[3]),
    ):
        ledger, ledger_sha = advance_run_state(
            repo,
            run_id=ledger["run_id"],
            expected_ledger_sha256=ledger_sha,
            next_state=state,
            transitioned_at=timestamp,
            artifacts={role: _artifact(f"test.{role}.v1")},
        )
    output = _no_portfolio_output(ledger, ledger_sha)
    ledger, terminal_sha = advance_run_state(
        repo,
        run_id=ledger["run_id"],
        expected_ledger_sha256=ledger_sha,
        next_state="SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        transitioned_at=TIMES[3],
        terminal_output=output,
    )
    output_binding = dict(ledger["artifacts"]["terminal_output"])
    output_bytes = (repo / output_binding["relative_path"]).read_bytes()

    with pytest.raises(V17StateMachineError, match="not permitted"):
        advance_run_state(
            repo,
            run_id=ledger["run_id"],
            expected_ledger_sha256=terminal_sha,
            next_state="HARD_STOP_INVALID_EVIDENCE",
            transitioned_at=TIMES[3],
            terminal_output=output,
        )
    readback, readback_sha = load_run_ledger(repo, ledger["run_id"])
    assert readback_sha == terminal_sha
    assert readback["state"] == "SHADOW_RANK_COMPLETE_NO_PORTFOLIO"
    assert (repo / output_binding["relative_path"]).read_bytes() == output_bytes
    receipts = list((repo / "results/v17_shadow/runs/cn-v17-state-test/receipts").glob("*.json"))
    assert len(receipts) == 1
    receipt = read_json(receipts[0])
    assert receipt["status"] == "UNPUBLISHED"
    assert receipt["ledger_commit_status"] == "NOT_COMMITTED"


def test_invalid_transition_writes_only_unpublished_receipt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    ledger, ledger_sha = _initialize(repo)
    run = repo / "results/v17_shadow/runs/cn-v17-state-test"
    ledger_before = (run / "ledger.json").read_bytes()
    with pytest.raises(V17StateMachineError, match="not permitted"):
        advance_run_state(
            repo,
            run_id=ledger["run_id"],
            expected_ledger_sha256=ledger_sha,
            next_state="PORTFOLIO_COMPLETE",
            transitioned_at=TIMES[1],
            artifacts={"portfolio": _artifact("test.portfolio.v1")},
        )
    assert (run / "ledger.json").read_bytes() == ledger_before
    assert not any((run / "events").iterdir())
    assert not any((repo / "results/v17_shadow/outcomes").iterdir())
    receipts = list((run / "receipts").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "UNPUBLISHED"
