from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest

import scripts.build_factor_v4_1_cycle_source as runner
from quant_investor.factors.governance_source_readback_v4_1 import (
    BLOCKER_REPORT_FILENAME,
    BoundCutoffInputsV4_1,
    INPUT_BINDING_SCHEMA_VERSION,
    PRECOMMITTED_STATE_FILENAME,
)


def _args(tmp_path: Path, *, run_id: str) -> object:
    absolute = str(tmp_path / "unused")
    sha = "1" * 64
    return runner.parse_args(
        [
            "--latest-pointer-path",
            absolute,
            "--expected-latest-pointer-sha256",
            sha,
            "--snapshot-manifest-path",
            absolute,
            "--expected-snapshot-manifest-sha256",
            sha,
            "--components-path",
            absolute,
            "--expected-components-sha256",
            sha,
            "--expected-full-a-semantic-sha256",
            sha,
            "--pit-generation-manifest-path",
            absolute,
            "--expected-pit-generation-manifest-sha256",
            sha,
            "--pit-membership-path",
            absolute,
            "--expected-pit-membership-sha256",
            sha,
            "--table-root",
            absolute,
            "--snapshot-id",
            "20260717T172132Z",
            "--analysis-start",
            "2026-07-17",
            "--cutoff-date",
            "2026-07-17",
            "--private-root",
            str(tmp_path / "private"),
            "--run-id",
            run_id,
            "--cycle-id",
            "cycle-v41-fixture",
            "--expected-state-sha256",
            "empty",
            "--expected-full-a-count",
            "2",
            "--expected-serving-inventory-count",
            "5728",
        ]
    )


def _bound() -> BoundCutoffInputsV4_1:
    symbols = ("000001.SZ", "000002.SZ")
    scope_sha = hashlib.sha256("\n".join(symbols).encode()).hexdigest()
    records = tuple(
        {
            "schema_version": "cn_pit_universe.v1",
            "symbol": symbol,
            "source_list_status": "L",
            "effective_from": "20200101",
            "effective_to": "",
            "list_date": "20200101",
            "delist_date": "",
            "membership_quality": "ok",
        }
        for symbol in symbols
    )
    binding = {
        "schema_version": INPUT_BINDING_SCHEMA_VERSION,
        "snapshot_id": "20260717T172132Z",
        "cutoff_date": "2026-07-17",
        "components": {
            "count": 2,
            "newline_set_sha256": scope_sha,
        },
        "pit_generation": {
            "row_count": 2,
            "historical_alias_table_evidence": [],
        },
    }
    return BoundCutoffInputsV4_1(
        binding=binding,
        calendar_sessions=("2026-07-17",),
        component_symbols=symbols,
        pit_records=records,
        bound_table_symbol_row_counts=(("000001.SZ", 1), ("000002.SZ", 1)),
    )


def test_runner_builds_validated_exploratory_precommit_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, run_id="success-v41")
    monkeypatch.setattr(runner, "_bind", lambda _args: _bound())

    result = runner.run(args)  # type: ignore[arg-type]

    assert result["readiness"] == "EXPLORATORY_PRECOMMITTED"
    assert result["qualification"] is False
    assert result["coverage"] == {
        "universe": "full_a",
        "component_count": 2,
        "component_semantic_sha256": _bound().binding["components"][
            "newline_set_sha256"
        ],
        "pit_record_count": 2,
        "bound_table_symbol_count": 2,
    }
    assert result["blockers"] == [
        "holdout_not_appended",
        "statistics_not_run",
        "verified_v4_replay_not_run",
        "qualification_not_evaluated",
    ]
    state_path = (
        tmp_path / "private" / "success-v41" / PRECOMMITTED_STATE_FILENAME
    )
    state = json.loads(state_path.read_text())
    assert state["state"] == "PRECOMMITTED"
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    assert not any(
        token in json.dumps(result).lower()
        for token in ("registry_write", "broker_order", "trade_receipt")
    )


def test_runner_source_rejection_publishes_only_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, run_id="source-blocked-v41")
    monkeypatch.setattr(runner, "_bind", lambda _args: _bound())
    monkeypatch.setattr(
        runner.source,
        "validate_pit_records_v4_1",
        lambda _records: (_ for _ in ()).throw(ValueError("PIT invalid")),
    )

    result = runner.run(args)  # type: ignore[arg-type]

    run_dir = tmp_path / "private" / "source-blocked-v41"
    assert result["readiness"] == "BLOCKED_FAIL_CLOSED"
    assert result["qualification"] is False
    assert (run_dir / BLOCKER_REPORT_FILENAME).is_file()
    assert not (run_dir / PRECOMMITTED_STATE_FILENAME).exists()
    blocker = json.loads((run_dir / BLOCKER_REPORT_FILENAME).read_text())
    assert blocker["input_binding_complete"] is True
    assert blocker["blockers"] == [
        {"code": "SOURCE_CONTRACT_REJECTED", "detail": "PIT invalid"}
    ]


def test_runner_input_rejection_publishes_incomplete_binding_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, run_id="input-blocked-v41")
    monkeypatch.setattr(
        runner,
        "_bind",
        lambda _args: (_ for _ in ()).throw(ValueError("pointer SHA mismatch")),
    )

    result = runner.run(args)  # type: ignore[arg-type]

    run_dir = tmp_path / "private" / "input-blocked-v41"
    blocker = json.loads((run_dir / BLOCKER_REPORT_FILENAME).read_text())
    assert result["readiness"] == "BLOCKED_FAIL_CLOSED"
    assert blocker["input_binding_complete"] is False
    assert blocker["blockers"] == [
        {
            "code": "INPUT_BINDING_REJECTED",
            "detail": "pointer SHA mismatch",
        }
    ]
    assert not (run_dir / PRECOMMITTED_STATE_FILENAME).exists()

