from __future__ import annotations

import json

import pytest

from quant_investor.cli import main as cli_main
from quant_investor.portfolio_cycle import (
    DECISION_INPUT_READINESS_SCHEMA_ID,
    PortfolioCycleError,
)

SHA = "a" * 64
CUTOFF = "2026-08-05T07:00:00Z"
HISTORICAL_LABEL = "aggressive_tech_manufacturing"


def _base_argv(workspace_root: str) -> list[str]:
    return [
        "portfolio",
        "cycle-status",
        "--workspace-root",
        workspace_root,
        "--historical-label",
        HISTORICAL_LABEL,
        "--decision-cutoff",
        CUTOFF,
    ]


def test_portfolio_cycle_status_help_is_registered_and_read_only(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["portfolio", "cycle-status", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    for expected in (
        "--workspace-root",
        "--strategy-id",
        "--historical-label",
        "--identity-path",
        "--identity-sha256",
        "--holdings-pointer-path",
        "--holdings-pointer-sha256",
        "--decision-cutoff",
    ):
        assert expected in help_text
    for forbidden in (
        "--commit",
        "--publish",
        "--activate",
        "--allow-live",
        "--strict-cn-verified",
        "--portfolio-policy-verified",
    ):
        assert forbidden not in help_text


def test_portfolio_cycle_status_requires_historical_label(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            [
                "portfolio",
                "cycle-status",
                "--workspace-root",
                str(tmp_path),
                "--decision-cutoff",
                CUTOFF,
            ]
        )

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "blocker_code": "ARGUMENTS_INVALID",
        "status": "BLOCKED",
    }
    assert captured.err == ""


@pytest.mark.parametrize(
    "extra",
    [
        ["--identity-path", "data/private/identity.json"],
        ["--identity-sha256", SHA],
        ["--holdings-pointer-path", "data/private/holdings/_active.json"],
        ["--holdings-pointer-sha256", SHA],
    ],
)
def test_portfolio_cycle_status_requires_path_sha_pairs(tmp_path, extra, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(_base_argv(str(tmp_path)) + extra)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "blocker_code": "ARGUMENTS_INVALID",
        "status": "BLOCKED",
    }
    assert captured.err == ""


@pytest.mark.parametrize(
    "option,value",
    [
        ("--identity-path", "/absolute/identity.json"),
        ("--identity-path", "data/private/../identity.json"),
        ("--identity-path", "data//private/identity.json"),
        ("--identity-path", "data\\private\\identity.json"),
        ("--holdings-pointer-path", "./data/private/holdings.json"),
    ],
)
def test_portfolio_cycle_status_rejects_noncanonical_paths(tmp_path, option, value, capsys) -> None:
    sha_option = "--identity-sha256" if option == "--identity-path" else "--holdings-pointer-sha256"
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(_base_argv(str(tmp_path)) + [option, value, sha_option, SHA])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "blocker_code": "ARGUMENTS_INVALID",
        "status": "BLOCKED",
    }
    assert captured.err == ""


def test_portfolio_cycle_status_rejects_non_ascii_path(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            _base_argv(str(tmp_path))
            + [
                "--identity-path",
                "data/private/身份.json",
                "--identity-sha256",
                SHA,
            ]
        )

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "blocker_code": "ARGUMENTS_INVALID",
        "status": "BLOCKED",
    }
    assert captured.err == ""


@pytest.mark.parametrize(
    "option,value",
    [
        ("--identity-sha256", "A" * 64),
        ("--holdings-pointer-sha256", "abc"),
    ],
)
def test_portfolio_cycle_status_rejects_invalid_sha(tmp_path, option, value, capsys) -> None:
    path_option, path_value = (
        ("--identity-path", "data/private/identity.json")
        if option == "--identity-sha256"
        else (
            "--holdings-pointer-path",
            "data/private/holdings/_active.json",
        )
    )
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(_base_argv(str(tmp_path)) + [path_option, path_value, option, value])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "blocker_code": "ARGUMENTS_INVALID",
        "status": "BLOCKED",
    }
    assert captured.err == ""


@pytest.mark.parametrize(
    "cutoff",
    [
        "2026-08-05T15:00:00",
        "2026-08-05T15:00:00+08:00",
        "2026-08-05T07:00:00.000000Z",
        "not-a-cutoff",
    ],
)
def test_portfolio_cycle_status_rejects_invalid_decision_cutoff(tmp_path, cutoff, capsys) -> None:
    argv = _base_argv(str(tmp_path))
    argv[-1] = cutoff
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(argv)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "blocker_code": "ARGUMENTS_INVALID",
        "status": "BLOCKED",
    }
    assert captured.err == ""


def test_portfolio_cycle_status_blocked_is_json_and_exit_two(monkeypatch, tmp_path, capsys) -> None:
    expected = {
        "schema_id": DECISION_INPUT_READINESS_SCHEMA_ID,
        "state": "BLOCKED",
        "blockers": ["STRATEGY_ID_UNCONFIRMED"],
        "synthetic_only": False,
        "operational_authority": False,
        "write_performed": False,
    }
    monkeypatch.setattr(
        cli_main,
        "run_portfolio_cycle_status",
        lambda **_: expected,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(_base_argv(str(tmp_path)))

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == expected


def test_portfolio_cycle_status_dispatches_exact_foundation_arguments(
    monkeypatch, tmp_path, capsys
) -> None:
    captured_kwargs = {}
    expected = {
        "schema_id": DECISION_INPUT_READINESS_SCHEMA_ID,
        "state": "FOUNDATION_VALIDATED",
        "blockers": [],
        "synthetic_only": True,
        "operational_authority": False,
        "write_performed": False,
    }

    def fake_status(**kwargs):
        captured_kwargs.update(kwargs)
        return expected

    monkeypatch.setattr(cli_main, "run_portfolio_cycle_status", fake_status)
    cli_main.main(
        _base_argv(str(tmp_path))
        + [
            "--strategy-id",
            "cn-tech",
            "--identity-path",
            "data/private/portfolio_identity/declaration.json",
            "--identity-sha256",
            SHA,
            "--holdings-pointer-path",
            "data/private/portfolio_holdings/_active.json",
            "--holdings-pointer-sha256",
            SHA,
        ]
    )

    assert json.loads(capsys.readouterr().out) == expected
    assert captured_kwargs["workspace_root"] == tmp_path
    assert captured_kwargs["strategy_id"] == "cn-tech"
    assert captured_kwargs["historical_label"] == HISTORICAL_LABEL
    assert captured_kwargs["decision_cutoff"] == CUTOFF
    assert set(captured_kwargs) == {
        "workspace_root",
        "strategy_id",
        "historical_label",
        "identity_path",
        "identity_sha256",
        "holdings_pointer_path",
        "holdings_pointer_sha256",
        "decision_cutoff",
    }


def test_portfolio_cycle_status_internal_error_is_redacted_json(
    monkeypatch, tmp_path, capsys
) -> None:
    def fail(**_):
        raise RuntimeError("sensitive internal path")

    monkeypatch.setattr(cli_main, "run_portfolio_cycle_status", fail)
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(_base_argv(str(tmp_path)))

    assert exc_info.value.code == 3
    captured = capsys.readouterr()
    assert captured.err == "quant-investor encountered an internal error\n"
    assert json.loads(captured.out) == {
        "blocker_code": "INTERNAL_ERROR",
        "status": "ERROR",
    }
    assert "sensitive" not in captured.out


def test_portfolio_cycle_status_security_error_is_stable_json(
    monkeypatch, tmp_path, capsys
) -> None:
    def fail(**_):
        raise PortfolioCycleError(
            "PORTFOLIO_CYCLE_STORAGE_SECURITY",
            "artifact symlink rejected",
        )

    monkeypatch.setattr(cli_main, "run_portfolio_cycle_status", fail)
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(_base_argv(str(tmp_path)))

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "blocker_code": "PORTFOLIO_CYCLE_STORAGE_SECURITY",
        "status": "BLOCKED",
    }


def test_portfolio_cycle_status_wrapper_passes_historical_label_only_as_constraint(
    monkeypatch, tmp_path
) -> None:
    from quant_investor.portfolio_cycle import readiness

    captured = {}

    def fake_derive(workspace_root, **kwargs):
        captured.update(workspace_root=workspace_root, **kwargs)
        return {"state": "BLOCKED"}

    monkeypatch.setattr(readiness, "derive_decision_input_readiness", fake_derive)
    result = cli_main.run_portfolio_cycle_status(
        workspace_root=tmp_path,
        strategy_id=None,
        historical_label=HISTORICAL_LABEL,
        identity_path=None,
        identity_sha256=None,
        holdings_pointer_path=None,
        holdings_pointer_sha256=None,
        decision_cutoff=CUTOFF,
    )

    assert result == {"state": "BLOCKED"}
    assert captured["workspace_root"] == tmp_path
    assert captured["expected_historical_label"] == HISTORICAL_LABEL
    assert captured["strategy_id"] is None
    assert captured["synthetic_only"] is False
    assert "historical_label" not in captured


def test_portfolio_cycle_status_never_creates_missing_workspace(tmp_path, capsys) -> None:
    missing = tmp_path / "not-created"

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(_base_argv(str(missing)))

    assert exc_info.value.code == 2
    assert not missing.exists()
    assert json.loads(capsys.readouterr().out)["state"] == "BLOCKED"
