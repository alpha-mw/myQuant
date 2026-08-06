from __future__ import annotations

import pytest

from quant_investor.cli import main as cli_main
from quant_investor.v17_mainline import V17MainlineError


@pytest.mark.parametrize(
    "argv",
    [
        ["research", "run", "--help"],
        ["market", "analyze", "--help"],
        ["market", "run", "--help"],
    ],
)
def test_v17_public_help_has_only_workspace_and_strategy(argv, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(argv)

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--workspace-root" in help_text
    assert "--strategy-id" in help_text
    for retired in ("US", "--market", "--decision-protocol", "--stocks"):
        assert retired not in help_text


@pytest.mark.parametrize(
    "argv",
    [
        ["research", "run", "--strategy-id", "cn-mainline", "--market", "US"],
        ["market", "analyze", "--strategy-id", "cn-mainline", "--decision-protocol", "retired"],
        ["market", "run", "--strategy-id", "cn-mainline", "--stocks", "000001.SZ"],
    ],
)
def test_v17_public_cli_rejects_legacy_arguments(argv) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(argv)
    assert exc_info.value.code == 2


@pytest.mark.parametrize("command", ["research", "analyze", "run"])
def test_v17_public_cli_dispatches_same_reader(monkeypatch, tmp_path, command, capsys) -> None:
    expected = {"schema_id": "myquant.v17.v4.mainline-public-run.v1"}
    captured = {}

    def fake_read_public_run(workspace_root, *, strategy_id, **kwargs):
        captured.update(workspace_root=workspace_root, strategy_id=strategy_id, kwargs=kwargs)
        return expected

    monkeypatch.setattr("quant_investor.v17_mainline.read_public_run", fake_read_public_run)
    argv = (
        ["research", "run"]
        if command == "research"
        else ["market", command]
    )
    cli_main.main(
        argv
        + [
            "--workspace-root",
            str(tmp_path),
            "--strategy-id",
            "cn-mainline",
        ]
    )

    assert "myquant.v17.v4.mainline-public-run.v1" in capsys.readouterr().out
    assert captured["workspace_root"] == tmp_path
    assert captured["strategy_id"] == "cn-mainline"


def test_programmatic_us_market_fails_closed() -> None:
    with pytest.raises(V17MainlineError) as exc_info:
        cli_main.run_market_analysis(
            workspace_root=".",
            strategy_id="cn-mainline",
            market="US",
        )
    assert exc_info.value.code == "V17_MARKET_UNSUPPORTED"


def test_programmatic_legacy_argument_fails_closed() -> None:
    with pytest.raises(V17MainlineError) as exc_info:
        cli_main.run_market_pipeline(
            workspace_root=".",
            strategy_id="cn-mainline",
            decision_protocol="retired",
        )
    assert exc_info.value.code == "V17_PUBLIC_ARGUMENTS_UNSUPPORTED"


def test_backtest_is_fixed_unavailable() -> None:
    with pytest.raises(V17MainlineError) as exc_info:
        cli_main.run_market_backtest(market="CN")
    assert exc_info.value.code == "V17_BACKTEST_UNAVAILABLE"


def test_market_help_hides_retired_producers_and_keeps_neutral_commands(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["market", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    for retired in ("macro-refresh", "fundamental-research-", "data-governance"):
        assert retired not in help_text
    assert "macro-maintain" in help_text
    for supported in (
        "maintain",
        "download",
        "fundamental-maintain",
        "fundamental-promote",
        "storage-validate",
        "materialize-features",
        "analyze",
        "run",
        "backtest",
    ):
        assert supported in help_text
