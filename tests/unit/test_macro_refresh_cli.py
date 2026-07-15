"""CLI contract tests for the production CN Macro refresh command."""

from __future__ import annotations

import json

import pytest

import quant_investor.cli.main as cli_main


def _required_args() -> list[str]:
    return [
        "market",
        "macro-refresh",
        "--market",
        "CN",
        "--run-id",
        "cn_macro_primary_20260714",
        "--expected-catalog-sha256",
        "1" * 64,
        "--expected-market-pointer-sha256",
        "2" * 64,
    ]


def test_macro_refresh_help_discloses_live_and_cas_requirements(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["market", "macro-refresh", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--allow-live" in output
    assert "--expected-catalog-sha256" in output
    assert "--expected-market-pointer-sha256" in output
    assert "--run-id" in output


def test_macro_refresh_dispatches_all_production_inputs(monkeypatch, capsys):
    captured: dict[str, object] = {}

    def _fake_run_macro_refresh(**kwargs):
        captured.update(kwargs)
        return {"status": "promoted", "run_id": kwargs["run_id"]}

    monkeypatch.setattr(cli_main, "run_macro_refresh", _fake_run_macro_refresh)
    cli_main.main(
        [
            *_required_args(),
            "--allow-live",
        ]
    )

    assert captured == {
        "market": "CN",
        "as_of": "",
        "data_root": "data/parquet/cn/macro_daily",
        "run_id": "cn_macro_primary_20260714",
        "expected_catalog_sha256": "1" * 64,
        "expected_market_pointer_sha256": "2" * 64,
        "allow_live": True,
    }
    assert json.loads(capsys.readouterr().out) == {
        "status": "promoted",
        "run_id": "cn_macro_primary_20260714",
    }


def test_macro_refresh_without_live_authorization_fails_before_dispatch(
    monkeypatch,
    capsys,
):
    called = False

    def _unexpected_run_macro_refresh(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        cli_main,
        "run_macro_refresh",
        _unexpected_run_macro_refresh,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(_required_args())

    assert exc_info.value.code == 2
    assert called is False
    assert "requires explicit --allow-live" in capsys.readouterr().err
