"""CLI contract tests for the v15 staged CN Macro publisher."""

from __future__ import annotations

import json

import pytest

import quant_investor.cli.main as cli_main


def _required_args() -> list[str]:
    return [
        "market",
        "macro-maintain",
        "--market",
        "CN",
        "--as-of",
        "2026-07-14",
        "--authoritative-refresh",
        "--run-id",
        "cn_macro_primary_20260714",
        "--expected-catalog-sha256",
        "1" * 64,
        "--expected-market-pointer-sha256",
        "2" * 64,
        "--nbs-cn-pmi-url",
        (
            "https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202606/"
            "t20260630_1964032.html"
        ),
    ]


def test_macro_maintain_help_discloses_live_and_cas_requirements(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["market", "macro-maintain", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--allow-live" in output
    assert "--expected-catalog-sha256" in output
    assert "--expected-market-pointer-sha256" in output
    assert "--run-id" in output
    assert "--nbs-cn-pmi-url" in output
    assert "--allow-tushare-fallback" in output


def test_macro_maintain_stages_all_production_inputs(monkeypatch, capsys):
    captured: dict[str, object] = {}

    def _fake_run_macro_maintenance(**kwargs):
        captured.update(kwargs)
        return {"status": "staged", "promoted": False, "run_id": kwargs["run_id"]}

    monkeypatch.setattr(
        cli_main,
        "run_macro_authoritative_maintenance",
        _fake_run_macro_maintenance,
    )
    cli_main.main(
        [
            *_required_args(),
            "--allow-live",
            "--allow-tushare-fallback",
        ]
    )

    assert captured == {
        "market": "CN",
        "as_of": "2026-07-14",
        "canonical_root": "data/parquet/cn/macro_daily",
        "staging_root": "results/v15/macro_observation_staging",
        "run_id": "cn_macro_primary_20260714",
        "expected_catalog_sha256": "1" * 64,
        "expected_market_pointer_sha256": "2" * 64,
        "allow_live": True,
        "nbs_cn_pmi_url": (
            "https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202606/"
            "t20260630_1964032.html"
        ),
        "allow_tushare_fallback": True,
    }
    assert json.loads(capsys.readouterr().out) == {
        "status": "staged",
        "promoted": False,
        "run_id": "cn_macro_primary_20260714",
    }


def test_macro_maintain_without_live_authorization_fails_before_dispatch(
    monkeypatch,
    capsys,
):
    called = False

    def _unexpected_run_macro_maintenance(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        cli_main,
        "run_macro_authoritative_maintenance",
        _unexpected_run_macro_maintenance,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(_required_args())

    assert exc_info.value.code == 2
    assert called is False
    assert "--authoritative-refresh requires --allow-live" in capsys.readouterr().err


def test_macro_promote_dispatches_only_staging_and_catalog_cas(monkeypatch, capsys):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_promotion",
        lambda **kwargs: captured.update(kwargs) or {"status": "promoted"},
    )
    cli_main.main(
        [
            "market",
            "macro-promote",
            "--staging-root",
            "/tmp/macro-stage/run-1",
            "--expected-catalog-sha256",
            "a" * 64,
        ]
    )
    assert captured == {
        "staging_root": "/tmp/macro-stage/run-1",
        "canonical_root": "data/parquet/cn/macro_daily",
        "expected_catalog_sha256": "a" * 64,
    }
    assert json.loads(capsys.readouterr().out) == {"status": "promoted"}
