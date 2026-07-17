"""CLI contract tests for the v15 staged CN Macro publisher."""

from __future__ import annotations

import json
from pathlib import Path

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
        "--expected-macro-observations-pointer-sha256",
        "3" * 64,
        "--expected-macro-release-calendar-pointer-sha256",
        "4" * 64,
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
    assert "--expected-macro-observations-pointer-sha256" in output
    assert "--expected-macro-release-calendar-pointer-sha256" in output
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
        "macro_observations_root": "data/parquet/cn/macro_observations",
        "expected_macro_observations_pointer_sha256": "3" * 64,
        "macro_release_calendar_root": str(
            (Path.cwd() / "data/parquet/cn/macro_release_calendar").resolve()
        ),
        "expected_macro_release_calendar_pointer_sha256": "4" * 64,
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


def test_macro_official_compile_dispatches_offline_bundle(monkeypatch, capsys):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_official_web_compilation",
        lambda **kwargs: captured.update(kwargs) or {"status": "OK"},
    )

    cli_main.main(
        [
            "market",
            "macro-official-compile",
            "--plan",
            "/tmp/plan.json",
            "--capture-manifest",
            "/tmp/capture.json",
            "--raw-root",
            "/tmp/raw",
            "--run-id",
            "official-1",
        ]
    )

    assert captured == {
        "plan_path": "/tmp/plan.json",
        "capture_manifest_path": "/tmp/capture.json",
        "raw_root": "/tmp/raw",
        "output_root": "results/v15/macro_official_web",
        "run_id": "official-1",
    }
    assert json.loads(capsys.readouterr().out) == {"status": "OK"}


def test_macro_release_calendar_publish_dispatches_all_hashes(
    monkeypatch,
    capsys,
):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_release_calendar_publish",
        lambda **kwargs: captured.update(kwargs) or {"idempotent": False},
    )

    cli_main.main(
        [
            "market",
            "macro-release-calendar-publish",
            "--plan",
            "/tmp/release-plan.json",
            "--expected-plan-sha256",
            "1" * 64,
            "--capture-manifest",
            "/tmp/release-capture.json",
            "--expected-capture-manifest-sha256",
            "2" * 64,
            "--raw-root",
            "/tmp/release-raw",
            "--market-open-days",
            "/tmp/open-days.json",
            "--expected-market-open-days-sha256",
            "3" * 64,
            "--run-id",
            "release-calendar-1",
            "--expected-pointer-sha256",
            "EMPTY",
        ]
    )

    assert captured == {
        "plan_path": "/tmp/release-plan.json",
        "expected_plan_sha256": "1" * 64,
        "capture_manifest_path": "/tmp/release-capture.json",
        "expected_capture_manifest_sha256": "2" * 64,
        "raw_root": "/tmp/release-raw",
        "market_open_days_path": "/tmp/open-days.json",
        "expected_market_open_days_sha256": "3" * 64,
        "canonical_root": str(
            (Path.cwd() / "data/parquet/cn/macro_release_calendar").resolve()
        ),
        "run_id": "release-calendar-1",
        "expected_pointer_sha256": "EMPTY",
    }
    assert json.loads(capsys.readouterr().out) == {"idempotent": False}


def _write_open_days(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "market-open-days.v1",
                "market": "CN",
                "open_dates": ["20260714", "20260715", "20260716"],
            }
        ),
        encoding="utf-8",
    )


def test_macro_official_refresh_dispatches_projection_bindings(
    monkeypatch,
    capsys,
    tmp_path,
):
    open_days = tmp_path / "open-days.json"
    _write_open_days(open_days)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_official_observation_refresh",
        lambda **kwargs: captured.update(kwargs) or {"status": "OK"},
    )

    cli_main.main(
        [
            "market",
            "macro-observation-official-refresh",
            "--official-manifest",
            "/tmp/official/manifest.json",
            "--expected-official-manifest-sha256",
            "1" * 64,
            "--expected-official-plan-sha256",
            "2" * 64,
            "--target-as-of",
            "20260716",
            "--decision-cutoff-at",
            "2026-07-16T15:00:00+08:00",
            "--market-open-days",
            str(open_days),
            "--expected-market-open-days-sha256",
            "3" * 64,
            "--run-id",
            "official-refresh-1",
            "--expected-pointer-sha256",
            "4" * 64,
        ]
    )

    assert captured["pinned_open_dates"] == [
        "20260714",
        "20260715",
        "20260716",
    ]
    assert captured["target_as_of"] == "20260716"
    assert captured["decision_cutoff_at"] == "2026-07-16T15:00:00+08:00"
    assert captured["expected_pointer_sha256"] == "4" * 64
    assert json.loads(capsys.readouterr().out) == {"status": "OK"}


def test_macro_local_roll_dispatches_projection_bindings(
    monkeypatch,
    capsys,
    tmp_path,
):
    open_days = tmp_path / "open-days.json"
    _write_open_days(open_days)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_local_breadth_roll",
        lambda **kwargs: captured.update(kwargs) or {"status": "OK"},
    )

    cli_main.main(
        [
            "market",
            "macro-local-observation-roll",
            "--snapshot-manifest",
            "/tmp/snapshot.json",
            "--expected-snapshot-manifest-sha256",
            "1" * 64,
            "--coverage-manifest",
            "/tmp/coverage.json",
            "--expected-coverage-manifest-sha256",
            "2" * 64,
            "--target-trade-date",
            "20260716",
            "--scope-artifact",
            "/tmp/scope.json",
            "--expected-scope-artifact-sha256",
            "3" * 64,
            "--target-as-of",
            "20260716",
            "--decision-cutoff-at",
            "2026-07-16T15:00:00+08:00",
            "--market-open-days",
            str(open_days),
            "--expected-market-open-days-sha256",
            "4" * 64,
            "--run-id",
            "local-roll-1",
            "--expected-pointer-sha256",
            "5" * 64,
        ]
    )

    assert captured["pinned_open_dates"] == [
        "20260714",
        "20260715",
        "20260716",
    ]
    assert captured["target_trade_date"] == "20260716"
    assert captured["target_as_of"] == "20260716"
    assert captured["expected_pointer_sha256"] == "5" * 64
    assert json.loads(capsys.readouterr().out) == {"status": "OK"}


def test_macro_observation_bootstrap_dispatches_all_hash_bindings(
    monkeypatch,
    capsys,
):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_production_observation_bundle",
        lambda **kwargs: captured.update(kwargs) or {"status": "OK"},
    )

    cli_main.main(
        [
            "market",
            "macro-observation-bootstrap",
            "--official-manifest",
            "/tmp/official/manifest.json",
            "--expected-official-manifest-sha256",
            "1" * 64,
            "--expected-official-plan-sha256",
            "2" * 64,
            "--local-binding-plan",
            "/tmp/local-binding-plan.json",
            "--expected-local-binding-plan-sha256",
            "3" * 64,
            "--as-of",
            "20260715",
            "--run-id",
            "observations-1",
            "--expected-pointer-sha256",
            "EMPTY",
        ]
    )

    assert captured == {
        "official_bundle_manifest_path": "/tmp/official/manifest.json",
        "expected_official_bundle_manifest_sha256": "1" * 64,
        "expected_official_plan_sha256": "2" * 64,
        "local_bootstrap_plan_path": "/tmp/local-binding-plan.json",
        "expected_local_bootstrap_plan_sha256": "3" * 64,
        "as_of": "20260715",
        "canonical_observations_root": (
            "data/parquet/cn/macro_observations"
        ),
        "run_id": "observations-1",
        "expected_pointer_sha256": "",
    }
    assert json.loads(capsys.readouterr().out) == {"status": "OK"}


def test_macro_local_observation_publish_dispatches_pointer_cas(
    monkeypatch,
    capsys,
):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_local_breadth_publish",
        lambda **kwargs: captured.update(kwargs) or {"status": "OK"},
    )

    cli_main.main(
        [
            "market",
            "macro-local-observation-publish",
            "--snapshot-manifest",
            "/tmp/snapshot.json",
            "--expected-snapshot-manifest-sha256",
            "4" * 64,
            "--coverage-manifest",
            "/tmp/coverage.json",
            "--expected-coverage-manifest-sha256",
            "6" * 64,
            "--target-trade-date",
            "20260715",
            "--scope-artifact",
            "/tmp/full-a-scope.json",
            "--expected-scope-artifact-sha256",
            "7" * 64,
            "--as-of",
            "20260715",
            "--run-id",
            "local-1",
            "--expected-pointer-sha256",
            "5" * 64,
        ]
    )

    assert captured == {
        "snapshot_manifest_path": "/tmp/snapshot.json",
        "expected_snapshot_manifest_sha256": "4" * 64,
        "coverage_manifest_path": "/tmp/coverage.json",
        "expected_coverage_manifest_sha256": "6" * 64,
        "target_trade_date": "20260715",
        "scope_artifact_path": "/tmp/full-a-scope.json",
        "expected_scope_artifact_sha256": "7" * 64,
        "as_of": "20260715",
        "canonical_observations_root": (
            "data/parquet/cn/macro_observations"
        ),
        "run_id": "local-1",
        "expected_pointer_sha256": "5" * 64,
    }
    assert json.loads(capsys.readouterr().out) == {"status": "OK"}


@pytest.mark.parametrize(
    ("command", "required_flags"),
    [
        (
            "macro-observation-bootstrap",
            (
                "--local-binding-plan",
                "--expected-local-binding-plan-sha256",
            ),
        ),
        (
            "macro-local-observation-publish",
            (
                "--snapshot-manifest",
                "--expected-snapshot-manifest-sha256",
                "--coverage-manifest",
                "--expected-coverage-manifest-sha256",
                "--target-trade-date",
                "--scope-artifact",
                "--expected-scope-artifact-sha256",
            ),
        ),
        (
            "macro-observation-official-refresh",
            (
                "--official-manifest",
                "--expected-official-manifest-sha256",
                "--target-as-of",
                "--decision-cutoff-at",
                "--market-open-days",
                "--expected-market-open-days-sha256",
            ),
        ),
        (
            "macro-local-observation-roll",
            (
                "--snapshot-manifest",
                "--coverage-manifest",
                "--target-trade-date",
                "--target-as-of",
                "--decision-cutoff-at",
                "--market-open-days",
            ),
        ),
        (
            "macro-release-calendar-publish",
            (
                "--plan",
                "--expected-plan-sha256",
                "--capture-manifest",
                "--expected-capture-manifest-sha256",
                "--market-open-days",
                "--expected-market-open-days-sha256",
            ),
        ),
    ],
)
def test_macro_observation_publication_help_discloses_exact_bindings(
    command,
    required_flags,
    capsys,
):
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["market", command, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for flag in required_flags:
        assert flag in output
