from __future__ import annotations

import pytest

from quant_investor.cli import main as cli_main
from quant_investor.market import fundamental_successor


def _taint_cli_args(tmp_path) -> list[str]:
    return [
        "market",
        "fundamental-maintain",
        "--market",
        "CN",
        "--universes",
        "full_a",
        "--as-of",
        "20260814",
        "--run-id",
        "taint-test",
        "--allow-live",
        "--taint-analysis-dry-run",
        "--audit-run-root",
        str(tmp_path / "audit"),
        "--canonical-predecessor-root",
        str(tmp_path / "canonical"),
        "--expected-pointer-sha256",
        "a" * 64,
        "--canonical-scope-path",
        str(tmp_path / "scope.json"),
        "--canonical-market-pointer-path",
        str(tmp_path / "market.json"),
        "--canonical-pit-pointer-path",
        str(tmp_path / "pit.json"),
        "--canonical-membership-path",
        str(tmp_path / "membership.parquet"),
        "--history-audit-path",
        str(tmp_path / "history.json"),
        "--expected-history-audit-sha256",
        "b" * 64,
    ]


def _append_first_cli_args(tmp_path) -> list[str]:
    return [
        "market",
        "fundamental-maintain",
        "--market",
        "CN",
        "--universes",
        "full_a",
        "--as-of",
        "20260814",
        "--run-id",
        "append-first-test",
        "--allow-live",
        "--safe-incremental-successor",
        "--append-first-successor",
        "--historical-taint-failure-evidence",
        f"{tmp_path / 'capture-failures'}#3564",
        "--successor-income-support",
        "689009.SH@20250630",
        "--successor-financial-support",
        "cashflow:920198.BJ@20260630",
        "--data-root",
        str(tmp_path / "staging"),
        "--checkpoint-root",
        str(tmp_path / "capture"),
        "--canonical-predecessor-root",
        str(tmp_path / "canonical"),
        "--expected-pointer-sha256",
        "a" * 64,
        "--canonical-scope-path",
        str(tmp_path / "scope.json"),
        "--canonical-market-pointer-path",
        str(tmp_path / "market.json"),
        "--canonical-pit-pointer-path",
        str(tmp_path / "pit.json"),
        "--canonical-membership-path",
        str(tmp_path / "membership.parquet"),
        "--history-audit-path",
        str(tmp_path / "history.json"),
        "--expected-history-audit-sha256",
        "b" * 64,
    ]


def test_canonical_subject_scope_is_authority_union_not_target_only() -> None:
    symbols, closure = fundamental_successor._subject_scope_closure(
        parent_subjects={"000001.SZ"},
        parent_evidence={"fundamental_period": {"subject_count": 1}},
        pit_expected_by_session={"20260807": {"600000.SH"}},
        observed_by_session={"20260807": {"688001.SH"}},
        target_scope={"000001.SZ", "920001.BJ"},
        target="20260807",
        authority_sha256={"authority": "a" * 64},
    )

    assert symbols == ["000001.SZ", "600000.SH", "688001.SH", "920001.BJ"]
    assert closure["subject_count"] == 4
    assert closure["frozen_before_provider_capture"] is True
    assert closure["alias_transformations"] == []


def test_safe_successor_promotion_defaults_to_read_only_preflight(
    monkeypatch, tmp_path, capsys
) -> None:
    captured = {}

    def fake_promote(**kwargs):
        captured.update(kwargs)
        return {"status": "PREFLIGHT_OK", "execute": kwargs["execute"]}

    monkeypatch.setattr(
        "quant_investor.market.fundamental_successor_promotion."
        "promote_successor_generation",
        fake_promote,
    )
    cli_main.main(
        [
            "market",
            "fundamental-promote",
            "--safe-incremental-successor",
            "--staging-root",
            str(tmp_path / "staging"),
            "--canonical-root",
            str(tmp_path / "canonical"),
            "--expected-pointer-sha256",
            "a" * 64,
        ]
    )

    assert captured["execute"] is False
    assert captured["journal_root"] is None
    assert captured["journal_run_id"] is None
    assert "PREFLIGHT_OK" in capsys.readouterr().out


def test_safe_successor_execute_requires_durable_journal(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            [
                "market",
                "fundamental-promote",
                "--safe-incremental-successor",
                "--execute",
                "--staging-root",
                str(tmp_path / "staging"),
                "--canonical-root",
                str(tmp_path / "canonical"),
                "--expected-pointer-sha256",
                "a" * 64,
            ]
        )
    assert exc_info.value.code == 2


def test_safe_successor_recovery_requires_exact_journal_identity() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            [
                "market",
                "fundamental-promote",
                "--safe-incremental-successor",
                "--recover",
            ]
        )
    assert exc_info.value.code == 2


def test_taint_dry_run_has_exclusive_non_staging_dispatch(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    captured = {}

    def fake_maintain(**kwargs):
        captured.update(kwargs)
        return {
            "taint_analysis_status": "PASS",
            "staging_written": False,
            "promotion_authorized": False,
        }

    monkeypatch.setattr(cli_main, "run_fundamental_maintenance", fake_maintain)
    cli_main.main(_taint_cli_args(tmp_path))
    assert captured["taint_analysis_dry_run"] is True
    assert captured["safe_incremental_successor"] is False
    assert captured["authoritative_full_rebuild"] is False
    assert captured["audit_run_root"] == str(tmp_path / "audit")
    assert captured["checkpoint_root"] is None
    assert '"staging_written": false' in capsys.readouterr().out.lower()


@pytest.mark.parametrize(
    "conflict",
    [
        ["--safe-incremental-successor"],
        ["--authoritative-full-rebuild"],
        ["--checkpoint-root", "/private/tmp/checkpoint"],
        ["--data-root", "/private/tmp/staging"],
    ],
)
def test_taint_dry_run_conflicts_fail_before_dispatch(
    monkeypatch,
    tmp_path,
    conflict,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_fundamental_maintenance",
        lambda **_kwargs: pytest.fail("conflict reached maintenance dispatch"),
    )
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main([*_taint_cli_args(tmp_path), *conflict])
    assert exc_info.value.code == 2


def test_taint_dry_run_blocked_receipt_exits_nonzero(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_fundamental_maintenance",
        lambda **_kwargs: {
            "taint_analysis_status": "BLOCKED",
            "staging_written": False,
        },
    )
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(_taint_cli_args(tmp_path))
    assert exc_info.value.code == 2
    assert '"taint_analysis_status": "blocked"' in (
        capsys.readouterr().out.lower()
    )


def test_taint_dry_run_help_discloses_live_capture_and_zero_publication(
    capsys,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["market", "fundamental-maintain", "--help"])
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--taint-analysis-dry-run" in help_text
    assert "live provider" in help_text
    assert "不写 staging、canonical 或 promotion" in help_text


def test_append_first_dispatch_seals_explicit_failure_evidence(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    captured = {}

    def fake_maintain(**kwargs):
        captured.update(kwargs)
        return {"status": "staged", "append_first": True}

    monkeypatch.setattr(cli_main, "run_fundamental_maintenance", fake_maintain)
    cli_main.main(_append_first_cli_args(tmp_path))

    assert captured["safe_incremental_successor"] is True
    assert captured["append_first_successor"] is True
    assert captured["historical_taint_evidence"] == [
        {
            "failure_root": str(tmp_path / "capture-failures"),
            "ordinal": 3564,
        }
    ]
    assert captured["income_support_dependencies"] == [
        {"ts_code": "689009.SH", "end_date": "20250630"}
    ]
    assert captured["financial_support_dependencies"] == [
        {
            "table": "cashflow",
            "ts_code": "920198.BJ",
            "end_date": "20260630",
        }
    ]
    assert '"append_first": true' in capsys.readouterr().out.lower()


def test_append_first_requires_replayable_failure_reference(tmp_path) -> None:
    args = _append_first_cli_args(tmp_path)
    marker = args.index("--historical-taint-failure-evidence")
    del args[marker : marker + 2]

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(args)

    assert exc_info.value.code == 2


def test_financial_support_closure_separates_rows_from_exact_absence() -> None:
    source_plan = {
        "requests": [
            {
                "table": "income",
                "partition_type": "EXACT_SYMBOL_REPORT_PERIOD_SUPPORT",
                "params": {"ts_code": "689009.SH", "period": "20250630"},
            },
            {
                "table": "cashflow",
                "partition_type": "EXACT_SYMBOL_REPORT_PERIOD_SUPPORT",
                "params": {"ts_code": "920198.BJ", "period": "20260630"},
            },
        ]
    }
    source_manifest = {
        "request_receipts": [
            {
                "accepted_count": 1,
                "status": "AVAILABLE",
                "receipt_sha256": "7" * 64,
            },
            {
                "accepted_count": 0,
                "status": "EMPTY",
                "receipt_sha256": "8" * 64,
            },
        ]
    }

    captured, absences = fundamental_successor._financial_support_closure(
        source_plan=source_plan,
        source_manifest=source_manifest,
        target_cutoff="20260814",
    )

    assert captured == [
        {"table": "income", "ts_code": "689009.SH", "end_date": "20250630"}
    ]
    assert absences == [
        {
            "status": "PROVEN_ABSENT",
            "table": "cashflow",
            "symbol": "920198.BJ",
            "end_date": "20260630",
            "available_through": "20260814",
            "evidence_sha256": "8" * 64,
        }
    ]
