from __future__ import annotations

import pytest

from quant_investor.cli import main as cli_main
from quant_investor.market import fundamental_successor


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
