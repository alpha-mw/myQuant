from __future__ import annotations

import json

import pytest

from quant_investor.cli.output import (
    CommandError,
    MachineArgumentParser,
    canonical_json_line,
    command_boundary,
    emit_json,
)


def test_canonical_json_line_is_compact_sorted_and_finite() -> None:
    assert canonical_json_line({"z": 0.5, "a": "因子"}) == '{"a":"因子","z":0.5}'
    with pytest.raises(ValueError):
        canonical_json_line({"value": float("nan")})


def test_emit_json_writes_exactly_one_machine_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    emit_json({"status": "OK", "blockers": []})
    captured = capsys.readouterr()
    assert captured.out == '{"blockers":[],"status":"OK"}\n'
    assert captured.err == ""


def test_expected_failure_uses_exit_two_and_safe_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def action() -> None:
        raise CommandError(
            "POINTER_CONFLICT",
            fields={
                "expected_pointer_sha256": "a" * 64,
                "observed_pointer_sha256": "b" * 64,
            },
        )

    with pytest.raises(SystemExit) as raised:
        command_boundary(action)
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "BLOCKED",
        "blocker_code": "POINTER_CONFLICT",
        "expected_pointer_sha256": "a" * 64,
        "observed_pointer_sha256": "b" * 64,
    }
    assert captured.err == ""
    assert captured.out.count("\n") == 1


def test_internal_failure_does_not_disclose_exception_or_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def action() -> None:
        raise RuntimeError("secret /private/data/location")

    with pytest.raises(SystemExit) as raised:
        command_boundary(action)
    assert raised.value.code == 3
    captured = capsys.readouterr()
    assert captured.out == ('{"blocker_code":"INTERNAL_ERROR","status":"ERROR"}\n')
    assert captured.err == "quant-investor encountered an internal error\n"
    assert "secret" not in captured.out + captured.err
    assert "/private" not in captured.out + captured.err


def test_internal_failure_keeps_traceback_only_in_private_diagnostics(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    diagnostics = tmp_path / "diagnostics"
    monkeypatch.setenv("QUANT_INVESTOR_DIAGNOSTICS_DIR", str(diagnostics))

    def action() -> None:
        raise RuntimeError("secret /private/data/location")

    with pytest.raises(SystemExit) as raised:
        command_boundary(action)
    assert raised.value.code == 3
    captured = capsys.readouterr()
    assert captured.out == ('{"blocker_code":"INTERNAL_ERROR","status":"ERROR"}\n')
    assert captured.err == "quant-investor encountered an internal error\n"
    (written,) = diagnostics.iterdir()
    assert written.stat().st_mode & 0o777 == 0o600
    text = written.read_text()
    assert "RuntimeError: secret /private/data/location" in text
    assert "in action" in text


def test_expected_failure_and_unwritable_diagnostics_leave_contract_unchanged(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    diagnostics = tmp_path / "diagnostics"
    monkeypatch.setenv("QUANT_INVESTOR_DIAGNOSTICS_DIR", str(diagnostics))

    def expected() -> None:
        raise CommandError("RESULT_NOT_FOUND")

    with pytest.raises(SystemExit) as raised:
        command_boundary(expected)
    assert raised.value.code == 2
    assert not diagnostics.exists()
    capsys.readouterr()

    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")
    monkeypatch.setenv("QUANT_INVESTOR_DIAGNOSTICS_DIR", str(blocked))

    def unexpected() -> None:
        raise RuntimeError("boom")

    with pytest.raises(SystemExit) as raised:
        command_boundary(unexpected)
    assert raised.value.code == 3
    assert capsys.readouterr().out == ('{"blocker_code":"INTERNAL_ERROR","status":"ERROR"}\n')


def test_argument_failure_is_canonical_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = MachineArgumentParser(prog="quant-investor")
    parser.add_argument("--required", required=True)
    with pytest.raises(SystemExit) as raised:
        parser.parse_args([])
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ('{"blocker_code":"ARGUMENTS_INVALID","status":"BLOCKED"}\n')
    assert captured.err == ""


def test_domain_exit_two_is_safely_projected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class DomainError(RuntimeError):
        exit_code = 2
        code = "OBJECT_INVALID"
        public_fields = {"object_sha256": "a" * 64}

    def action() -> None:
        raise DomainError("do not expose /secret/path")

    with pytest.raises(SystemExit) as raised:
        command_boundary(action)
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "BLOCKED",
        "blocker_code": "OBJECT_INVALID",
        "object_sha256": "a" * 64,
    }
    assert "/secret" not in captured.out + captured.err
