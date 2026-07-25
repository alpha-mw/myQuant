from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import quant_investor.cli.main as cli_main

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


@pytest.mark.parametrize(
    ("command", "runner_name", "arguments", "expected"),
    [
        (
            "v17-source-maintain",
            "run_v17_source_maintain",
            [
                "--plan",
                "plan.json",
                "--expected-plan-sha256",
                SHA_A,
                "--expected-manifest-sha256",
                SHA_B,
            ],
            {
                "plan_path": "plan.json",
                "expected_plan_sha256": SHA_A,
                "expected_manifest_sha256": SHA_B,
            },
        ),
        (
            "v17-risk-policy-seal",
            "run_v17_risk_policy_seal",
            [
                "--owner-mandate",
                "owner.json",
                "--output",
                "data/private/v17_sources/objects/risk.json",
                "--expected-owner-mandate-sha256",
                SHA_A,
                "--validation-cutoff",
                "2026-07-22T07:00:00Z",
            ],
            {
                "owner_mandate_path": "owner.json",
                "output_path": "data/private/v17_sources/objects/risk.json",
                "expected_owner_mandate_sha256": SHA_A,
                "validation_cutoff": "2026-07-22T07:00:00Z",
            },
        ),
        (
            "v17-shadow-prepare",
            "run_v17_shadow_prepare",
            [
                "--request",
                "prepare.json",
                "--expected-request-sha256",
                SHA_A,
                "--expected-ledger-sha256",
                SHA_B,
            ],
            {
                "request_path": "prepare.json",
                "expected_request_sha256": SHA_A,
                "expected_ledger_sha256": SHA_B,
            },
        ),
        (
            "v17-shadow-receive",
            "run_v17_shadow_receive",
            [
                "--run-id",
                "run-1",
                "--response",
                "response.json",
                "--expected-response-sha256",
                SHA_A,
                "--expected-ledger-sha256",
                SHA_B,
                "--expected-latest-sha256",
                SHA_C,
                "--failed-at",
                "2026-07-22T07:01:00Z",
            ],
            {
                "run_id": "run-1",
                "response_path": "response.json",
                "expected_response_sha256": SHA_A,
                "expected_ledger_sha256": SHA_B,
                "expected_latest_sha256": SHA_C,
                "failed_at": "2026-07-22T07:01:00Z",
            },
        ),
        (
            "v17-shadow-finalize",
            "run_v17_shadow_finalize",
            [
                "--run-id",
                "run-1",
                "--finalization",
                "finalization.json",
                "--expected-finalization-sha256",
                SHA_A,
                "--expected-ledger-sha256",
                SHA_B,
                "--expected-latest-sha256",
                SHA_C,
                "--failed-at",
                "2026-07-22T07:02:00Z",
            ],
            {
                "run_id": "run-1",
                "finalization_path": "finalization.json",
                "expected_finalization_sha256": SHA_A,
                "expected_ledger_sha256": SHA_B,
                "expected_latest_sha256": SHA_C,
                "failed_at": "2026-07-22T07:02:00Z",
            },
        ),
        (
            "v17-shadow-status",
            "run_v17_shadow_status",
            ["--run-id", "run-1"],
            {"run_id": "run-1"},
        ),
        (
            "v17-shadow-latest-repair",
            "run_v17_shadow_latest_repair",
            [
                "--run-id",
                "run-1",
                "--expected-ledger-sha256",
                SHA_B,
                "--expected-latest-sha256",
                SHA_C,
                "--repaired-at",
                "2026-07-22T07:03:00Z",
            ],
            {
                "run_id": "run-1",
                "expected_ledger_sha256": SHA_B,
                "expected_latest_sha256": SHA_C,
                "repaired_at": "2026-07-22T07:03:00Z",
            },
        ),
    ],
)
def test_shared_market_cli_dispatches_all_v17_shadow_commands(
    command: str,
    runner_name: str,
    arguments: list[str],
    expected: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    captured: dict[str, Any] = {}
    printed: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"command": command, "authority": False}

    monkeypatch.setattr(cli_main, runner_name, fake_runner)
    monkeypatch.setattr(cli_main, "_print_json", printed.append)

    cli_main.main(
        [
            "market",
            command,
            "--repo-root",
            str(repo_root),
            *arguments,
        ]
    )

    assert captured == {"repo_root": repo_root.absolute(), **expected}
    assert printed == [{"command": command, "authority": False}]


@pytest.mark.parametrize(
    "command",
    [
        "v17-source-maintain",
        "v17-risk-policy-seal",
        "v17-shadow-prepare",
        "v17-shadow-receive",
        "v17-shadow-finalize",
        "v17-shadow-status",
        "v17-shadow-latest-repair",
    ],
)
def test_v17_shadow_command_help_is_available_without_writes(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["market", command, "--help"])

    assert exc_info.value.code == 0
    assert list(tmp_path.iterdir()) == []


def test_risk_policy_seal_missing_owner_file_exits_two_without_writes(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            [
                "market",
                "v17-risk-policy-seal",
                "--repo-root",
                str(tmp_path),
                "--owner-mandate",
                str(tmp_path / "missing-owner.json"),
                "--output",
                "data/private/v17_sources/objects/risk.json",
                "--expected-owner-mandate-sha256",
                SHA_A,
                "--validation-cutoff",
                "2026-07-22T07:00:00Z",
            ]
        )

    assert exc_info.value.code == 2
    assert list(tmp_path.iterdir()) == []
