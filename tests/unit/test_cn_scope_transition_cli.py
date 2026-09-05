from pathlib import Path
import subprocess

import pytest

from quant_investor.cli.main import _build_parser
from quant_investor.market.daily_maintenance import DailyMaintenanceError, run_cn_daily_maintenance


def test_paired_request_options_and_unchanged_default():
    args = [
        "market",
        "daily-maintain",
        "--market",
        "CN",
        "--workspace-root",
        "/private/tmp/scope",
        "--run-root",
        "/private/tmp/scope/data/private/cn_daily_maintenance",
        "--mode",
        "execute",
    ]
    plain = _build_parser().parse_args(args)
    assert plain.scope_transition_request is None
    assert plain.expected_scope_transition_sha256 is None
    paired = _build_parser().parse_args(
        args
        + [
            "--scope-transition-request",
            "/private/tmp/request.json",
            "--expected-scope-transition-sha256",
            "a" * 64,
        ]
    )
    assert str(paired.scope_transition_request) == "/private/tmp/request.json"
    assert paired.expected_scope_transition_sha256 == "a" * 64


@pytest.mark.parametrize("scope_request,sha", [("/private/tmp/request.json", ""), (None, "a" * 64)])
def test_unpaired_options_fail_before_any_run_root_write(tmp_path, scope_request, sha):
    root = tmp_path / "must-not-exist"
    with pytest.raises(DailyMaintenanceError, match="REQUIRED_TOGETHER"):
        run_cn_daily_maintenance(
            workspace_root=tmp_path,
            run_root=root,
            mode="execute",
            scope_transition_request=scope_request,
            expected_scope_transition_sha256=sha,
        )
    assert not root.exists()


def test_launcher_rejects_unpaired_request_before_credential_access(tmp_path):
    launcher = Path(__file__).resolve().parents[2] / "scripts/operations/run_cn_daily_slot.sh"
    import sys

    result = subprocess.run(
        [
            str(launcher),
            "--python",
            sys.executable,
            "--expected-import-root",
            str(tmp_path),
            "--workspace-root",
            str(tmp_path),
            "--run-root",
            str(tmp_path / "run"),
            "--attempt-slot",
            "2020",
            "--scope-transition-request",
            str(tmp_path / "request.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "SCOPE_TRANSITION_ARGUMENTS_REQUIRED_TOGETHER" in result.stderr
    assert not (tmp_path / "run").exists()
