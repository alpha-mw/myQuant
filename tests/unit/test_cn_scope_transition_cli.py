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


def test_retirement_requires_transition_before_any_credential_access(tmp_path):
    import sys

    launcher = Path(__file__).resolve().parents[2] / "scripts/operations/run_cn_daily_slot.sh"
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
            "--retire-coverage-declaration-sha256",
            "a" * 64,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "REQUIRED_FOR_DECLARATION_RETIREMENT" in result.stderr
    assert not (tmp_path / "run").exists()


def test_launcher_passes_exact_retirement_and_scope_args(tmp_path):
    import sys, json

    launcher = Path(__file__).resolve().parents[2] / "scripts/operations/run_cn_daily_slot.sh"
    fake = tmp_path / "python"
    fake.write_text(
        "#!"
        + sys.executable
        + "\n"
        + """import sys,json
from pathlib import Path
root=Path("""
        + repr(str(tmp_path))
        + """)
a=sys.argv[1:]
if '-c' in a:
    code=a[a.index('-c')+1]
    print(str(root/'quant_investor/__init__.py') if 'import pathlib,quant_investor' in code else 'unit-test-token')
elif 'credential-preflight' in a:
    p=Path(a[a.index('--run-root')+1])/'credential_preflight'/(a[a.index('--receipt-id')+1]+'.json')
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
elif 'daily-maintain' in a:
    (root/'args.json').write_text(json.dumps(a))
else:
    raise SystemExit('unexpected launcher route')
"""
    )
    fake.chmod(0o755)
    request = tmp_path / "request.json"
    result = subprocess.run(
        [
            str(launcher),
            "--python",
            str(fake),
            "--expected-import-root",
            str(tmp_path),
            "--workspace-root",
            str(tmp_path),
            "--run-root",
            str(tmp_path / "run"),
            "--attempt-slot",
            "2020",
            "--scope-transition-request",
            str(request),
            "--expected-scope-transition-sha256",
            "b" * 64,
            "--retire-coverage-declaration-sha256",
            "a" * 64,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads((tmp_path / "args.json").read_text())
    assert argv[argv.index("--scope-transition-request") + 1] == str(request)
    assert argv[argv.index("--expected-scope-transition-sha256") + 1] == "b" * 64
    assert argv[argv.index("--retire-coverage-declaration-sha256") + 1] == "a" * 64
