"""The real CLI/shell reject mixed authority before touching runtime state."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from quant_investor.cli.main import _dispatch
from quant_investor.market.daily_maintenance import DailyMaintenanceError

PAIR = [
    "--factor-loop-context",
    "/private/tmp/context.json",
    "--expected-factor-loop-context-sha256",
    "a" * 64,
]
TRANSITION = [
    "--scope-transition-request",
    "/private/tmp/scope.json",
    "--expected-scope-transition-sha256",
    "b" * 64,
]
RETIRE = ["--retire-coverage-declaration-sha256", "c" * 64]


@pytest.mark.parametrize(
    "extra,mode,slot",
    [
        (PAIR[:2], "execute", "2020"),
        (PAIR[2:], "execute", "2020"),
        (["--recover-only"], "execute", "2020"),
        (PAIR, "shadow", "2020"),
        (PAIR, "execute", "1820"),
        (PAIR + ["--recover-only"], "execute", "auto"),
        (TRANSITION, "shadow", "2020"),
        (TRANSITION, "execute", "1620"),
        (TRANSITION[:2], "execute", "2020"),
        (TRANSITION[2:], "execute", "2020"),
        (RETIRE, "execute", "2020"),
        (TRANSITION + PAIR, "execute", "2020"),
        (TRANSITION + ["--recover-only"], "execute", "2020"),
        (RETIRE + PAIR, "execute", "2020"),
        (TRANSITION[:2] + PAIR[2:], "execute", "2020"),
    ],
)
def test_cli_invalid_modes_leave_every_runtime_byte_untouched(tmp_path, extra, mode, slot):
    protected = tmp_path / "results/factors/observations/original.json"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"immutable-original")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    run = tmp_path / "data/private/cn_daily_maintenance"
    with pytest.raises(DailyMaintenanceError):
        _dispatch(
            [
                "market",
                "daily-maintain",
                "--market",
                "CN",
                "--workspace-root",
                str(tmp_path),
                "--run-root",
                str(run),
                "--mode",
                mode,
                "--attempt-slot",
                slot,
                *extra,
            ]
        )
    assert not run.exists()
    assert {
        p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    } == before


@pytest.mark.parametrize(
    "extra,slot",
    [
        (PAIR[:2], "2020"),
        (PAIR[2:], "2020"),
        (PAIR, "1620"),
        (TRANSITION + PAIR, "2020"),
        (RETIRE + PAIR, "2020"),
        (TRANSITION[:2] + PAIR[2:], "2020"),
        (PAIR[:-1] + ["g" * 64], "2020"),
        (TRANSITION[:-1] + ["g" * 64], "2020"),
        (TRANSITION + RETIRE[:-1] + ["g" * 64], "2020"),
    ],
)
def test_shell_rejects_before_even_running_the_interpreter(tmp_path, extra, slot):
    interpreter = tmp_path / "python"
    called = tmp_path / "interpreter-called"
    interpreter.write_text(f"#!/bin/sh\ntouch '{called}'\nexit 99\n")
    interpreter.chmod(0o700)
    launcher = Path(__file__).resolve().parents[2] / "scripts/operations/run_cn_daily_slot.sh"
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    result = subprocess.run(
        [
            str(launcher),
            "--python",
            str(interpreter),
            "--expected-import-root",
            str(tmp_path),
            "--workspace-root",
            str(tmp_path),
            "--run-root",
            str(tmp_path / "run"),
            "--attempt-slot",
            slot,
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr
    assert not called.exists()
    assert not (tmp_path / "run").exists()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before


def test_shell_factor_recovery_precedes_credentials_and_keeps_auxiliary_failure(tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!" + sys.executable + "\n" + f"root={str(tmp_path)!r}\n" + """
import json,sys
from pathlib import Path
root=Path(root)
a=sys.argv[1:]
def log(stage):
    with (root/'calls.jsonl').open('a') as output:
        output.write(json.dumps(stage)+'\\n')
if '-c' in a:
    code=a[a.index('-c')+1]
    if 'write_launcher_record' in code:
        phase='ENDED' if 'phase="ENDED"' in code else 'STARTED'
        log(phase)
        (Path(a[a.index('-c')+2])/'launcher_attempts'/a[a.index('-c')+3]).mkdir(parents=True,exist_ok=True)
        print('{}')
    elif 'import pathlib,quant_investor' in code:
        log('IMPORT'); print(root/'quant_investor/__init__.py')
    else:
        log('CREDENTIAL_READ'); print('unit-test-token')
elif 'credential-preflight' in a:
    log('CREDENTIAL_RECEIPT')
    p=Path(a[a.index('--run-root')+1])/'credential_preflight'/(a[a.index('--receipt-id')+1]+'.json')
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
elif 'daily-maintain' in a:
    assert '--factor-loop-context' in a and '--scope-transition-request' not in a
    log('RECOVERY' if '--recover-only' in a else 'MAINTENANCE')
    print(json.dumps({'status':'NO_ACTION' if '--recover-only' in a else 'PARTIAL'}))
    raise SystemExit(0 if '--recover-only' in a else 2)
else:
    raise SystemExit('unexpected launcher route')
""")
    fake.chmod(0o700)
    launcher = Path(__file__).resolve().parents[2] / "scripts/operations/run_cn_daily_slot.sh"
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
            *PAIR,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr
    assert [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()] == [
        "IMPORT",
        "STARTED",
        "RECOVERY",
        "CREDENTIAL_READ",
        "CREDENTIAL_RECEIPT",
        "MAINTENANCE",
        "ENDED",
    ]
    assert "unit-test-token" not in result.stdout + result.stderr
    attempts = list((tmp_path / "run/launcher_attempts").iterdir())
    assert len(attempts) == 1
    assert json.loads((attempts[0] / "recovery.stdout.json").read_text())["status"] == "NO_ACTION"
    assert json.loads((attempts[0] / "maintenance.stdout.json").read_text())["status"] == "PARTIAL"
