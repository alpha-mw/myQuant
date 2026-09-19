"""Real shell decisions with a controlled interpreter; no credential or provider access."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "scripts/operations/run_cn_daily_slot.sh"


def arguments(root, python):
    return [
        str(LAUNCHER),
        "--python",
        str(python),
        "--workspace-root",
        str(root),
        "--run-root",
        str(root / "data/private/cn_daily_maintenance"),
        "--expected-import-root",
        str(root),
        "--attempt-slot",
        "2020",
        "--daily-production-request",
        "requests/auto.json",
        "--expected-daily-production-request-sha256",
        "a" * 64,
        "--release-repository-root",
        str(root / "repository"),
        "--release-install-input",
        "install.json",
        "--expected-release-install-input-sha256",
        "b" * 64,
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "partial",
        "missing_value",
        "absolute_ref",
        "traversal",
        "double_slash",
        "unicode",
        "sha",
        "slot",
        "root",
        "factor",
        "scope",
    ],
)
def test_bad_dag_arguments_stop_before_interpreter(tmp_path, fault):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\ntouch " + str(tmp_path / "called") + "\nexit 99\n")
    python.chmod(0o700)
    args = arguments(tmp_path, python)
    if fault == "partial":
        del args[-2:]
    elif fault == "missing_value":
        del args[-1:]
    elif fault in {"absolute_ref", "traversal", "double_slash", "unicode"}:
        args[args.index("--daily-production-request") + 1] = {
            "absolute_ref": "/tmp/request.json",
            "traversal": "x/../request.json",
            "double_slash": "x//request.json",
            "unicode": "输入.json",
        }[fault]
    elif fault == "sha":
        args[-1] = "G" * 64
    elif fault == "slot":
        args[args.index("--attempt-slot") + 1] = "1620"
    elif fault == "root":
        args[args.index("--run-root") + 1] = str(tmp_path / "other")
    elif fault == "factor":
        args += [
            "--factor-loop-context",
            str(tmp_path / "context.json"),
            "--expected-factor-loop-context-sha256",
            "c" * 64,
        ]
    else:
        args += [
            "--scope-transition-request",
            str(tmp_path / "scope.json"),
            "--expected-scope-transition-sha256",
            "c" * 64,
        ]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 2 and not (tmp_path / "called").exists()
    assert not (tmp_path / "data").exists()


def interpreter(root, *, mode, dispatch_exit, changed=False, end_failed=False):
    file = root / "python"
    code = """
import hashlib,json,os,sys
from pathlib import Path
a=sys.argv[1:]
def log(stage,**extra):
 with (root/'calls.jsonl').open('a') as f:
  f.write(json.dumps({'stage':stage,'token_present':'TUSHARE_TOKEN' in os.environ,**extra})+'\\n')
if '-c' in a:
 code=a[a.index('-c')+1]
 if 'import pathlib,quant_investor' in code:
  log('IMPORT');print(root/'quant_investor/__init__.py')
 elif 'write_launcher_record' in code:
  phase='ENDED' if 'phase="ENDED"' in code else 'STARTED'
  log(phase,exit_code=int(a[-1]) if phase=='ENDED' else None)
  if phase=='ENDED' and end_failed:raise SystemExit(1)
  (Path(a[a.index('-c')+2])/'launcher_attempts'/a[a.index('-c')+3]).mkdir(parents=True,exist_ok=True)
  print('{}')
 elif 'daily_launch' in code:
  stage=a[a.index('--mode')+1];log(stage.upper())
  if stage!='inspect':
   p=root/a[a.index('--inspection')+1]
   assert hashlib.sha256(p.read_bytes()).hexdigest()==a[a.index('--expected-inspection-sha256')+1]
  if changed and stage=='validate':raise SystemExit(2)
  if stage=='emit':print(' {"status":"NO_ACTION"}');raise SystemExit(0)
  if stage=='recover':
   log('DAILY_CLOSE',no_producers=True);assert 'TUSHARE_TOKEN' not in os.environ
   print(json.dumps({'status':'COMPLETE' if dispatch_exit==0 else 'PARTIAL'}));raise SystemExit(dispatch_exit)
  if stage=='inspect':print(json.dumps({'fixture_inspection_mode':mode}))
  raise SystemExit(mode)
 else:
  log('CREDENTIAL_READ');print('unit-test-token')
elif 'credential-preflight' in a:
 log('CREDENTIAL_RECEIPT')
 p=Path(a[a.index('--run-root')+1])/'credential_preflight'/(a[a.index('--receipt-id')+1]+'.json')
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
elif 'daily-close' in a:
 log('DAILY_CLOSE',no_producers='--no-producers' in a)
 assert (os.environ.get('TUSHARE_TOKEN')=='unit-test-token')==(mode==11)
 assert ('--no-producers' in a)==(mode==10)
 print(json.dumps({'status':'COMPLETE' if dispatch_exit==0 else 'PARTIAL'}))
 raise SystemExit(dispatch_exit)
else:
 log('FORBIDDEN_LEGACY');raise SystemExit(99)
"""
    file.write_text(
        "#!"
        + sys.executable
        + "\n"
        + "from pathlib import Path\n"
        + f"root=Path({str(root)!r})\nmode={mode!r}\n"
        + f"dispatch_exit={dispatch_exit!r}\nchanged={changed!r}\n"
        + f"end_failed={end_failed!r}\n"
        + code
    )
    file.chmod(0o700)
    return file


@pytest.mark.parametrize(
    "mode,dispatch_exit,expected",
    [(0, 0, 0), (10, 0, 0), (10, 2, 2), (11, 0, 0), (11, 2, 2), (11, 7, 3), (2, 0, 2), (3, 0, 3)],
)
def test_dag_branch_isolated_and_receipts_keep_actual_exit(tmp_path, mode, dispatch_exit, expected):
    fake = interpreter(tmp_path, mode=mode, dispatch_exit=dispatch_exit)
    result = subprocess.run(
        arguments(tmp_path, fake),
        capture_output=True,
        text=True,
        env={**os.environ, "TUSHARE_TOKEN": "inherited-test-only"},
    )
    assert result.returncode == expected, result.stderr
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    stages = [row["stage"] for row in rows]
    assert "FORBIDDEN_LEGACY" not in stages and stages[:3] == ["IMPORT", "STARTED", "INSPECT"]
    assert stages[-1] == "ENDED" and rows[-1]["exit_code"] == expected
    if mode == 0:
        assert stages == ["IMPORT", "STARTED", "INSPECT", "VALIDATE", "EMIT", "ENDED"]
    elif mode in (10, 11):
        assert stages.count("DAILY_CLOSE") == 1
        assert ("CREDENTIAL_READ" in stages) == (mode == 11)
    else:
        assert "VALIDATE" not in stages and "CREDENTIAL_READ" not in stages
    assert all(not row["token_present"] for row in rows if row["stage"] != "DAILY_CLOSE")
    assert "unit-test-token" not in result.stdout + result.stderr
    attempt = next((tmp_path / "data/private/cn_daily_maintenance/launcher_attempts").iterdir())
    assert (attempt / "inspection.stdout.json").is_file() and (
        attempt / "inspection.stderr.log"
    ).is_file()


def test_changed_inspection_never_reaches_credentials(tmp_path):
    result = subprocess.run(
        arguments(tmp_path, interpreter(tmp_path, mode=11, dispatch_exit=0, changed=True)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    stages = [
        json.loads(line)["stage"] for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert stages == ["IMPORT", "STARTED", "INSPECT", "VALIDATE", "ENDED"]


def test_missing_ended_receipt_cannot_report_launcher_success(tmp_path):
    fake = interpreter(tmp_path, mode=0, dispatch_exit=0, end_failed=True)
    result = subprocess.run(arguments(tmp_path, fake), capture_output=True, text=True)
    assert result.returncode == 3
    assert "CN_DAILY_LAUNCHER_END_RECEIPT_UNAVAILABLE" in result.stderr
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert rows[-1]["stage"] == "ENDED" and rows[-1]["exit_code"] == 0
    assert not any(row["stage"] in {"CREDENTIAL_READ", "DAILY_CLOSE"} for row in rows)
