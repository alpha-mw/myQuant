"""Real zsh routing with controlled interpreter, never real credentials/providers."""

import json
import os
import subprocess

import pytest

from test_daily_launcher_dag_profile import arguments as old_arguments, interpreter


def arguments(root, python):
    args = old_arguments(root, python)
    index = args.index("--daily-production-request")
    stop = index + 4
    args[index:stop] = [
        "--daily-source-config",
        "config.json",
        "--expected-daily-source-config-sha256",
        "c" * 64,
    ]
    return args


def source_interpreter(
    root,
    *,
    source_mode,
    dag_mode,
    source_exit=0,
    closed=False,
    selection="requests/generated source.json",
):
    executable = interpreter(root, mode=dag_mode, dispatch_exit=0)
    content = executable.read_text()
    prefix = (
        f"source_mode={source_mode!r}\nsource_exit={source_exit!r}\n"
        f"closed={closed!r}\nselection={selection!r}\n"
    )
    content = content.replace(
        "import hashlib,json,os,sys\n", prefix + "import hashlib,json,os,sys\n", 1
    )
    branch = """elif 'quant_investor.cli.daily_sources' in a:
 stage=a[a.index('--mode')+1]
 log('SOURCE_'+stage.upper(),no_providers='--no-providers' in a)
 if stage in ('emit','select'):
  p=root/a[a.index('--inspection')+1]
  assert hashlib.sha256(p.read_bytes()).hexdigest()==a[a.index('--expected-inspection-sha256')+1]
 if stage=='inspect' and not (root/'source-done').exists():
  print(json.dumps({'mode':'NON_TRADING_DAY' if closed else 'REQUEST_AVAILABLE'}))
  raise SystemExit(source_mode)
 if stage=='provision':
  assert ('TUSHARE_TOKEN' in os.environ)==(source_mode==11)
  assert ('--no-providers' in a)==(source_mode==10)
  if source_exit:
   print(json.dumps({'status':'BLOCKED'}));raise SystemExit(source_exit)
  (root/'source-done').write_text('done')
 if stage=='select':
  if closed:raise SystemExit(10)
  print(selection);print('d'*64);raise SystemExit(0)
 print(json.dumps({'mode':'NON_TRADING_DAY' if closed else 'REQUEST_AVAILABLE'}))
 raise SystemExit(0)
"""
    content = content.replace(
        "elif 'credential-preflight' in a:\n", branch + "elif 'credential-preflight' in a:\n", 1
    )
    executable.write_text(content)
    return executable


@pytest.mark.parametrize("source_mode,dag_mode", [(s, d) for s in (0, 10, 11) for d in (0, 10, 11)])
def test_source_to_existing_dag_routes_and_credentials_are_scoped(tmp_path, source_mode, dag_mode):
    python = source_interpreter(tmp_path, source_mode=source_mode, dag_mode=dag_mode)
    result = subprocess.run(
        arguments(tmp_path, python),
        capture_output=True,
        text=True,
        env={**os.environ, "TUSHARE_TOKEN": "must-not-inherit"},
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    stages = [row["stage"] for row in rows]
    assert stages[0:3] == ["IMPORT", "STARTED", "SOURCE_INSPECT"]
    assert stages[-1] == "ENDED" and "FORBIDDEN_LEGACY" not in stages
    assert stages.count("CREDENTIAL_READ") == int(source_mode == 11 or dag_mode == 11)
    assert stages.count("CREDENTIAL_RECEIPT") == int(source_mode == 11 or dag_mode == 11)
    for row in rows:
        if row["stage"] not in {"SOURCE_PROVISION", "DAILY_CLOSE"}:
            assert row["token_present"] is False
    assert "unit-test-token" not in result.stdout + result.stderr


def test_nontrading_day_finishes_before_dag(tmp_path):
    python = source_interpreter(tmp_path, source_mode=11, dag_mode=11, closed=True)
    result = subprocess.run(arguments(tmp_path, python), capture_output=True, text=True)
    assert (
        result.returncode == 0
        and json.loads(result.stdout.splitlines()[-2])["mode"] == "NON_TRADING_DAY"
    )
    stages = [
        json.loads(line)["stage"] for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert "INSPECT" not in stages and "DAILY_CLOSE" not in stages and stages[-1] == "ENDED"


@pytest.mark.parametrize("exit_code", [2, 3, 7])
def test_source_failure_never_falls_through(tmp_path, exit_code):
    python = source_interpreter(tmp_path, source_mode=11, dag_mode=11, source_exit=exit_code)
    result = subprocess.run(arguments(tmp_path, python), capture_output=True, text=True)
    assert result.returncode == (2 if exit_code == 2 else 3)
    stages = [
        json.loads(line)["stage"] for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert not {"INSPECT", "DAILY_CLOSE", "FORBIDDEN_LEGACY"} & set(stages)


@pytest.mark.parametrize("fault", ["both", "partial", "absolute", "sha", "slot"])
def test_invalid_source_profile_is_rejected_before_interpreter(tmp_path, fault):
    python = source_interpreter(tmp_path, source_mode=11, dag_mode=11)
    args = arguments(tmp_path, python)
    if fault == "both":
        args.extend(
            [
                "--daily-production-request",
                "request.json",
                "--expected-daily-production-request-sha256",
                "a" * 64,
            ]
        )
    elif fault == "partial":
        args.remove("--expected-daily-source-config-sha256")
        args.remove("c" * 64)
    elif fault == "absolute":
        args[args.index("--daily-source-config") + 1] = "/tmp/config.json"
    elif fault == "sha":
        args[args.index("--expected-daily-source-config-sha256") + 1] = "bad"
    else:
        args[args.index("--attempt-slot") + 1] = "1620"
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 2 and not (tmp_path / "calls.jsonl").exists()


def test_selection_is_literal_and_never_evaluated(tmp_path):
    selection = "requests/$(touch MUST_NOT_EXIST).json"
    python = source_interpreter(tmp_path, source_mode=0, dag_mode=0, selection=selection)
    result = subprocess.run(
        arguments(tmp_path, python), capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "MUST_NOT_EXIST").exists()


def test_invalid_selected_path_stops_before_dag(tmp_path):
    python = source_interpreter(tmp_path, source_mode=0, dag_mode=11, selection="../escape.json")
    result = subprocess.run(arguments(tmp_path, python), capture_output=True, text=True)
    assert result.returncode == 2
    stages = [
        json.loads(line)["stage"] for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert "INSPECT" not in stages and "CREDENTIAL_READ" not in stages
