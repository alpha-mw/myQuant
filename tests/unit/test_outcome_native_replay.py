"""Recorded-release route and bounded child mechanics; verifier seams are explicit."""

from copy import deepcopy
import sys

import pytest

from quant_investor.operations import outcome_native_replay as module
from quant_investor.operations.completed_handoff_snapshot import _mint_snapshot
from quant_investor.operations.daily_contract import EOD_NODE_IDS, ContractError
from test_production_outcome_daily_admission import _put


def context(root, monkeypatch, *, trade_date="20260820", commit="a" * 40):
    repository = root / "repository"
    repository.mkdir()
    ledger = {
        "classification": "CONTEMPORANEOUS",
        "prospective": True,
        "synthetic": False,
        "recomputed": False,
    }
    ledger_ref = _put(root, "controlled/ledger.json", ledger)
    install = {
        "release_install_evidence": {
            "payload": {
                "python_executable": "/verified/recorded/bin/python",
                "final_commit": commit,
                "final_tree": "b" * 40,
                "installed_code_manifest_sha256": "c" * 64,
            }
        }
    }
    install_ref = _put(root, "controlled/install.json", install)
    completion = {"schema_version": "cn-daily-eod-completion.v2", "synthetic": False}
    completion_ref = _put(root, "controlled/completion.json", completion)
    docs = {
        "completion": (completion_ref, completion),
        "ledger": (ledger_ref, ledger),
        "release_install_input": (install_ref, install),
    }
    for role in ("materialization", "handoff", "recipe", "loop_context", "logical_claim"):
        value = {"release_repository_root": str(repository)} if role == "loop_context" else {}
        docs[role] = (_put(root, "controlled/" + role + ".json", value), value)
    snapshot = _mint_snapshot(
        workspace=str(root),
        trade_date=trade_date,
        documents=tuple(
            (role, ref["path"], ref["sha256"], (root / ref["path"]).read_bytes())
            for role, (ref, _) in docs.items()
        ),
    )
    monkeypatch.setattr(
        module,
        "inspect_recorded_completion",
        lambda **kw: {"recorded_completion": completion, "completed_handoff_snapshot": snapshot},
    )
    verified = []
    monkeypatch.setattr(
        module, "verify_archived_handoff_context", lambda value: verified.append(value)
    )
    monkeypatch.setattr(module, "_invoke_active_completion_replay", lambda **kw: (False, None))
    envelope = {
        "schema_version": module.ENVELOPE_SCHEMA,
        "native_schema_version": "cn-daily-eod-native-readback.v1",
        "completion_ref": completion_ref,
        "trade_date": trade_date,
        "native_replay_validated": True,
        "validated_nodes": sorted(EOD_NODE_IDS),
        "synthetic": False,
        "ledger": {
            **ledger,
            "ledger_ref": ledger_ref,
            "validation_scope": "NATIVE_LEDGER_DERIVATION",
            "native_business_replay_required": True,
        },
    }
    calls = []
    monkeypatch.setattr(
        module, "_child_replay", lambda **kw: (calls.append(kw) or deepcopy(envelope))
    )
    args = {"workspace": str(root), "trade_date": trade_date, "completion_ref": completion_ref}
    return args, snapshot, envelope, calls, verified


def test_standalone_uses_only_the_recorded_interpreter_and_identity(tmp_path, monkeypatch):
    args, snapshot, envelope, calls, verified = context(tmp_path, monkeypatch)
    value = module.replay_outcome_daily_evidence(**args)
    assert verified == [snapshot]
    assert calls == [
        {
            "python": "/verified/recorded/bin/python",
            "request": {
                **args,
                "release_install_ref": snapshot.reference("release_install_input"),
                "repository_root": str(tmp_path / "repository"),
            },
        }
    ]
    assert value["prospective"] is True and value["recomputed"] is False
    assert value["native_identity"]["final_commit"] == "a" * 40
    assert value["native_identity"]["operation"] == "completion_replay"


def test_matching_active_result_has_the_same_identity_and_no_child(tmp_path, monkeypatch):
    args, snapshot, envelope, calls, verified = context(tmp_path, monkeypatch)
    child_value = module.replay_outcome_daily_evidence(**args)
    native = {key: value for key, value in envelope.items() if key != "native_schema_version"}
    native["schema_version"] = envelope["native_schema_version"]
    monkeypatch.setattr(module, "_invoke_active_completion_replay", lambda **kw: (True, native))
    active_value = module.replay_outcome_daily_evidence(**args)
    assert len(calls) == 1
    assert active_value["native_identity"] == child_value["native_identity"]


def test_distinct_archived_days_keep_their_recorded_release_identity(tmp_path, monkeypatch):
    identities = []
    for day, commit in (("20260820", "a" * 40), ("20260821", "f" * 40)):
        root = tmp_path / day
        root.mkdir()
        with monkeypatch.context() as patch:
            args, _, _, calls, _ = context(root, patch, trade_date=day, commit=commit)
            value = module.replay_outcome_daily_evidence(**args)
            assert value["trade_date"] == day and len(calls) == 1
            assert value["native_identity"]["final_commit"] == commit
            identities.append(value["native_identity"])
    assert identities[0] != identities[1]


def test_active_failure_does_not_fall_back_to_child(tmp_path, monkeypatch):
    args, _, _, calls, _ = context(tmp_path, monkeypatch)

    def fail(**kwargs):
        raise ContractError("CONTROLLED_ACTIVE_FAILURE")

    monkeypatch.setattr(module, "_invoke_active_completion_replay", fail)
    with pytest.raises(ContractError, match="CONTROLLED_ACTIVE_FAILURE"):
        module.replay_outcome_daily_evidence(**args)
    assert calls == []


@pytest.mark.parametrize(
    "fault", ["ref", "date", "nodes", "boolean", "ledger_sha", "classification", "extra"]
)
def test_child_metadata_cannot_bypass_native_contract(tmp_path, monkeypatch, fault):
    args, _, envelope, _, _ = context(tmp_path, monkeypatch)
    if fault == "ref":
        envelope["completion_ref"] = {"path": "other.json", "sha256": "e" * 64}
    elif fault == "date":
        envelope["trade_date"] = "20260821"
    elif fault == "nodes":
        envelope["validated_nodes"].pop()
    elif fault == "boolean":
        envelope["native_replay_validated"] = 1
    elif fault == "ledger_sha":
        envelope["ledger"]["ledger_ref"] = {"path": "other.json", "sha256": "e" * 64}
    elif fault == "classification":
        envelope["ledger"]["classification"] = "LATE_REGISTERED"
    else:
        envelope["allow_override"] = True
    with pytest.raises(ContractError):
        module.replay_outcome_daily_evidence(**args)


def test_old_result_recomputed_comes_only_from_verified_bound_ledger(tmp_path, monkeypatch):
    args, _, envelope, _, _ = context(tmp_path, monkeypatch)
    envelope["ledger"].pop("recomputed")
    assert module.replay_outcome_daily_evidence(**args)["recomputed"] is False


def test_snapshot_drift_after_native_call_is_rejected(tmp_path, monkeypatch):
    args, snapshot, envelope, _, _ = context(tmp_path, monkeypatch)

    def child(**kwargs):
        (tmp_path / snapshot.reference("ledger")["path"]).write_bytes(b"changed")
        return envelope

    monkeypatch.setattr(module, "_child_replay", child)
    with pytest.raises(ContractError, match="COMPLETED_SNAPSHOT_CHANGED"):
        module.replay_outcome_daily_evidence(**args)


def test_recorded_v1_is_excluded_without_inventing_native_release(tmp_path, monkeypatch):
    args, _, _, calls, verified = context(tmp_path, monkeypatch)
    monkeypatch.setattr(
        module,
        "inspect_recorded_completion",
        lambda **kw: {
            "recorded_completion": {
                "schema_version": "cn-daily-eod-completion.v1",
                "synthetic": False,
            }
        },
    )
    value = module.replay_outcome_daily_evidence(**args)
    assert value["classification"] == "UNKNOWN_LEGACY"
    assert value["validation_scope"] == "RECORDED_EOD_WITHOUT_NATIVE_RELEASE"
    assert value["native_identity"] is None and calls == [] and verified == []


def test_real_child_environment_is_isolated_and_input_is_canonical(tmp_path, monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "never-copy")
    monkeypatch.setenv("OPENAI_API_KEY", "never-copy")
    monkeypatch.setenv("HTTPS_PROXY", "never-copy")
    monkeypatch.setattr(
        module,
        "RUNNER_SOURCE",
        """
import json, os, sys
request=json.load(sys.stdin)
sys.stdout.write(json.dumps({"keys":sorted(request),"environment":dict(os.environ)},sort_keys=True,separators=(",",":")))
""",
    )
    result = module._child_replay(python=sys.executable, request={"marker": "literal"})
    assert result["keys"] == ["marker"]
    env = result["environment"]
    assert not {"TUSHARE_TOKEN", "OPENAI_API_KEY", "HTTPS_PROXY"} & set(env)
    assert env["PYTHONPATH"] == "" and env["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.parametrize("fault", ["timeout", "stdout", "stderr", "malformed"])
def test_real_child_bounds_and_owned_process_cleanup(tmp_path, monkeypatch, fault):
    if fault == "timeout":
        monkeypatch.setattr(module, "TIMEOUT_SECONDS", 0.1)
        program = "import time; time.sleep(10)"
    elif fault == "stdout":
        program = (
            f"import sys,time; sys.stdout.write('x'*{module.MAX_OUTPUT + 1}); "
            "sys.stdout.flush(); time.sleep(10)"
        )
    elif fault == "stderr":
        program = (
            f"import sys,time; sys.stderr.write('x'*{module.MAX_ERROR + 1}); "
            "sys.stderr.flush(); time.sleep(10)"
        )
    else:
        program = "print('not canonical JSON')"
    monkeypatch.setattr(module, "RUNNER_SOURCE", program)
    original = module.subprocess.Popen
    processes = []

    def spawn(*args, **kwargs):
        assert args[0][:3] == [sys.executable, "-I", "-B"]
        assert kwargs["start_new_session"] is True
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    with pytest.raises(ValueError):
        module._child_replay(python=sys.executable, request={})
    assert len(processes) == 1 and processes[0].poll() is not None


@pytest.mark.parametrize(
    "action", ["network", "source_write", "relative_source_delete", "relative_temp_cleanup"]
)
def test_literal_runner_guard_protects_sources_and_allows_temp_cleanup(
    tmp_path, monkeypatch, action
):
    source = tmp_path / "retained.json"
    source.write_bytes(b"retained")
    prefix = module.RUNNER_SOURCE.split('reference = request["release_install_ref"]', 1)[0]
    operation = {
        "network": "import socket; socket.socket()",
        "source_write": (
            "pathlib.Path(request['workspace'],'retained.json').write_bytes(b'changed')"
        ),
        "relative_source_delete": (
            "fd=os.open(request['workspace'],os.O_RDONLY); " "os.unlink('retained.json',dir_fd=fd)"
        ),
        "relative_temp_cleanup": (
            "(temporary/'cleanup').write_bytes(b'ok'); fd=os.open(temporary,os.O_RDONLY); "
            "os.unlink('cleanup',dir_fd=fd); os.close(fd)"
        ),
    }[action]
    suffix = (
        "\nblocked=False\ntry:\n    "
        + operation
        + "\nexcept RuntimeError:\n    blocked=True\n"
        + "sys.stdout.buffer.write(canonical_json_bytes({'blocked':blocked}))\n"
    )
    monkeypatch.setattr(module, "RUNNER_SOURCE", prefix + suffix)
    value = module._child_replay(
        python=sys.executable,
        request={
            "workspace": str(tmp_path),
            "trade_date": "20260820",
            "completion_ref": {},
            "release_install_ref": {},
            "repository_root": str(tmp_path),
        },
    )
    assert value["blocked"] is (action != "relative_temp_cleanup")
    assert source.read_bytes() == b"retained"
