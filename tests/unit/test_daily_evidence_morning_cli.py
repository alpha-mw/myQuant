"""Public v1/v2 routing and canonical failure exits; native replay tested separately."""

from contextlib import contextmanager
import hashlib
import json
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.cli.main import main
from quant_investor.cli import morning_v2
from quant_investor.operations.daily_contract import ContractError


def args(root, request):
    raw = canonical_json_bytes(request)
    path = root / "request.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    return [
        "research",
        "morning-strategy",
        "--workspace-root",
        str(root),
        "--request",
        "request.json",
        "--expected-request-sha256",
        hashlib.sha256(raw).hexdigest(),
    ]


def request():
    ref = {"path": "input.json", "sha256": "a" * 64}
    return {
        "schema_version": "morning-strategy-request.v2",
        "action": "REPLAY",
        "run_date": "20260904",
        "previous_completion_ref": {
            "path": "results/operations/daily_production/CN/20260903/completion.v1.json",
            "sha256": "a" * 64,
        },
        "quote_capture_ref": ref,
        "quote_raw_ref": ref,
        "owner_policy_ref": ref,
        "output_ref": None,
    }


def group(root):
    raw = canonical_json_bytes({})
    p = root / "release.json"
    p.write_bytes(raw)
    p.chmod(0o600)
    return [
        "--release-repository-root",
        str(root),
        "--release-install-input",
        "release.json",
        "--expected-release-install-input-sha256",
        hashlib.sha256(raw).hexdigest(),
    ]


def output(capsys):
    stdout = capsys.readouterr().out
    assert len(stdout.splitlines()) == 1
    value = json.loads(stdout)
    assert (
        stdout
        == json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    return value


@pytest.mark.parametrize("suffix", [[], ["--release-repository-root", "/unused"]])
def test_v2_incomplete_group_rejected_without_legacy_fallback(
    tmp_path, monkeypatch, capsys, suffix
):
    import quant_investor.intelligence as intelligence

    monkeypatch.setattr(
        intelligence, "run_morning_strategy", lambda **kw: pytest.fail("v1 fallback")
    )
    with pytest.raises(SystemExit) as exc:
        main(args(tmp_path, request()) + suffix)
    assert exc.value.code == 2
    assert output(capsys)["blocker_code"] == "MORNING_V2_RELEASE_ARGUMENTS_REQUIRED"


def test_v1_rejects_group_and_preserves_original_handler(tmp_path, monkeypatch, capsys):
    import quant_investor.intelligence as intelligence

    calls = []
    monkeypatch.setattr(
        intelligence, "run_morning_strategy", lambda **kw: (calls.append(kw) or {"legacy": True})
    )
    command = args(tmp_path, {"action": "PREFLIGHT"})
    main(command)
    assert output(capsys) == {"legacy": True}
    with pytest.raises(SystemExit) as exc:
        main(command + group(tmp_path))
    assert exc.value.code == 2
    assert output(capsys)["blocker_code"] == "MORNING_V1_RELEASE_ARGUMENTS_FORBIDDEN"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kind,exit_code,code",
    [
        ("business", 2, "MORNING_V2_EVIDENCE_REJECTED"),
        ("runtime", 2, "MORNING_V2_RUNTIME_REJECTED"),
        ("unexpected", 3, "INTERNAL_ERROR"),
        ("malformed_result", 3, "INTERNAL_ERROR"),
    ],
)
def test_v2_failure_boundary_and_context_cleanup(
    tmp_path, monkeypatch, capsys, kind, exit_code, code
):
    calls = []

    def consumer(**kwargs):
        calls.append("consumer")
        if kind == "business":
            raise ContractError("private detail /do/not/expose")
        if kind == "unexpected":
            raise RuntimeError("private traceback")
        return {"arbitrary": "success"}

    @contextmanager
    def bridge(**kwargs):
        calls.append("enter")
        if kind == "runtime":
            raise ContractError("runtime invalid")
        try:
            yield {"morning": consumer}
        finally:
            calls.append("cleanup")

    monkeypatch.setattr(morning_v2, "verified_native_context", bridge)
    with pytest.raises(SystemExit) as exc:
        main(args(tmp_path, request()) + group(tmp_path))
    assert exc.value.code == exit_code
    assert output(capsys)["blocker_code"] == code
    assert calls[-1] == ("enter" if kind == "runtime" else "cleanup")


def test_unknown_schema_cannot_select_v1(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args(tmp_path, {"schema_version": "morning-strategy-request.v99"}))
    assert exc.value.code == 2
    assert output(capsys)["blocker_code"] == "MORNING_REQUEST_SCHEMA_UNSUPPORTED"


@pytest.mark.parametrize("reject_readback", [False, True])
def test_cli_seal_binds_original_request_and_requires_live_receipt_replay(
    tmp_path, monkeypatch, capsys, reject_readback
):
    from test_daily_evidence_morning_seal import fixture, module as sealer

    reference, _, _ = fixture(tmp_path, monkeypatch)
    document = json.loads((tmp_path / reference["path"]).read_bytes())
    events = []

    def seal(**kwargs):
        events.append("seal")
        assert kwargs["request_ref"] == reference
        return sealer.seal_morning_consumer(**kwargs)

    def readback(**kwargs):
        events.append("readback")
        result = sealer.read_morning_consumer_receipt(**kwargs)
        if reject_readback:
            result["command_status"] = "REPLAY_VERIFIED"
        return result

    @contextmanager
    def bridge(**kwargs):
        events.append("enter")
        try:
            yield {"morning_seal": seal, "morning_receipt": readback}
        finally:
            events.append("cleanup")

    monkeypatch.setattr(morning_v2, "verified_native_context", bridge)
    command = args(tmp_path, document) + group(tmp_path)
    if reject_readback:
        with pytest.raises(SystemExit) as caught:
            main(command)
        assert caught.value.code == 3
        assert output(capsys)["status"] != "COMPLETE"
    else:
        main(command)
        result = output(capsys)
        assert result["command_status"] == "PUBLISHED"
        assert result["receipt"]["request_ref"] == reference
        main(command)
        assert output(capsys)["command_status"] == "NO_ACTION"
    assert events[:4] == ["enter", "seal", "readback", "cleanup"]
