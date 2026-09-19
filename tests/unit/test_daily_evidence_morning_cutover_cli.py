"""Explicit public v2 cutover route, with controlled installed/native evidence."""

from contextlib import contextmanager
import json
import pytest
from quant_investor.cli.main import main
from quant_investor.cli import morning_cutover as cli
from test_daily_evidence_morning_cutover_v2 import fixture, module
from test_daily_evidence_morning_cli import args, output


def command(root, document):
    values = args(root, document)
    values[1] = "morning-cutover"
    return values


@pytest.mark.parametrize("corrupt", [False, True])
def test_v2_cli_requires_bound_recommendation_and_never_applies_schedule(
    tmp_path, monkeypatch, capsys, corrupt
):
    ref, install, _ = fixture(tmp_path, monkeypatch, "DUAL_RUN", 2)
    document = json.loads((tmp_path / ref["path"]).read_bytes())
    events = []

    def recommend(**kwargs):
        assert kwargs["request_ref"] == ref and kwargs["release_install_ref"] == install
        result = module.recommend_morning_cutover(**kwargs)
        if corrupt:
            result["receipt"]["application_performed"] = True
        return result

    @contextmanager
    def bridge(**kwargs):
        events.append("enter")
        try:
            yield {"morning_cutover": recommend}
        finally:
            events.append("cleanup")

    monkeypatch.setattr(cli, "verified_native_context", bridge)
    invocation = command(tmp_path, document) + [
        "--release-repository-root",
        str(tmp_path),
        "--release-install-input",
        install["path"],
        "--expected-release-install-input-sha256",
        install["sha256"],
    ]
    if corrupt:
        with pytest.raises(SystemExit) as caught:
            main(invocation)
        assert caught.value.code == 3
        assert output(capsys)["blocker_code"] == "INTERNAL_ERROR"
    else:
        main(invocation)
        value = output(capsys)
        assert value["receipt"]["next_schedule_state"] == "MORNING_PRIMARY"
        assert value["receipt"]["application_performed"] is False
        assert value["receipt"]["current_schedule_state_basis"] == "OWNER_DECLARATION"
        main(invocation)
        assert output(capsys)["command_status"] == "NO_ACTION"
    assert events[:2] == ["enter", "cleanup"]


def test_v2_cutover_requires_runtime_and_unknown_schema_never_uses_v1(
    tmp_path, monkeypatch, capsys
):
    ref, _, _ = fixture(tmp_path, monkeypatch, "EVENING_PRIMARY", 0)
    document = json.loads((tmp_path / ref["path"]).read_bytes())
    monkeypatch.setattr(
        "quant_investor.intelligence.evaluate_morning_cutover",
        lambda **kwargs: pytest.fail("v1 fallback"),
    )
    with pytest.raises(SystemExit) as caught:
        main(command(tmp_path, document))
    assert caught.value.code == 2
    assert output(capsys)["blocker_code"] == "MORNING_CUTOVER_RELEASE_ARGUMENTS_REQUIRED"
    document["schema_version"] = "morning-strategy-cutover-request.v99"
    with pytest.raises(SystemExit) as caught:
        main(command(tmp_path, document))
    assert caught.value.code == 2
    assert output(capsys)["blocker_code"] == "MORNING_CUTOVER_SCHEMA_UNSUPPORTED"


def test_v1_cutover_route_unchanged_and_rejects_runtime_group(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "quant_investor.intelligence.evaluate_morning_cutover", lambda **kwargs: {"legacy": True}
    )
    invocation = command(tmp_path, {})
    main(invocation)
    assert output(capsys) == {"legacy": True}
    with pytest.raises(SystemExit) as caught:
        main(invocation + ["--release-repository-root", str(tmp_path)])
    assert caught.value.code == 2
    assert output(capsys)["blocker_code"] == "MORNING_CUTOVER_V1_RELEASE_ARGUMENTS_FORBIDDEN"
