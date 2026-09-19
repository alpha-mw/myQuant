"""Registered launch v3 and typed expiry details through the public CLI wire."""

from contextlib import contextmanager
from copy import deepcopy
import json

import pytest

from _public_catchup_fixture import put
from _daily_preparation_fixture import snapshot
from test_registered_automatic_recovery import fixture, inspect
from quant_investor.cli import daily_launch as cli
from quant_investor.cli.daily_production import _evidence_error
from quant_investor.cli.output import CommandError
from quant_investor.operations.automatic_catchup_contract import AutomaticCatchupError
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_launch_contract import validate_launch_inspection
from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage
from scripts import daily_production, daily_launch_inspection


def test_cli_recover_uses_committed_scope_and_emits_expired_completed_refs(
    tmp_path, monkeypatch, capsys
):
    case = fixture(tmp_path, monkeypatch)
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        return daily_production.dispatch_daily_request(**kwargs, synthetic=True)

    @contextmanager
    def installed(**kwargs):
        yield {
            "daily_launch_inspection": lambda **kw: daily_launch_inspection.inspect_daily_launch(
                **kw, synthetic=True
            ),
            "daily_close": execute,
        }

    monkeypatch.setattr(cli, "verified_native_context", installed)
    ref, install = case["current"]["auto_request_ref"], case["config"]["release_install_ref"]
    args = [
        "--workspace-root",
        str(tmp_path),
        "--request",
        ref["path"],
        "--expected-request-sha256",
        ref["sha256"],
        "--release-repository-root",
        str(tmp_path),
        "--release-install-input",
        install["path"],
        "--expected-release-install-input-sha256",
        install["sha256"],
    ]
    capsys.readouterr()
    before = snapshot(tmp_path)
    assert cli.main(args + ["--mode", "inspect"]) == 10
    raw = capsys.readouterr().out.encode()
    assert json.loads(raw)["recovery_scope"] == "COMMITTED_DAG_RECOVERY"
    assert snapshot(tmp_path) == before and calls == []
    saved = put(
        tmp_path,
        "data/private/cn_daily_maintenance/launcher_attempts/"
        "slot-2020-20260826T122000Z-77/inspection.stdout.json",
        raw,
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            args
            + [
                "--mode",
                "recover",
                "--inspection",
                saved["path"],
                "--expected-inspection-sha256",
                saved["sha256"],
            ]
        )
    assert stopped.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["blocker_code"] == "AUTO_PUBLICATION_EXPIRED"
    assert result["status"] == "BLOCKED"
    assert [row["trade_date"] for row in result["completed_eod_refs"]] == ["20260825"]
    assert calls[0]["committed_recovery_only"] is True and "no_producers" not in calls[0]
    assert AutomaticRunStorage(str(tmp_path)).pending()["state"] == "IDLE"


@pytest.mark.parametrize("fault", ["extra", "scope", "mode"])
def test_launch_v3_scope_cannot_be_widened(tmp_path, monkeypatch, fault):
    case = fixture(tmp_path, monkeypatch)
    value = deepcopy(inspect(tmp_path, case))
    if fault == "extra":
        value["allow_live"] = True
    elif fault == "scope":
        value["recovery_scope"] = "FRESH_PRODUCERS"
    else:
        value["mode"] = "PRODUCER_REQUIRED"
    with pytest.raises(ContractError):
        validate_launch_inspection(
            value,
            request_ref=case["current"]["auto_request_ref"],
            release_install_ref=case["config"]["release_install_ref"],
        )


@pytest.mark.parametrize(
    "fault", [None, "wrong_path", "wrong_date", "duplicate", "unsorted", "authority", "wrong_code"]
)
def test_completed_eod_error_fields_are_exact_and_non_authorizing(fault):
    rows = [
        {
            "trade_date": day,
            "completion_ref": {
                "path": f"results/operations/daily_production/CN/{day}/completion.v1.json",
                "sha256": "a" * 64,
            },
        }
        for day in ("20260824", "20260825")
    ]
    code = "AUTO_PUBLICATION_EXPIRED"
    if fault == "wrong_path":
        rows[0]["completion_ref"]["path"] = "completion.v1.json"
    elif fault == "wrong_date":
        rows[0]["trade_date"] = "20260821"
    elif fault == "duplicate":
        rows.append(rows[-1])
    elif fault == "unsorted":
        rows.reverse()
    elif fault == "authority":
        rows[0]["execution_authorized"] = True
    elif fault == "wrong_code":
        code = "AUTO_PENDING_RECOVERY_UNCONFIRMED"
    if fault is None:
        error = AutomaticCatchupError(code, completed_eod_refs=rows)
        with pytest.raises(CommandError) as emitted:
            _evidence_error(error, automatic=True)
        assert emitted.value.fields == {"completed_eod_refs": rows}
    else:
        with pytest.raises(ContractError):
            AutomaticCatchupError(code, completed_eod_refs=rows)
