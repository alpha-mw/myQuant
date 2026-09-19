"""V2 recommendation publication with controlled native evidence boundaries."""

from datetime import datetime, timezone
import hashlib
import importlib
import json
import sys
from pathlib import Path
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.morning_cutover_contract import cutover_path
from test_daily_evidence_morning_receipt import receipt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
module = importlib.import_module("scripts.daily_morning_cutover")


def fixture(root, monkeypatch, state, count, fault=None, version="v2"):
    def put(path, value):
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        parent = p.parent
        while parent != root:
            parent.chmod(0o700)
            parent = parent.parent
        raw = canonical_json_bytes(value)
        p.write_bytes(raw)
        p.chmod(0o600)
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}

    ledger = put(
        "ledger.json",
        {
            "schema_version": "cn-daily-evidence-ledger.v1",
            "trade_date": "20260827",
            "classification": "CONTEMPORANEOUS",
            "prospective": True,
            "synthetic": fault == "synthetic",
            "recomputed": False,
            "authority": dict(FALSE_AUTHORITY),
        },
    )
    eod = put(
        "results/operations/daily_production/CN/20260827/completion.v1.json",
        {
            "prospective_ledger_ref": ledger,
            "native_validation_completed_at": "2026-08-27T13:00:00Z",
        },
    )
    install = put("release.json", {})
    refs = []
    for day in (["20260826", "20260827"] if count == 2 else ["20260827"] if count == 1 else []):
        value = receipt()
        if day == "20260826":
            value = json.loads(
                json.dumps(value)
                .replace("20260826", "20260825")
                .replace("20260827", "20260826")
                .replace("2026-08-27", "2026-08-26")
            )
        if version == "v3":
            value.update(
                schema_version="morning-strategy-run.v3",
                threshold_policy_refs={
                    "trailing": {"path": "trailing.json", "sha256": "c" * 64},
                    "initial_stop": {"path": "stop.json", "sha256": "d" * 64},
                },
                threshold_review_sha256="e" * 64,
                review_summary_state="PARTIAL_RESEARCH_REVIEW",
            )
            value["output_ref"]["path"] = value["output_ref"]["path"].replace(".v2.md", ".v3.md")
        refs.append(
            put(f"results/operations/morning_strategy/CN/{day}/0945-run.{version}.json", value)
        )
    request = {
        "schema_version": "morning-strategy-cutover-request.v2",
        "target_date": "20260827",
        "daily_completion_ref": eod,
        "morning_receipts": refs,
        "current_schedule_state": state,
    }
    reference = put("request.json", request)
    events = []

    def select(**kwargs):
        events.append("eod")
        return {
            "schema_version": "cn-daily-eligible-evidence-selection.v1",
            "market": "CN",
            "trade_date": "20260827",
            "completion_ref": eod,
            "ledger_ref": ledger,
            "eligibility_scope": "LOCAL_COORDINATOR_AVAILABILITY",
            "factor_admission": False,
            "authority": dict(FALSE_AUTHORITY),
        }

    def read(**kwargs):
        events.append("morning")
        ref = kwargs["receipt_ref"]
        return {
            "schema_version": "morning-strategy-seal-result." + version,
            "command_status": "REPLAY_VERIFIED" if fault == "replay" else "RECEIPT_VERIFIED",
            "receipt_ref": ref,
            "receipt": json.loads((root / ref["path"]).read_bytes()),
        }

    monkeypatch.setattr(module, "select_eligible_daily_evidence", select)
    monkeypatch.setattr(module, "read_morning_consumer_receipt", read)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 27, 13, 30, tzinfo=timezone.utc)

    monkeypatch.setattr(module, "datetime", Clock)
    return reference, install, events


@pytest.mark.parametrize(
    "state,count,next_state",
    [
        ("EVENING_PRIMARY", 0, "DUAL_RUN"),
        ("DUAL_RUN", 0, "DUAL_RUN"),
        ("DUAL_RUN", 1, "DUAL_RUN"),
        ("DUAL_RUN", 2, "MORNING_PRIMARY"),
        ("MORNING_PRIMARY", 0, "DUAL_RUN"),
        ("MORNING_PRIMARY", 1, "MORNING_PRIMARY"),
    ],
)
def test_v2_native_evidence_drives_recommendation_only(
    tmp_path, monkeypatch, state, count, next_state
):
    ref, install, events = fixture(tmp_path, monkeypatch, state, count)

    def forbidden(*args, **kwargs):
        pytest.fail("cutover reached network or writer lock")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("quant_investor.operations.journal_storage.JournalStorage.lock", forbidden)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = module.recommend_morning_cutover(
        workspace=str(tmp_path), request_ref=ref, release_install_ref=install
    )
    record = result["receipt"]
    assert record["next_schedule_state"] == next_state
    assert record["consecutive_morning_success_count"] == count
    assert (
        record["application_performed"] is False
        and record["current_schedule_state_basis"] == "OWNER_DECLARATION"
    )
    after = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    assert set(after) - set(before) == {str(tmp_path / cutover_path("20260827"))}
    assert all(after[p][0] == raw for p, raw in before.items())
    again = module.recommend_morning_cutover(
        workspace=str(tmp_path), request_ref=ref, release_install_ref=install
    )
    assert again["command_status"] == "NO_ACTION" and again["receipt"] == record
    assert events.count("eod") == 2 and events.count("morning") == count * 2
    assert after == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("fault", ["synthetic", "replay"])
def test_invalid_native_evidence_never_publishes_cutover(tmp_path, monkeypatch, fault):
    ref, install, _ = fixture(tmp_path, monkeypatch, "DUAL_RUN", 1, fault)
    with pytest.raises(ContractError):
        module.recommend_morning_cutover(
            workspace=str(tmp_path), request_ref=ref, release_install_ref=install
        )
    assert not (tmp_path / cutover_path("20260827")).exists()


def test_changed_same_day_declaration_cannot_replace_receipt(tmp_path, monkeypatch):
    ref, install, _ = fixture(tmp_path, monkeypatch, "DUAL_RUN", 2)
    first = module.recommend_morning_cutover(
        workspace=str(tmp_path), request_ref=ref, release_install_ref=install
    )
    saved = (tmp_path / first["receipt_ref"]["path"]).read_bytes()
    path = tmp_path / ref["path"]
    request = json.loads(path.read_bytes())
    request["current_schedule_state"] = "MORNING_PRIMARY"
    raw = canonical_json_bytes(request)
    path.write_bytes(raw)
    ref = {"path": ref["path"], "sha256": hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ContractError, match="IMMUTABLE_CONFLICT"):
        module.recommend_morning_cutover(
            workspace=str(tmp_path), request_ref=ref, release_install_ref=install
        )
    assert (tmp_path / first["receipt_ref"]["path"]).read_bytes() == saved


def rewrite(root, ref, value):
    raw = canonical_json_bytes(value)
    (root / ref["path"]).write_bytes(raw)
    return {"path": ref["path"], "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.mark.parametrize(
    "fault", ["duplicate", "reversed", "extra", "bad_selector", "changed_source"]
)
def test_cutover_invalid_or_changed_inputs_cannot_publish(tmp_path, monkeypatch, fault):
    ref, install, _ = fixture(tmp_path, monkeypatch, "DUAL_RUN", 2)
    request = json.loads((tmp_path / ref["path"]).read_bytes())
    if fault == "duplicate":
        request["morning_receipts"] = [request["morning_receipts"][0]] * 2
    elif fault == "reversed":
        request["morning_receipts"].reverse()
    elif fault == "extra":
        request["scheduler_origin_verified"] = True
    if fault in {"duplicate", "reversed", "extra"}:
        ref = rewrite(tmp_path, ref, request)
    elif fault == "bad_selector":
        original = module.select_eligible_daily_evidence
        monkeypatch.setattr(
            module,
            "select_eligible_daily_evidence",
            lambda **kw: {**original(**kw), "factor_admission": True},
        )
    else:
        original = module.read_morning_consumer_receipt

        def changed(**kwargs):
            result = original(**kwargs)
            (tmp_path / "ledger.json").write_bytes(b"changed during replay")
            return result

        monkeypatch.setattr(module, "read_morning_consumer_receipt", changed)
    with pytest.raises(ContractError):
        module.recommend_morning_cutover(
            workspace=str(tmp_path), request_ref=ref, release_install_ref=install
        )
    assert not (tmp_path / cutover_path("20260827")).exists()


def test_nonconsecutive_morning_successes_do_not_promote(tmp_path, monkeypatch):
    ref, install, _ = fixture(tmp_path, monkeypatch, "DUAL_RUN", 2)
    request = json.loads((tmp_path / ref["path"]).read_bytes())
    first = request["morning_receipts"][0]
    value = json.loads((tmp_path / first["path"]).read_bytes())
    value = json.loads(
        json.dumps(value)
        .replace("20260825", "20260824")
        .replace("20260826", "20260825")
        .replace("2026-08-26", "2026-08-25")
    )
    path = "results/operations/morning_strategy/CN/20260825/0945-run.v2.json"
    (tmp_path / path).parent.mkdir(mode=0o700)
    (tmp_path / path).write_bytes(canonical_json_bytes(value))
    (tmp_path / path).chmod(0o600)
    request["morning_receipts"][0] = {
        "path": path,
        "sha256": hashlib.sha256((tmp_path / path).read_bytes()).hexdigest(),
    }
    ref = rewrite(tmp_path, ref, request)
    result = module.recommend_morning_cutover(
        workspace=str(tmp_path), request_ref=ref, release_install_ref=install
    )
    assert result["receipt"]["consecutive_morning_success_count"] == 1
    assert result["receipt"]["next_schedule_state"] == "DUAL_RUN"


def test_v3_receipts_are_verified_without_applying_cutover(tmp_path, monkeypatch):
    ref, install, events = fixture(tmp_path, monkeypatch, "DUAL_RUN", 1, version="v3")
    result = module.recommend_morning_cutover(
        workspace=str(tmp_path), request_ref=ref, release_install_ref=install
    )
    assert result["receipt"]["application_performed"] is False
    assert result["receipt"]["consecutive_morning_success_count"] == 1
    assert result["receipt"]["morning_receipts"][0]["path"].endswith("0945-run.v3.json")
    assert events == ["eod", "morning"]
