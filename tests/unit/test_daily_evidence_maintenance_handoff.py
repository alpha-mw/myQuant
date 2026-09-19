"""Real immutable handoff storage with controlled native validation seams."""

from types import SimpleNamespace
import hashlib
import json
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.system.store import object_ref_for_artifact
from quant_investor.operations import maintenance_handoff as producer
from quant_investor.operations.daily_contract import ContractError
from test_daily_evidence_production_request import request, recipe
from test_unified_factor_manifest import _release


def context(root, monkeypatch, policy=None):
    def put(name, value):
        raw = canonical_json_bytes(value)
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        p.chmod(0o600)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    release = _release()
    release_ref = put("release.json", release)
    install = put("install.json", {"deployed_release": release})
    ctx = put("context.json", {"release_install_input_ref": install})
    rec = recipe()
    if policy is not None:
        rec["policy_refs"]["prospective"] = put("prediction-policy.json", policy)
    rec.update(release_ref=release_ref, release_install_ref=install, factor_loop_context_ref=ctx)
    recipe_ref = put("recipe.json", rec)
    req = request()
    req.update(release_install_ref=install, recipe_ref=recipe_ref)
    req_ref = put("request.json", req)
    claim = put("maintenance/logical_tasks/20260904-2020-execute/claim.json", {"fixture": "claim"})
    started = put(
        "maintenance/attempts/one/started.json",
        {"started_at": "2026-09-04T07:00:00Z", "state": "STARTED", "mode": "execute"},
    )
    from test_daily_evidence_requested_session import capture

    calendar_capture = capture("2026-09-04T20:20:00+08:00")
    raw_path = root / "calendar-raw.json"
    raw_path.write_bytes(calendar_capture.raw_response_bytes)
    raw_path.chmod(0o600)
    rawcal = {
        "path": "calendar-raw.json",
        "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    calendar = put(
        "calendar.json",
        {**calendar_capture.receipt, "raw_response_path": str(root / rawcal["path"])},
    )
    market = put("market.json", {"fixture": "market"})
    snapshot = put("snapshot.json", {"fixture": "snapshot"})
    factor = put("factor.json", {"fixture": "factor"})
    terminal = put("terminal.json", {"finished_at": "2026-09-04T07:01:00Z"})
    core = put(
        "maintenance/attempts/one/core-completion.json",
        {
            "schema_version": "cn-daily-maintenance-core.v1",
            "producer": "quant_investor.market.daily_maintenance",
            "scope": "FACTOR_INPUTS_ONLY",
            "other_authority": "NONE",
            "target_date": "20260904",
            "mode": "execute",
            "status": "CORE_COMPLETE",
            "maintenance_status": "IN_PROGRESS",
            "blockers": [],
            "logical_claim_ref": claim,
            "started_ref": started,
            "close_session_receipt_ref": calendar,
            "stage_results": [
                {
                    "stage": "MARKET",
                    "evidence": {
                        "pointer_path": str(root / market["path"]),
                        "pointer_sha256": market["sha256"],
                        "snapshot_manifest_path": str(root / snapshot["path"]),
                        "snapshot_manifest_sha256": snapshot["sha256"],
                    },
                }
            ],
        },
    )
    core_path = root / core["path"]
    checkpoint = json.loads(core_path.read_text())
    market_row = checkpoint["stage_results"][0]
    checkpoint["stage_results"] = [{"stage": "PIT"}, market_row, {"stage": "HISTORY"}]
    for row in checkpoint["stage_results"]:
        row.update(status="READY", blockers=[])
    checkpoint["stage_refs"] = [
        put(
            "maintenance/attempts/one/stage-" + row["stage"] + ".json",
            {"state": "STAGE_COMPLETED", "result": row},
        )
        for row in checkpoint["stage_results"]
    ]
    core = put(core["path"], checkpoint)
    handoff = put("core-handoff.json", {"fixture": "handoff"})
    monkeypatch.setattr(
        producer,
        "read_factor_loop_context",
        lambda **kw: (
            {"release_install_input_ref": install},
            {"release_ref": object_ref_for_artifact(release)},
        ),
    )
    monkeypatch.setattr(
        producer, "validate_daily_maintenance_receipt", lambda **kw: {"target_date": "20260904"}
    )
    monkeypatch.setattr(
        producer, "inspect_handoff_slot_claim", lambda **kw: {"maintenance_run_root": "maintenance"}
    )
    monkeypatch.setattr(
        producer,
        "inspect_core_handoff",
        lambda **kw: {"factor_pointer_ref": factor, "node_terminal_refs": {"factor": terminal}},
    )
    monkeypatch.setattr(
        producer,
        "FactorProductionStore",
        lambda workspace: SimpleNamespace(
            inspect_recorded_research_inputs=lambda **kw: {
                "market_pointer_sha256": market["sha256"],
                "market_manifest_sha256": snapshot["sha256"],
                "factor_generation": {
                    "payload": {"deployed_release_ref": object_ref_for_artifact(release)}
                },
            }
        ),
    )
    return (
        dict(
            workspace=str(root),
            request_ref=req_ref,
            state={"core_checkpoint_ref": core, "core_handoff_ref": handoff},
        ),
        market,
    )


def test_current_handoff_rejects_rehashed_wrong_predecessor_before_write(tmp_path, monkeypatch):
    from _public_catchup_fixture import put

    args, _ = context(tmp_path, monkeypatch)
    request_ref = args["request_ref"]
    req = json.loads((tmp_path / request_ref["path"]).read_bytes())
    rec = json.loads((tmp_path / req["recipe_ref"]["path"]).read_bytes())
    rec["previous_completion_ref"][
        "path"
    ] = "results/operations/daily_production/CN/20260902/completion.v1.json"
    req["recipe_ref"] = put(tmp_path, req["recipe_ref"]["path"], rec)
    args["request_ref"] = put(tmp_path, request_ref["path"], req)
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    with pytest.raises(ContractError, match="MAINTENANCE_HANDOFF_CURRENT_PREDECESSOR_INVALID"):
        producer.publish_maintenance_handoff(**args)
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


def test_publish_retains_inputs_and_repeats_identical_bytes(tmp_path, monkeypatch):
    args, market = context(tmp_path, monkeypatch)
    ref = producer.publish_maintenance_handoff(**args)
    data = json.loads((tmp_path / ref["path"]).read_text())
    assert data["schema_version"] == producer.SCHEMA
    assert data["prospective_policy_ref"] is None
    assert data["market_pointer_ref"]["path"] != market["path"]
    assert data["market_pointer_ref"]["sha256"] == market["sha256"]
    assert (tmp_path / data["market_pointer_ref"]["path"]).read_bytes() == (
        tmp_path / market["path"]
    ).read_bytes()
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    assert producer.publish_maintenance_handoff(**args) == ref
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    assert all(v is False for v in data["authority"].values())


def test_changed_producer_source_cannot_replace_existing_handoff(tmp_path, monkeypatch):
    args, market = context(tmp_path, monkeypatch)
    ref = producer.publish_maintenance_handoff(**args)
    original = (tmp_path / ref["path"]).read_bytes()
    (tmp_path / market["path"]).write_bytes(b"changed")
    with pytest.raises(ContractError, match="SOURCE_SHA_MISMATCH"):
        producer.publish_maintenance_handoff(**args)
    assert (tmp_path / ref["path"]).read_bytes() == original


def test_cannot_retrofit_new_handoff_after_completed_eod(tmp_path, monkeypatch):
    from quant_investor.operations.daily_journal import DailyJournal

    args, _ = context(tmp_path, monkeypatch)
    journal = DailyJournal(str(tmp_path), "20260904")
    journal.storage.write(str(journal.root / "completion.v1.json"), b"{}")
    with pytest.raises(ContractError, match="EOD_ALREADY_COMPLETED"):
        producer.publish_maintenance_handoff(**args)
    assert not (tmp_path / journal.root / "executions").exists()


def test_old_core_without_claim_is_blocked_without_publication(tmp_path, monkeypatch):
    args, _ = context(tmp_path, monkeypatch)
    ref = args["state"]["core_checkpoint_ref"]
    path = tmp_path / ref["path"]
    value = json.loads(path.read_text())
    value.pop("logical_claim_ref")
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    ref["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ContractError, match="NATIVE_REFS_MISSING"):
        producer.publish_maintenance_handoff(**args)
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("fault", ["historical_v2", "unknown_version", "v1_historical_field"])
def test_unsupported_core_cannot_publish_unreadable_handoff(tmp_path, monkeypatch, fault):
    args, _ = context(tmp_path, monkeypatch)
    ref = args["state"]["core_checkpoint_ref"]
    path = tmp_path / ref["path"]
    value = json.loads(path.read_bytes())
    if fault == "v1_historical_field":
        value["historical_session_ref"] = None
    else:
        value["schema_version"] = (
            "cn-daily-maintenance-core.v2" if fault == "historical_v2" else "unknown"
        )
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    ref["sha256"] = hashlib.sha256(raw).hexdigest()
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    # Native support alone cannot expand the handoff's own supported version set.
    monkeypatch.setattr(
        producer,
        "validate_daily_maintenance_receipt",
        lambda **kw: pytest.fail("unsupported core reached native validator"),
    )
    with pytest.raises(ContractError, match="MAINTENANCE_HANDOFF_CORE_CONTRACT_INVALID"):
        producer.publish_maintenance_handoff(**args)
    assert not (tmp_path / "results").exists()
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


def test_readback_rejects_historical_field_in_ordinary_core(tmp_path, monkeypatch):
    args, _ = context(tmp_path, monkeypatch)
    ref = producer.publish_maintenance_handoff(**args)
    handoff_path = tmp_path / ref["path"]
    handoff = json.loads(handoff_path.read_bytes())
    core_ref = handoff["maintenance_core_ref"]
    core_path = tmp_path / core_ref["path"]
    checkpoint = json.loads(core_path.read_bytes())
    checkpoint["historical_session_ref"] = None
    core_raw = canonical_json_bytes(checkpoint)
    core_path.write_bytes(core_raw)
    core_ref["sha256"] = hashlib.sha256(core_raw).hexdigest()
    handoff_raw = canonical_json_bytes(handoff)
    handoff_path.write_bytes(handoff_raw)
    ref["sha256"] = hashlib.sha256(handoff_raw).hexdigest()
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ContractError, match="MAINTENANCE_HANDOFF_CORE_CONTRACT_INVALID"):
        producer.read_maintenance_handoff(workspace=str(tmp_path), handoff_ref=ref)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_readback_uses_retained_request_recipe_and_market_after_head_change(tmp_path, monkeypatch):
    from quant_investor.market import close_session_authority

    args, market = context(tmp_path, monkeypatch)
    ref = producer.publish_maintenance_handoff(**args)
    monkeypatch.setattr(
        close_session_authority,
        "replay_close_session_authority",
        lambda *a: SimpleNamespace(receipt={"target_trade_date": "20260904"}),
    )
    monkeypatch.setattr(
        producer,
        "validate_daily_maintenance_receipt",
        lambda **kw: pytest.fail("current-head maintenance validator called"),
    )
    (tmp_path / market["path"]).write_bytes(b"new current head")
    (tmp_path / "request.json").unlink()
    (tmp_path / "recipe.json").unlink()
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    value = producer.read_maintenance_handoff(workspace=str(tmp_path), handoff_ref=ref)
    assert value["handoff_ref"] == ref
    assert value["execution_authorized"] is False and value["native_replay_required"] is True
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    retained = tmp_path / value["handoff"]["market_pointer_ref"]["path"]
    retained.write_bytes(b"tampered")
    with pytest.raises(ContractError, match="READBACK_SHA_MISMATCH"):
        producer.read_maintenance_handoff(workspace=str(tmp_path), handoff_ref=ref)


def test_v2_retains_policy_and_reads_it_without_original(tmp_path, monkeypatch):
    from quant_investor.operations.daily_contract import GRAPH_SHA256
    from quant_investor.market import close_session_authority

    policy = dict(
        schema_version="cn-daily-prediction-policy.v1",
        trade_date="20260904",
        graph_sha256=GRAPH_SHA256,
        session_rule="CN_SSE_SZSE_OPEN",
        prediction_deadline="2026-09-04T15:00:00Z",
    )
    args, _ = context(tmp_path, monkeypatch, policy=policy)
    original = (tmp_path / "prediction-policy.json").read_bytes()
    ref = producer.publish_maintenance_handoff(**args)
    value = json.loads((tmp_path / ref["path"]).read_bytes())
    selected = value["prospective_policy_ref"]
    assert selected["sha256"] == hashlib.sha256(original).hexdigest()
    assert (tmp_path / selected["path"]).read_bytes() == original
    assert producer.publish_maintenance_handoff(**args) == ref
    (tmp_path / "prediction-policy.json").unlink()
    monkeypatch.setattr(
        close_session_authority,
        "replay_close_session_authority",
        lambda *a: SimpleNamespace(receipt={"target_trade_date": "20260904"}),
    )
    readback = producer.read_maintenance_handoff(workspace=str(tmp_path), handoff_ref=ref)
    assert readback["handoff"]["prospective_policy_ref"] == selected
    # A late policy is recorded as late custody; the handoff makes no admission claim.
    assert readback["execution_authorized"] is False
    (tmp_path / selected["path"]).write_bytes(b"tampered retained policy")
    with pytest.raises(ContractError, match="READBACK_SHA_MISMATCH"):
        producer.read_maintenance_handoff(workspace=str(tmp_path), handoff_ref=ref)


def test_legacy_handoff_replays_but_cannot_be_upgraded(tmp_path, monkeypatch):
    from quant_investor.market import close_session_authority

    args, _ = context(tmp_path, monkeypatch)
    ref = producer.publish_maintenance_handoff(**args)
    path = tmp_path / ref["path"]
    value = json.loads(path.read_bytes())
    value["schema_version"] = producer.LEGACY_SCHEMA
    value.pop("prospective_policy_ref")
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    legacy_ref = {"path": ref["path"], "sha256": hashlib.sha256(raw).hexdigest()}
    monkeypatch.setattr(
        close_session_authority,
        "replay_close_session_authority",
        lambda *a: SimpleNamespace(receipt={"target_trade_date": "20260904"}),
    )
    assert (
        producer.read_maintenance_handoff(workspace=str(tmp_path), handoff_ref=legacy_ref)[
            "handoff"
        ]
        == value
    )
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ContractError, match="LEGACY_PATH_OCCUPIED"):
        producer.publish_maintenance_handoff(**args)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_malformed_selected_policy_cannot_publish_handoff(tmp_path, monkeypatch):
    args, _ = context(tmp_path, monkeypatch, policy={"prediction_deadline": "2026-09-04T15:00:00Z"})
    with pytest.raises(ContractError, match="PREDICTION_POLICY_CONTRACT_INVALID"):
        producer.publish_maintenance_handoff(**args)
    assert not (tmp_path / "results").exists()


def test_v2_policy_nullability_must_match_retained_recipe(tmp_path, monkeypatch):
    args, _ = context(tmp_path, monkeypatch)
    ref = producer.publish_maintenance_handoff(**args)
    path = tmp_path / ref["path"]
    value = json.loads(path.read_bytes())
    value["prospective_policy_ref"] = {"path": "invented.json", "sha256": "a" * 64}
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    changed = {"path": ref["path"], "sha256": hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ContractError, match="POLICY_BINDING_INVALID"):
        producer.read_maintenance_handoff(workspace=str(tmp_path), handoff_ref=changed)


def test_recorded_reader_rejects_caller_refs_instead_of_capability(tmp_path):
    with pytest.raises(ContractError, match="CAPABILITY_REQUIRED"):
        producer.read_recorded_maintenance_handoff({"path": "handoff.json", "sha256": "a" * 64})


def test_live_reader_never_falls_back_to_archived_install(tmp_path, monkeypatch):
    args, _ = context(tmp_path, monkeypatch)
    ref = producer.publish_maintenance_handoff(**args)

    def running_required(**kw):
        raise ValueError("RUNNING_INSTALL_REQUIRED")

    monkeypatch.setattr(producer, "read_factor_loop_context", running_required)
    import quant_investor.operations.archived_handoff_context as archived

    monkeypatch.setattr(
        archived,
        "verify_archived_handoff_context",
        lambda *a: pytest.fail("live reader selected archive path"),
    )
    with pytest.raises(ValueError, match="RUNNING_INSTALL_REQUIRED"):
        producer.read_maintenance_handoff(workspace=str(tmp_path), handoff_ref=ref)
