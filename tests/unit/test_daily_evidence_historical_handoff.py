"""Real handoff/proof storage and Calendar replay; unrelated native boundaries controlled."""

import json
from pathlib import Path

import pytest

from quant_investor.operations import maintenance_handoff as module
from quant_investor.operations.maintenance_handoff_contract import handoff_task_date
from quant_investor.operations.daily_contract import ContractError
from quant_investor.market.historical_session import build_historical_session, CORE_SCHEMA, FILENAME
from test_daily_evidence_maintenance_handoff import context as ordinary_context
from test_daily_evidence_requested_session import capture
from test_unified_factor_production_rollover import _write
from quant_investor.operations.completed_handoff_snapshot import _mint_snapshot
from quant_investor.operations import archived_handoff_context as archive
from quant_investor.system.store import object_ref_for_artifact
import hashlib
from contextlib import contextmanager


def context(root, monkeypatch, *, routed=False):
    args, _ = ordinary_context(root, monkeypatch)
    core_ref = args["state"]["core_checkpoint_ref"]
    path = root / core_ref["path"]
    core = json.loads(path.read_bytes())
    attempt = path.parent
    started_path = root / core["started_ref"]["path"]
    started_sha = _write(
        started_path, {"state": "STARTED", "mode": "execute", "started_at": "2026-09-07T07:00:00Z"}
    )
    value = capture("2026-09-07T20:20:00+08:00")
    raw_path = attempt / "close-session.raw.json"
    _write(raw_path, value.raw_response_bytes)
    calendar = {**value.receipt, "raw_response_path": str(raw_path)}
    calendar_path = attempt / "close-session-receipt.json"
    calendar_sha = _write(calendar_path, calendar)
    proof = build_historical_session(
        requested_trade_date="20260904",
        previous_trade_date="20260903",
        calendar_bytes=calendar_path.read_bytes(),
        raw=value.raw_response_bytes,
    )
    proof_path = attempt / FILENAME
    proof_sha = _write(proof_path, proof)
    core.update(
        schema_version=CORE_SCHEMA,
        provider_activity={},
        sealed_at="2026-09-07T13:00:00Z",
        started_ref={"path": str(started_path), "sha256": started_sha},
        close_session_receipt_ref={"path": str(calendar_path), "sha256": calendar_sha},
        historical_session_ref={"path": str(proof_path), "sha256": proof_sha},
    )
    core_ref["sha256"] = _write(path, core)
    from _public_catchup_fixture import bind_existing_recipe, put

    request = json.loads((root / args["request_ref"]["path"]).read_bytes())
    recipe = json.loads((root / request["recipe_ref"]["path"]).read_bytes())
    if routed:
        from quant_investor.operations.dashboard_serving_contract import POLICY

        recipe.update(
            schema_version="cn-daily-execute-recipe.v4",
            theme_acquisition_ref=None,
            corporate_action_context_ref=recipe["release_install_ref"],
            dashboard_publication_policy=POLICY,
        )
        recipe["research_sources"]["theme_source_ref"] = recipe["release_install_ref"]
    original_raw = put(root, "root-calendar.raw.json", value.raw_response_bytes)
    original_calendar = put(
        root,
        "root-calendar.json",
        {
            **value.receipt,
            "raw_response_path": str(root / original_raw["path"]),
        },
    )
    args["request_ref"], args["_catchup_binding_ref"], _ = bind_existing_recipe(
        root,
        recipe_value=recipe,
        calendar_ref=original_calendar,
        raw_calendar_ref=original_raw,
        routed=routed,
    )
    claims = []

    def claim(**kwargs):
        day = handoff_task_date(kwargs["handoff"], kwargs["started"])
        claims.append(day)
        return {"maintenance_run_root": "maintenance", "run_date": day}

    monkeypatch.setattr(module, "inspect_handoff_slot_claim", claim)
    return args, core, claims


@pytest.mark.parametrize("routed", [False, True])
def test_historical_publish_read_and_repeat_preserve_proof_and_claim(tmp_path, monkeypatch, routed):
    args, core, claims = context(tmp_path, monkeypatch, routed=routed)
    ref = module.publish_maintenance_handoff(**args)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    read = module.read_maintenance_handoff(workspace=str(tmp_path), handoff_ref=ref)
    handoff = read["handoff"]
    assert handoff["schema_version"] == module.HISTORICAL_SCHEMA
    assert handoff["historical_session_ref"] == {
        "path": str(Path(core["historical_session_ref"]["path"]).relative_to(tmp_path)),
        "sha256": core["historical_session_ref"]["sha256"],
    }
    assert module.publish_maintenance_handoff(**args) == ref
    assert claims == ["20260904"] * 3
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    assert read["execution_authorized"] is False
    if routed:
        assert args["_catchup_binding_ref"]["path"].endswith("binding.v2.json")
        assert read["recipe"]["publish_current_dashboard"] is False


@pytest.mark.parametrize(
    "fault", ["proof_sha", "proof_path", "core_version", "extra", "time", "predecessor"]
)
def test_historical_bad_core_cannot_publish(tmp_path, monkeypatch, fault):
    args, core, _ = context(tmp_path, monkeypatch)
    if fault == "proof_sha":
        core["historical_session_ref"]["sha256"] = "0" * 64
    elif fault == "proof_path":
        core["historical_session_ref"]["path"] = str(tmp_path / FILENAME)
    elif fault == "core_version":
        core["schema_version"] = "cn-daily-maintenance-core.v1"
    elif fault == "extra":
        core["prospective"] = True
    elif fault == "time":
        core["sealed_at"] = "2026-09-07T06:00:00Z"
    else:
        request_ref = args["request_ref"]
        request_path = tmp_path / request_ref["path"]
        request = json.loads(request_path.read_bytes())
        recipe_ref = request["recipe_ref"]
        recipe_path = tmp_path / recipe_ref["path"]
        recipe = json.loads(recipe_path.read_bytes())
        recipe["previous_completion_ref"][
            "path"
        ] = "results/operations/daily_production/CN/20260902/completion.v1.json"
        recipe_ref["sha256"] = _write(recipe_path, recipe)
        request_ref["sha256"] = _write(request_path, request)
    ref = args["state"]["core_checkpoint_ref"]
    ref["sha256"] = _write(tmp_path / ref["path"], core)
    with pytest.raises(ContractError):
        module.publish_maintenance_handoff(**args)
    assert not list((tmp_path / "results").rglob("maintenance-handoff.v1.json"))


@pytest.mark.parametrize("fault", ["missing", "schema", "proof_ref", "premature", "raw"])
def test_historical_readback_rejects_mutation(tmp_path, monkeypatch, fault):
    args, core, _ = context(tmp_path, monkeypatch)
    ref = module.publish_maintenance_handoff(**args)
    path = tmp_path / ref["path"]
    value = json.loads(path.read_bytes())
    if fault == "missing":
        value.pop("historical_session_ref")
    elif fault == "schema":
        value["schema_version"] = module.SCHEMA
        value.pop("historical_session_ref")
    elif fault == "proof_ref":
        value["historical_session_ref"]["path"] = value["calendar_ref"]["path"]
        value["historical_session_ref"]["sha256"] = value["calendar_ref"]["sha256"]
    elif fault == "premature":
        value["sealed_at"] = "2026-09-07T12:30:00Z"
    else:
        _write(
            Path(core["historical_session_ref"]["path"]).with_name("close-session.raw.json"), b"{}"
        )
    ref["sha256"] = _write(path, value)
    with pytest.raises(ContractError):
        module.read_maintenance_handoff(workspace=str(tmp_path), handoff_ref=ref)


def test_completed_historical_reader_uses_snapshot_and_recorded_sources(tmp_path, monkeypatch):
    args, _, _ = context(tmp_path, monkeypatch)
    ref = module.publish_maintenance_handoff(**args)
    handoff = json.loads((tmp_path / ref["path"]).read_bytes())
    recipe = json.loads((tmp_path / handoff["recipe_ref"]["path"]).read_bytes())
    docs = []

    def add(role, selected):
        raw = (tmp_path / selected["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == selected["sha256"]
        docs.append((role, selected["path"], selected["sha256"], raw))

    for role, selected in [
        ("handoff", ref),
        ("recipe", handoff["recipe_ref"]),
        ("loop_context", recipe["factor_loop_context_ref"]),
        ("release_install_input", handoff["release_install_ref"]),
        ("logical_claim", handoff["logical_claim_ref"]),
    ]:
        add(role, selected)
    for role in ("completion", "ledger", "materialization"):
        path = "completed-fixture/" + role + ".json"
        add(role, {"path": path, "sha256": _write(tmp_path / path, {"fixture": role})})
    snapshot = _mint_snapshot(workspace=str(tmp_path), trade_date="20260904", documents=tuple(docs))
    release = json.loads((tmp_path / handoff["release_ref"]["path"]).read_bytes())
    monkeypatch.setattr(
        archive,
        "verify_archived_handoff_context",
        lambda value: {
            "context": {"release_install_input_ref": handoff["release_install_ref"]},
            "installation": {"release_ref": object_ref_for_artifact(release)},
            "claim": {
                "claim_ref": handoff["logical_claim_ref"],
                "run_date": "20260904",
                "maintenance_run_root": "maintenance",
            },
        },
    )

    def forbidden(**kwargs):
        pytest.fail("historical reader must not use current install or current maintenance heads")

    monkeypatch.setattr(module, "read_factor_loop_context", forbidden)
    monkeypatch.setattr(module, "validate_daily_maintenance_receipt", forbidden)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = module.read_recorded_maintenance_handoff(snapshot)
    assert result["handoff_ref"] == ref and result["execution_authorized"] is False
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_historical_proof_changed_before_publication_is_not_retained(tmp_path, monkeypatch):
    args, core, _ = context(tmp_path, monkeypatch)
    original = module.DailyJournal.locked

    @contextmanager
    def changed(journal):
        with original(journal):
            path = Path(core["historical_session_ref"]["path"])
            path.write_bytes(path.read_bytes() + b" ")
            yield

    monkeypatch.setattr(module.DailyJournal, "locked", changed)
    with pytest.raises(ContractError, match="SOURCE_CHANGED"):
        module.publish_maintenance_handoff(**args)
    assert not list((tmp_path / "results").rglob("maintenance-handoff.v1.json"))
    assert not list((tmp_path / "results").rglob("request-*.json"))
