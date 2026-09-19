"""Bootstrap wire/recovery with real retained bytes and explicit native proof seams."""

from copy import deepcopy
from contextlib import contextmanager
import hashlib
from types import SimpleNamespace

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.bootstrap_launch_contract import (
    BootstrapLaunchInputs,
    execution_paths,
)
from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS, GRAPH_SHA256
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY
from quant_investor.operations.daily_launch_contract import validate_launch_inspection
from quant_investor.operations.dependency_diagnostics import DependencyInputError
from test_daily_evidence_research_timing import timing_recipe
from test_daily_evidence_execute_wiring import module as materialization
from scripts import daily_bootstrap_launch as native


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent
    while parent != root:
        parent.chmod(0o700)
        parent = parent.parent
    raw = value if isinstance(value, bytes) else canonical_json_bytes(value)
    path.write_bytes(raw)
    path.chmod(0o600)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def fixture(root, monkeypatch, *, handoff=True, cutoff=True):
    request, recipe = timing_recipe()
    other = put(root, "input.json", {})
    request["release_install_ref"] = recipe["release_install_ref"] = other
    recipe["previous_completion_ref"] = None
    declaration = {
        "schema_version": "cn-daily-bootstrap.v1",
        "market": "CN",
        "strategy_id": "aggressive_tech_manufacturing",
        "graph_sha256": GRAPH_SHA256,
        "first_trade_date": request["target_trade_date"],
        "previous_trade_date": "20260903",
        "factor_parent_pointer_sha256": "b" * 64,
        "store_preimages": recipe["store_preimages"],
        "authority": dict(FALSE_AUTHORITY),
    }
    recipe["bootstrap_ref"] = put(root, "bootstrap.json", declaration)
    request["recipe_ref"] = put(root, "recipe.json", recipe)
    request_ref = put(root, "request.json", request)
    paths = execution_paths(request_ref, request)
    recovered = None
    if handoff:
        put(root, paths["request_ref"]["path"], (root / request_ref["path"]).read_bytes())
        put(root, paths["recipe_ref"]["path"], (root / request["recipe_ref"]["path"]).read_bytes())
        document = {
            "request_ref": paths["request_ref"],
            "recipe_ref": paths["recipe_ref"],
            "release_install_ref": other,
            "release_ref": recipe["release_ref"],
            "trade_date": request["target_trade_date"],
            "core_handoff_ref": other,
            "maintenance_core_ref": other,
            "logical_claim_ref": {
                "path": (
                    "data/private/cn_daily_maintenance/logical_tasks/"
                    "20260904-2020-execute/claim.json"
                ),
                "sha256": "c" * 64,
            },
        }
        handoff_ref = put(root, paths["handoff"], document)
        recovered = {
            "handoff": document,
            "handoff_ref": handoff_ref,
            "request": request,
            "recipe": recipe,
        }
        monkeypatch.setattr(native, "read_maintenance_handoff", lambda **kw: deepcopy(recovered))
        if cutoff:
            put(
                root,
                paths["cutoff"],
                {
                    "request_ref": paths["request_ref"],
                    "maintenance_handoff_ref": handoff_ref,
                    "core_handoff_ref": other,
                    "trade_date": request["target_trade_date"],
                },
            )
    monkeypatch.setattr(native, "read_cutoff_inputs", lambda **kw: {})
    return request, request_ref, recipe, paths, recovered


def inspect(root, request, ref):
    return native.inspect_bootstrap_launch(
        workspace=str(root),
        request_ref=ref,
        release_install_ref=request["release_install_ref"],
        synthetic=True,
    )


def snapshot(root):
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_original_and_retained_paths_differ_but_exact_bytes_bind(tmp_path, monkeypatch):
    request, ref, _, paths, recovered = fixture(tmp_path, monkeypatch)
    value = BootstrapLaunchInputs(
        workspace=str(tmp_path), request_ref=ref, release_install_ref=request["release_install_ref"]
    )
    assert ref != paths["request_ref"]
    value.bind(recovered)
    alias = put(tmp_path, "other/request.json", (tmp_path / ref["path"]).read_bytes())
    BootstrapLaunchInputs(
        workspace=str(tmp_path),
        request_ref=alias,
        release_install_ref=request["release_install_ref"],
    ).bind(recovered)


@pytest.mark.parametrize(
    "fault", ["request_bytes", "recipe_bytes", "recipe_path", "handoff_path", "missing_original"]
)
def test_retained_or_missing_original_identity_cannot_be_substituted(tmp_path, monkeypatch, fault):
    request, ref, _, paths, recovered = fixture(tmp_path, monkeypatch)
    if fault == "missing_original":
        (tmp_path / ref["path"]).unlink()
    elif fault.endswith("bytes"):
        key = "request_ref" if fault == "request_bytes" else "recipe_ref"
        (tmp_path / paths[key]["path"]).write_bytes(b"different")
    elif fault == "recipe_path":
        recovered["handoff"]["recipe_ref"] = put(
            tmp_path, "wrong-recipe.json", (tmp_path / request["recipe_ref"]["path"]).read_bytes()
        )
    else:
        recovered["handoff_ref"] = {
            "path": "wrong-handoff.json",
            "sha256": recovered["handoff_ref"]["sha256"],
        }
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        value = BootstrapLaunchInputs(
            workspace=str(tmp_path),
            request_ref=ref,
            release_install_ref=request["release_install_ref"],
        )
        value.bind(recovered)


def test_fresh_bootstrap_inspection_is_non_mutating(tmp_path, monkeypatch):
    request, ref, _, _, _ = fixture(tmp_path, monkeypatch, handoff=False)
    calls = []
    monkeypatch.setattr(native, "_fresh_controls", lambda value, **kw: calls.append(kw))
    before = snapshot(tmp_path)
    value = inspect(tmp_path, request, ref)
    assert value["mode"] == "PRODUCER_REQUIRED" and value["recovery_scope"] == "NONE"
    assert calls == [{"warm": False}] and snapshot(tmp_path) == before


@pytest.mark.parametrize("missing", [False, True])
def test_warm_committed_inspection_skips_mutable_heads_and_accepts_only_typed_missing(
    tmp_path, monkeypatch, missing
):
    request, ref, _, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        native, "_fresh_controls", lambda *a, **kw: pytest.fail("warm path read current heads")
    )
    if missing:

        def read(**kwargs):
            raise DependencyInputError("CUTOFF_COMMITTED_OBJECT_MISSING")

        monkeypatch.setattr(native, "read_cutoff_inputs", read)
    before = snapshot(tmp_path)
    value = inspect(tmp_path, request, ref)
    assert value["mode"] == "LOCAL_REPAIR" and value["recovery_scope"] == "COMMITTED_DAG_RECOVERY"
    assert snapshot(tmp_path) == before


def test_conflicting_committed_objects_never_enable_repair(tmp_path, monkeypatch):
    request, ref, _, _, _ = fixture(tmp_path, monkeypatch)

    def fail(**kwargs):
        raise DependencyInputError("CUTOFF_COMMITTED_OBJECT_CONFLICT")

    monkeypatch.setattr(native, "read_cutoff_inputs", fail)
    with pytest.raises(DependencyInputError):
        inspect(tmp_path, request, ref)


@pytest.mark.parametrize("scope", ["NONE", "SERVING_ONLY", "COMMITTED_DAG_RECOVERY"])
def test_bootstrap_inspection_scope_cannot_be_relabelled(tmp_path, monkeypatch, scope):
    request, ref, _, _, _ = fixture(tmp_path, monkeypatch)
    value = inspect(tmp_path, request, ref)
    value["recovery_scope"] = scope
    if scope == "NONE":
        with pytest.raises(ContractError):
            validate_launch_inspection(
                value,
                request_ref=ref,
                release_install_ref=request["release_install_ref"],
                target_trade_date=request["target_trade_date"],
            )
    else:
        validate_launch_inspection(
            value,
            request_ref=ref,
            release_install_ref=request["release_install_ref"],
            target_trade_date=request["target_trade_date"],
        )


def completed(root, monkeypatch, request, paths, recovered):
    from quant_investor.operations import completion_readback

    journal = DailyJournal(str(root), request["target_trade_date"])
    ref = put(root, str(journal.root / "completion.v1.json"), {})
    cap = SimpleNamespace(recheck=lambda: None)
    monkeypatch.setattr(
        completion_readback,
        "inspect_recorded_completion",
        lambda **kw: {"completed_handoff_snapshot": cap},
    )
    monkeypatch.setattr(native, "read_recorded_maintenance_handoff", lambda value: recovered)
    monkeypatch.setattr(
        native,
        "replay_native_completion",
        lambda **kw: {
            **kw,
            "native_replay_validated": True,
            "validated_nodes": sorted(EOD_NODE_IDS),
            "synthetic": True,
        },
    )
    return ref


@pytest.mark.parametrize(
    "serving,mode,scope",
    [(True, "COMPLETE_READ_ONLY", "NONE"), (False, "LOCAL_REPAIR", "SERVING_ONLY")],
)
def test_completed_bootstrap_uses_frozen_proof_and_correct_serving_scope(
    tmp_path, monkeypatch, serving, mode, scope
):
    from scripts import daily_catchup

    request, ref, _, paths, recovered = fixture(tmp_path, monkeypatch)
    completion_ref = completed(tmp_path, monkeypatch, request, paths, recovered)
    monkeypatch.setattr(daily_catchup, "_serving_gate", lambda *a, **kw: serving)
    monkeypatch.setattr(
        native, "_fresh_controls", lambda *a, **kw: pytest.fail("completed path read current heads")
    )
    before = snapshot(tmp_path)
    value = inspect(tmp_path, request, ref)
    assert (value["mode"], value["recovery_scope"]) == (mode, scope)
    if serving:
        assert value["result"]["days"][0]["completion_ref"] == completion_ref
    assert snapshot(tmp_path) == before


def test_committed_recovery_runs_downstream_but_never_upstream(tmp_path, monkeypatch):
    from scripts import daily_completion

    request, ref, _, _, _ = fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(native, "read_auxiliary_stage_records", lambda **kw: {"stages": {}})

    def materialize(**kwargs):
        kwargs["journal"]._require_lock()
        assert kwargs["_execute_theme"] is False and kwargs["_committed_cutoff_ref"]
        calls.append("committed-materialization")
        return SimpleNamespace(native_inputs_ref={"path": "native.json", "sha256": "a" * 64})

    monkeypatch.setattr(materialization, "materialize_locked", materialize)
    monkeypatch.setattr(
        daily_completion,
        "run_materialized_native_input",
        lambda **kw: calls.append("downstream") or {"status": "COMPLETE"},
    )
    monkeypatch.setattr(
        "quant_investor.market.daily_factor_loop.DailyFactorLoop",
        lambda **kw: pytest.fail("loop constructed"),
    )
    monkeypatch.setattr(
        "quant_investor.market.daily_maintenance.run_cn_daily_maintenance",
        lambda **kw: pytest.fail("maintenance called"),
    )
    result = native.recover_bootstrap_committed(
        workspace=str(tmp_path),
        request_ref=ref,
        release_install_ref=request["release_install_ref"],
        synthetic=True,
    )
    assert result["status"] == "COMPLETE" and calls == ["committed-materialization", "downstream"]


def test_missing_cutoff_blocks_before_auxiliary_or_downstream_work(tmp_path, monkeypatch):
    request, ref, _, _, _ = fixture(tmp_path, monkeypatch, cutoff=False)
    monkeypatch.setattr(
        native,
        "read_auxiliary_stage_records",
        lambda **kw: pytest.fail("read auxiliary before committed proof"),
    )
    with pytest.raises(ContractError, match="COMMITTED_RECOVERY_REQUIRED"):
        native.recover_bootstrap_committed(
            workspace=str(tmp_path),
            request_ref=ref,
            release_install_ref=request["release_install_ref"],
            synthetic=True,
        )


@pytest.mark.parametrize("fault", [None, "wrong_date", "has_core", "busy", "changed"])
def test_only_exact_finalized_closed_session_can_emit_no_action(tmp_path, monkeypatch, fault):
    request, ref, _, _, _ = fixture(tmp_path, monkeypatch, handoff=False)
    monkeypatch.setattr(native, "_claim_exists", lambda value: True)
    result = {
        "target_date": request["target_trade_date"],
        "request_target_trade_date": request["target_trade_date"],
        "requested_session_result": {"classification": "CONFIRMED_CLOSED"},
    }
    if fault == "wrong_date":
        result["target_date"] = "20260903"
    elif fault == "has_core":
        result["core_completion_ref"] = {"path": "core.json", "sha256": "f" * 64}

    @contextmanager
    def finalized(**kwargs):
        assert kwargs["run_date"] == request["target_trade_date"]
        if fault == "busy":
            raise ContractError("MAINTENANCE_AUXILIARY_RUNNING")
        yield result
        if fault == "changed":
            raise ContractError("FINALIZED_MAINTENANCE_CHANGED")

    monkeypatch.setattr(native, "locked_finalized_maintenance_replay", finalized)
    monkeypatch.setattr(
        native, "_fresh_controls", lambda *a, **kw: pytest.fail("closed session read fresh heads")
    )
    before = snapshot(tmp_path)
    if fault is not None:
        with pytest.raises(ContractError):
            inspect(tmp_path, request, ref)
    else:
        value = inspect(tmp_path, request, ref)
        assert value["mode"] == "COMPLETE_READ_ONLY" and value["recovery_scope"] == "NONE"
        assert (
            value["result"]["business_state"] == "NON_TRADING_DAY" and value["result"]["days"] == []
        )
    assert snapshot(tmp_path) == before


def test_automatic_lock_race_blocks_bootstrap_before_business_writer(tmp_path, monkeypatch):
    from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage
    from scripts import daily_production

    request, ref, _, _, _ = fixture(tmp_path, monkeypatch)
    assert inspect(tmp_path, request, ref)["mode"] == "LOCAL_REPAIR"
    monkeypatch.setattr(
        daily_production, "execute_daily_recipe", lambda **kw: pytest.fail("raced into writer")
    )
    with AutomaticRunStorage(str(tmp_path)).locked():
        with pytest.raises(ContractError, match="AUTO_RUN_BUSY"):
            daily_production.dispatch_daily_request(
                workspace=str(tmp_path),
                request_ref=ref,
                release_install_ref=request["release_install_ref"],
                synthetic=True,
                committed_recovery_only=True,
            )


def test_existing_strategy_lock_is_held_during_bootstrap_execution(tmp_path, monkeypatch):
    from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage
    from scripts import daily_production

    request, ref, _, _, _ = fixture(tmp_path, monkeypatch)
    entered = []

    def execute(**kwargs):
        assert kwargs["committed_recovery_only"] is True
        with pytest.raises(ContractError, match="AUTO_RUN_BUSY"):
            with AutomaticRunStorage(str(tmp_path)).locked():
                pytest.fail("second controller acquired lock")
        entered.append(True)
        return {"status": "BLOCKED", "completion_ref": None}

    monkeypatch.setattr(daily_production, "execute_daily_recipe", execute)
    result = daily_production.dispatch_daily_request(
        workspace=str(tmp_path),
        request_ref=ref,
        release_install_ref=request["release_install_ref"],
        synthetic=True,
        committed_recovery_only=True,
    )
    assert entered == [True] and result["business_state"] == "INCOMPLETE"
    assert AutomaticRunStorage(str(tmp_path)).pending() is None


def test_public_no_producers_does_not_gain_bootstrap_writer_authority(tmp_path, monkeypatch):
    from scripts import daily_production

    request, ref, _, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        daily_production, "execute_daily_recipe", lambda **kw: pytest.fail("widened no-producers")
    )
    with pytest.raises(ContractError, match="AUTO_NO_PRODUCERS_FLAG_INVALID"):
        daily_production.dispatch_daily_request(
            workspace=str(tmp_path),
            request_ref=ref,
            release_install_ref=request["release_install_ref"],
            synthetic=True,
            no_producers=True,
        )


def test_committed_cutoff_disappearance_blocks_before_any_new_plan(tmp_path, monkeypatch):
    request, ref, recipe, paths, recovered = fixture(tmp_path, monkeypatch)
    journal = DailyJournal(str(tmp_path), request["target_trade_date"])
    stored = journal.storage.read(paths["cutoff"])
    cutoff_ref = {"path": paths["cutoff"], "sha256": stored.byte_sha256}
    (tmp_path / paths["cutoff"]).unlink()
    monkeypatch.setattr(
        materialization,
        "prepare_materialized_store_plan",
        lambda **kw: pytest.fail("new plan selected"),
    )
    with journal.locked(), pytest.raises(ContractError, match="COMMITTED_CUTOFF_MISMATCH"):
        materialization.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={}, _committed_cutoff_ref=cutoff_ref
        )
