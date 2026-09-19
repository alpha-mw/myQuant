"""Real native close/adoption with no second CAS and preserved pre-close context."""

from copy import deepcopy
import hashlib
import json

import pytest

from _native_daily_store_fixture import NativeStoreFixture, DAYS, write
from scripts.daily_production_store_adapter import prepare_store_plan, StoreCloseAdapter, native
from scripts.daily_store_adoption import verify_store_plan_binding, PREIMAGES
from scripts.daily_store_materialization import prepare_materialized_store_plan
from scripts import daily_materialization as materializer
from quant_investor.operations.decision_recipe import read_decision_recipe
from quant_investor.contracts import canonical_json_bytes
from test_daily_evidence_daily_materialization import context


def complete(root):
    fixture = NativeStoreFixture(root)
    args = fixture.advance(DAYS[0])
    prepared = prepare_store_plan(args)
    ref = {"path": prepared["plan_path"], "sha256": prepared["plan_sha256"]}
    release = {"path": "release.json", "sha256": write(root / "release.json", {"synthetic": True})}
    adapter = StoreCloseAdapter(
        arguments=args, trade_date="20260824", plan_ref=ref, release_ref=release
    )
    adapter.execute(adapter.template())
    return fixture, adapter, ref


def inventory(root):
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_native_close_is_adopted_and_replayed_without_second_cas(tmp_path, monkeypatch):
    fixture, adapter, ref = complete(tmp_path)
    first = adapter.probe(adapter.template()).outcome
    before = inventory(tmp_path)
    adopted = prepare_store_plan(fixture.arguments())
    assert adopted["status"] == "PLAN_ADOPTED"
    assert {"path": adopted["plan_path"], "sha256": adopted["plan_sha256"]} == ref
    assert inventory(tmp_path) == before

    def forbidden(*args, **kwargs):
        pytest.fail("adopted replay selected current or invoked financial/recovery writer")

    monkeypatch.setattr(native, "_pointer_sha", forbidden)
    monkeypatch.setattr(native, "load_registered_catalog", forbidden)
    monkeypatch.setattr(native, "publish_catalog", forbidden)
    monkeypatch.setattr(native, "recover_close_completion", forbidden)
    assert adapter.probe(adapter.template()).outcome == first
    adapter.execute(adapter.template())
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("name", list(PREIMAGES))
def test_adoption_rejects_any_other_changed_preimage(tmp_path, name):
    fixture, adapter, ref = complete(tmp_path)
    args = fixture.arguments()
    args[name] = "e" * 64
    before = inventory(tmp_path)
    with pytest.raises(ValueError, match="NON_STORE_PREIMAGE_MISMATCH"):
        verify_store_plan_binding(arguments=args, plan_ref=ref, plan=adapter.plan)
    assert inventory(tmp_path) == before


def test_missing_old_source_cannot_be_reconstructed_for_adoption(tmp_path):
    fixture, adapter, _ = complete(tmp_path)
    path = native._completion_path(fixture.root, adapter.plan["transaction_id"]).with_name(
        "source-pointer.v1.json"
    )
    path.unlink()
    before = inventory(tmp_path)
    with pytest.raises(Exception):
        prepare_store_plan(fixture.arguments())
    assert inventory(tmp_path) == before


@pytest.mark.parametrize(
    "fault",
    ["wrong_bytes", "symlink", "partial_legacy", "corrupt_pending_source", "legacy_unsealed"],
)
def test_invalid_commit_custody_never_falls_back_to_execution(tmp_path, monkeypatch, fault):
    fixture, adapter, _ = complete(tmp_path)
    completion = native._completion_path(fixture.root, adapter.plan["transaction_id"])
    source = completion.with_name("source-pointer.v1.json")
    if fault in {"wrong_bytes", "corrupt_pending_source"}:
        source.write_bytes(b"{}\n")
        if fault == "corrupt_pending_source":
            completion.unlink()
    elif fault == "symlink":
        target = source.with_name("aliased-source.json")
        source.rename(target)
        source.symlink_to(target)
    elif fault == "legacy_unsealed":
        source.unlink()
        completion.unlink()
    else:
        source.unlink()
        completion.with_name("committed-pointer.v1.json").unlink()
    before = inventory(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid custody reached current lookup, legacy fallback or writer")

    for name in (
        "_pointer_sha",
        "load_registered_catalog",
        "publish_catalog",
        "recover_close_completion",
        "inspect_close_commit",
    ):
        monkeypatch.setattr(native, name, forbidden)
    with pytest.raises(Exception):
        adapter.execute(adapter.template())
    assert inventory(tmp_path) == before


def test_complete_legacy_without_source_retains_read_only_output_compatibility(tmp_path):
    fixture, adapter, _ = complete(tmp_path)
    expected = adapter.probe(adapter.template()).outcome
    source = native._completion_path(fixture.root, adapter.plan["transaction_id"]).with_name(
        "source-pointer.v1.json"
    )
    source.unlink()
    before = inventory(tmp_path)
    assert adapter.probe(adapter.template()).outcome == expected
    adapter.execute(adapter.template())
    assert inventory(tmp_path) == before


def test_adoption_cannot_predate_native_commit(tmp_path):
    fixture, adapter, ref = complete(tmp_path)
    before = inventory(tmp_path)
    with pytest.raises(ValueError, match="CUSTODY_BEFORE_COMMIT"):
        verify_store_plan_binding(
            arguments=fixture.arguments(),
            plan_ref=ref,
            plan=adapter.plan,
            custody_at="2026-01-01T00:00:00Z",
        )
    assert inventory(tmp_path) == before


def test_real_materialization_adopts_existing_day_and_retains_prior_book(tmp_path, monkeypatch):
    journal, recovered, book = context(tmp_path)
    with journal.locked():
        prepared = prepare_materialized_store_plan(journal=journal, recovered=recovered)
    adapter = StoreCloseAdapter(
        arguments=prepared["store_arguments"],
        trade_date=journal.trade_date,
        plan_ref=prepared["store_plan_ref"],
        release_ref=recovered["handoff"]["release_ref"],
    )
    adapter.execute(adapter.template())
    current = book.root / "_record_store/current.v1.json"
    closed = current.read_bytes()
    updated = deepcopy(recovered)
    updated["recipe"]["store_preimages"]["store_pointer_ref"]["sha256"] = hashlib.sha256(
        closed
    ).hexdigest()
    recipe_raw = canonical_json_bytes(updated["recipe"])
    (tmp_path / updated["handoff"]["recipe_ref"]["path"]).write_bytes(recipe_raw)
    updated["handoff"]["recipe_ref"]["sha256"] = hashlib.sha256(recipe_raw).hexdigest()
    raw = canonical_json_bytes(updated["handoff"])
    (tmp_path / updated["handoff_ref"]["path"]).write_bytes(raw)
    updated["handoff_ref"]["sha256"] = hashlib.sha256(raw).hexdigest()
    with journal.locked():
        materialized = materializer.materialize_locked(
            journal=journal, recovered=updated, auxiliary={"stages": {}}
        )
    assert current.read_bytes() == closed
    assert materialized.inputs.store_plan_ref == prepared["store_plan_ref"]
    bound = read_decision_recipe(
        workspace=tmp_path,
        trade_date=journal.trade_date,
        recipe_ref=materialized.inputs.decision_recipe_ref,
        research_request_ref=materialized.inputs.research_request_ref,
        store_plan_ref=materialized.inputs.store_plan_ref,
    )
    assert bound["portfolio"]["payload"]["source_effective_trade_date"] == "20260821"
    assert (
        bound["portfolio"]["payload"]["source_record_id"]
        == prepared["native_plan"]["source_active_record_id"]
    )
    assert (
        bound["portfolio"]["payload"]["source_record_id"] != json.loads(closed)["active_record_id"]
    )
    current.unlink()
    before = inventory(tmp_path)
    with journal.locked():
        replay = materializer.materialize_locked(
            journal=journal, recovered=updated, auxiliary={"stages": {}}
        )
    assert replay.status == "NO_ACTION" and replay.inputs == materialized.inputs
    assert inventory(tmp_path) == before
    source = native._completion_path(
        book.root, prepared["native_plan"]["transaction_id"]
    ).with_name("source-pointer.v1.json")
    source.unlink()
    with journal.locked(), pytest.raises(Exception):
        materializer.materialize_locked(
            journal=journal, recovered=updated, auxiliary={"stages": {}}
        )
