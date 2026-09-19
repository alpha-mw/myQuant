"""Request registration tests; no native EOD, installed release or live input claim."""

from datetime import datetime, timedelta
import json

import pytest

from quant_investor.operations import daily_preparation as module
from quant_investor.operations.daily_preparation_contract import (
    PreparationError,
    validate_config,
    preparation_root,
)
from quant_investor.operations.automatic_catchup_contract import run_path, document_ref
from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.dashboard_serving_contract import (
    PREFIX,
    HEAD_JSON,
    HEAD_JS,
    head_js,
    sealed,
)
from quant_investor.contracts import canonical_json_bytes
from _daily_preparation_fixture import NOW, calendar, config, native_inputs, put, snapshot


def prepare(root, ref, refs, **kwargs):
    return module.prepare_daily_request(
        workspace=str(root),
        config_ref=ref,
        calendar_ref=refs[0],
        raw_calendar_ref=refs[1],
        now=kwargs.pop("now", NOW),
        **kwargs,
    )


def forbidden(*args, **kwargs):
    raise AssertionError("forbidden fresh-source/producer read")


def head(root, day="20260824"):
    completion = put(
        root,
        f"results/operations/daily_production/CN/{day}/completion.v1.json",
        {"synthetic": True},
    )
    document = sealed(
        {
            "schema_version": "cn-daily-completed-head.v1",
            "trade_date": day,
            "completion_ref": completion,
            "previous_head_sha256": None,
            "registered_at": "2026-08-24T12:19:00Z",
            "authority": dict(FALSE_AUTHORITY),
        }
    )
    put(root, f"{PREFIX}/{HEAD_JSON}", document)
    put(root, f"{PREFIX}/{HEAD_JS}", head_js(canonical_json_bytes(document)))
    return completion


def test_native_bootstrap_registration_and_unchanged_repeat(tmp_path, monkeypatch):
    book, value, ref = native_inputs(tmp_path, monkeypatch)
    refs = calendar(tmp_path)
    before = snapshot(tmp_path)
    result = prepare(tmp_path, ref, refs)
    assert result["status"] == "REGISTERED_INPUTS_ONLY"
    request = json.loads((tmp_path / result["request_ref"]["path"]).read_bytes())
    recipe = json.loads((tmp_path / request["recipe_ref"]["path"]).read_bytes())
    assert request["action"] == "EXECUTE" and recipe["research_sources"]["as_of"] is None
    assert recipe["policy_refs"]["prospective"] is None
    assert recipe["research_sources"]["macro"]["mode"] == "MAINTENANCE_STAGE"
    after = snapshot(tmp_path)
    assert all(after[path] == state for path, state in before.items())
    monkeypatch.setattr(module, "_native_sources", forbidden)
    monkeypatch.setattr(module, "FactorProductionStore", forbidden)
    monkeypatch.setattr(module, "_locator", forbidden)
    assert prepare(tmp_path, ref, refs, now=NOW + timedelta(days=1)) == result
    assert snapshot(tmp_path) == after


@pytest.mark.parametrize(
    "now,holidays",
    [
        (datetime.fromisoformat("2026-08-22T20:20:00+08:00"), ()),
        (NOW, ("20260824",)),
    ],
)
def test_closed_session_no_journal_or_source_reads(tmp_path, monkeypatch, now, holidays):
    _, ref = config(tmp_path)
    refs = calendar(tmp_path, now, holidays=holidays)
    before = snapshot(tmp_path)
    monkeypatch.setattr(module, "_native_sources", forbidden)
    monkeypatch.setattr(module, "AutomaticRunStorage", forbidden)
    result = prepare(tmp_path, ref, refs, now=now)
    assert result["status"] == "NON_TRADING_DAY"
    assert result["request_ref"] is result["commitment_ref"] is None
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("fault", ["old", "future", "naive", "raw"])
def test_bad_calendar_before_any_write(tmp_path, fault):
    _, ref = config(tmp_path)
    refs = calendar(tmp_path)
    now = NOW
    if fault == "old":
        now += timedelta(days=1)
    elif fault == "future":
        now -= timedelta(seconds=2)
    elif fault == "naive":
        now = now.replace(tzinfo=None)
    else:
        (tmp_path / refs[1]["path"]).write_bytes(b"{}")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError):
        prepare(tmp_path, ref, refs, now=now)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "deadline", ["9:45:00", "24:00:00", "12:60:00", "12:00", True, "12:00:00Z"]
)
def test_deadline_has_no_implicit_format_or_default(tmp_path, deadline):
    value, _ = config(tmp_path, deadline=deadline)
    with pytest.raises(PreparationError, match="DEADLINE_INVALID"):
        validate_config(value)


def test_explicit_deadline_uses_shanghai_date(tmp_path, monkeypatch):
    _, value, _ = native_inputs(tmp_path, monkeypatch)
    value["prediction_deadline_local_time"] = "21:45:00"
    ref = put(tmp_path, "prepare-fixture/config.json", value)
    result = prepare(tmp_path, ref, calendar(tmp_path))
    commitment = json.loads((tmp_path / result["commitment_ref"]["path"]).read_bytes())
    policy = commitment["objects"][0]["document"]
    assert policy["prediction_deadline"] == "2026-08-24T13:45:00Z"


@pytest.mark.parametrize("mode", ["HEAD", "SEED"])
def test_same_day_locator_registers_empty_collection(tmp_path, monkeypatch, mode):
    seed = head(tmp_path)
    if mode == "SEED":
        (tmp_path / PREFIX / HEAD_JSON).unlink()
        (tmp_path / PREFIX / HEAD_JS).unlink()
    _, ref = config(tmp_path, seed=seed)
    refs = calendar(tmp_path)
    monkeypatch.setattr(module, "_native_sources", forbidden)
    monkeypatch.setattr(module, "_static", forbidden)
    monkeypatch.setattr(module, "FactorProductionStore", forbidden)
    result = prepare(tmp_path, ref, refs)
    commitment = json.loads((tmp_path / result["commitment_ref"]["path"]).read_bytes())
    assert len(commitment["objects"]) == 2
    assert commitment["objects"][0]["document"]["recipes"] == {}
    request = commitment["objects"][1]["document"]
    assert request["seed_completion_ref"] == (seed if mode == "SEED" else None)
    assert commitment["locator"]["configured_seed_ref"] == seed


def test_previous_seed_generates_current_recipe_only(tmp_path, monkeypatch):
    _, value, _ = native_inputs(tmp_path, monkeypatch)
    value["seed_completion_ref"] = put(
        tmp_path,
        "results/operations/daily_production/CN/20260820/completion.v1.json",
        {"synthetic": True},
    )
    ref = put(tmp_path, "prepare-fixture/config.json", value)
    monkeypatch.setattr(module, "FactorProductionStore", forbidden)
    result = prepare(tmp_path, ref, calendar(tmp_path))
    request = json.loads((tmp_path / result["request_ref"]["path"]).read_bytes())
    collection = json.loads((tmp_path / request["recipe_ref"]["path"]).read_bytes())
    assert set(collection["recipes"]) == {"20260824"}  # No invented Aug21 historical recipe.
    assert collection["recipes"]["20260824"]["store_preimages"]["store_pointer_ref"] is None


def test_request_write_crash_then_next_day_exact_repair(tmp_path, monkeypatch):
    _, _, ref = native_inputs(tmp_path, monkeypatch)
    refs = calendar(tmp_path)
    original = module.JournalStorage.write

    def crash(storage, path, raw, **kwargs):
        if path.endswith("/request.json"):
            raise OSError("synthetic crash")
        return original(storage, path, raw, **kwargs)

    monkeypatch.setattr(module.JournalStorage, "write", crash)
    with pytest.raises(OSError, match="synthetic crash"):
        prepare(tmp_path, ref, refs)
    path = preparation_root(ref, "20260824") + "/commitment.v1.json"
    committed = json.loads((tmp_path / path).read_bytes())
    expected = committed["objects"][-1]
    before = snapshot(tmp_path)
    monkeypatch.setattr(module.JournalStorage, "write", original)
    monkeypatch.setattr(module, "_native_sources", forbidden)
    monkeypatch.setattr(module, "_locator", forbidden)
    monkeypatch.setattr(module, "FactorProductionStore", forbidden)
    result = prepare(tmp_path, ref, refs, now=NOW + timedelta(days=1))
    assert result["request_ref"] == expected["ref"]
    after = snapshot(tmp_path)
    assert set(after) - set(before) == {expected["ref"]["path"]}
    assert all(after[p] == v for p, v in before.items())


def test_corrupt_existing_object_prevents_other_repairs(tmp_path, monkeypatch):
    _, _, ref = native_inputs(tmp_path, monkeypatch)
    refs = calendar(tmp_path)
    result = prepare(tmp_path, ref, refs)
    commitment = json.loads((tmp_path / result["commitment_ref"]["path"]).read_bytes())
    (tmp_path / commitment["objects"][0]["ref"]["path"]).unlink()
    (tmp_path / result["request_ref"]["path"]).write_bytes(b"{}")
    before = snapshot(tmp_path)
    with pytest.raises(PreparationError, match="OBJECT_CONFLICT"):
        prepare(tmp_path, ref, refs)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("matching", [False, True])
def test_pending_lease_is_preserved(tmp_path, monkeypatch, matching):
    _, _, ref = native_inputs(tmp_path, monkeypatch)
    refs = calendar(tmp_path)
    result = prepare(tmp_path, ref, refs)
    pending_ref = result["request_ref"] if matching else {"path": "other.json", "sha256": "b" * 64}
    with AutomaticRunStorage(str(tmp_path)).locked() as lock:
        lock.set_pending(
            {
                "schema_version": "cn-daily-catchup-pending.v1",
                "state": "ACTIVE",
                "auto_request_ref": pending_ref,
                "resolution_ref": document_ref(run_path(pending_ref, "resolution.v1.json"), {}),
            }
        )
    before = snapshot(tmp_path)
    if matching:
        assert prepare(tmp_path, ref, refs) == result
    else:
        with pytest.raises(PreparationError, match="PENDING_REQUEST") as caught:
            prepare(tmp_path, ref, refs)
        assert caught.value.fields["pending_request_ref"] == pending_ref
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("with_benchmark", [False, True])
def test_native_missing_dates_reported_before_factor_or_registration(
    tmp_path, monkeypatch, with_benchmark
):
    _, _, ref = native_inputs(tmp_path, monkeypatch)
    if with_benchmark:
        root = tmp_path / "data/parquet/cn/benchmarks"
        old = module.benchmark.load_generation(root)
        rows = old["rows"] + [
            {**row, "date": "2026-08-25", "value_date": "2026-08-25"}
            for row in old["rows"]
            if str(row["date"]) == "2026-08-24"
        ]
        module.benchmark.publish_generation(
            root,
            rows=rows,
            generation_id="synthetic-20260825",
            captured_at="2026-08-25T12:00:00Z",
            expected_pointer_sha256=old["pointer_sha256"],
            acquisition_receipt_ref=put(
                tmp_path, "prepare-fixture/new-capture.json", {"synthetic": True}
            ),
        )
    now = NOW + timedelta(days=1)
    refs = calendar(tmp_path, now)
    before = snapshot(tmp_path)
    monkeypatch.setattr(module, "FactorProductionStore", forbidden)
    with pytest.raises(PreparationError, match="NATIVE_DATES_MISSING") as caught:
        prepare(tmp_path, ref, refs, now=now)
    assert caught.value.fields == {
        "missing_event_dates": ["20260825"],
        "missing_benchmark_dates": [] if with_benchmark else ["20260825"],
    }
    after = snapshot(tmp_path)
    assert all(after[p] == state for p, state in before.items())
    assert not (tmp_path / preparation_root(ref, "20260825")).exists()


def test_stale_compatibility_is_not_native_benchmark_authority(tmp_path, monkeypatch):
    _, _, ref = native_inputs(tmp_path, monkeypatch)
    (tmp_path / module.BENCHMARK_CSV).write_bytes(b"date,ts_code,close\n")
    with pytest.raises(PreparationError, match="BENCHMARK_COMPATIBILITY_CHANGED"):
        prepare(tmp_path, ref, calendar(tmp_path))
    assert not (tmp_path / preparation_root(ref, "20260824")).exists()


def test_invalid_factor_parent_stops_bootstrap(tmp_path, monkeypatch):
    _, _, ref = native_inputs(tmp_path, monkeypatch)
    factory = module.FactorProductionStore

    def bad_parent(workspace):
        factor = factory(workspace)
        original = factor.verify_active
        factor.verify_active = lambda: {**original(), "as_of": "20260820"}
        return factor

    monkeypatch.setattr(module, "FactorProductionStore", bad_parent)
    with pytest.raises(PreparationError, match="FACTOR_PARENT_INVALID"):
        prepare(tmp_path, ref, calendar(tmp_path))


@pytest.mark.parametrize("surface", ["pointer", "head"])
def test_fresh_source_drift_before_commitment_stops_registration(tmp_path, monkeypatch, surface):
    _, _, ref = native_inputs(tmp_path, monkeypatch)
    refs = calendar(tmp_path)
    original = module._static

    def drift(*args):
        original(*args)
        if surface == "pointer":
            path = tmp_path / module.PREIMAGES["store_pointer_ref"]
            path.write_bytes(path.read_bytes())
        else:
            head(tmp_path)

    monkeypatch.setattr(module, "_static", drift)
    with pytest.raises(PreparationError, match="SOURCE_CHANGED"):
        prepare(tmp_path, ref, refs)
    assert not (tmp_path / preparation_root(ref, "20260824")).exists()


def test_head_drift_after_request_keeps_original_commitment_for_recovery(tmp_path, monkeypatch):
    _, _, ref = native_inputs(tmp_path, monkeypatch)
    refs = calendar(tmp_path)
    original = module.JournalStorage.write

    def drift(storage, path, raw, **kwargs):
        result = original(storage, path, raw, **kwargs)
        if path.endswith("/request.json"):
            head(tmp_path)
        return result

    monkeypatch.setattr(module.JournalStorage, "write", drift)
    with pytest.raises(PreparationError, match="SOURCE_CHANGED"):
        prepare(tmp_path, ref, refs)
    path = preparation_root(ref, "20260824") + "/commitment.v1.json"
    commitment = json.loads((tmp_path / path).read_bytes())
    assert commitment["locator"]["mode"] == "NONE"
    before = snapshot(tmp_path)
    monkeypatch.setattr(module.JournalStorage, "write", original)
    monkeypatch.setattr(module, "_locator", forbidden)
    monkeypatch.setattr(module, "_native_sources", forbidden)
    result = prepare(tmp_path, ref, refs, now=NOW + timedelta(days=1))
    assert result["request_ref"] == commitment["objects"][-1]["ref"]
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("fault", ["extra", "escape", "derivation", "locator", "calendar_alias"])
def test_commitment_mutation_or_calendar_alias_never_repairs(tmp_path, monkeypatch, fault):
    _, _, ref = native_inputs(tmp_path, monkeypatch)
    refs = calendar(tmp_path)
    result = prepare(tmp_path, ref, refs)
    path = result["commitment_ref"]["path"]
    value = json.loads((tmp_path / path).read_bytes())
    if fault == "extra":
        value["ignored"] = True
    elif fault == "escape":
        value["objects"][-1]["ref"]["path"] = "outside.json"
    elif fault == "derivation":
        value["objects"][-1]["document"]["action"] = "PLAN"
    elif fault == "locator":
        value["locator"]["selected_seed_ref"] = {"path": "seed.json", "sha256": "b" * 64}
    else:
        refs = (
            put(
                tmp_path,
                "prepare-fixture/alias-calendar.json",
                (tmp_path / refs[0]["path"]).read_bytes(),
            ),
            refs[1],
        )
    put(tmp_path, path, value)
    (tmp_path / result["request_ref"]["path"]).unlink()
    before = snapshot(tmp_path)
    with pytest.raises(ValueError):
        prepare(tmp_path, ref, refs)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("fault", ["symlink", "hardlink", "writable", "executable", "fifo"])
def test_source_alias_and_unsafe_modes_are_rejected(tmp_path, fault):
    import os
    from quant_investor.system.errors import SystemSecurityError

    target = put(tmp_path, "source.json", {"synthetic": True})
    if fault == "symlink":
        (tmp_path / "alias.json").symlink_to(tmp_path / "source.json")
        target = {**target, "path": "alias.json"}
    elif fault == "hardlink":
        os.link(tmp_path / "source.json", tmp_path / "alias.json")
    elif fault == "writable":
        (tmp_path / "source.json").chmod(0o666)
    elif fault == "executable":
        (tmp_path / "source.json").chmod(0o700)
    else:
        (tmp_path / "source.json").unlink()
        os.mkfifo(tmp_path / "source.json", 0o600)
    with pytest.raises(SystemSecurityError):
        module.Sources(str(tmp_path)).raw(target)


def test_readonly_repository_source_stays_readonly(tmp_path):
    ref = put(tmp_path, "policy.json", {"synthetic": True})
    (tmp_path / ref["path"]).chmod(0o644)
    before = snapshot(tmp_path)
    source = module.Sources(str(tmp_path))
    assert source.document(ref) == {"synthetic": True}
    source.recheck()
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "leaf", ["prediction-policy.json", "bootstrap.json", "recipe.json", "collection.json"]
)
def test_each_generated_object_can_recover_without_new_selection(tmp_path, monkeypatch, leaf):
    _, value, _ = native_inputs(tmp_path, monkeypatch)
    value["prediction_deadline_local_time"] = "21:45:00"
    if leaf == "collection.json":
        value["seed_completion_ref"] = put(
            tmp_path,
            "results/operations/daily_production/CN/20260821/completion.v1.json",
            {"synthetic": True},
        )
    ref = put(tmp_path, "prepare-fixture/config.json", value)
    refs = calendar(tmp_path)
    original = module.JournalStorage.write

    def crash(storage, path, raw, **kwargs):
        if path.endswith("/" + leaf):
            raise OSError("synthetic object interruption")
        return original(storage, path, raw, **kwargs)

    monkeypatch.setattr(module.JournalStorage, "write", crash)
    with pytest.raises(OSError, match="synthetic object interruption"):
        prepare(tmp_path, ref, refs)
    path = preparation_root(ref, "20260824") + "/commitment.v1.json"
    commitment = json.loads((tmp_path / path).read_bytes())
    before = snapshot(tmp_path)
    monkeypatch.setattr(module.JournalStorage, "write", original)
    monkeypatch.setattr(module, "_native_sources", forbidden)
    monkeypatch.setattr(module, "_locator", forbidden)
    monkeypatch.setattr(module, "FactorProductionStore", forbidden)
    result = prepare(tmp_path, ref, refs, now=NOW + timedelta(days=1))
    after = snapshot(tmp_path)
    assert all(after[p] == state for p, state in before.items())
    assert result["request_ref"] == commitment["objects"][-1]["ref"]
    assert set(after) - set(before) == {
        item["ref"]["path"] for item in commitment["objects"] if item["ref"]["path"] not in before
    }


@pytest.mark.parametrize("fault", ["mirror", "future", "closed"])
def test_invalid_head_locator_before_native_sources(tmp_path, monkeypatch, fault):
    head(
        tmp_path,
        "20260825" if fault == "future" else "20260823" if fault == "closed" else "20260824",
    )
    if fault == "mirror":
        (tmp_path / PREFIX / HEAD_JS).write_bytes(b"invalid mirror")
    _, ref = config(tmp_path)
    monkeypatch.setattr(module, "_native_sources", forbidden)
    with pytest.raises(ValueError):
        prepare(tmp_path, ref, calendar(tmp_path))
    assert not (tmp_path / preparation_root(ref, "20260824")).exists()


def test_fresh_foreign_pending_does_not_read_native_sources(tmp_path, monkeypatch):
    _, ref = config(tmp_path)
    refs = calendar(tmp_path)
    pending_ref = {"path": "other.json", "sha256": "b" * 64}
    with AutomaticRunStorage(str(tmp_path)).locked() as lock:
        lock.set_pending(
            {
                "schema_version": "cn-daily-catchup-pending.v1",
                "state": "ACTIVE",
                "auto_request_ref": pending_ref,
                "resolution_ref": document_ref(run_path(pending_ref, "resolution.v1.json"), {}),
            }
        )
    before = snapshot(tmp_path)
    monkeypatch.setattr(module, "_native_sources", forbidden)
    monkeypatch.setattr(module, "_locator", forbidden)
    with pytest.raises(PreparationError, match="PENDING_REQUEST"):
        prepare(tmp_path, ref, refs)
    assert snapshot(tmp_path) == before
