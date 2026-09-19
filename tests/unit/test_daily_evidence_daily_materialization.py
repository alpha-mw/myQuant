"""Compose real research/Store input writers and native decoder from a controlled handoff."""

from datetime import datetime, timezone
from pathlib import Path
import sys
import hashlib
import json
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from quant_investor.contracts import canonical_json_bytes
from quant_investor.intelligence.storage import approved_theme_policy_v2
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.market_data_reader import MarketDataUnavailableError
from quant_investor.operations.daily_contract import ContractError
from scripts import daily_materialization as materializer
from test_daily_evidence_store_materialization import context as store_context


def context(root, version=1):
    journal, recovered, book, arguments = store_context(root)

    def put(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        current = path.parent
        while current != root:
            current.chmod(0o700)
            current = current.parent
        raw = canonical_json_bytes(value)
        path.write_bytes(raw)
        path.chmod(0o600)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    def ref(path):
        return {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    reader = MarketDataReader(market="CN", data_root=root / "data", mode_policy="strict")
    manifest = Path(reader.snapshot()["manifest_path"])
    snapshot = ref(manifest)
    pointer = put("fixtures/factor-pointer.json", {"fixture": "factor"})
    low = put("fixtures/low-observation.json", {"fixture": "LOW"})
    w80 = put("fixtures/w80-observation.json", {"fixture": "W80"})
    pool = put("fixtures/pool.json", {"fixture": "Top100"})
    nodes = {
        name: put(
            "fixtures/" + name + "-terminal.json", {"state": "SUCCEEDED", "output_refs": outputs}
        )
        for name, outputs in {
            "calendar": {},
            "low_observation": {"LOW": low},
            "w80_observation": {"W80": w80},
            "top100": {"manifest.json": pool},
        }.items()
    }
    core = put("fixtures/core-handoff.json", {"node_refs": nodes})
    source_pointer = put("fixtures/fundamental-pointer.json", {"fixture": "pointer bytes"})
    fundamental = put(
        "fixtures/fundamental-source.json",
        {
            "available_at": "2026-08-21T08:00:00Z",
            "pointer": source_pointer,
            "daily_parquet": source_pointer,
        },
    )
    macro = put("fixtures/macro-source.json", {"fixture": "Macro descriptor"})
    recipe = recovered["recipe"]
    recipe["schema_version"] = f"cn-daily-execute-recipe.v{version}"
    if version == 2:
        recipe["theme_acquisition_ref"] = None
    recipe.update(
        strategy_id="aggressive_tech_manufacturing",
        publish_current_dashboard=True,
        previous_completion_ref={
            "path": "results/operations/daily_production/CN/20260821/completion.v1.json",
            "sha256": "b" * 64,
        },
        dashboard_sources={
            "benchmark_ref": ref(root / "portfolio_dashboard/inputs/cn_index_benchmark.csv"),
            "risk_free_ref": ref(root / "portfolio_dashboard/inputs/cn_govt_bond_yield.csv"),
        },
    )
    recipe["policy_refs"]["research"] = put(
        "fixtures/research-policy.json", approved_theme_policy_v2()
    )
    recipe["research_sources"] = {
        "as_of": "2026-08-24T13:30:00Z",
        "industry_source_ref": None,
        "theme_source_ref": None,
        "exposure_rows_ref": None,
        "fundamental": {"mode": "PINNED", "source_ref": fundamental},
        "macro": {"mode": "PINNED", "source_ref": macro},
    }
    if version == 2:
        recipe["research_sources"]["theme_source_ref"] = put(
            "fixtures/theme-source.json", {"fixture": "pinned Theme descriptor"}
        )
    recipe_ref = put("fixtures/recipe.json", recipe)
    release_ref = put("fixtures/release.json", {"fixture": "release"})
    handoff = recovered["handoff"]
    handoff.update(
        recipe_ref=recipe_ref,
        release_ref=release_ref,
        request_ref={"path": "request.json", "sha256": "a" * 64},
        core_handoff_ref=core,
        factor_pointer_ref=pointer,
        market_snapshot_ref=snapshot,
        sealed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    execution = journal.root / "executions" / ("a" * 64)
    recovered["handoff_ref"] = put(str(execution / "maintenance-handoff.v1.json"), handoff)
    return journal, recovered, book


@pytest.mark.parametrize("version", [1, 2])
def test_materialization_uses_native_decoder_and_does_not_replan_on_resume(
    tmp_path, monkeypatch, version
):
    journal, recovered, book = context(tmp_path, version)
    store_pointer = book.root / "_record_store/current.v1.json"
    before = store_pointer.read_bytes()
    with journal.locked():
        first = materializer.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    assert first.status == "MATERIALIZED"
    assert first.inputs.previous_trade_date == "20260821"
    assert first.inputs.publish_current_dashboard is True
    native_document = json.loads((tmp_path / first.native_inputs_ref["path"]).read_bytes())
    assert native_document["schema_version"] == "cn-daily-native-inputs.v3"
    assert native_document["decision_recipe_ref"] == first.inputs.decision_recipe_ref
    assert first.inputs.decision_recipe_ref is not None
    from quant_investor.operations.decision_recipe import read_decision_recipe

    bound = read_decision_recipe(
        workspace=tmp_path,
        trade_date=journal.trade_date,
        recipe_ref=first.inputs.decision_recipe_ref,
        research_request_ref=first.inputs.research_request_ref,
        store_plan_ref=first.inputs.store_plan_ref,
    )
    assert bound["portfolio"]["payload"]["prospective"] is False
    assert set(first.inputs.adjustment_market_refs) == set(book.stocks)
    assert store_pointer.read_bytes() == before
    record = json.loads((tmp_path / first.materialization_ref["path"]).read_text())
    assert record["native_inputs_ref"] == first.native_inputs_ref
    assert record["maintenance_handoff_ref"] == recovered["handoff_ref"]
    from quant_investor.operations.materialization_contract import layout_for_recipe

    schema, filename, fields = layout_for_recipe(recovered["recipe"])
    assert set(record) == fields
    assert record["schema_version"] == schema
    assert first.materialization_ref["path"].endswith("/" + filename)
    if version == 2:
        assert record["theme_source_handoff_ref"] is None
    # The native input loader must retain the original preimages after Store moves.
    store_pointer.write_bytes(b"changed current Store pointer")

    def forbidden(**kwargs):
        pytest.fail("replanned or rebuilt on recorded materialization replay")

    monkeypatch.setattr(materializer, "prepare_materialized_store_plan", forbidden)
    monkeypatch.setattr(materializer, "publish_research_input", forbidden)
    monkeypatch.setattr(materializer, "materialized_adjustment_refs", forbidden)
    with journal.locked():
        again = materializer.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    assert again.status == "NO_ACTION"
    assert again.materialization_ref == first.materialization_ref
    assert again.native_inputs_ref == first.native_inputs_ref
    assert again.inputs == first.inputs
    native_path = tmp_path / first.native_inputs_ref["path"]
    native_path.write_bytes(b"tampered")
    with journal.locked():
        with pytest.raises(ContractError, match="REF_SHA_MISMATCH"):
            materializer.materialize_locked(
                journal=journal, recovered=recovered, auxiliary={"stages": {}}
            )


def test_recorded_materialization_rejects_changed_held_prices(tmp_path):
    journal, recovered, _ = context(tmp_path)
    with journal.locked():
        first = materializer.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    ref = next(iter(first.inputs.adjustment_market_refs.values()))
    (tmp_path / ref["path"]).write_bytes(b"changed held prices")
    with journal.locked(), pytest.raises(MarketDataUnavailableError):
        materializer.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )


def test_materialized_context_runs_under_original_lock_without_reloading(tmp_path, monkeypatch):
    from scripts.daily_native_registry import NativeDailyRegistry
    from scripts.daily_completion import run_and_seal_materialized_input
    import scripts.daily_completion as completion
    import scripts.daily_native_inputs as decoder

    journal, recovered, book = context(tmp_path)
    before = (book.root / "_record_store/current.v1.json").read_bytes()
    with journal.locked():
        loaded = materializer.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
        registry = NativeDailyRegistry(
            str(tmp_path), journal.trade_date, loaded.inputs, journal=journal
        )
        assert registry.runner.journal is journal

        def forbidden(*args, **kwargs):
            pytest.fail("reloaded inputs or reacquired day lock")

        monkeypatch.setattr(journal, "locked", forbidden)
        monkeypatch.setattr(decoder, "load_native_inputs", forbidden)
        monkeypatch.setattr(completion, "load_native_inputs", forbidden)
        result = run_and_seal_materialized_input(
            registry, materialized=loaded, resume=True, synthetic=True
        )
    # The controlled handoff lacks native Factor authority: the actual adapters
    # must block, rather than allowing a green input receipt to imply completion.
    assert result["status"] != "COMPLETE"
    assert result["completion_ref"] is None
    assert (book.root / "_record_store/current.v1.json").read_bytes() == before
    assert journal.storage.read(str(journal.root / "completion.v1.json")) is None


def test_shared_journal_requires_matching_context_and_owned_lock(tmp_path):
    from quant_investor.operations.daily_runner import DayRunner

    journal, _, _ = context(tmp_path)
    with pytest.raises(ContractError, match="JOURNAL_LOCK_REQUIRED"):
        DayRunner(str(tmp_path), journal.trade_date, {}, journal=journal)
    with journal.locked():
        with pytest.raises(ContractError, match="SHARED_JOURNAL_CONTEXT_MISMATCH"):
            DayRunner(str(tmp_path), "20260825", {}, journal=journal)
        other = tmp_path / "other-workspace"
        other.mkdir()
        with pytest.raises(ContractError, match="SHARED_JOURNAL_CONTEXT_MISMATCH"):
            DayRunner(str(other), journal.trade_date, {}, journal=journal)


@pytest.mark.parametrize("changed", ["none", "previous", "event", "store_args", "publication_type"])
def test_loaded_input_recheck_without_decoder_or_current_store(tmp_path, monkeypatch, changed):
    from dataclasses import replace
    import scripts.daily_native_inputs as native

    journal, recovered, book = context(tmp_path)
    with journal.locked():
        first = materializer.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    (book.root / "_record_store/current.v1.json").write_bytes(b"advanced current pointer")
    monkeypatch.setattr(native, "load_native_inputs", lambda **kw: pytest.fail("second decoder"))
    inputs = first.inputs
    if changed == "previous":
        inputs = replace(inputs, previous_trade_date="20260820")
    elif changed == "event":
        inputs = replace(inputs, event_pointer_sha256="a" * 64)
    elif changed == "store_args":
        inputs = replace(
            inputs,
            store_arguments={**inputs.store_arguments, "expected_store_pointer_sha": "b" * 64},
        )
    elif changed == "publication_type":
        inputs = replace(inputs, publish_current_dashboard=1)
    args = dict(
        workspace=str(tmp_path),
        input_ref=first.native_inputs_ref,
        trade_date=journal.trade_date,
        inputs=inputs,
    )
    if changed == "none":
        native.verify_loaded_native_inputs(**args)
    else:
        with pytest.raises(ContractError, match="CONTEXT_MISMATCH"):
            native.verify_loaded_native_inputs(**args)


def test_materialization_full_readback_uses_no_writer_or_decoder(tmp_path, monkeypatch):
    journal, recovered, book = context(tmp_path)
    with journal.locked():
        first = materializer.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    (book.root / "_record_store/current.v1.json").write_bytes(b"advanced Store")

    def forbidden(*a, **kw):
        pytest.fail("readback wrote or decoded again")

    monkeypatch.setattr(materializer, "load_native_inputs", forbidden)
    monkeypatch.setattr(journal.storage, "write", forbidden)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with journal.locked():
        record = materializer.verify_materialized_inputs(
            journal=journal, recovered=recovered, materialized=first
        )
    assert record["native_inputs_ref"] == first.native_inputs_ref
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("version", [1, 2])
def test_readonly_materialization_replay_takes_no_lock(tmp_path, monkeypatch, version):
    journal, recovered, _ = context(tmp_path, version)
    with journal.locked():
        first = materializer.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )

    def forbidden(*a, **kw):
        pytest.fail("readonly replay acquired lock or wrote")

    monkeypatch.setattr(journal, "locked", forbidden)
    monkeypatch.setattr(journal.storage, "write", forbidden)
    materializer.verify_materialized_inputs(
        journal=journal, recovered=recovered, materialized=first, readonly=True
    )


def test_v2_combined_materialized_entry_preserves_incomplete_native_state(tmp_path):
    from scripts.daily_native_registry import NativeDailyRegistry
    from scripts.daily_completion import run_and_seal_materialized_input

    journal, recovered, book = context(tmp_path)
    before = (book.root / "_record_store/current.v1.json").read_bytes()
    with journal.locked():
        first = materializer.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
        registry = NativeDailyRegistry(
            str(tmp_path), journal.trade_date, first.inputs, journal=journal
        )
        result = run_and_seal_materialized_input(
            registry, materialized=first, resume=True, synthetic=True
        )
    assert result["status"] != "COMPLETE" and result["completion_ref"] is None
    assert (book.root / "_record_store/current.v1.json").read_bytes() == before
    assert not (tmp_path / "results/prospective").exists()


def bootstrap_context(root, monkeypatch):
    journal, recovered, book = context(root)
    from quant_investor.operations.bootstrap import read_bootstrap_declaration

    declaration = dict(
        schema_version="cn-daily-bootstrap.v1",
        market="CN",
        strategy_id="aggressive_tech_manufacturing",
        first_trade_date="20260824",
        previous_trade_date="20260821",
        graph_sha256=materializer.GRAPH_SHA256,
        factor_parent_pointer_sha256="b" * 64,
        store_preimages=recovered["recipe"]["store_preimages"],
        authority=materializer.FALSE_AUTHORITY,
    )

    def rewrite(path, value):
        raw = canonical_json_bytes(value)
        target = root / path
        target.write_bytes(raw)
        target.chmod(0o600)
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}

    recipe = recovered["recipe"]
    recipe["previous_completion_ref"] = None
    recipe["target_trade_date"] = "20260824"
    recipe["bootstrap_ref"] = rewrite("bootstrap.json", declaration)
    recovered["handoff"]["recipe_ref"] = rewrite("fixtures/recipe.json", recipe)
    recovered["handoff_ref"] = rewrite(recovered["handoff_ref"]["path"], recovered["handoff"])

    def native_proof(**kw):
        value = read_bootstrap_declaration(workspace=str(root), recipe=kw["recovered"]["recipe"])
        return {
            "declaration": value,
            "previous_trade_date": "20260821",
            "ordered_open_dates": ["20260821", "20260824"],
            "execution_authorized": False,
        }

    monkeypatch.setattr(materializer.bootstrap_evidence, "verify_bootstrap_native", native_proof)
    return journal, recovered, book


def test_bootstrap_materializes_native_store_plan_and_replays_after_store_advance(
    tmp_path, monkeypatch
):
    journal, recovered, book = bootstrap_context(tmp_path, monkeypatch)
    with journal.locked():
        first = materializer.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    assert first.inputs.previous_trade_date == "20260821"
    (book.root / "_record_store/current.v1.json").write_bytes(b"advanced Store")
    monkeypatch.setattr(
        materializer,
        "prepare_materialized_store_plan",
        lambda **kw: pytest.fail("replanned bootstrap"),
    )
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with journal.locked():
        again = materializer.materialize_locked(
            journal=journal, recovered=recovered, auxiliary={"stages": {}}
        )
    assert again.materialization_ref == first.materialization_ref and again.inputs == first.inputs
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_bootstrap_existing_predecessor_rejected_before_input_publication(tmp_path, monkeypatch):
    journal, recovered, _ = bootstrap_context(tmp_path, monkeypatch)
    with journal.locked():
        journal.storage.write(str(journal.root.parent / "20260821" / "completion.v1.json"), b"{}")
        before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        with pytest.raises(ContractError, match="BOOTSTRAP_EXISTING_EOD"):
            materializer.materialize_locked(
                journal=journal, recovered=recovered, auxiliary={"stages": {}}
            )
        assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("version", [1, 2])
def test_materialization_conflicting_version_prevents_writes(tmp_path, monkeypatch, version):
    journal, recovered, _ = context(tmp_path, version)
    other = 2 if version == 1 else 1
    path = journal.root / "executions" / ("a" * 64) / f"materialization.v{other}.json"
    with journal.locked():
        journal.storage.write(str(path), b"{}")

        def forbidden(*args, **kwargs):
            pytest.fail("conflicting version reached writer")

        monkeypatch.setattr(journal.storage, "write", forbidden)
        with pytest.raises(ContractError, match="VERSION_CONFLICT"):
            materializer.materialize_locked(
                journal=journal, recovered=recovered, auxiliary={"stages": {}}
            )


def test_acquisition_without_handoff_is_provider_free_and_writes_nothing(tmp_path, monkeypatch):
    from quant_investor.operations import theme_handoff_publish

    journal, recovered, _ = context(tmp_path, 2)
    recipe = recovered["recipe"]
    acquisition = {
        "schema_version": "cn-daily-theme-acquisition.v1",
        "provider_priority": ["TUSHARE_DC", "TUSHARE_TDX"],
        "fallback_mode": "NATIVE_DC_PARTITION_FALLBACK",
        "maximum_companies": 100,
    }
    policy_raw = canonical_json_bytes(acquisition)
    (tmp_path / "policy.json").write_bytes(policy_raw)
    (tmp_path / "policy.json").chmod(0o600)
    recipe["theme_acquisition_ref"] = {
        "path": "policy.json",
        "sha256": hashlib.sha256(policy_raw).hexdigest(),
    }
    recipe["research_sources"]["theme_source_ref"] = None
    recipe_raw = canonical_json_bytes(recipe)
    (tmp_path / recovered["handoff"]["recipe_ref"]["path"]).write_bytes(recipe_raw)
    recovered["handoff"]["recipe_ref"]["sha256"] = hashlib.sha256(recipe_raw).hexdigest()
    handoff_raw = canonical_json_bytes(recovered["handoff"])
    (tmp_path / recovered["handoff_ref"]["path"]).write_bytes(handoff_raw)
    recovered["handoff_ref"]["sha256"] = hashlib.sha256(handoff_raw).hexdigest()

    def forbidden(*args, **kwargs):
        pytest.fail("missing handoff reached producer or writer")

    monkeypatch.setattr(theme_handoff_publish, "publish_theme_handoff", forbidden)
    with journal.locked():
        monkeypatch.setattr(journal.storage, "write", forbidden)
        with pytest.raises(ContractError, match="THEME_HANDOFF_REQUIRED"):
            materializer.materialize_locked(
                journal=journal, recovered=recovered, auxiliary={"stages": {}}
            )
