"""Exercise Store materialization through its real native plan-only producer."""

from pathlib import Path
import sys
import hashlib
import json
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _native_daily_store_fixture import NativeStoreFixture
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.daily_contract import ContractError
from scripts.daily_store_materialization import prepare_materialized_store_plan


def context(root):
    book = NativeStoreFixture(root)
    arguments = book.advance("2026-08-24")

    def ref(path):
        return {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def put(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(value))
        path.chmod(0o600)
        return ref(path)

    preimages = {
        "store_pointer_ref": ref(book.root / "_record_store/current.v1.json"),
        "event_pointer_ref": ref(book.root / "_event_store/current.v1.json"),
        "benchmark_pointer_ref": ref(root / "data/parquet/cn/benchmarks/_latest.json"),
    }
    recipe = {
        "schema_version": "cn-daily-execute-recipe.v1",
        "store_preimages": preimages,
        "policy_refs": {
            "store": {"path": arguments["policy_path"], "sha256": arguments["policy_sha"]}
        },
        "retrospective_ref": None,
    }
    recipe_ref = put("fixtures/recipe.json", recipe)
    handoff = {
        "trade_date": "20260824",
        "recipe_ref": recipe_ref,
        "market_pointer_ref": ref(root / "data/parquet/cn/_latest.json"),
        "calendar_ref": ref(book.calendar_path),
    }
    recovered = {
        "handoff": handoff,
        "recipe": recipe,
        "handoff_ref": put("fixtures/handoff.json", handoff),
    }
    return DailyJournal(str(root), "20260824"), recovered, book, arguments


def test_real_native_plan_preserves_preimages_and_does_not_commit_store(tmp_path):
    journal, recovered, book, arguments = context(tmp_path)
    pointer = book.root / "_record_store/current.v1.json"
    before = pointer.read_bytes()
    with journal.locked():
        result = prepare_materialized_store_plan(journal=journal, recovered=recovered)
    assert result["store_arguments"] == arguments
    assert result["execution_authorized"] is False
    assert (
        result["native_plan"]["preimages"]["store_pointer_sha256"]
        == arguments["expected_store_pointer_sha"]
    )
    plan = tmp_path / result["store_plan_ref"]["path"]
    assert hashlib.sha256(plan.read_bytes()).hexdigest() == result["store_plan_ref"]["sha256"]
    assert pointer.read_bytes() == before
    with journal.locked():
        again = prepare_materialized_store_plan(journal=journal, recovered=recovered)
    assert again["store_plan_ref"] == result["store_plan_ref"]
    assert pointer.read_bytes() == before


def test_recipe_preimage_drift_blocks_native_plan(tmp_path):
    journal, recovered, book, _ = context(tmp_path)
    pointer = book.root / "_record_store/current.v1.json"
    before = pointer.read_bytes()
    event_ref = recovered["recipe"]["store_preimages"]["event_pointer_ref"]
    (tmp_path / event_ref["path"]).write_bytes(b"changed")
    with journal.locked():
        with pytest.raises(ContractError, match="SOURCE_SHA_MISMATCH"):
            prepare_materialized_store_plan(journal=journal, recovered=recovered)
    assert pointer.read_bytes() == before


def test_held_adjustment_refs_use_frozen_market_and_exact_native_ledger(tmp_path):
    from scripts.daily_store_materialization import materialized_adjustment_refs
    from quant_investor.market.market_data_reader import MarketDataReader

    journal, recovered, book, _ = context(tmp_path)
    market = MarketDataReader(market="CN", data_root=tmp_path / "data", mode_policy="strict")
    manifest = Path(market.snapshot()["manifest_path"])
    snapshot_ref = {
        "path": str(manifest.relative_to(tmp_path)),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    recovered["handoff"]["market_snapshot_ref"] = snapshot_ref
    raw = canonical_json_bytes(recovered["handoff"])
    handoff_path = tmp_path / recovered["handoff_ref"]["path"]
    handoff_path.write_bytes(raw)
    recovered["handoff_ref"]["sha256"] = hashlib.sha256(raw).hexdigest()
    before = (book.root / "_record_store/current.v1.json").read_bytes()
    with journal.locked():
        prepared = prepare_materialized_store_plan(journal=journal, recovered=recovered)
        # Subsequent source capture must not consult the mutable Market alias.
        (tmp_path / "data/parquet/cn/_latest.json").write_bytes(b"changed current head")
        refs = materialized_adjustment_refs(journal=journal, recovered=recovered, prepared=prepared)
    assert set(refs) == set(book.stocks)
    for ref in refs.values():
        assert hashlib.sha256((tmp_path / ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    assert (book.root / "_record_store/current.v1.json").read_bytes() == before
