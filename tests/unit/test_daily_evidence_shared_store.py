"""Store joins existing Factor Market without replacing its pointer/PIT/bars."""

import hashlib
from _native_daily_factor_fixture import NativeFactorInputs, strict_market_from_factor_inputs
from _native_daily_store_fixture import NativeStoreFixture, DAYS
from scripts.daily_production_store_adapter import prepare_store_plan, StoreCloseAdapter


def test_native_store_consumes_shared_market_without_mutating_it(tmp_path):
    source = NativeFactorInputs(tmp_path / "factor-inputs", count=10)
    reader, pit = strict_market_from_factor_inputs(
        tmp_path, source.day(3), pit_observed_at="2026-08-27T00:00:00Z"
    )
    data = tmp_path / "data"
    protected = [
        data / "parquet/cn/_latest.json",
        data / "cn_universe/cn_index_components.json",
        *list((data / "parquet/cn/reference").rglob("*")),
        *list((data / "parquet/cn/_snapshots").rglob("*")),
    ]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    fixture = NativeStoreFixture(tmp_path, stock_symbols=source.symbols[:7], preserve_market=True)
    for day in DAYS[:4]:
        args = fixture.advance(day)
    plan = prepare_store_plan(args)
    assert plan["missing_dates"] == DAYS[:4]
    adapter = StoreCloseAdapter(
        arguments=args,
        trade_date="20260827",
        plan_ref={"path": plan["plan_path"], "sha256": plan["plan_sha256"]},
        release_ref={
            "path": "SYNTHETIC-FIXTURE.json",
            "sha256": hashlib.sha256(
                (tmp_path / "SYNTHETIC-FIXTURE.json").read_bytes()
            ).hexdigest(),
        },
    )
    adapter.execute(adapter.template())
    assert adapter.probe(adapter.template()).outcome.state.value == "SUCCEEDED"
    assert before == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()
    }
    assert reader.snapshot()["latest_complete_trade_date"] == "20260827"
