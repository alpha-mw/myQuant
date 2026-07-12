from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

import quant_investor.market.fundamental_generation as fundamental_generation
import quant_investor.market.fundamental_mart as fundamental_mart
from quant_investor.factors.pit_fundamentals import (
    build_fundamental_metric_matrices,
    load_fundamental_pit_series,
)
from quant_investor.market.fundamental_mart import (
    DERIVED_DAILY_FIELDS,
    FundamentalReadinessError,
    _fetch_tushare_tables,
    _resolve_symbols_from_parquet_universe,
    run_cn_fundamental_maintenance,
    write_fundamental_mart,
)
from quant_investor.market.fundamental_generation import (
    load_fundamental_pointer,
    publish_fundamental_generation,
)


def _raw_tables() -> dict[str, pd.DataFrame]:
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    fina_rows = []
    income_rows = []
    balance_rows = []
    cashflow_rows = []
    daily_rows = []
    forecast_rows = []
    for idx, symbol in enumerate(symbols, start=1):
        sector = ["bank", "industrial", "healthcare"][idx - 1]
        for end_date, ann_date, profit, ocf, capex, roe in (
            ("20221231", "20230428", 80.0 + idx, 100.0 + idx, 10.0, 8.0 + idx),
            ("20231231", "20240430", 100.0 + idx, 130.0 + idx, 20.0, 12.0 + idx),
        ):
            fina_rows.append(
                {
                    "ts_code": symbol,
                    "end_date": end_date,
                    "ann_date": ann_date,
                    "f_ann_date": ann_date,
                    "roe_dt": roe,
                    "roa": 5.0 + idx,
                    "debt_to_assets": 45.0 + idx,
                    "netprofit_yoy": "",
                    "ocf_to_profit": "",
                }
            )
            income_rows.append(
                {
                    "ts_code": symbol,
                    "end_date": end_date,
                    "ann_date": ann_date,
                    "f_ann_date": ann_date,
                    "n_income_attr_p": profit,
                }
            )
            balance_rows.append(
                {
                    "ts_code": symbol,
                    "end_date": end_date,
                    "ann_date": ann_date,
                    "f_ann_date": ann_date,
                    "total_liab": 400.0 + idx,
                    "total_assets": 1000.0 + idx,
                }
            )
            cashflow_rows.append(
                {
                    "ts_code": symbol,
                    "end_date": end_date,
                    "ann_date": ann_date,
                    "f_ann_date": ann_date,
                    "n_cashflow_act": ocf,
                    "c_pay_acq_const_fiolta": capex,
                }
            )
        for trade_date in ("20240429", "20240430", "20240502", "20240510"):
            daily_rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date,
                    "total_mv": 100000.0 * idx,
                    "sector": sector,
                }
            )
        forecast_rows.append(
            {
                "ts_code": symbol,
                "ann_date": "20240429",
                "end_date": "20240630",
                "type": "预增",
                "p_change_min": 5.0 + idx,
                "p_change_max": 15.0 + idx,
                "summary": "fixture forecast",
                "change_reason": "fixture",
            }
        )
    fina_rows.append(
        {
            "ts_code": "000001.SZ",
            "end_date": "20231231",
            "ann_date": "20240510",
            "f_ann_date": "20240510",
            "roe_dt": 20.0,
            "roa": 7.0,
            "debt_to_assets": 40.0,
        }
    )
    fina_rows.append(
        {
            "ts_code": "000004.SZ",
            "end_date": "20231231",
            "roe_dt": 10.0,
        }
    )
    return {
        "fina_indicator": pd.DataFrame(fina_rows),
        "income": pd.DataFrame(income_rows),
        "balancesheet": pd.DataFrame(balance_rows),
        "cashflow": pd.DataFrame(cashflow_rows),
        "daily_basic": pd.DataFrame(daily_rows),
        "forecast": pd.DataFrame(forecast_rows),
    }


def _write_parquet_market_data(root, symbols):
    data_root = root / "data"
    parquet_root = data_root / "parquet" / "cn"
    bars_root = parquet_root / "bars"
    serving_root = data_root / "parquet_serving" / "cn" / "bars"
    manifest_path = parquet_root / "_snapshots" / "fixture.json"
    rows = []
    for symbol in symbols:
        rows.append(
            {
                "ts_code": symbol,
                "trade_date": "20240510",
                "open": 10.0,
                "high": 10.5,
                "low": 9.9,
                "close": 10.2,
                "vol": 1000.0,
                "amount": 10000.0,
            }
        )
        serving_dir = serving_root / f"symbol={symbol}"
        serving_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([rows[-1]]).to_parquet(serving_dir / "bars.parquet", index=False)
    (bars_root / "year=2024").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(bars_root / "year=2024" / "part.parquet", index=False)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"snapshot_id": "fixture"}), encoding="utf-8")
    (parquet_root / "_latest.json").write_text(
        json.dumps(
            {
                "status": "OK",
                "snapshot_id": "fixture",
                "latest_complete_trade_date": "20240510",
                "latest_trade_date": "20240510",
                "table_root": str(bars_root),
                "derived_serving_root": str(serving_root),
                "manifest_path": str(manifest_path),
            }
        ),
        encoding="utf-8",
    )
    return data_root


def test_fundamental_mart_pit_join_readiness_and_quarantine(tmp_path):
    artifacts, readiness = write_fundamental_mart(
        _raw_tables(),
        data_root=tmp_path / "clean" / "cn_fundamental",
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="fixture",
    )

    daily = pd.read_parquet(artifacts.fundamental_daily_path)
    symbol_daily = daily[daily["ts_code"] == "000001.SZ"].set_index("trade_date")
    assert symbol_daily.loc["2024-04-29", "fin_roe"] == 0.09
    assert symbol_daily.loc["2024-04-30", "fin_roe"] == 0.13
    assert symbol_daily.loc["2024-05-02", "fin_roe"] == 0.13
    assert symbol_daily.loc["2024-05-10", "fin_roe"] == 0.20
    assert symbol_daily.loc["2024-04-30", "fin_net_profit_yoy"] > 0.0
    assert symbol_daily.loc["2024-04-30", "fin_fcf_to_profit"] > 0.0
    assert symbol_daily.loc["2024-04-30", "fcf_to_price"] > 0.0
    assert symbol_daily.loc["2024-04-30", "forecast_revision"] == 0.11
    assert readiness["gate2_passed"] is True
    assert readiness["coverage_rate"] >= 0.60
    quarantine = pd.read_parquet(artifacts.quarantine_path)
    assert "missing_ts_code_end_date_or_announcement_date" in set(quarantine["quarantine_reason"])
    assert artifacts.readiness_json_path.exists()
    assert json.loads(artifacts.readiness_json_path.read_text())["gate2_passed"] is True


def test_generation_pointer_carries_verified_source_priority(tmp_path):
    data_root = tmp_path / "clean" / "cn_fundamental"
    evidence_path = tmp_path / "tushare_readiness.json"
    evidence_path.write_text(
        json.dumps(
            {
                "provider_status": "live_tushare_partial",
                "provider_manifest": {"provider": "tushare"},
            }
        ),
        encoding="utf-8",
    )
    write_fundamental_mart(
        _raw_tables(),
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="verified-source",
        source="canonical_raw_offline_rebuild",
        provider_manifest={
            "source_priority": "tushare_primary",
            "source_provenance": "verified_local_tushare_refresh_manifests",
            "provenance_evidence": [str(evidence_path)],
        },
    )

    pointer = load_fundamental_pointer(data_root)
    assert pointer is not None
    assert pointer["metadata"]["source_priority"] == "tushare_primary"
    assert (
        pointer["metadata"]["source_provenance"]
        == "verified_local_tushare_refresh_manifests"
    )


def test_offline_generation_rejects_unverified_tushare_priority(tmp_path):
    data_root = tmp_path / "clean" / "cn_fundamental"

    with pytest.raises(ValueError, match="tushare_primary requires"):
        write_fundamental_mart(
            _raw_tables(),
            data_root=data_root,
            raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
            reports_root=tmp_path / "reports" / "fundamental_readiness",
            run_id="forged-source",
            source="canonical_raw_offline_rebuild",
            provider_manifest={
                "source_priority": "tushare_primary",
                "source_provenance": "unverified_caller_claim",
            },
        )

    assert load_fundamental_pointer(data_root) is None


def _only_symbol(
    tables: dict[str, pd.DataFrame],
    symbol: str,
) -> dict[str, pd.DataFrame]:
    return {
        name: (
            frame[frame["ts_code"] == symbol].copy()
            if "ts_code" in frame.columns
            else frame.copy()
        )
        for name, frame in tables.items()
    }


def test_partial_refresh_preserves_prior_symbols_in_new_generation(tmp_path):
    data_root = tmp_path / "clean" / "cn_fundamental"
    first, _ = write_fundamental_mart(
        _raw_tables(),
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="generation-one",
        source="first_source",
    )
    second, readiness = write_fundamental_mart(
        _only_symbol(_raw_tables(), "000001.SZ"),
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="generation-two",
        source="second_source",
    )

    daily = pd.read_parquet(second.fundamental_daily_path)
    pointer = load_fundamental_pointer(data_root)
    assert set(daily["ts_code"]) == {
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
    }
    assert pointer is not None
    assert pointer["generation_id"] == "generation-two"
    assert first.fundamental_daily_path.exists()
    assert readiness["merge"]["fundamental_daily"][
        "retained_existing_rows"
    ] > 0


def test_partial_exact_key_does_not_replace_more_complete_pit_row(tmp_path):
    data_root = tmp_path / "clean" / "cn_fundamental"
    write_fundamental_mart(
        _raw_tables(),
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="complete-generation",
        source="complete_source",
    )
    partial = _only_symbol(_raw_tables(), "000001.SZ")
    partial["fina_indicator"] = partial["fina_indicator"].assign(
        roa=None,
        debt_to_assets=None,
    )
    artifacts, _ = write_fundamental_mart(
        partial,
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="partial-generation",
        source="partial_source",
    )

    period = pd.read_parquet(artifacts.fundamental_period_path)
    row = period[
        (period["ts_code"] == "000001.SZ")
        & (period["end_date"] == "20231231")
        & (
            pd.to_datetime(period["availability_date"])
            == pd.Timestamp("2024-04-30")
        )
    ].iloc[-1]
    assert row["fin_roa"] == 0.06
    assert row["fin_debt_to_assets"] > 0.0
    assert row["source"] == "complete_source"


def test_complete_exact_key_refresh_wins_on_quality_tie(tmp_path):
    data_root = tmp_path / "clean" / "cn_fundamental"
    write_fundamental_mart(
        _raw_tables(),
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="old-generation",
        source="old_source",
    )
    refreshed = _only_symbol(_raw_tables(), "000001.SZ")
    target = (
        (refreshed["fina_indicator"]["end_date"] == "20231231")
        & (refreshed["fina_indicator"]["ann_date"] == "20240430")
    )
    refreshed["fina_indicator"].loc[target, "roe_dt"] = 18.0
    artifacts, _ = write_fundamental_mart(
        refreshed,
        data_root=data_root,
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="new-generation",
        source="new_source",
    )

    period = pd.read_parquet(artifacts.fundamental_period_path)
    row = period[
        (period["ts_code"] == "000001.SZ")
        & (period["end_date"] == "20231231")
        & (
            pd.to_datetime(period["availability_date"])
            == pd.Timestamp("2024-04-30")
        )
    ].iloc[-1]
    assert row["fin_roe"] == 0.18
    assert row["source"] == "new_source"


def test_generation_pointer_failure_cleans_final_directory_for_retry(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "cn"
    tables = {
        "fundamental_period": pd.DataFrame(columns=["ts_code"]),
        "fundamental_daily": pd.DataFrame(
            [{"ts_code": "000001.SZ", "trade_date": "20240510"}]
        ),
        "fundamental_quarantine": pd.DataFrame(columns=["ts_code"]),
    }
    original_write = fundamental_generation._atomic_write_json
    pointer_failed = False

    def fail_pointer_once(path, payload):
        nonlocal pointer_failed
        if (
            path.name == fundamental_generation.FUNDAMENTAL_POINTER_FILENAME
            and not pointer_failed
        ):
            pointer_failed = True
            raise OSError("simulated pointer write failure")
        return original_write(path, payload)

    monkeypatch.setattr(
        fundamental_generation,
        "_atomic_write_json",
        fail_pointer_once,
    )
    with pytest.raises(OSError, match="simulated pointer write failure"):
        publish_fundamental_generation(
            root=root,
            run_id="retryable-generation",
            tables=tables,
            metadata={"run_id": "retryable-generation"},
        )
    assert not (
        root
        / fundamental_generation.FUNDAMENTAL_GENERATIONS_DIRNAME
        / "retryable-generation"
    ).exists()

    monkeypatch.setattr(
        fundamental_generation,
        "_atomic_write_json",
        original_write,
    )
    paths, pointer = publish_fundamental_generation(
        root=root,
        run_id="retryable-generation",
        tables=tables,
        metadata={"run_id": "retryable-generation"},
    )

    assert pointer["generation_id"] == "retryable-generation"
    assert all(path.exists() for path in paths.values())


def test_offline_canonical_rebuild_can_skip_duplicate_raw_csv_snapshots(
    tmp_path,
):
    raw_snapshot_root = tmp_path / "snapshots" / "fundamental"
    artifacts, readiness = write_fundamental_mart(
        _raw_tables(),
        data_root=tmp_path / "cn",
        raw_snapshot_root=raw_snapshot_root,
        reports_root=tmp_path / "reports",
        run_id="offline-rebuild",
        source="canonical_raw_offline_rebuild",
        write_raw_snapshots=False,
    )

    assert artifacts.fundamental_daily_path.exists()
    assert readiness["raw_snapshot_written"] is False
    assert not list(raw_snapshot_root.rglob("*.csv"))


def test_disjoint_symbol_recovery_uses_append_merge_path(monkeypatch):
    existing = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240510",
                "end_date": 20231231,
                "availability_date": pd.Timestamp("2024-04-30").date(),
                "fin_roe": 0.10,
            }
        ]
    )
    incoming = pd.DataFrame(
        [
            {
                "ts_code": "000002.SZ",
                "trade_date": "20240510",
                "end_date": "20231231",
                "availability_date": pd.Timestamp("2024-04-30"),
                "fin_roe": 0.20,
            }
        ]
    )

    def fail_slow_key_path(*_args, **_kwargs):
        raise AssertionError("disjoint recovery must not normalize every row")

    monkeypatch.setattr(
        fundamental_mart,
        "_merge_key_values",
        fail_slow_key_path,
    )
    merged, stats = fundamental_mart._merge_fundamental_table(
        existing,
        incoming,
        key_fields=("ts_code", "trade_date", "end_date"),
        quality_fields=("fin_roe",),
    )

    assert set(merged["ts_code"]) == {"000001.SZ", "000002.SZ"}
    assert pd.api.types.is_integer_dtype(merged["end_date"])
    assert all(isinstance(value, date) for value in merged["availability_date"])
    assert stats["merge_path"] == "disjoint_symbol_append"


def test_empty_refresh_retains_existing_without_rowwise_key_normalization(
    monkeypatch,
):
    existing = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240510",
                "fin_roe": 0.10,
            }
        ]
    )

    def fail_slow_key_path(*_args, **_kwargs):
        raise AssertionError("empty refresh must not normalize existing rows")

    monkeypatch.setattr(
        fundamental_mart,
        "_merge_key_values",
        fail_slow_key_path,
    )
    merged, stats = fundamental_mart._merge_fundamental_table(
        existing,
        pd.DataFrame(),
        key_fields=("ts_code", "trade_date"),
        quality_fields=("fin_roe",),
    )

    pd.testing.assert_frame_equal(merged, existing)
    assert stats["merge_path"] == "retain_existing_no_incoming"


def test_quarantine_merge_aligns_existing_canonical_schema():
    existing = pd.DataFrame(
        [{"ts_code": "000001.SZ", "end_date": 20231231, "reason": "old"}]
    )
    incoming = pd.DataFrame(
        [{"ts_code": "000002.SZ", "end_date": "20240331", "reason": "new"}]
    )

    merged, _stats = fundamental_mart._merge_quarantine_table(
        existing,
        incoming,
    )

    assert pd.api.types.is_integer_dtype(merged["end_date"])


def test_readiness_symbol_coverage_is_capped_and_reports_scope_surplus():
    period = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "availability_date": "2024-04-30"},
            {"ts_code": "000002.SZ", "availability_date": "2024-04-30"},
        ]
    )
    daily = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240510",
                **{field: 1.0 for field in DERIVED_DAILY_FIELDS},
                "sector": "bank",
                "size_bucket": "large",
            },
            {
                "ts_code": "000002.SZ",
                "trade_date": "20240510",
                **{field: 1.0 for field in DERIVED_DAILY_FIELDS},
                "sector": "health",
                "size_bucket": "small",
            },
        ]
    )

    readiness = fundamental_mart.build_readiness_payload(
        daily,
        period,
        pd.DataFrame(),
        run_id="scope-surplus",
        expected_symbol_count=1,
    )

    assert readiness["symbol_coverage_rate"] == 1.0
    assert readiness["symbol_scope_surplus_count"] == 1


def test_readiness_requires_explicit_expected_scope_when_requested():
    readiness = fundamental_mart.build_readiness_payload(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        run_id="missing-scope",
        require_expected_symbol_scope=True,
    )

    assert readiness["gate2_passed"] is False
    assert readiness["symbol_coverage_rate"] == 0.0
    assert "expected_symbol_scope_missing" in readiness["blockers"]


def test_legacy_fetched_at_fallback_disabled_for_production(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "period": "20231231",
                "metric_name": "operating_cashflow",
                "value": 100.0,
                "fetched_at": "2024-05-10",
            }
        ]
    ).to_parquet(metadata_dir / "fundamental_series.parquet", index=False)

    production = load_fundamental_pit_series(
        metadata_dir=metadata_dir,
        mart_root=tmp_path / "missing_mart",
        allow_legacy_fallback=False,
    )
    diagnostic = load_fundamental_pit_series(
        metadata_dir=metadata_dir,
        mart_root=tmp_path / "missing_mart",
        allow_legacy_fallback=True,
    )

    assert production.empty
    assert not diagnostic.empty
    assert diagnostic["source"].str.contains("fetched_at_fallback").any()


def test_metric_matrices_read_canonical_mart_without_legacy(tmp_path):
    artifacts, _readiness = write_fundamental_mart(
        _raw_tables(),
        data_root=tmp_path / "clean" / "cn_fundamental",
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="fixture",
    )
    dates = pd.to_datetime(["2024-04-29", "2024-04-30"])
    matrices, diagnostics = build_fundamental_metric_matrices(
        dates,
        ["000001.SZ"],
        metrics=DERIVED_DAILY_FIELDS,
        mart_root=artifacts.data_root,
        allow_legacy_fallback=False,
    )

    assert diagnostics["legacy_fallback_allowed"] is False
    assert matrices["fin_roe"].loc[pd.Timestamp("2024-04-29"), "000001.SZ"] == 0.09
    assert matrices["fin_roe"].loc[pd.Timestamp("2024-04-30"), "000001.SZ"] == 0.13
    assert matrices["fcf_to_price"].loc[pd.Timestamp("2024-04-30"), "000001.SZ"] > 0.0


def test_daily_basic_uses_local_stock_list_sector(tmp_path, monkeypatch):
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "industry": "银行"},
            {"ts_code": "000002.SZ", "industry": "全国地产"},
            {"ts_code": "000003.SZ", "industry": "医疗服务"},
        ]
    ).to_parquet(metadata_root / "stock_list.parquet", index=False)
    monkeypatch.setattr(fundamental_mart, "DEFAULT_METADATA_ROOT", metadata_root)
    tables = _raw_tables()
    tables["daily_basic"] = tables["daily_basic"].drop(columns=["sector"])

    artifacts, _readiness = write_fundamental_mart(
        tables,
        data_root=tmp_path / "clean" / "cn_fundamental",
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
        run_id="fixture",
    )

    daily = pd.read_parquet(artifacts.fundamental_daily_path)
    sector_by_symbol = daily.groupby("ts_code")["sector"].first().to_dict()
    assert sector_by_symbol["000001.SZ"] == "银行"
    assert sector_by_symbol["000002.SZ"] == "全国地产"


def test_fundamental_maintenance_offline_input_writes_expected_artifacts(
    tmp_path,
    monkeypatch,
):
    market_data_root = _write_parquet_market_data(
        tmp_path,
        ["000001.SZ", "000002.SZ", "000003.SZ"],
    )
    monkeypatch.setattr(
        fundamental_mart,
        "DEFAULT_MARKET_DATA_ROOT",
        market_data_root,
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for table, frame in _raw_tables().items():
        frame.to_parquet(raw_dir / f"{table}.parquet", index=False)

    result = run_cn_fundamental_maintenance(
        market="CN",
        universes="full_a",
        years=5,
        as_of="20240510",
        raw_input_dir=raw_dir,
        data_root=tmp_path / "clean" / "cn_fundamental",
        raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
        reports_root=tmp_path / "reports" / "fundamental_readiness",
    )

    assert result["provider_status"] == "offline_input"
    assert result["readiness"]["gate2_passed"] is True
    assert result["readiness"]["expected_symbol_count"] == 3
    assert result["readiness"]["raw_row_counts"]["fina_indicator"] > 0
    assert (tmp_path / "clean" / "cn_fundamental" / "latest_manifest.json").exists()
    manifest = json.loads((tmp_path / "clean" / "cn_fundamental" / "latest_manifest.json").read_text())
    assert manifest["raw_row_counts"]["daily_basic"] > 0


def test_offline_partial_scope_fails_closed_without_publishing_pointer(
    tmp_path,
    monkeypatch,
):
    market_data_root = _write_parquet_market_data(
        tmp_path,
        ["000001.SZ", "000002.SZ", "000003.SZ"],
    )
    monkeypatch.setattr(
        fundamental_mart,
        "DEFAULT_MARKET_DATA_ROOT",
        market_data_root,
    )
    reports_root = tmp_path / "reports" / "fundamental_readiness"
    data_root = tmp_path / "clean" / "cn_fundamental"

    with pytest.raises(FundamentalReadinessError) as exc_info:
        run_cn_fundamental_maintenance(
            market="CN",
            universes="full_a",
            as_of="20240510",
            raw_tables=_only_symbol(_raw_tables(), "000001.SZ"),
            data_root=data_root,
            raw_snapshot_root=tmp_path / "snapshots" / "fundamental",
            reports_root=reports_root,
        )

    readiness = exc_info.value.readiness
    assert readiness["gate2_passed"] is False
    assert readiness["symbol_coverage_rate"] == pytest.approx(1 / 3)
    assert "symbol_coverage_below_95pct" in readiness["blockers"]
    assert list(reports_root.glob("cn_fundamental_20240510_*.json"))
    assert load_fundamental_pointer(data_root) is None


def test_full_a_universe_resolves_from_parquet_serving_inventory(tmp_path):
    data_root = _write_parquet_market_data(
        tmp_path,
        ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
    )

    symbols = _resolve_symbols_from_parquet_universe(data_root, ["full_a"])

    assert symbols == ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]


def test_full_a_universe_intersects_canonical_components_with_serving(tmp_path):
    data_root = _write_parquet_market_data(
        tmp_path,
        ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
    )
    components_path = data_root / "cn_universe" / "cn_index_components.json"
    components_path.parent.mkdir(parents=True)
    components_path.write_text(
        json.dumps(
            {
                "full_a": ["000001.SZ", "000002.SZ", "000003.SZ"],
            }
        ),
        encoding="utf-8",
    )

    symbols = _resolve_symbols_from_parquet_universe(data_root, ["full_a"])

    assert symbols == ["000001.SZ", "000002.SZ", "000003.SZ"]


def test_live_fetch_records_partial_provider_errors(monkeypatch):
    class _FakePro:
        def __init__(self):
            self.calls = []

        def fina_indicator(self, ts_code, start_date="", end_date="", fields=""):
            self.calls.append(("fina_indicator", ts_code, start_date, end_date))
            if ts_code == "000002.SZ":
                raise RuntimeError("quota limited")
            return pd.DataFrame(
                [
                    {
                        "ts_code": ts_code,
                        "end_date": "20231231",
                        "ann_date": "20240430",
                        "roe": 10.0,
                    }
                ]
            )

        def income(self, **kwargs):
            self.calls.append(("income", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

        def balancesheet(self, **kwargs):
            self.calls.append(("balancesheet", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

        def cashflow(self, **kwargs):
            self.calls.append(("cashflow", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

        def daily_basic(self, **kwargs):
            self.calls.append(("daily_basic", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

        def forecast(self, **kwargs):
            self.calls.append(("forecast", kwargs.get("ts_code"), kwargs.get("start_date"), kwargs.get("end_date")))
            return pd.DataFrame()

    monkeypatch.setattr("quant_investor.market.fundamental_mart.time.sleep", lambda _seconds: None)
    pro = _FakePro()

    tables, manifest = _fetch_tushare_tables(
        ["000001.SZ", "000002.SZ"],
        years=5,
        as_of="20240510",
        workers=4,
        pro=pro,
    )

    assert len(tables["fina_indicator"]) == 1
    assert manifest["requests_failed"] == 1
    assert manifest["errors"][0]["symbol"] == "000002.SZ"
    assert manifest["raw_row_counts"]["fina_indicator"] == 1
    first_start_by_table = {}
    for table, _symbol, start_date, _end_date in pro.calls:
        first_start_by_table.setdefault(table, start_date)
    assert first_start_by_table["daily_basic"] == "20190510"
    assert first_start_by_table["fina_indicator"] == "20170510"
    assert first_start_by_table["income"] == "20170510"
    assert first_start_by_table["forecast"] == "20170510"
    assert manifest["daily_start_date"] == "20190510"
    assert manifest["financial_start_date"] == "20170510"
